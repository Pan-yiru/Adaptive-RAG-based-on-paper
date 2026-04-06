"""
Adaptive RAG: 自适应检索增强生成系统
根据问题类型自动选择最优策略：
- 事实题：使用 Baseline RAG（直接检索）
- 复杂题/推理题：使用 Advanced RAG（问题分解 + Rerank）
"""

import os
import re
import json
import tempfile
import shutil
from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv
import dashscope
import requests
import numpy as np

from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ============================================================
# 1. 基础配置
# ============================================================

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


class QwenEmbeddings(Embeddings):
    """通义千问 Embedding 类"""
    def __init__(self, model="text-embedding-v4", api_key=None, base_url=None, batch_size=10):
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def embed_documents(self, texts):
        texts = [text.strip() for text in texts if text.strip()]
        if not texts:
            return []
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            completion = self.client.embeddings.create(model=self.model, input=batch_texts)
            embeddings.extend([item.embedding for item in completion.data])
        return embeddings

    def embed_query(self, text):
        text = text.strip()
        if not text:
            return []
        completion = self.client.embeddings.create(model=self.model, input=text)
        return completion.data[0].embedding


# ============================================================
# 2. 文档处理

class DocumentProcessor:
    """文档处理器：加载、清洗、切分"""

    @staticmethod
    def load_pdf(url: str) -> List:
        """从 URL 加载 PDF"""
        try:
            print(f"正在加载 PDF: {url}")
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.write(response.content)
            temp_path = temp_file.name
            temp_file.close()

            try:
                loader = PyMuPDFLoader(temp_path)
                docs = loader.load()
                print(f"✅ 加载完成，共 {len(docs)} 页")
                return docs
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            print(f"❌ PDF 加载失败: {e}")
            return []

    @staticmethod
    def clean_docs(docs: List) -> List:
        """清洗文档内容"""
        cleaned = []
        for doc in docs:
            content = doc.page_content
            content = re.sub(r'Page \d+ of \d+', '', content)
            content = re.sub(r'arXiv:\d+\.\d+v\d+ \[cs\.CL\] \d+ \w+ \d+', '', content)
            content = content.replace("-\n", "")
            content = re.sub(r'\n+', '\n', content).strip()

            if len(content) > 10:
                doc.page_content = content
                cleaned.append(doc)
        print(f"✅ 清洗完成，处理 {len(cleaned)} 页")
        return cleaned

    @staticmethod
    def split_docs(docs: List, chunk_size: int = 600, chunk_overlap: int = 60) -> List:
        """切分文档"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        print(f"✅ 切分完成，共 {len(chunks)} 个片段")
        return chunks


# ============================================================
# 3. 检索器创建

def create_retriever(chunks: List, weights: List[float] = [0.4, 0.6]) -> EnsembleRetriever:
    """创建混合检索器（BM25 + Dense）"""
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=QwenEmbeddings(
            model="text-embedding-v4",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            batch_size=10
        )
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6] # 给予向量检索稍高权重
    )

    print("✅ 混合检索器创建完成")
    return retriever


# ============================================================
# 4. 问题分类器

class QuestionClassifier:
    """问题类型分类器：LLM Router + 关键词混合策略"""

    # 复杂问题的关键词
    COMPLEX_KEYWORDS = [
        "为什么", "如何", "怎样", "原理", "机制", "区别", "差异",
        "对比", "比较", "分析", "影响", "关系", "优势", "劣势",
        "权衡", "优缺点", "适用场景"
    ]

    # 推理问题的关键词
    REASONING_KEYWORDS = [
        "如果", "假设", "推测", "推断", "可能", "应该", "会怎样",
        "场景", "情况下", "变化", "导致", "结果", "性能瓶颈"
    ]

    # 事实问题的关键词（高置信度）
    FACTUAL_KEYWORDS = [
        "多少", "是什么", "哪些", "几个", "参数", "公式", "取值",
        "配置", "层", "维度", "大小", "数量", "模型", "架构"
    ]

    def __init__(self, llm=None):
        """
        初始化分类器

        参数:
        - llm: 可选的 LLM 实例（用于语义路由）
        """
        self.llm = llm or ChatOpenAI(
            model_name="qwen-turbo",
            temperature=0,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            max_tokens=256
        )

        # LLM Router prompt
        self.router_prompt = ChatPromptTemplate.from_template(
            """你是一个专业的问题分类器。请分析用户问题并判断其类型。

问题类型定义：
1. simple (事实题): 可以直接从文档中找到明确答案的问题
   - 例如：参数取值、配置细节、具体数字、是什么等
   - 特征：答案在文档中有明确对应，不需要推理或综合

2. complex (复杂题): 需要综合多个信息点或深入理解的问题
   - 例如：为什么、如何、原理、区别、分析等
   - 特征：需要整合文档多处信息，或需要理解概念间的关系

3. reasoning (推理题): 需要假设、推断或超出文档明确内容的推理
   - 例如：假设场景、推测结果、推断影响等
   - 特征：包含假设词（如果、假设），或需要基于文档进行推理

问题：{question}

请严格按照以下 JSON 格式返回（不要输出其他内容）：
{{
  "question_type": "simple/complex/reasoning",
  "confidence": 0.0-1.0,
  "reasoning": "分类理由"
}}
"""
        )

        self.router_chain = self.router_prompt | self.llm | StrOutputParser()

    def _keyword_classify(self, question: str) -> tuple:
        """
        基于关键词的快速分类

        返回: (类型, 置信度)
        """
        # 检查推理关键词
        reasoning_count = sum(1 for kw in self.REASONING_KEYWORDS if kw in question)
        if reasoning_count >= 1:
            confidence = min(0.9, 0.5 + reasoning_count * 0.1)
            return "reasoning", confidence

        # 检查复杂关键词
        complex_count = sum(1 for kw in self.COMPLEX_KEYWORDS if kw in question)
        if complex_count >= 1:
            confidence = min(0.85, 0.5 + complex_count * 0.1)
            return "complex", confidence

        # 检查事实关键词（高置信度）
        factual_count = sum(1 for kw in self.FACTUAL_KEYWORDS if kw in question)
        if factual_count >= 1:
            confidence = min(0.95, 0.6 + factual_count * 0.1)
            return "simple", confidence

        # 问题长度作为判断依据（低置信度）
        if len(question) > 45:
            return "complex", 0.4

        # 默认为简单问题（低置信度）
        return "simple", 0.3

    def classify(self, question: str, use_llm_router: bool = True, confidence_threshold: float = 0.6) -> dict:
        """
        分类问题类型（混合策略）

        参数:
        - question: 用户问题
        - use_llm_router: 是否使用 LLM Router（默认 True）
        - confidence_threshold: 关键词分类的置信度阈值（低于此值时使用 LLM Router）

        返回:
        {
            "type": "simple/complex/reasoning",
            "confidence": 0.0-1.0,
            "method": "keyword/llm_router",
            "reasoning": "分类理由"
        }
        """
        # 1. 先尝试快速关键词分类
        keyword_type, keyword_confidence = self._keyword_classify(question)

        # 2. 如果不使用 LLM Router 或置信度足够高，直接返回关键词分类结果
        if not use_llm_router or keyword_confidence >= confidence_threshold:
            return {
                "type": keyword_type,
                "confidence": keyword_confidence,
                "method": "keyword",
                "reasoning": f"关键词匹配（置信度: {keyword_confidence:.2f}）"
            }

        # 3. 置信度低时，使用 LLM Router 进行语义分类
        try:
            print(f"  🔄 关键词分类置信度较低 ({keyword_confidence:.2f})，使用 LLM Router...")
            router_output = self.router_chain.invoke({"question": question})

            # 解析 JSON 输出
            import json
            import re

            # 尝试提取 JSON
            json_match = re.search(r'\{[^{}]*\}', router_output)
            if json_match:
                router_result = json.loads(json_match.group())

                return {
                    "type": router_result.get("question_type", keyword_type),
                    "confidence": router_result.get("confidence", 0.8),
                    "method": "llm_router",
                    "reasoning": router_result.get("reasoning", "LLM语义分析")
                }

        except Exception as e:
            print(f"  ⚠️ LLM Router 失败 ({e})，使用关键词分类结果")

        # 4. LLM Router 失败时，回退到关键词分类
        return {
            "type": keyword_type,
            "confidence": keyword_confidence,
            "method": "keyword_fallback",
            "reasoning": f"LLM Router 失败，回退到关键词分类"
        }

    def should_use_decomposition(self, question: str) -> bool:
        """判断是否需要问题分解"""
        result = self.classify(question)
        return result["type"] in ["complex", "reasoning"]


# ============================================================
# 5. Baseline RAG 策略

class BaselineRAG:
    """Baseline RAG: 直接检索策略（完全对齐 rag_baseline.ipynb）"""

    def __init__(self, chunks, llm=None):
        """
        初始化 Baseline RAG（对齐 rag_baseline.ipynb 逻辑）

        参数:
        - chunks: 切分后的文档块列表
        - llm: 可选的 LLM 实例
        """
        # 创建 Chroma 向量检索器
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=QwenEmbeddings(
                model="text-embedding-v4",
                api_key=os.environ["DASHSCOPE_API_KEY"],
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                batch_size=10
            )
        )

        # 创建检索器（只用向量检索，与 rag_baseline.ipynb 一致）
        self.retriever = self.vectorstore.as_retriever()

        # LLM 配置（与 rag_baseline.ipynb 一致）
        self.llm = llm or ChatOpenAI(
            model_name="qwen-turbo",
            temperature=0,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            max_tokens=1024
        )

        # Prompt 模板
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""你是一个严谨的学术助手。
        请严格根据下方提供的【参考资料】来回答用户的问题。
        要求：
        1. 只能使用资料中的事实，严禁发挥。
        2. 如果资料未提及，请回答"资料中未找到相关内容"。
        3. 涉及计算复杂度或具体数值（如 d_model）时，必须完全与原文一致
        参考资料：{context}
        待回答的问题：{question}
        请用专业、简洁的中文回答："""
        )

        # format_docs 函数
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Chain: Retriever -> Format -> Prompt -> LLM -> Output Parser
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def answer(self, question: str, top_k: int = 3) -> Tuple[str, List[str]]:
        """
        生成答案（完全对齐 rag_baseline.ipynb 逻辑）

        返回: (答案, 检索到的上下文列表)
        """
        # 先检索获取上下文
        docs = self.retriever.invoke(question)
        docs = [doc for doc in docs if getattr(doc, "page_content", "").strip()]
        retrieved_contexts = [doc.page_content for doc in docs]

        # 使用 LCEL chain 生成答案
        answer = self.chain.invoke(question)

        return answer, retrieved_contexts


# ============================================================
# 6. Advanced RAG 策略

class AdvancedRAG:
    """Advanced RAG: 问题分解 + Rerank 策略（适用于复杂题和推理题）"""

    def __init__(self, retriever, llm=None):
        self.retriever = retriever
        self.llm = llm or ChatOpenAI(
            model_name="qwen-turbo",
            temperature=0,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            max_tokens=1024
        )

        # 问题分解 prompt - 要求JSON格式输出
        self.decompose_prompt = ChatPromptTemplate.from_template(
            """你是一个擅长拆解复杂问题的助手。
    将问题分解为 3-5 个互补的子问题：
    1. 每个子问题应能独立检索
    2. 优先提取关键概念
    3. 包含同义词或相关术语
    4. 避免重复

    问题：{question}

    请以JSON数组格式返回子问题，例如：
    ["子问题1", "子问题2", "子问题3"]

    子问题："""
        )

        # 子问题答案生成 prompt
        self.subqa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""你是一个严谨的学术助手。
请根据以下参考资料回答问题。

参考资料：
{context}

问题：{question}

要求：
1. 只能使用参考资料中的事实
2. 如果资料未提及，请回答"资料中未找到相关内容"
3. 答案简洁明了

回答："""
        )

        # 最终答案生成 prompt
        self.final_prompt = ChatPromptTemplate.from_template(
            """你是最终答案整合助手。
请严格依据下面的子问题回答与参考文档进行整合，不要补充任何外部知识。

子问题回答：
{subqa_context}

参考文档：
{doc_context}

最终问题：{question}

要求：
- 答案简洁明了，1-2 句话
- 如果资料不足，明确说"我不知道"
- 保持逻辑连贯

最终回答："""
        )

        self.decompose_chain = self.decompose_prompt | self.llm | StrOutputParser()
        self.subqa_chain = self.subqa_prompt | self.llm | StrOutputParser()
        self.final_chain = self.final_prompt | self.llm | StrOutputParser()

    @staticmethod
    def _deduplicate_docs(docs: List) -> List:
        """按 page_content 去重并保持顺序"""
        unique_docs = []
        seen = set()
        for doc in docs:
            text = getattr(doc, "page_content", "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            unique_docs.append(doc)
        return unique_docs

    def _parse_sub_questions(self, raw_output: str) -> List[str]:
        """解析子问题"""
        candidates = []
        try:
            parsed = json.loads(raw_output)
            if isinstance(parsed, list):
                candidates = [str(item) for item in parsed]
        except:
            pass

        if not candidates:
            candidates = [line for line in str(raw_output).split("\n") if line.strip()]

        # 清理和去重
        cleaned, seen = [], set()
        for item in candidates:
            text = re.sub(r"^\s*(?:[-*•]|\d+[\.)])\s*", "", str(item)).strip()
            if text and text.lower() not in seen:
                seen.add(text.lower())
                cleaned.append(text)

        return cleaned[:6]

    def _rerank_documents(self, query: str, docs: List, top_n: int = 3) -> List:
        """使用 GTE-Rerank 对文档重排"""
        if not docs:
            return docs

        doc_texts = [d.page_content for d in docs]
        try:
            response = dashscope.TextReRank.call(
                model="gte-rerank-v2",
                query=query,
                documents=doc_texts,
                top_n=top_n
            )
            if response.status_code == 200:
                return [docs[item.index] for item in response.output.results]
        except Exception as e:
            print(f"⚠️ Rerank 失败: {e}")

        return docs[:top_n]

    def answer(self, question: str, top_k: int = 3) -> Tuple[str, List[str]]:
        """
        生成答案

        返回: (答案, 检索到的上下文列表)
        """
        # 1. 问题分解
        try:
            raw_sub_questions = self.decompose_chain.invoke({"question": question})
            sub_questions = self._parse_sub_questions(raw_sub_questions)
        except Exception as e:
            print(f"⚠️ 子问题生成失败，使用原问题: {e}")
            sub_questions = [question]

        if not sub_questions:
            sub_questions = [question]

        # 2. 对每个子问题检索并收集所有文档
        all_retrieved_docs = []
        seen_docs = set()

        for sub_question in sub_questions:
            # 检索
            retrieved_docs = self.retriever.invoke(sub_question)
            retrieved_docs = [doc for doc in retrieved_docs if getattr(doc, "page_content", "").strip()]

            # 收集所有去重文档
            for doc in retrieved_docs:
                text = doc.page_content.strip()
                if text and text not in seen_docs:
                    seen_docs.add(text)
                    all_retrieved_docs.append(doc)

        # 3. Rerank - 对所有检索到的文档进行重排，只保留top_k个最相关的
        reranked_docs = self._rerank_documents(question, all_retrieved_docs, top_n=top_k)

        # 4. 基于reranked后的文档生成子问题答案
        sub_answers = []
        doc_contexts = [doc.page_content for doc in reranked_docs]
        context_text = "\n\n".join(doc_contexts)

        for sub_question in sub_questions:
            if context_text:
                answer = self.subqa_chain.invoke({"context": context_text, "question": sub_question})
            else:
                answer = "资料中未找到相关内容"
            sub_answers.append(answer)

        # 5. 生成最终答案 - 子问题答案 + reranked文档
        subqa_context = "\n\n".join(
            [f"子问题 {i+1}: {sub_q}\n回答: {sub_a}" for i, (sub_q, sub_a) in enumerate(zip(sub_questions, sub_answers))]
        )
        doc_context = "\n\n".join([f"文档 {i+1}: {doc.page_content}" for i, doc in enumerate(reranked_docs)])
        final_answer = self.final_chain.invoke({
            "subqa_context": subqa_context,
            "doc_context": doc_context,
            "question": question,
        })

        # 6. 返回最终答案和reranked后的文档上下文（保持top_k数量）
        retrieved_contexts = [doc.page_content for doc in reranked_docs]

        return final_answer, retrieved_contexts


# ============================================================
# 7. Adaptive RAG 主类

class AdaptiveRAG:
    """自适应 RAG 系统：根据问题类型自动选择最优策略"""

    def __init__(self, pdf_url: str, chunk_size: int = 600, chunk_overlap: int = 60):
        """
        初始化 Adaptive RAG 系统

        参数:
        - pdf_url: PDF 文档 URL
        - chunk_size: 文档切分大小
        - chunk_overlap: 文档切分重叠大小
        """
        print("="*80)
        print("🚀 初始化 Adaptive RAG 系统")
        print("="*80)

        # 1. 加载和处理文档
        docs = DocumentProcessor.load_pdf(pdf_url)
        docs = DocumentProcessor.clean_docs(docs)
        chunks = DocumentProcessor.split_docs(docs, chunk_size, chunk_overlap)

        # 2. 创建检索器（用于 AdvancedRAG）
        self.retriever = create_retriever(chunks)

        # 3. 初始化两种策略
        # BaselineRAG 使用自己的 Chroma 向量检索（对齐 rag_baseline.ipynb）
        self.baseline = BaselineRAG(chunks)
        # AdvancedRAG 使用 EnsembleRetriever（BM25 + Dense）
        self.advanced = AdvancedRAG(self.retriever)

        # 4. 初始化问题分类器（LLM Router + 关键词混合策略）
        self.classifier = QuestionClassifier()

        # 5. 统计信息
        self.stats = {
            "simple": 0,
            "complex": 0,
            "reasoning": 0
        }

        print("✅ Adaptive RAG 系统初始化完成\n")

    def answer(self, question: str, top_k: int = 3, force_strategy: str = None) -> Tuple[str, str, List[str]]:
        """
        自适应回答问题

        参数:
        - question: 用户问题
        - top_k: 检索返回的文档数量
        - force_strategy: 强制使用指定策略 ("baseline" 或 "advanced")

        返回: (答案, 策略类型, 检索上下文)
        """
        # 判断问题类型（使用 LLM Router + 关键词混合策略）
        if force_strategy:
            strategy = force_strategy
            qtype = "advanced" if force_strategy == "advanced" else "simple"
            classification_method = "force"
        else:
            classification_result = self.classifier.classify(question)
            qtype = classification_result["type"]
            strategy = "baseline" if qtype == "simple" else "advanced"
            classification_method = classification_result["method"]

        # 更新统计
        self.stats[qtype] += 1

        # 选择策略
        if strategy == "baseline":
            print(f"📝 [Baseline] 事实题 ({classification_method}): {question[:50]}...")
            answer, contexts = self.baseline.answer(question, top_k)
        else:
            print(f"🔍 [Advanced] {qtype}题 ({classification_method}): {question[:50]}...")
            answer, contexts = self.advanced.answer(question, top_k)

        return answer, strategy, contexts

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {"simple": 0, "complex": 0, "reasoning": 0}


# ============================================================
# 8. 评测接口

class AdaptiveRAGEvaluator:
    """Adaptive RAG 评测器"""

    def __init__(self, adaptive_rag: AdaptiveRAG):
        self.rag = adaptive_rag

        # 评测模型
        self.evaluator_llm = ChatOpenAI(
            model_name="qwen-max",
            temperature=0,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            max_tokens=1024
        )

        self.evaluator_embeddings = QwenEmbeddings(
            model="text-embedding-v4",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            batch_size=10
        )

    def answer_for_eval(self, question: str) -> Tuple[str, List[str]]:
        """为评测生成答案"""
        answer, strategy, contexts = self.rag.answer(question)
        return answer, contexts

    def evaluate(self, questions: List, ground_truths: List) -> Dict:
        """
        运行评测（简化版，返回基本统计）

        完整评测使用 run_ragas_evaluation() 函数
        """
        results = []
        self.rag.reset_stats()

        for idx, (question, reference) in enumerate(zip(questions, ground_truths), start=1):
            print(f"[{idx}/{len(questions)}] {question[:50]}...")
            answer, strategy, contexts = self.rag.answer(question)

            results.append({
                "question": question,
                "answer": answer,
                "reference": reference,
                "strategy": strategy,
                "contexts_count": len(contexts)
            })

        stats = self.rag.get_stats()
        total = sum(stats.values())

        print("\n" + "="*80)
        print("📊 策略使用统计")
        print("="*80)
        print(f"事实题 (Baseline): {stats['simple']} ({stats['simple']/total*100:.1f}%)")
        print(f"复杂题 (Advanced): {stats['complex']} ({stats['complex']/total*100:.1f}%)")
        print(f"推理题 (Advanced): {stats['reasoning']} ({stats['reasoning']/total*100:.1f}%)")
        print("="*80)

        return {
            "results": results,
            "stats": stats,
            "total": total
        }


# ============================================================
# 9. Ragas 评测接口

def run_ragas_evaluation(adaptive_rag: AdaptiveRAG,
                         eval_questions: List,
                         eval_ground_truths: List,
                         evaluator_model_name: str = "qwen-max",
                         limit: Optional[int] = None,
                         indices: Optional[List] = None) -> any:
    """
    运行 Ragas 评测

    参数:
    - adaptive_rag: AdaptiveRAG 实例
    - eval_questions: 评测问题列表
    - eval_ground_truths: 标准答案列表
    - evaluator_model_name: 评测模型名称
    - limit: 限制评测数量
    - indices: 指定评测索引

    返回: 评测结果 DataFrame
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness, AnswerRelevancy,
        LLMContextRecall, LLMContextPrecisionWithReference,
        AnswerCorrectness
    )
    import pandas as pd

    # 评测器配置
    evaluator_llm = ChatOpenAI(
        model_name=evaluator_model_name,
        temperature=0,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        max_tokens=1024
    )

    evaluator_embeddings = QwenEmbeddings(
        model="text-embedding-v4",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        batch_size=10
    )

    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        AnswerCorrectness(llm=evaluator_llm),
    ]

    # 选择评测数据
    if indices is not None:
        selected_questions = [eval_questions[i] for i in indices]
        selected_truths = [eval_ground_truths[i] for i in indices]
    elif limit is not None:
        selected_questions = eval_questions[:limit]
        selected_truths = eval_ground_truths[:limit]
    else:
        selected_questions = eval_questions
        selected_truths = eval_ground_truths

    # 生成答案
    evaluator = AdaptiveRAGEvaluator(adaptive_rag)
    records = []

    adaptive_rag.reset_stats()

    for idx, (question, reference) in enumerate(zip(selected_questions, selected_truths), start=1):
        print(f"[{idx}/{len(selected_questions)}] {question[:50]}...")
        response, retrieved_contexts = evaluator.answer_for_eval(question)

        records.append({
            "user_input": question,
            "response": response,
            "reference": reference,
            "retrieved_contexts": retrieved_contexts,
        })

    # 运行评测
    dataset = Dataset.from_list(records)

    print("🧪 Ragas 正在进行评测...")
    eval_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
        show_progress=True,
    )

    # 合并结果
    score_df = eval_result.to_pandas()
    metric_cols = [col for col in [
        "faithfulness", "answer_relevancy", "context_recall",
        "llm_context_precision_with_reference", "answer_correctness"
    ] if col in score_df.columns]

    merged_df = pd.concat([
        pd.DataFrame(records),
        score_df[metric_cols].reset_index(drop=True)
    ], axis=1)

    # 显示统计
    stats = adaptive_rag.get_stats()
    total = sum(stats.values())

    print("\n" + "="*80)
    print("📊 策略使用统计")
    print("="*80)
    print(f"事实题 (Baseline): {stats['simple']} ({stats['simple']/total*100:.1f}%)")
    print(f"复杂题 (Advanced): {stats['complex']} ({stats['complex']/total*100:.1f}%)")
    print(f"推理题 (Advanced): {stats['reasoning']} ({stats['reasoning']/total*100:.1f}%)")
    print("="*80)

    return merged_df


# ============================================================
# 10. 工具函数

def sample_stratified_evaluation(total_samples: int = 5) -> List:
    """
    分层采样评测问题

    返回: 采样的索引列表
    """
    # 题型分布
    basic_indices = list(range(0, 38))      # Q1-38: 基础知识题 (76%)
    complex_indices = list(range(38, 45))   # Q39-45: 复杂题 (14%)
    reasoning_indices = list(range(45, 50)) # Q46-50: 推理题 (10%)

    # 按比例分配
    basic_count = max(1, int(total_samples * 0.76))
    complex_count = max(1, int(total_samples * 0.14))
    reasoning_count = max(1, int(total_samples * 0.10))

    # 调整总和
    total_allocated = basic_count + complex_count + reasoning_count
    if total_allocated > total_samples:
        if reasoning_count > 0:
            reasoning_count -= (total_allocated - total_samples)
        elif complex_count > 0:
            complex_count -= (total_allocated - total_samples)
        else:
            basic_count -= (total_allocated - total_samples)
    elif total_allocated < total_samples:
        basic_count += (total_samples - total_allocated)

    # 随机采样
    np.random.seed(42)
    sampled_indices = (
        np.random.choice(basic_indices, size=basic_count, replace=False).tolist() +
        np.random.choice(complex_indices, size=complex_count, replace=False).tolist() +
        np.random.choice(reasoning_indices, size=reasoning_count, replace=False).tolist()
    )

    print(f"✅ 分层采样完成：共 {total_samples} 题")
    print(f"   - 基础知识题: {basic_count} 题")
    print(f"   - 复杂题: {complex_count} 题")
    print(f"   - 推理题: {reasoning_count} 题")
    print(f"   - 采样题号: {sorted(sampled_indices)}\n")

    return sorted(sampled_indices)


# ============================================================
# 11. 主程序入口

if __name__ == "__main__":
    pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"

    rag = AdaptiveRAG(pdf_url)

    # 测试问题
    test_questions = [
        "Transformer 的 base 配置使用了多少层？",  # 事实题
        "Transformer 为什么需要多头注意力机制？",  # 复杂题
        "如果去掉残差连接会怎样？",  # 推理题
    ]

    print("\n" + "="*80)
    print("🧪 测试自适应策略")
    print("="*80 + "\n")

    for question in test_questions:
        answer, strategy, contexts = rag.answer(question)
        print(f"\n问题: {question}")
        print(f"策略: {strategy}")
        print(f"答案: {answer}\n")
        print("-"*80)

    # 显示统计
    stats = rag.get_stats()
    print(f"\n统计: {stats}")
