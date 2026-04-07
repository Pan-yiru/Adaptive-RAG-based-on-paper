"""
Adaptive RAG: 自适应检索增强生成系统
根据问题类型自动选择最优策略：
- 事实题：使用 Baseline RAG（直接检索）
- 复杂题/推理题：使用 Advanced RAG（问题分解 + Rerank）
"""

import os
import re
import json
import uuid
import tempfile
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional, TypedDict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import dashscope
import requests
import numpy as np

try:
    import redis as redis_lib
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

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
# 1.5 会话记忆（Redis 短期记忆）
# ============================================================

class SessionMemory:
    """
    基于 Redis 的会话短期记忆
    - 每个 session_id 存储最近 max_turns 轮对话
    - Redis 不可用时自动降级为内存存储（当次进程内有效）
    """

    def __init__(self, host: str = "localhost", port: int = 6379,
                 db: int = 0, max_turns: int = 10):
        self.max_turns = max_turns
        self._fallback: Dict[str, List] = {}  # Redis 不可用时的内存回退
        if REDIS_AVAILABLE:
            try:
                self._redis = redis_lib.Redis(
                    host=host, port=port, db=db,
                    decode_responses=True,
                    socket_connect_timeout=2
                )
                self._redis.ping()
                self.available = True
                print("✅ Redis 连接成功，会话记忆已启用")
            except Exception:
                self._redis = None
                self.available = False
                print("⚠️ Redis 不可用，降级为内存会话记忆")
        else:
            self._redis = None
            self.available = False
            print("⚠️ redis 未安装，降级为内存会话记忆")

    def get_history(self, session_id: str) -> List[Dict]:
        """获取最近 max_turns 轮对话记录"""
        key = f"session:{session_id}"
        try:
            if self._redis:
                raw = self._redis.lrange(key, -self.max_turns * 2, -1)
                return [json.loads(m) for m in raw]
        except Exception:
            pass
        raw_fallback = self._fallback.get(key, [])[-self.max_turns * 2:]
        # _fallback 存储的是 JSON 字符串，需要解析后再返回
        return [json.loads(m) if isinstance(m, str) else m for m in raw_fallback]

    def add_turn(self, session_id: str, question: str, answer: str):
        """追加一轮对话并裁剪超长记录"""
        key = f"session:{session_id}"
        messages = [
            json.dumps({"role": "user",      "content": question}, ensure_ascii=False),
            json.dumps({"role": "assistant", "content": answer},   ensure_ascii=False),
        ]
        try:
            if self._redis:
                pipe = self._redis.pipeline()
                for msg in messages:
                    pipe.rpush(key, msg)
                pipe.ltrim(key, -self.max_turns * 2, -1)
                pipe.execute()
                return
        except Exception:
            pass
        # 内存回退
        existing = self._fallback.get(key, [])
        existing.extend(messages)
        self._fallback[key] = existing[-self.max_turns * 2:]

    def clear(self, session_id: str):
        """清空指定会话的历史"""
        key = f"session:{session_id}"
        try:
            if self._redis:
                self._redis.delete(key)
                return
        except Exception:
            pass
        self._fallback.pop(key, None)

    def list_sessions(self) -> List[str]:
        """返回所有 session_id 列表（从 Redis SCAN 或内存回退）"""
        try:
            if self._redis:
                keys = []
                cursor = 0
                while True:
                    cursor, batch = self._redis.scan(cursor, match="session:*", count=100)
                    keys.extend(k[len("session:"):] for k in batch)
                    if cursor == 0:
                        break
                return keys
        except Exception:
            pass
        return [k[len("session:"):] for k in self._fallback]


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

def create_retriever(chunks: List, weights: List[float] = None, vectorstore=None) -> EnsembleRetriever:
    """创建混合检索器（BM25 + Dense）"""
    if weights is None:
        weights = [0.4, 0.6]
    if vectorstore is None:
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
        weights=weights
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


# ============================================================
# 5. Baseline RAG 策略

class BaselineRAG:
    """Baseline RAG: 直接检索策略（完全对齐 rag_baseline.ipynb）"""

    def __init__(self, chunks=None, llm=None, vectorstore=None):
        """
        初始化 Baseline RAG（对齐 rag_baseline.ipynb 逻辑）

        参数:
        - chunks: 切分后的文档块列表（vectorstore 已提供时可为 None）
        - llm: 可选的 LLM 实例
        - vectorstore: 可选的预构建 Chroma 向量库（避免重复 Embedding）
        """
        # 复用外部向量库或新建
        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=QwenEmbeddings(
                    model="text-embedding-v4",
                    api_key=os.environ["DASHSCOPE_API_KEY"],
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    batch_size=10
                )
            )

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

    def answer(self, question: str, top_k: int = 3) -> Tuple[str, List[str]]:
        """
        生成答案（完全对齐 rag_baseline.ipynb 逻辑）

        返回: (答案, 检索到的上下文列表)
        """
        # 按 top_k 创建检索器
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})

        # 检索并过滤空文档
        docs = retriever.invoke(question)
        docs = [doc for doc in docs if getattr(doc, "page_content", "").strip()]
        retrieved_contexts = [doc.page_content for doc in docs]

        # 用检索到的文档构建 context 后生成答案
        context = "\n\n".join(retrieved_contexts)
        prompt_value = self.prompt.format(context=context, question=question)
        response = self.llm.invoke(prompt_value)
        answer_text = response.content if hasattr(response, "content") else str(response)

        return answer_text, retrieved_contexts

    def stream_answer(self, question: str, top_k: int = 3):
        """
        Baseline 的流式接口（单步生成，yield 一个 node 事件 + done 事件）。
        """
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(question)
        docs = [doc for doc in docs if getattr(doc, "page_content", "").strip()]
        retrieved_contexts = [doc.page_content for doc in docs]
        sources = [
            {
                "index": i + 1,
                "content": doc.page_content,
                "page": doc.metadata.get("page", "-"),
                "source": doc.metadata.get("source", "-"),
            }
            for i, doc in enumerate(docs)
        ]
        yield {"type": "node", "node": "retrieve",
               "data": {"all_docs": docs}}

        context = "\n\n".join(retrieved_contexts)
        prompt_value = self.prompt.format(context=context, question=question)
        response = self.llm.invoke(prompt_value)
        answer_text = response.content if hasattr(response, "content") else str(response)

        yield {"type": "node", "node": "generate",
               "data": {"final_answer": answer_text}}
        yield {"type": "done", "answer": answer_text,
               "contexts": retrieved_contexts, "sources": sources}


# ============================================================
# 6. Advanced RAG 策略

class AdvancedRAGState(TypedDict):
    """LangGraph 图状态"""
    question: str
    contextualized_question: str  # 代词还原后的完整问题
    history: List[Dict]           # 当前 session 的历史对话
    sub_questions: List[str]
    all_docs: List                # 检索合并后的全量文档
    reranked_docs: List           # Rerank 后的 Top-K 文档
    top_score: float              # Rerank 最高相关度分数
    sub_answers: List[str]
    final_answer: str
    sources: List[Dict]           # 引用来源列表（供前端可视化）
    retry_count: int              # 已触发改写次数
    top_k: int                    # 最终保留文档数
    expand_k: int                 # 检索扬展系数（每次改写后扩大单次检索数量）
    hyde_query: str               # HyDE 假设性文档文本（retry==1 时生成，替换子问题做向量检索）


class AdvancedRAG:
    """
    Advanced RAG：基于 LangGraph 的反思智能体

    流程：Plan → Retrieve → Rerank → Verify ─┬→ Generate
                                    ↑         └→ Rewrite ─┘
    - Verify  ：Rerank 最高分 < RERANK_THRESHOLD 且未超重试上限时触发 Rewrite
    - Rewrite ：改写子问题，重新检索（最多 MAX_RETRIES 次）
    - Consistency：所有子问题共享同一套 Top-K 文档，消除证据冲突
    """

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
- 在答案中用 [1]、[2] 等标注引用的参考文档编号（如"...注意力机制 [1]"）

最终回答："""
        )

        # 查询改写 prompt（检索质量不足时触发）
        self.rewrite_prompt = ChatPromptTemplate.from_template(
            """你是一个查询改写助手。当前检索结果相关度不足，请改写子问题以提升检索效果。

原始问题：{question}
当前子问题：
{sub_questions_str}

改写要求：
1. 使用更具体的学术术语或同义词
2. 补充相关背景概念
3. 保持子问题的独立可检索性
4. 数量与原来保持一致

请以JSON数组格式返回改写后的子问题：
["改写后子问题1", "改写后子问题2", ...]"""
        )

        # HyDE prompt：生成假设性文档文本，切换向量空间改善 semantic match
        self.hyde_prompt = ChatPromptTemplate.from_template(
            """请根据以下问题，生成一段假设性的参考文档摘录（2-3 句话）。
这段文字将用于在学术文档库中做向量相似度检索，要求：
1. 使用与学术论文相近的表达方式和专业术语
2. 包含问题所涉及的核心技术概念
3. 风格像是从教材、论文或技术报告中直接摘录的句子
（注意：这不是你对问题的真实回答，只是一段帮助检索到正确文档的假设性文本）

问题：{question}

假设性文档摘录（只输出文本本身，不要任何前缀或解释）："""
        )

        # 问题还原 prompt（利用历史对话消除代词指代）
        self.contextualize_prompt = ChatPromptTemplate.from_template(
            """根据以下对话历史，将用户的最新问题改写为一个独立、完整的问题（还原其中的代词和指代关系）。
如果问题不含任何代词或指代，原样返回即可。只输出改写后的问题，不要解释。

对话历史：
{history}

最新问题：{question}

改写后的问题："""
        )

        self.decompose_chain     = self.decompose_prompt     | self.llm | StrOutputParser()
        self.subqa_chain         = self.subqa_prompt         | self.llm | StrOutputParser()
        self.final_chain         = self.final_prompt         | self.llm | StrOutputParser()
        self.rewrite_chain       = self.rewrite_prompt       | self.llm | StrOutputParser()
        self.hyde_chain          = self.hyde_prompt          | self.llm | StrOutputParser()
        self.contextualize_chain = self.contextualize_prompt | self.llm | StrOutputParser()

        # 构建 LangGraph 流程图
        self.graph = self._build_graph()

    RERANK_THRESHOLD   = 0.3   # 相关度低于此值时触发改写
    NO_ANSWER_THRESHOLD = 0.1   # 相关度低于此值时语料库中确实无相关内容，直接拒绝回答
    MAX_RETRIES        = 2     # 最大改写重试次数
    EXPAND_MULTIPLIER  = 3     # 第一次改写后展宽单次检索 k 的倍数

    # ----------------------------------------------------------
    # 工具方法

    def _parse_sub_questions(self, raw_output: str) -> List[str]:
        """解析子问题"""
        candidates = []
        try:
            parsed = json.loads(raw_output)
            if isinstance(parsed, list):
                candidates = [str(item) for item in parsed]
        except (json.JSONDecodeError, ValueError):
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

    def _rerank_with_scores(self, query: str, docs: List, top_n: int = 3) -> Tuple[List, float]:
        """使用 GTE-Rerank 重排文档，返回 (reranked_docs, top_relevance_score)"""
        if not docs:
            return [], 0.0

        doc_texts = [d.page_content for d in docs]
        try:
            response = dashscope.TextReRank.call(
                model="gte-rerank-v2",
                query=query,
                documents=doc_texts,
                top_n=top_n,
                return_documents=False
            )
            if response.status_code == 200:
                results = response.output.results
                reranked = [docs[item.index] for item in results]
                top_score = results[0].relevance_score if results else 0.0
                return reranked, top_score
        except Exception as e:
            print(f"⚠️ Rerank 失败: {e}")

        return docs[:top_n], 0.0

    # ----------------------------------------------------------
    # LangGraph 节点

    def _contextualize_node(self, state: AdvancedRAGState) -> dict:
        """Contextualize：利用会话历史还原代词，将问题改写为独立完整的表述"""
        history = state.get("history") or []
        question = state["question"]
        if not history:
            return {"contextualized_question": question}
        history_str = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
            for m in history[-6:]
        )
        try:
            result = self.contextualize_chain.invoke({
                "history": history_str,
                "question": question,
            })
            contextualized = result.strip() or question
        except Exception as e:
            print(f"⚠️ 问题还原失败: {e}")
            contextualized = question
        if contextualized != question:
            print(f"  💬 问题还原: '{question[:40]}' → '{contextualized[:40]}'")
        return {"contextualized_question": contextualized}

    def _plan_node(self, state: AdvancedRAGState) -> dict:
        """Plan：将还原后的完整问题分解为互补子问题"""
        effective_q = state.get("contextualized_question") or state["question"]
        try:
            raw = self.decompose_chain.invoke({"question": effective_q})
            sub_questions = self._parse_sub_questions(raw)
        except Exception as e:
            print(f"⚠️ 子问题生成失败，使用原问题: {e}")
            sub_questions = [effective_q]
        if not sub_questions:
            sub_questions = [effective_q]
        print(f"  📋 分解为 {len(sub_questions)} 个子问题")
        return {"sub_questions": sub_questions}

    def _retrieve_node(self, state: AdvancedRAGState) -> dict:
        """Retrieve：对所有子问题检索，合并去重；HyDE 模式下以假设性文本替代子问题向量"""
        all_docs = []
        seen = set()
        k_per_query = state.get("expand_k", 10)
        # HyDE 模式：用假设性文档文本做向量检索，切换到文档向量空间
        hyde_query = state.get("hyde_query", "")
        queries = [hyde_query] if hyde_query else state["sub_questions"]
        mode = "HyDE" if hyde_query else "子问题"
        for query in queries:
            docs = self.retriever.invoke(query, config={"configurable": {"k": k_per_query}})
            # EnsembleRetriever 不保证 config 传递 k，直接截取
            docs = [d for d in docs if getattr(d, "page_content", "").strip()]
            for doc in docs:
                text = doc.page_content.strip()
                if text and text not in seen:
                    seen.add(text)
                    all_docs.append(doc)
        print(f"  🔎 检索完成（{mode}），共 {len(all_docs)} 个去重文档（k={k_per_query}）")
        return {"all_docs": all_docs}

    def _rerank_node(self, state: AdvancedRAGState) -> dict:
        """Rerank：针对还原后的完整问题全局重排，确保所有子问题共享同一套证据"""
        effective_q = state.get("contextualized_question") or state["question"]
        reranked_docs, top_score = self._rerank_with_scores(
            effective_q, state["all_docs"], top_n=state["top_k"]
        )
        print(f"  🏆 Rerank 完成，最高相关度: {top_score:.3f}")
        return {"reranked_docs": reranked_docs, "top_score": top_score}

    def _rewrite_node(self, state: AdvancedRAGState) -> dict:
        """
        改写策略（两阶段兜底）：
        - 第 1 次：改写子问题 + 展宽单次检索 k（捕获更多候选文档）
        - 第 2 次：放弃子问题分解，直接用还原后的完整问题单路检索（Baseline 充层）
        """
        effective_q = state.get("contextualized_question") or state["question"]
        retry = state["retry_count"]

        if retry == 0:
            # 第一次：改写子问题，同时扩大检索数量
            sub_questions_str = "\n".join(f"{i+1}. {q}" for i, q in enumerate(state["sub_questions"]))
            try:
                raw = self.rewrite_chain.invoke({
                    "question": effective_q,
                    "sub_questions_str": sub_questions_str,
                })
                rewritten = self._parse_sub_questions(raw)
                if not rewritten:
                    rewritten = state["sub_questions"]
            except Exception as e:
                print(f"⚠️ 查询改写失败: {e}")
                rewritten = state["sub_questions"]
            new_expand_k = state.get("expand_k", 10) * self.EXPAND_MULTIPLIER
            print(f"  ✏️ 第 1 次改写：{len(rewritten)} 个子问题，检索扩展至 k={new_expand_k}")
            return {"sub_questions": rewritten, "retry_count": 1, "expand_k": new_expand_k}
        else:
            # 第二次：HyDE - 生成假设性文档文本，切换向量空间改善 semantic match
            try:
                hyde_text = self.hyde_chain.invoke({"question": effective_q}).strip()
                if not hyde_text:
                    hyde_text = effective_q
            except Exception as e:
                print(f"⚠️ HyDE 生成失败: {e}")
                hyde_text = effective_q
            new_expand_k = state.get("expand_k", 30) * self.EXPAND_MULTIPLIER
            print(f"  🔬 第 2 次：HyDE 向量切换，假设性文档长度={len(hyde_text)}，k={new_expand_k}")
            return {
                "sub_questions": [effective_q],   # 生成时仍以原始问题为准
                "hyde_query":    hyde_text,        # 检索时切换为假设性文档向量
                "retry_count":   2,
                "expand_k":      new_expand_k,
            }

    def _generate_node(self, state: AdvancedRAGState) -> dict:
        """Generate：所有子问题共享同一套 Top-K 文档，消除证据冲突"""
        reranked_docs = state["reranked_docs"]
        sub_questions = state["sub_questions"]
        # 使用还原后的完整问题作为最终回答依据
        question = state.get("contextualized_question") or state["question"]

        # 硬拒绝：相关度极低，语料库中根本不含相关内容，不调用 LLM 防止幻觉
        if state.get("top_score", 1.0) < self.NO_ANSWER_THRESHOLD:
            no_answer = (
                f"抱歉，当前文档库中未找到与该问题相关的内容（最高相关度 {state['top_score']:.2f}），"
                "无法作答。请尝试换一个问题，或确认该问题是否在文档覆盖范围内。"
            )
            print(f"  🚫 相关度极低 ({state['top_score']:.3f} < {self.NO_ANSWER_THRESHOLD})，拒绝生成")
            return {"sub_answers": [], "final_answer": no_answer, "sources": []}

        context_text = "\n\n".join(doc.page_content for doc in reranked_docs)

        sub_answers = []
        for sub_q in sub_questions:
            ans = (
                self.subqa_chain.invoke({"context": context_text, "question": sub_q})
                if context_text else "资料中未找到相关内容"
            )
            sub_answers.append(ans)

        subqa_context = "\n\n".join(
            f"子问题 {i+1}: {sub_q}\n回答: {sub_a}"
            for i, (sub_q, sub_a) in enumerate(zip(sub_questions, sub_answers))
        )
        doc_context = "\n\n".join(
            f"文档 {i+1}: {doc.page_content}" for i, doc in enumerate(reranked_docs)
        )
        final_answer = self.final_chain.invoke({
            "subqa_context": subqa_context,
            "doc_context": doc_context,
            "question": question,
        })

        # 低置信度标注：所有重试后 top_score 仍未过阈值，明确告知用户证据不足
        if state.get("top_score", 1.0) < self.RERANK_THRESHOLD:
            final_answer = (
                f"⚠️ 文档中未找到强相关证据（相关度 {state['top_score']:.2f} < {self.RERANK_THRESHOLD}），"
                f"以下回答基于有限信息，仅供参考：\n\n{final_answer}"
            )

        # 构建引用来源列表（供前端可视化）
        sources = [
            {
                "index": i + 1,
                "content": doc.page_content,
                "page": doc.metadata.get("page", "-"),
                "source": doc.metadata.get("source", "-"),
            }
            for i, doc in enumerate(reranked_docs)
        ]
        return {"sub_answers": sub_answers, "final_answer": final_answer, "sources": sources}

    # ----------------------------------------------------------
    # 条件路由

    def _verify_route(self, state: AdvancedRAGState) -> str:
        """
        验证路由：两阶段兜底策略
        - 第 0 次失败 → Rewrite（改写子问题 + 扩展k）
        - 第 1 次失败 → Rewrite（单路直接检索原始问题）
        - 第 2 次失败 → 不再改写，直接生成（防止无限循环）
        """
        score   = state["top_score"]
        retries = state["retry_count"]
        if score < self.RERANK_THRESHOLD and retries < self.MAX_RETRIES:
            strategy_hint = "改写+扩展k" if retries == 0 else "单路直接检索"
            print(f"  ⚠️ 相关度不足 ({score:.3f} < {self.RERANK_THRESHOLD})，第 {retries+1} 次兜底：{strategy_hint}")
            return "rewrite"
        if score >= self.RERANK_THRESHOLD and retries >= 1:
            print(f"  ✅ 兜底后相关度提升至: {score:.3f}，进入生成")
        return "generate"

    # ----------------------------------------------------------
    # 图构建

    def _build_graph(self):
        """构建 LangGraph 流程图"""
        workflow = StateGraph(AdvancedRAGState)
        workflow.add_node("contextualize", self._contextualize_node)
        workflow.add_node("plan",          self._plan_node)
        workflow.add_node("retrieve",      self._retrieve_node)
        workflow.add_node("rerank",        self._rerank_node)
        workflow.add_node("rewrite",       self._rewrite_node)
        workflow.add_node("generate",      self._generate_node)

        workflow.set_entry_point("contextualize")
        workflow.add_edge("contextualize", "plan")
        workflow.add_edge("plan",          "retrieve")
        workflow.add_edge("retrieve",      "rerank")
        workflow.add_conditional_edges(
            "rerank",
            self._verify_route,
            {"rewrite": "rewrite", "generate": "generate"}
        )
        workflow.add_edge("rewrite",  "retrieve")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def answer(self, question: str, top_k: int = 3,
               history: List[Dict] = None) -> Tuple[str, List[str]]:
        """
        生成答案（LangGraph Contextualize→Plan→Retrieve→Rerank→Verify→Generate 流程）

        参数:
        - history: 当前 session 的历史对话（由 AdaptiveRAG 从 Redis 注入）

        返回: (答案, 检索上下文列表)
        """
        initial_state: AdvancedRAGState = {
            "question":               question,
            "contextualized_question": "",
            "history":                history or [],
            "sub_questions":          [],
            "all_docs":               [],
            "reranked_docs":          [],
            "top_score":              0.0,
            "sub_answers":            [],
            "final_answer":           "",
            "sources":                [],
            "retry_count":            0,
            "top_k":                  top_k,
            "expand_k":               10,  # 每个子问题默认检索 10 篇
            "hyde_query":             "",  # HyDE 假设性文档文本（默认为空）
        }
        result = self.graph.invoke(initial_state)
        return result["final_answer"], [doc.page_content for doc in result["reranked_docs"]]

    def stream_answer(self, question: str, top_k: int = 3,
                      history: List[Dict] = None):
        """
        流式生成答案，逐节点 yield 中间状态事件。

        事件格式:
        - {"type": "node", "node": <str>, "data": <dict>}  每个节点完成后
        - {"type": "done", "answer": <str>, "contexts": <list>, "sources": <list>}
        """
        initial_state: AdvancedRAGState = {
            "question":               question,
            "contextualized_question": "",
            "history":                history or [],
            "sub_questions":          [],
            "all_docs":               [],
            "reranked_docs":          [],
            "top_score":              0.0,
            "sub_answers":            [],
            "final_answer":           "",
            "sources":                [],
            "retry_count":            0,
            "top_k":                  top_k,
            "expand_k":               10,  # 每个子问题默认检索 10 篇
            "hyde_query":             "",  # HyDE 假设性文档文本（默认为空）
        }
        accumulated: Dict = dict(initial_state)
        for event in self.graph.stream(initial_state, stream_mode="updates"):
            node_name = list(event.keys())[0]
            node_data = event[node_name]
            accumulated.update(node_data)
            yield {"type": "node", "node": node_name, "data": node_data}

        yield {
            "type":     "done",
            "answer":   accumulated.get("final_answer", ""),
            "contexts": [doc.page_content for doc in accumulated.get("reranked_docs", [])],
            "sources":  accumulated.get("sources", []),
        }


# ============================================================
# 7. Adaptive RAG 主类

class AdaptiveRAG:
    """自适应 RAG 系统：根据问题类型自动选择最优策略"""

    def __init__(self, pdf_url: str, chunk_size: int = 600, chunk_overlap: int = 60,
                 persist_dir: str = None, session_id: str = None,
                 redis_host: str = None, redis_port: int = None,
                 chroma_host: str = None, chroma_port: int = None):
        """
        初始化 Adaptive RAG 系统

        参数:
        - pdf_url: PDF 文档 URL
        - chunk_size: 文档切分大小
        - chunk_overlap: 文档切分重叠大小
        - persist_dir: ChromaDB 嵌入模式持久化目录（None 表示内存模式）
        - session_id: 当前会话 ID（None 时自动生成 UUID）
        - redis_host / redis_port: Redis 连接参数（优先读取 REDIS_HOST/REDIS_PORT 环境变量）
        - chroma_host / chroma_port: ChromaDB HTTP 服务器地址（优先读取 CHROMA_HOST/CHROMA_PORT 环境变量）
        """
        # 环境变量覆盖默认值
        redis_host  = redis_host  or os.getenv("REDIS_HOST",  "localhost")
        redis_port  = redis_port  or int(os.getenv("REDIS_PORT",  "6379"))
        chroma_host = chroma_host or os.getenv("CHROMA_HOST", "") or None
        chroma_port = chroma_port or int(os.getenv("CHROMA_PORT", "8000"))

        print("="*80)
        print("🚀 初始化 Adaptive RAG 系统")
        print("="*80)

        # 会话 ID
        self.session_id = session_id or str(uuid.uuid4())

        # 1. 加载和处理文档
        docs = DocumentProcessor.load_pdf(pdf_url)
        docs = DocumentProcessor.clean_docs(docs)
        chunks = DocumentProcessor.split_docs(docs, chunk_size, chunk_overlap)

        # 2. 向 chunks 注入 timestamp 和 session_id 元数据（为个性化知识打底）
        ingest_time = datetime.now(timezone.utc).isoformat()
        for chunk in chunks:
            chunk.metadata.update({
                "timestamp":  ingest_time,
                "session_id": self.session_id,
            })

        # 3. 创建共享 Embedding 实例
        embeddings = QwenEmbeddings(
            model="text-embedding-v4",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            batch_size=10
        )

        # 4. 创建共享向量库
        if chroma_host:
            # HTTP 模式：连接独立的 ChromaDB 容器
            import chromadb
            _client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            # 如果集合已存在则复用，否则创建并写入文档
            existing = [c.name for c in _client.list_collections()]
            if "rag_docs" in existing:
                print(f"🌐 连接已有 ChromaDB 集合 (http://{chroma_host}:{chroma_port})")
                shared_vectorstore = Chroma(
                    client=_client,
                    collection_name="rag_docs",
                    embedding_function=embeddings,
                )
            else:
                print(f"🌐 在 ChromaDB 服务器上新建集合 (http://{chroma_host}:{chroma_port})")
                shared_vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    client=_client,
                    collection_name="rag_docs",
                )
        elif persist_dir and os.path.isdir(persist_dir) and os.listdir(persist_dir):
            # 嵌入模式：加载已有持久化目录
            print(f"📂 加载已有向量库: {persist_dir}")
            shared_vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
        else:
            chroma_kwargs = {"documents": chunks, "embedding": embeddings}
            if persist_dir:
                os.makedirs(persist_dir, exist_ok=True)
                chroma_kwargs["persist_directory"] = persist_dir
                print(f"💾 新建持久化向量库: {persist_dir}")
            shared_vectorstore = Chroma.from_documents(**chroma_kwargs)

        # 5. 创建混合检索器（AdvancedRAG 使用，复用共享向量库）
        self.retriever = create_retriever(chunks, vectorstore=shared_vectorstore)

        # 6. 初始化两种策略（均复用共享向量库，无需重复 Embedding）
        self.baseline = BaselineRAG(chunks, vectorstore=shared_vectorstore)
        self.advanced = AdvancedRAG(self.retriever)

        # 7. 初始化问题分类器（LLM Router + 关键词混合策略）
        self.classifier = QuestionClassifier()

        # 8. 会话记忆（Redis 短期记忆，Redis 不可用时自动降级为内存）
        self.memory = SessionMemory(host=redis_host, port=redis_port)

        # 9. 统计信息
        self.stats = {
            "simple": 0,
            "complex": 0,
            "reasoning": 0
        }

        print(f"✅ Adaptive RAG 系统初始化完成（session: {self.session_id}）\n")

    def answer(self, question: str, top_k: int = 3, force_strategy: str = None,
               session_id: str = None) -> Tuple[str, str, List[str]]:
        """
        自适应回答问题

        参数:
        - question: 用户问题
        - top_k: 检索返回的文档数量
        - force_strategy: 强制使用指定策略 ("baseline" 或 "advanced")
        - session_id: 会话 ID（覆盖实例默认值）

        返回: (答案, 策略类型, 检索上下文)
        """
        sid = session_id or self.session_id

        # 从 Redis（或内存回退）加载当前 session 的历史对话
        history = self.memory.get_history(sid)

        # 判断问题类型（使用 LLM Router + 关键词混合策略）
        if force_strategy:
            strategy = force_strategy
            qtype = "complex" if force_strategy == "advanced" else "simple"
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
            # 历史仅用于代词还原：若首轮或无历史则直接检索，否则用 contextualize 还原后再走 Baseline
            if history:
                ctx_q = self.advanced.contextualize_chain.invoke({
                    "history": "\n".join(
                        f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
                        for m in history[-6:]
                    ),
                    "question": question,
                })
                ctx_q = ctx_q.strip() or question
            else:
                ctx_q = question
            answer, contexts = self.baseline.answer(ctx_q, top_k)
        else:
            print(f"🔍 [Advanced] {qtype}题 ({classification_method}): {question[:50]}...")
            # 将历史注入 AdvancedRAG，由 Contextualize 节点完成代词还原
            answer, contexts = self.advanced.answer(question, top_k, history=history)

        # 将本轮对话写入 Redis（或内存回退）
        self.memory.add_turn(sid, question, answer)

        return answer, strategy, contexts

    def stream_answer(self, question: str, top_k: int = 3,
                      force_strategy: str = None,
                      session_id: str = None):
        """
        流式问答接口，供 Streamlit UI 使用。

        逐步 yield 事件：
        - {"type": "strategy", "strategy": ..., "qtype": ..., "method": ...}
        - {"type": "node", "node": ..., "data": ...}  （来自 LangGraph 节点）
        - {"type": "done",  "answer": ..., "contexts": ..., "sources": ..., "strategy": ...}
        """
        sid = session_id or self.session_id
        history = self.memory.get_history(sid)

        # 路由决策
        if force_strategy:
            strategy = force_strategy
            qtype = "complex" if force_strategy == "advanced" else "simple"
            classification_method = "force"
        else:
            classification_result = self.classifier.classify(question)
            qtype = classification_result["type"]
            strategy = "baseline" if qtype == "simple" else "advanced"
            classification_method = classification_result["method"]

        self.stats[qtype] += 1
        yield {"type": "strategy", "strategy": strategy,
               "qtype": qtype, "method": classification_method}

        if strategy == "baseline":
            # 同步代词还原（与 answer() 保持一致）
            if history:
                ctx_q = self.advanced.contextualize_chain.invoke({
                    "history": "\n".join(
                        f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
                        for m in history[-6:]
                    ),
                    "question": question,
                })
                ctx_q = ctx_q.strip() or question
            else:
                ctx_q = question
            gen = self.baseline.stream_answer(ctx_q, top_k)
        else:
            gen = self.advanced.stream_answer(question, top_k, history=history)

        answer = ""
        contexts = []
        sources = []
        # 透传 node 事件；拦截 done 事件（避免发送两次 done），统一在循环后发送
        for event in gen:
            if event["type"] == "done":
                answer   = event["answer"]
                contexts = event["contexts"]
                sources  = event["sources"]
            else:
                yield event

        # 写回记忆
        self.memory.add_turn(sid, question, answer)

        # 发送唯一的 done 事件（含 strategy 字段，供 UI 展示）
        yield {"type": "done", "answer": answer, "contexts": contexts,
               "sources": sources, "strategy": strategy}

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

    # 调整总和（逐步减1，避免任意 count 变为负数）
    total_allocated = basic_count + complex_count + reasoning_count
    while total_allocated > total_samples:
        if reasoning_count > 1:
            reasoning_count -= 1
        elif complex_count > 1:
            complex_count -= 1
        else:
            basic_count -= 1
        total_allocated -= 1
    if total_allocated < total_samples:
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
