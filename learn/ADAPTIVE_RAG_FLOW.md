# Adaptive RAG 系统流程文档

## 一、系统初始化流程

```
AdaptiveRAG.__init__(pdf_url, chunk_size=600, chunk_overlap=60)
│
├─ 1. 文档处理
│   ├─ DocumentProcessor.load_pdf(url)
│   │   └─ requests.get → tempfile → PyMuPDFLoader → 返回 List[Document]
│   ├─ DocumentProcessor.clean_docs(docs)
│   │   └─ 正则清洗页码/arXiv 水印/多余换行
│   └─ DocumentProcessor.split_docs(docs, chunk_size, chunk_overlap)
│       └─ RecursiveCharacterTextSplitter → List[Document] (chunks)
│
├─ 2. 创建共享向量库（仅调用一次 Embedding API）
│   └─ Chroma.from_documents(chunks, QwenEmbeddings)
│       └─ → shared_vectorstore
│
├─ 3. 创建混合检索器（AdvancedRAG 用）
│   └─ create_retriever(chunks, vectorstore=shared_vectorstore)
│       ├─ BM25Retriever (k=10)
│       ├─ vector_retriever = shared_vectorstore.as_retriever(k=10)
│       └─ EnsembleRetriever(weights=[0.4, 0.6]) → self.retriever
│
├─ 4. 初始化 BaselineRAG（复用 shared_vectorstore，无额外 Embedding）
│   └─ BaselineRAG(chunks, vectorstore=shared_vectorstore)
│
├─ 5. 初始化 AdvancedRAG
│   └─ AdvancedRAG(self.retriever)
│
└─ 6. 初始化 QuestionClassifier（LLM Router + 关键词混合）
```

---

## 二、问答主流程

```
AdaptiveRAG.answer(question, top_k=3, force_strategy=None)
│
├─ [若 force_strategy 指定] 直接跳到对应策略
│
├─ QuestionClassifier.classify(question)
│   ├─ 步骤 1：关键词快速分类
│   │   ├─ 检查推理关键词 → reasoning（置信度≥0.5）
│   │   ├─ 检查复杂关键词 → complex（置信度≥0.5）
│   │   ├─ 检查事实关键词 → simple（置信度≥0.6）
│   │   └─ 问题长度 > 45 字 → complex（置信度=0.4）
│   │
│   └─ 步骤 2：若关键词置信度 < 0.6，启用 LLM Router
│       ├─ qwen-turbo → JSON 输出 {question_type, confidence, reasoning}
│       └─ 解析失败时回退到关键词分类结果
│
├─ 路由决策
│   ├─ type == "simple"              → strategy = "baseline"
│   └─ type == "complex"/"reasoning" → strategy = "advanced"
│
├─ [Baseline 路径] BaselineRAG.answer(question, top_k)
│   ├─ vectorstore.as_retriever(k=top_k).invoke(question)
│   ├─ 过滤空文档
│   ├─ prompt.format(context, question)
│   └─ llm.invoke → 返回 (answer_text, retrieved_contexts)
│
└─ [Advanced 路径] AdvancedRAG.answer(question, top_k)
    ├─ 步骤 1：问题分解
    │   └─ decompose_chain → JSON 数组 ["子问题1", ...]（3~5 个）
    ├─ 步骤 2：多子问题并行检索
    │   └─ 对每个子问题调用 EnsembleRetriever，去重合并所有文档
    ├─ 步骤 3：GTE-Rerank-v2 重排
    │   └─ dashscope.TextReRank → 保留 top_k 最相关文档
    ├─ 步骤 4：子问题答案生成
    │   └─ 对每个子问题 → subqa_chain(context=reranked_docs, question=sub_q)
    └─ 步骤 5：最终答案整合
        └─ final_chain(subqa_context + doc_context + question) → (answer, contexts)
```

---

## 三、评测流程

```
run_ragas_evaluation(adaptive_rag, eval_questions, eval_ground_truths, ...)
│
├─ 可选：sample_stratified_evaluation(total_samples)
│   └─ 分层采样（基础题 76% / 复杂题 14% / 推理题 10%）→ indices
│
├─ 遍历选定问题
│   └─ AdaptiveRAGEvaluator.answer_for_eval(question)
│       └─ AdaptiveRAG.answer → (response, retrieved_contexts)
│
├─ 构建 Ragas Dataset
│   └─ {user_input, response, reference, retrieved_contexts}
│
└─ ragas.evaluate(dataset, metrics=[...])
    ├─ Faithfulness          （答案对检索内容的忠实度）
    ├─ AnswerRelevancy        （答案与问题的相关性）
    ├─ LLMContextRecall       （上下文召回率）
    ├─ LLMContextPrecisionWithReference（上下文精度）
    └─ AnswerCorrectness      （答案正确性）
        └─ → DataFrame（含各题分数 + 策略统计）
```

---

## 四、修复记录（v1 → v2）

| # | 位置 | 问题 | 修复方式 |
|---|---|---|---|
| ① | `AdaptiveRAG.answer` | `force_strategy="advanced"` 导致 `stats` KeyError | `qtype` 改为 `"complex"` |
| ② | `create_retriever` | `weights` 参数被忽略，函数内硬编码 | 改用传入的 `weights` |
| ③ | `BaselineRAG.answer` | `top_k` 参数从未生效 | 每次调用按 `top_k` 动态创建 retriever |
| ④ | `AdvancedRAG` | `_deduplicate_docs` 是死代码 | 删除 |
| ⑤ | `QuestionClassifier.classify` | 方法内重复 `import json/re` | 删除冗余 import |
| ⑥ | 顶层 import | `import shutil` 从未使用 | 删除 |
| ⑦ | `AdaptiveRAG.__init__` | `BaselineRAG` 和 `create_retriever` 各自独立调用 Embedding，重复两次 | 创建 `shared_vectorstore` 统一传入，仅 Embed 一次 |
| ⑧ | `AdvancedRAG._parse_sub_questions` | 裸 `except:` 会捕获 `SystemExit`/`KeyboardInterrupt` | 改为 `except (json.JSONDecodeError, ValueError)` |
| ⑨ | `BaselineRAG.__init__` | `self.chain` 和 `format_docs` 定义后在生产路径中仍绕过 Chain 直接调用 LLM | 移除预构建 Chain，`answer()` 统一走 `llm.invoke` |
