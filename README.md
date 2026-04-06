# Adaptive RAG

自适应检索增强生成系统，根据问题类型自动选择最优 RAG 策略。

## 核心思路

对输入问题进行分类，再路由到对应的 RAG 管道：

| 问题类型 | 特征关键词 | 使用策略 |
|---|---|---|
| 事实题 (simple) | 多少、是什么、哪些、参数… | **Baseline RAG** — 直接向量检索 |
| 复杂题 (complex) | 为什么、如何、原理、对比… | **Advanced RAG** — 问题分解 + Rerank |
| 推理题 (reasoning) | 如果、假设、推测、导致… | **Advanced RAG** — 问题分解 + Rerank |

## 系统架构

```
AdaptiveRAG
├── DocumentProcessor       # PDF 加载 / 清洗 / 切分
├── QuestionClassifier      # LLM Router + 关键词混合分类
│   ├── _keyword_classify() # 快速关键词匹配（高置信度时直接使用）
│   └── LLM Router          # 置信度低时调用 qwen-turbo 做语义分类
├── BaselineRAG             # Chroma 向量检索 → LCEL Chain
├── AdvancedRAG             # 问题分解 → EnsembleRetriever → GTE-Rerank
│   ├── 问题分解            # 分解为 3–5 个互补子问题
│   ├── BM25 + Dense 检索   # 混合检索（权重 0.4 / 0.6）
│   └── GTE-Rerank-v2       # 重排后取 top-k 文档
└── AdaptiveRAGEvaluator    # Ragas 5 指标评测接口
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件，填入阿里云 DashScope API Key：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

### 3. 初始化系统

```python
from Adaptive_RAG import AdaptiveRAG

rag = AdaptiveRAG(
    pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
    chunk_size=600,
    chunk_overlap=60
)
```

### 4. 提问

```python
answer, strategy, contexts = rag.answer("Transformer 的 base 配置使用了多少层？")
print(f"策略: {strategy}")   # baseline / advanced
print(f"答案: {answer}")
```

支持强制指定策略：

```python
answer, strategy, contexts = rag.answer(question, force_strategy="advanced")
```

### 5. 查看策略使用统计

```python
stats = rag.get_stats()
# {'simple': 2, 'complex': 3, 'reasoning': 1}
```

## 评测

系统集成 [Ragas](https://github.com/explodinggradients/ragas) 5 项指标：

- **Faithfulness** — 答案对检索内容的忠实度
- **Answer Relevancy** — 答案与问题的相关性
- **Context Recall** — 上下文召回率
- **Context Precision** — 上下文精度
- **Answer Correctness** — 答案正确性

### 分层采样快速评测

```python
from Adaptive_RAG import run_ragas_evaluation, sample_stratified_evaluation
from evaluation_dataset import eval_questions_50, eval_ground_truths_50

indices = sample_stratified_evaluation(total_samples=10)  # 按题型分层采样
report = run_ragas_evaluation(
    adaptive_rag=rag,
    eval_questions=eval_questions_50,
    eval_ground_truths=eval_ground_truths_50,
    indices=indices
)
```

采样比例与评测集题型分布一致（基础题 76%、复杂题 14%、推理题 10%）。

### 全量评测

```python
report = run_ragas_evaluation(
    adaptive_rag=rag,
    eval_questions=eval_questions_50,
    eval_ground_truths=eval_ground_truths_50
)
```

## 测试 Notebook

打开 [adaptive_rag_test.ipynb](adaptive_rag_test.ipynb) 可按步骤执行：

1. 初始化系统
2. 测试不同类型问题，观察策略路由结果
3. 查看策略使用统计
4. 运行分层采样评测
5. 与 Baseline / Advanced 方法对比

## 模型依赖

| 用途 | 模型 |
|---|---|
| Embedding | `text-embedding-v4` (DashScope) |
| 问题分类 / 答案生成 | `qwen-turbo` (DashScope) |
| Rerank | `gte-rerank-v2` (DashScope) |
| Ragas 评测 | `qwen-max` (DashScope) |

## 文件说明

| 文件 | 说明 |
|---|---|
| `Adaptive_RAG.py` | 核心实现 |
| `adaptive_rag_test.ipynb` | 测试与评测 Notebook |
| `evaluation_dataset.py` | 50 题评测数据集 |
| `requirements.txt` | 依赖列表 |
