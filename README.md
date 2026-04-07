# Adaptive RAG

自适应检索增强生成系统，根据问题类型自动选择最优 RAG 策略，并在检索质量不足时通过三阶段兜底机制提升答案可靠性。

## 核心思路

对输入问题进行分类，再路由到对应的 RAG 管道：

| 问题类型 | 特征关键词 | 使用策略 |
|---|---|---|
| 事实题 (simple) | 多少、是什么、哪些、参数… | **Baseline RAG** — 直接向量检索 |
| 复杂题 (complex) | 为什么、如何、原理、对比… | **Advanced RAG** — 问题分解 + Rerank + 反思重写 |
| 推理题 (reasoning) | 如果、假设、推测、导致… | **Advanced RAG** — 问题分解 + Rerank + 反思重写 |

## 系统架构

```
AdaptiveRAG
├── DocumentProcessor         # PDF 加载 / 清洗 / 切分
├── QuestionClassifier        # LLM Router + 关键词混合分类
│   ├── _keyword_classify()   # 快速关键词匹配（高置信度时直接使用）
│   └── LLM Router            # 置信度低时调用 qwen-turbo 做语义分类
├── BaselineRAG               # Chroma 向量检索 → LCEL Chain
├── AdvancedRAG               # LangGraph 反思智能体
│   ├── Contextualize         # 利用对话历史还原代词指代
│   ├── Plan（问题分解）       # 分解为 3–5 个互补子问题
│   ├── Retrieve              # BM25 + Dense 混合检索（权重 0.4 / 0.6）
│   ├── Rerank                # GTE-Rerank-v2 全局重排，取 top-k 文档
│   ├── Verify（路由判断）     # 相关度阈值路由：通过 → Generate，不足 → Rewrite
│   ├── Rewrite（三阶段兜底）  # 见下方「兜底策略」
│   └── Generate              # 子问题分步回答 + 最终整合，含低置信度标注
├── SessionMemory             # Redis 短期记忆（不可用时自动降级为内存）
└── AdaptiveRAGEvaluator      # Ragas 5 指标评测接口
```

## Advanced RAG 兜底策略

当 Rerank 最高分（`top_score`）低于阈值时，系统按以下三个阶段逐步升级检索策略：

```
初始检索   sub_questions × k=10  ──→ top_score ≥ 0.3 ✅ → Generate（正常）
              ↓ top_score < 0.3
第 1 次兜底  改写子问题 × k=30   ──→ top_score ≥ 0.3 ✅ → Generate（正常）
              ↓ top_score < 0.3
第 2 次兜底  HyDE 假设文档 × k=90 ──→ top_score ≥ 0.3 ✅ → Generate（正常）
              ↓ top_score 仍低
最终路由    top_score ≥ 0.1 → Generate + ⚠️ 证据不足警告
            top_score < 0.1 → 🚫 拒绝回答（语料库中确实无相关内容）
```

| 阶段 | 策略 | 说明 |
|---|---|---|
| 第 1 次兜底 | 改写子问题 + 扩展 k（10→30） | 换关键词角度，增加候选池 |
| 第 2 次兜底 | HyDE — 生成假设性文档文本做向量检索（k=90） | 切换到文档向量空间，解决词汇不匹配问题 |
| 最终生成 | 0.1 ≤ score < 0.3，加 ⚠️ 前缀 | 弱相关证据可参考，但明示不确定性 |
| 拒绝回答 | score < 0.1 | 语料盲区，不调用 LLM 防止幻觉 |

**HyDE 原理：** 让 LLM 先生成一段"论文风格的假设性摘录"，用这段文字的向量去检索，比用自然语言问题检索更贴近文档的向量空间，能找到语义相关但词汇不同的段落。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入阿里云 DashScope API Key：

```bash
cp .env.example .env
```

```env
DASHSCOPE_API_KEY=your_api_key_here
```

### 3. 运行 Streamlit UI

```bash
streamlit run app.py
```

### 4. 代码调用

```python
from Adaptive_RAG import AdaptiveRAG

rag = AdaptiveRAG(
    pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
    chunk_size=600,
    chunk_overlap=60,
    persist_dir="./chroma_db"   # 可选，持久化向量库避免重复 Embedding
)

answer, strategy, contexts = rag.answer("Transformer 的 base 配置使用了多少层？")
print(f"策略: {strategy}")   # baseline / advanced
print(f"答案: {answer}")
```

支持强制指定策略：

```python
answer, strategy, contexts = rag.answer(question, force_strategy="advanced")
```

支持多轮会话（自动维护历史）：

```python
answer, strategy, contexts = rag.answer(
    "它的注意力头数是多少？",   # 代词"它"会被还原为上文提到的模型
    session_id="my_session"
)
```

### 5. 查看策略使用统计

```python
stats = rag.get_stats()
# {'simple': 2, 'complex': 3, 'reasoning': 1}
```

## Docker 部署

项目提供完整的三服务编排，一键启动：

```bash
docker-compose up -d
```

| 服务 | 镜像 | 说明 |
|---|---|---|
| `app` | 本地构建 | Streamlit UI，端口 8501 |
| `redis` | redis:7-alpine | 会话记忆持久化（AOF） |
| `chromadb` | chromadb/chroma:latest | 向量库 HTTP 服务，端口 8000 |

Redis 不可用时系统自动降级为内存会话记忆，功能不受影响。

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
5. 全量 50 题 Baseline vs Adaptive 对比 + 可视化

## 模型依赖

| 用途 | 模型 |
|---|---|
| Embedding | `text-embedding-v4` (DashScope) |
| 问题分类 / 答案生成 / 改写 / HyDE | `qwen-turbo` (DashScope) |
| Rerank | `gte-rerank-v2` (DashScope) |
| Ragas 评测 | `qwen-max` (DashScope) |

## 文件说明

| 文件 | 说明 |
|---|---|
| `Adaptive_RAG.py` | 核心实现 |
| `app.py` | Streamlit 前端 |
| `adaptive_rag_test.ipynb` | 测试与评测 Notebook |
| `evaluation_dataset.py` | 50 题评测数据集 |
| `requirements.txt` | 依赖列表 |
| `.env.example` | 环境变量模板 |
| `Dockerfile` | 应用镜像构建文件 |
| `docker-compose.yml` | 三服务编排配置 |
| `learn/` | 早期探索 Notebook 及参考实现 |
