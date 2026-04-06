# Advanced RAG 性能下降根本原因分析

## 问题症状
- Advanced RAG（问题分解版）的 **Answer Relevancy**（回答准确率）低于 Baseline
- Advanced RAG 的 **Context Recall**（上下文准确率）低于 Baseline

## 根本原因

### 1️⃣ **问题分解导致的语义偏离**（最关键）

**Baseline 流程：**
```
原问题 → 直接检索 → 获取高相关文档 → 直接生成答案
例如："Transformer采用了什么架构？" → [编码器-解码器文档] → 答案准确
```

**Advanced RAG 流程：**
```
原问题 → LLM分解成子问题 → 多次检索 → 多份答案 → 合成最终答案
例如："Transformer采用了什么架构？" 
      ↓
被分解为：
  - "自注意力机制的基本概念是什么？"
  - "如何计算查询、键和值向量？"
  - "注意力权重是如何生成的？"
  - "缩放与softmax操作的作用是什么？"
  - "多头注意力机制是如何工作的？"
      ↓
这些子问题针对细节/机制，而不是主要的"架构"问题
```

**问题：** LLM 将"架构"问题拆解成了"机制细节"问题，导致检索到的文档侧重于注意力机制而非整体架构。

---

### 2️⃣ **上下文污染与冗余**

**Baseline：**
- 单次检索 → 获取与原问题最相关的 Top-K 文档
- 上下文简洁、高度相关

**Advanced RAG：**
```python
def retrieve_and_rag(...):
    for sub_question in sub_questions:  # 5-6 个子问题
        retrieved_docs = retriever.invoke(sub_question)  # 每次K=10
        retrieved_docs = rerank_documents(sub_question, retrieved_docs, top_n=3)  # 取Top3
        
        for doc in retrieved_docs:
            used_contexts.append(text)  # 累积所有文档
```

**结果：**
- 可能获取 **重复文档**（多个子问题都检索到同一文档）
- 可能获取 **低相关文档**（某个子问题的 Top3 可能与原问题无关）
- 最终合并的 `used_contexts` 冗余、杂乱，影响 **Context Recall** 评分

**Ragas 的 Context Recall 定义：** 检索文档与标准答案的匹配程度
→ 冗余/离题的文档会拉低这个指标

---

### 3️⃣ **答案合成阶段的信息丧失**

**Advanced RAG 的答案生成流程（三层LLM调用）：**
```
第一层：LLM 分解问题 → 5个子问题
    ↓
第二层：对每个子问题 LLM 生成答案 → 5个片段答案
    例如：
    Q1 答案: "自注意力允许模型关注不同位置..."
    Q2 答案: "查询向量通过投影获得..."
    Q3 答案: "权重通过softmax计算..."
    ...
    ↓
第三层：合成函数再次调用 LLM
    build_final_answer(main_question, sub_questions, sub_answers)
    → LLM 需要从 5 个片段答案合成一个最终答案
    → **这是信息丧失的关键** ❌
```

**信息丧失点：**
1. 子问题答案可能不完整（每个答案对应一个子问题）
2. 合成 LLM 需要**推断**这些片段之间的关系
3. 原标准答案是**"Transformer采用标准编码器-解码器架构"**
4. 但合成 LLM 收到的是 5 个关于"注意力机制细节"的答案
5. 无法准确合成出与标准答案相符的答案 → **Answer Relevancy 下降**

---

### 4️⃣ **Baseline 的评估优势**

```python
def answer_question_for_eval(main_question: str):
    """运行 baseline RAG 流程，并返回答案与检索上下文"""
    docs = retriever.invoke(main_question)  # ← 直接针对原问题
    retrieved_contexts = [doc.page_content for doc in docs]
    response = get_answer_from_pdf(main_question, retriever)
    return response, retrieved_contexts
```

**优势：**
- 检索针对**原问题**，不经过 LLM分解这一不稳定步骤
- 获取的文档与**标准答案**的匹配度高
- 最终答案是一次 LLM 调用的直接结果，无合成步骤

---

## 数据对比预期

| 指标 | Baseline | Advanced RAG | 原因 |
|------|----------|-------------|------|
| Faithfulness | ✅ 较高 | ⚠️ 略低 | 合成步骤可能引入虚假信息 |
| Answer Relevancy | ✅ 较高 | ❌ 明显偏低 | 问题分解导致答案偏离原问题 |
| Context Recall | ✅ 较高 | ❌ 明显偏低 | 多次检索的上下文与标准答案匹配度低 |
| Context Precision | ✅ 较高 | ⚠️ 略低 | 冗余上下文拉低精准度 |

---

## 根本问题总结

**Advanced RAG 性能更差的根本原因：**

1. **问题分解是一把双刃剑**
   - 对复杂问题有帮助（需要多视角分析）
   - 但对**事实型问题**（Transformer 论文的 90%）有害
   - 因为事实型问题应该**一次准确检索**，而不是分解

2. **与评估集的不匹配**
   - 评估集的标准答案是针对**完整原问题**的
   - Advanced RAG 的答案是针对**拆解的子问题**的合成
   - 两者在语义上的偏差导致评分低

3. **多步骤 → 多个失败点**
   - Baseline：问题 → 检索 → 答案（2 步）
   - Advanced：问题 → 分解 → 检索×5 → 答案×5 → 合成（6 步）
   - 每一步都可能引入错误

---

## 改进建议

### 短期方案
1. **为问题分解添加回源验证**
   ```python
   # 分解后，检查子问题是否与原问题相关
   if not_semantically_similar(sub_question, main_question):
       use_main_question_directly()  # 降级为原问题检索
   ```

2. **减少子问题数量**
   ```python
   # 改为 2-3 个核心问题，而非 5-6 个
   max_sub_questions = 3
   ```

3. **简化合成逻辑**
   ```python
   # 不用第三个 LLM 调用合成，直接拼接答案
   final_answer = "\n".join(sub_answers)
   ```

### 长期方案
1. **区分问题类型**
   - 事实型问题 → 直接 Baseline RAG
   - 复杂/多维问题 → Advanced（分解）RAG

2. **重新设计问题分解**
   - 改为"论文章节分解"而非"机制分解"
   - 分解应该沿着论文的**逻辑结构**，而非**概念架构**

3. **引入自适应检索**
   - 根据问题类型动态选择 RAG 策略
   - 评估分解的有效性，低于阈值则降级

---

## 建议优先级

🔴 **高优先级** → 短期方案 1（回源验证）
🟡 **中优先级** → 短期方案 2（减少子问题）
🟢 **本次实验** → 保持现状，用 Baseline 作为生产版本
