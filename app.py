"""
Adaptive RAG · Streamlit 对话界面
功能：
  - 侧边栏会话管理（从 Redis 加载历史会话列表）
  - st.chat_message 多轮对话
  - Agent 中间状态实时展示（st.status）
  - 引用溯源可视化（点击展开原文片段）
"""

import os
import re
import uuid

import streamlit as st
from dotenv import load_dotenv

from Adaptive_RAG import AdaptiveRAG

load_dotenv()

# ── 常量（优先读取环境变量，方便 Docker 覆盖）────────────────
PDF_URL     = os.getenv("PDF_URL", "https://arxiv.org/pdf/1706.03762.pdf")
PERSIST_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
REDIS_HOST  = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT  = int(os.getenv("REDIS_PORT", "6379"))
TOP_K       = 3

# LangGraph 节点 → 中文描述
NODE_LABELS: dict = {
    "contextualize": "🔗 理解上下文，还原代词",
    "plan":          "📋 分解子问题",
    "retrieve":      "🔍 混合检索文档",
    "rerank":        "🏆 GTE 精排文档",
    "rewrite":       "✏️ 改写查询（相关度不足）",
    "generate":      "✍️ 整合答案",
}


# ── 缓存 RAG 实例（整个 Streamlit 进程内只初始化一次）────────
@st.cache_resource(show_spinner="⏳ 正在加载知识库，首次启动需要约 30 秒...")
def load_rag() -> AdaptiveRAG:
    return AdaptiveRAG(
        pdf_url=PDF_URL,
        persist_dir=PERSIST_DIR,
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
    )


# ── 引用渲染 ─────────────────────────────────────────────────
def render_citations(answer: str, sources: list[dict]) -> None:
    """解析答案中的 [N] 标记，渲染可展开的引用卡片。"""
    if not sources:
        return
    cited_indices = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
    cited_sources = [s for s in sources if s["index"] in cited_indices]
    if not cited_sources:
        return

    st.divider()
    st.caption("📎 引用来源")
    for src in cited_sources:
        label = f"[{src['index']}]  第 {src['page']} 页"
        with st.expander(label, expanded=False):
            content = src["content"]
            preview = content[:800] + "…" if len(content) > 800 else content
            st.markdown(f"> {preview}")


# ── 侧边栏 ────────────────────────────────────────────────────
def render_sidebar(rag: AdaptiveRAG) -> None:
    with st.sidebar:
        st.title("🗂 会话管理")

        # 新建会话
        if st.button("＋ 新建会话", use_container_width=True, type="primary"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages   = []
            st.rerun()

        st.divider()

        # 历史会话列表（来自 Redis / 内存回退）
        sessions = rag.memory.list_sessions()
        current  = st.session_state.get("session_id", "")

        if sessions:
            st.caption("历史会话")
            for sid in reversed(sessions):          # 最新在前
                label    = f"{'▶ ' if sid == current else ''}{sid[:12]}…"
                btn_type = "primary" if sid == current else "secondary"
                if st.button(label, key=f"sess_{sid}",
                             use_container_width=True, type=btn_type):
                    # 从 Redis 恢复该会话的消息列表
                    history  = rag.memory.get_history(sid)
                    messages = []
                    for i in range(0, len(history), 2):
                        if i < len(history):
                            messages.append({
                                "role":    "user",
                                "content": history[i]["content"],
                                "sources": [],
                            })
                        if i + 1 < len(history):
                            messages.append({
                                "role":    "assistant",
                                "content": history[i + 1]["content"],
                                "sources": [],
                            })
                    st.session_state.session_id = sid
                    st.session_state.messages   = messages
                    st.rerun()
        else:
            st.caption("暂无历史会话")

        st.divider()

        # 当前会话信息
        st.caption(f"**当前 Session**\n`{current[:16]}…`")

        if st.button("🗑 清除当前会话记录", use_container_width=True):
            rag.memory.clear(current)
            st.session_state.messages = []
            st.toast("已清除当前会话记录", icon="🗑")
            st.rerun()

        # 系统参数展示
        with st.expander("⚙️ 系统参数"):
            st.write(f"- **Top-K**: {TOP_K}")
            st.write(f"- **Rerank 阈值**: {rag.advanced.RERANK_THRESHOLD}")
            st.write(f"- **最大改写次数**: {rag.advanced.MAX_RETRIES}")
            stats = rag.get_stats()
            total = sum(stats.values()) or 1
            st.write("**策略分布**")
            st.progress(stats["simple"] / total, text=f"Baseline: {stats['simple']}")
            adv = stats["complex"] + stats["reasoning"]
            st.progress(adv / total, text=f"Advanced: {adv}")


# ── 主界面 ────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Adaptive RAG · 学术问答",
        page_icon="🤖",
        layout="wide",
    )

    # 初始化 session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 加载 RAG（缓存）
    rag = load_rag()

    # 侧边栏
    render_sidebar(rag)

    # 页头
    st.title("🤖 Adaptive RAG · 学术问答助手")
    st.caption(
        "基于 *Attention Is All You Need* 的智能问答 · "
        "事实题走 Baseline，复杂/推理题走 Advanced（LangGraph 反思智能体） · "
        "支持多轮对话与引用溯源"
    )
    st.divider()

    # 渲染历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_citations(msg["content"], msg["sources"])

    # 输入框
    if prompt := st.chat_input("请输入问题，支持追问（如：「它为什么这样设计？」）"):

        # 显示用户消息
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "sources": []}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成回答
        with st.chat_message("assistant"):
            status_box      = st.status("🤔 Agent 正在分析问题…", expanded=True)
            answer_ph       = st.empty()   # 答案占位
            citations_ph    = st.empty()   # 引用占位

            final_answer = ""
            sources      = []
            strategy     = ""

            try:
                for event in rag.stream_answer(
                    prompt, TOP_K,
                    session_id=st.session_state.session_id,
                ):
                    etype = event["type"]

                    if etype == "strategy":
                        strategy   = event["strategy"]
                        qtype      = event.get("qtype", "")
                        s_label    = (
                            "事实题 · Baseline RAG"
                            if strategy == "baseline"
                            else f"{qtype}题 · Advanced RAG (LangGraph)"
                        )
                        status_box.update(
                            label=f"⚙️ 策略已确定：{s_label}",
                            expanded=True,
                        )

                    elif etype == "node":
                        node      = event["node"]
                        data      = event["data"]
                        node_desc = NODE_LABELS.get(node, node)

                        # 构造附加细节
                        detail = ""
                        if node == "plan" and "sub_questions" in data:
                            subs   = data["sub_questions"]
                            detail = f"（{len(subs)} 个子问题）"
                            with status_box:
                                st.write(f"{node_desc} {detail}")
                                for i, sq in enumerate(subs, 1):
                                    st.caption(f"  {i}. {sq}")
                        elif node == "retrieve" and "all_docs" in data:
                            detail = f"（共 {len(data['all_docs'])} 个去重片段）"
                            with status_box:
                                st.write(f"{node_desc} {detail}")
                        elif node == "rerank" and "top_score" in data:
                            score  = data["top_score"]
                            detail = f"（最高相关度 {score:.3f}）"
                            with status_box:
                                st.write(f"{node_desc} {detail}")
                        elif node == "rewrite" and "retry_count" in data:
                            detail = f"（第 {data['retry_count']} 次改写）"
                            with status_box:
                                st.write(f"{node_desc} {detail}")
                        else:
                            with status_box:
                                st.write(f"{node_desc}")

                    elif etype == "done":
                        final_answer = event["answer"]
                        sources      = event.get("sources", [])
                        strategy     = event.get("strategy", strategy)

                status_box.update(label="✅ 回答生成完毕", state="complete", expanded=False)
                answer_ph.markdown(final_answer)

                # 渲染引用来源
                with citations_ph.container():
                    render_citations(final_answer, sources)

            except Exception as exc:
                status_box.update(label="❌ 生成出错", state="error", expanded=True)
                final_answer = f"⚠️ 生成答案时出错：{exc}"
                answer_ph.markdown(final_answer)

        # 写入 session state（用于页面重渲染时回显）
        st.session_state.messages.append({
            "role":    "assistant",
            "content": final_answer,
            "sources": sources,
        })


if __name__ == "__main__":
    main()
