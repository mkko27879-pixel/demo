r"""论文问答 Agent 的前端。

布局有意不用 st.chat_message：它自带头像，而且没有左右对齐的参数。这里用原生
容器自己排——用户消息靠右一个气泡，AI 回答靠左纯文本，两侧都不带头像。

运行（在 web 目录下，这样 .streamlit/config.toml 才会被读到）：
    ..\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
"""

import streamlit as st

import api_client as api
from ui import (
    format_answer,
    is_arxiv_url,
    normalize_markdown,
    render_conversation_panel,
    render_paper_list,
    render_sources,
    render_thinking,
    render_user_message,
    wait_until_indexed,
)

st.set_page_config(
    page_title="论文问答",
    page_icon=":material/menu_book:",
    layout="wide",
)

SUGGESTIONS = {
    "库里有哪些论文？": "库里现在有哪些论文？",
    "48Ca 那篇讲了什么？": "48Ca 那篇论文主要研究了什么？",
    "Tau 重建效率是多少？": "Tau lepton 的重建效率是多少？",
    "Figure 2 画的是什么？": "arxiv:2601.00597 里 Figure 2 的横轴和纵轴分别是什么？",
}


def new_chat() -> None:
    st.session_state.messages = []
    st.session_state.conversation_id = None
    st.session_state.focus_paper_id = None
    st.session_state.focus_label = None
    st.rerun()


def open_chat(conversation_id: str) -> None:
    """把服务端存的会话读回来。刷新页面后靠这个恢复上下文。"""
    try:
        conversation = api.get_conversation(conversation_id)
    except Exception as exc:
        st.error(f"打开会话失败：{exc}")
        return

    st.session_state.messages = [
        {
            "role": message["role"],
            "content": message["content"],
            "sources": message.get("sources") or [],
            "search_rounds": [],
        }
        for message in conversation["messages"]
    ]
    st.session_state.conversation_id = conversation_id
    # 换会话等于换上下文，"当前聚焦"要跟着清掉
    st.session_state.focus_paper_id = None
    st.session_state.focus_label = None
    st.rerun()


def render_history() -> None:
    for message in st.session_state.messages:
        if message["role"] == "user":
            render_user_message(message["content"])
            continue
        render_thinking(message)
        st.markdown(normalize_markdown(message["content"]))
        render_sources(message.get("sources") or [])


def answer(question: str) -> None:
    """发一轮问答：边收边渲染，等待期间先占个"正在检索"的位。"""
    sink: dict = {}

    # 占位符要在 write_stream 之前建，这样它才落在答案上方。
    # 等待第一个 token 的那几秒里，用户看到的是这里的"正在检索"，不是空白。
    slot = st.empty()
    # 文案不用"正在检索"：寒暄类问题 agent 根本不检索，而且这时候也还不知道
    # 它要跑几轮
    slot.status(":shimmer[正在思考…]", type="compact")

    body = st.empty()
    try:
        with body.container():
            streamed = st.write_stream(
                api.stream_tokens(
                    question,
                    st.session_state.conversation_id,
                    st.session_state.top_k,
                    sink,
                    focus_paper_id=st.session_state.focus_paper_id,
                )
            )
    except Exception as exc:
        slot.empty()
        st.error(f"请求失败：{exc}")
        return

    # 流完了重画一遍，原因是 write_stream 边收边解析 markdown，每个 chunk 到达时
    # 它只看到手里那半句话，表格这类要整块才能解析的结构会退化成竖线原文。
    #
    # 重画前必须先 empty()：直接对同一个元素先 write_stream 再 markdown，Streamlit
    # 不保证是替换关系，两段内容会叠在一起——表现就是答案前面多出一大截流式残留，
    # 后面才是真正的正文。
    display, linked_sources = format_answer(streamed, sink.get("sources") or [])
    body.empty()
    with body.container():
        st.markdown(normalize_markdown(display))

    # 跑完了，把占位换成真正的检索摘要（几轮、引用几条、花了多少 token）
    slot.empty()
    with slot.container():
        render_thinking(sink, live=True)
    render_sources(linked_sources)

    if sink.get("conversation_id"):
        st.session_state.conversation_id = sink["conversation_id"]

    st.session_state.messages.append({
        "role": "assistant",
        "content": display or sink.get("answer", ""),
        "sources": linked_sources,
        "search_rounds": sink.get("search_rounds") or [],
        "usage": sink.get("usage") or {},
    })


def ask(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    render_user_message(question)
    answer(question)


def ingest_pdf(files: list) -> None:
    """对话栏里拖进来的 PDF 按"加入知识库"处理。"""
    for uploaded in files:
        with st.status(f"正在上传 {uploaded.name}…", type="step") as status:
            try:
                created = api.upload_pdf(uploaded.name, uploaded.getvalue())
            except Exception as exc:
                status.update(label=f"{uploaded.name} 上传失败：{exc}", state="error")
                continue

            ok, detail = wait_until_indexed(created["paper_id"])
            status.update(
                label=(f"{uploaded.name} 已入库：{detail}" if ok
                       else f"{uploaded.name} 入库失败：{detail}"),
                state="complete" if ok else "error",
            )
            if ok:
                # 刚加进来的这篇就是用户接下来要问的，自动聚焦过去
                st.session_state.focus_paper_id = created["paper_id"]
                st.session_state.focus_label = uploaded.name


def ingest_arxiv(url: str) -> None:
    """粘贴 arXiv 链接就是加入知识库，不必再找个按钮。"""
    with st.status(f"正在下载并解析 {url}…", type="step") as status:
        try:
            created = api.add_arxiv_paper(url)
        except Exception as exc:
            status.update(label=f"提交失败：{exc}", state="error")
            return

        ok, detail = wait_until_indexed(created["paper_id"])
        status.update(
            label=(f"已入库：{detail}" if ok else f"入库失败：{detail}"),
            state="complete" if ok else "error",
        )
        if ok:
            st.session_state.focus_paper_id = created["paper_id"]
            st.session_state.focus_label = url


# ---- session state：集中初始化 ----
for key, value in {
    "messages": [],
    "conversation_id": None,
    "top_k": 3,
    "focus_paper_id": None,
    "focus_label": None,
}.items():
    st.session_state.setdefault(key, value)

backend_ok = api.backend_ready()

# ---- 侧边栏 ----
with st.sidebar:
    st.markdown("### :material/menu_book: 论文问答")
    st.caption("Agent 自主决定检索几轮 · 答案带来源和页码")

    if not backend_ok:
        st.error("连不上后端。先在项目根目录启动：\n\n`uv run uvicorn main:app --reload`")

    st.divider()
    render_conversation_panel(new_chat, open_chat)
    st.divider()
    render_paper_list()
    st.divider()
    st.session_state.top_k = st.slider(
        "每次检索的片段数", min_value=1, max_value=10,
        value=st.session_state.top_k,
        help="调大能给模型更多上下文，但也更容易混进无关片段",
    )

# ---- 主区 ----
if not st.session_state.messages:
    st.markdown("### 有什么想问的？")
    st.caption(
        "把 PDF 拖进下面的输入框、直接粘贴 arXiv 链接就能入库，"
        "也可以对已有论文提问——答案会标注来源、页码和实际检索轮次。"
    )
    picked = st.pills("试试这些问题：", list(SUGGESTIONS), label_visibility="collapsed")
    if picked:
        ask(SUGGESTIONS[picked])
        st.rerun()
else:
    render_history()

# 这行提示要放在 chat_input 之前：输入框是固定在页面底部的，写在它后面会被压住
if st.session_state.focus_label:
    focus_columns = st.columns([6, 1], vertical_alignment="center")
    with focus_columns[0]:
        st.caption(f"当前聚焦：{st.session_state.focus_label}"
                   " · 提问会优先在这一篇里检索")
    with focus_columns[1]:
        if st.button("回到全库", key="clear_focus"):
            st.session_state.focus_paper_id = None
            st.session_state.focus_label = None
            st.rerun()

submission = st.chat_input(
    "问点什么，或把 PDF 拖进来、粘贴 arXiv 链接…",
    accept_file=True,
    file_type=["pdf"],
    submit_mode="stop",
)

if submission:
    files = submission.files or []
    text = (submission.text or "").strip()

    if files:
        ingest_pdf(files)

    if not text:
        st.rerun()                      # 只是上传了附件，没有要问的问题
    elif is_arxiv_url(text) and not files:
        ingest_arxiv(text)              # 贴链接 = 加入知识库
        st.rerun()
    else:
        ask(text)
