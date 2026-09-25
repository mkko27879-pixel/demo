"""界面片段。放在这里是为了让 streamlit_app.py 保持"从上到下的脚本"形态。"""

import html
import re
import time

import streamlit as st

import api_client as api

# 后端状态 -> 界面上的中文标签
STATUS_LABEL = {
    "uploaded": ":gray[待解析]",
    "downloading": ":blue[下载中]",
    "indexing": ":orange[解析中]",
    "indexed": ":green[已就绪]",
    "failed": ":red[失败]",
}

# 认得出 arXiv 链接就够了：完整的 arxiv.org 链接，或者直接贴编号
_ARXIV_URL = re.compile(r"arxiv\.org/(abs|pdf|html)/|^\d{4}\.\d{4,5}(v\d+)?$")
# 单个列表项：缩进 + 无序标记（- * +）或有序标记（1. / 12.）+ 空格
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d{1,3}\.)\s+\S")
# 列表标记直接跟在句末标点或右括号之后，中间可能连空格都没有：
#   "……复杂度[upload:x#25]。-下一个要点"（模型实际就是这么写的）
# 前面限定是标点/括号，所以 "20-30"、"Lenz-Jensen" 这类正文里的连字符不会中招
_INLINE_LIST_ITEM = re.compile(r"(?<=[。！？；：!?;\]】）)])\s*[-*+]\s*(?=\S)")
# 模型写的引用：[upload:58975818caec48eaa4e48384b238ca07#6]
_CITATION = re.compile(r"\[([^\[\]#\s]+)#(\d+)\]")
# 加粗小标题后面直接接内容："**主要结论**-弱相互作用时……"
_HEADING_GLUED = re.compile(r"(\*\*[^*]+\*\*)[-—](?=\S)")


def format_answer(answer: str, sources: list[dict]) -> tuple[str, list[dict]]:
    """整理答案正文和来源列表，做两件模型不管的事。

    1. 长引用换编号。[upload:58975818caec48eaa4e48384b238ca07#6] 有 46 个字符，
       一篇回答里出现十几次，正文会被这串 id 切得读不下去。这里按首次出现的
       顺序换成 [1][2]，来源列表也按同样顺序重排，两边对得上号。
    2. 拆开粘在加粗标题后的破折号。"**主要结论**-弱相互作用时……" 是模型常犯的
       写法，标题和正文糊在一起，看着像错字。
    """
    text = _HEADING_GLUED.sub(r"\1\n\n", strip_copied_lead(answer, sources))

    order: list[tuple[str, int]] = []

    def renumber(match: re.Match) -> str:
        key = (match.group(1), int(match.group(2)))
        if key not in order:
            order.append(key)
        return f"[{order.index(key) + 1}]"

    text = _CITATION.sub(renumber, text)

    by_key = {(s["source"], s["chunk_id"]): s for s in sources}
    linked = [{**by_key[key], "index": index}
              for index, key in enumerate(order, start=1) if key in by_key]
    # 检索到、但正文没引用到的片段附在后面，编号接着排
    for source in sources:
        if (source["source"], source["chunk_id"]) not in order:
            linked.append({**source, "index": len(linked) + 1})
    return text, linked


def strip_copied_lead(answer: str, sources: list[dict], min_chars: int = 60) -> str:
    """切掉答案开头照搬资料原文的那一截。

    模型读完片段后经常先把片段开头复述一遍再给结论——问"这篇主要讲了什么"时尤其
    明显，用户看到的就是"一大段原文，最后才是总结"。原因是指示打架："只能依据
    片段作答、不许编造"让它倾向引用原文自证，而"用你自己的话总结"又不让它抄，
    于是它折中成先引后结。提示词里说过好几轮都没压住，所以这里直接比对：答案开头
    能和某个片段的首部逐字对上多少字，就把那一截切掉。

    比对长度不到 min_chars 就不动，免得把正常的一句短引用也切了。
    """
    for source in sources:
        text = (source.get("text") or "").strip()
        if not text:
            continue
        matched = 0
        while (matched < len(answer) and matched < len(text)
               and answer[matched] == text[matched]):
            matched += 1
        if matched >= min_chars:
            return answer[matched:].lstrip()
    return answer


def normalize_markdown(text: str) -> str:
    """给列表补空行，让 Markdown 真的能渲染成列表。

    CommonMark 里有序列表可以中断段落，**无序列表不行**。所以模型写的

        主要发现
        - 弱相互作用时……
        - 能级统计……

    会渲染成「主要发现 - 弱相互作用时…… - 能级统计……」一整段，看着就是
    没结构的一坨。提示词里要求过模型自己留空行，但它经常不听，所以在渲染前
    统一补上——这比反复调提示词可靠。

    要处理的两种坏写法：
      1. 列表标记紧跟在句号或右括号后面（"……复杂度[#25]。-下一个要点"），
         连行首都算不上，得先拆到新行、并补上标记后的空格；
      2. 列表开始前没有空行。

    反过来，列表项之间**不能**留空行：那会让 markdown 渲染成"松散列表"，
    每个要点之间空一大截，看起来比不给列表还乱。
    """
    text = _INLINE_LIST_ITEM.sub("\n- ", text)

    out: list[str] = []
    in_list = False
    for line in text.split("\n"):
        if _LIST_ITEM.match(line):
            if in_list:
                while out and not out[-1].strip():   # 列表中间的空行直接丢掉
                    out.pop()
            elif out and out[-1].strip():            # 列表开头补一个空行
                out.append("")
            out.append(line)
            in_list = True
            continue

        if not line.strip():
            if not in_list:                          # 非列表区域的空行保留（分段用）
                out.append("")
            continue

        if in_list:
            # 列表到此结束，补一个空行收尾；否则紧跟着的这一段会被 markdown
            # 当成最后一个列表项的续行
            out.append("")
        in_list = False
        out.append(line)

    return "\n".join(out)


def is_arxiv_url(text: str) -> bool:
    """输入框里贴的是不是 arXiv 链接。

    贴链接比找按钮顺手——所以不单独做"添加论文"的入口，直接在对话栏里认。
    """
    return bool(_ARXIV_URL.search(text.strip()))


def paper_title(paper: dict) -> str:
    return paper.get("title") or paper.get("original_filename") or paper["paper_id"]


def render_user_message(text: str) -> None:
    """用户消息靠右，用气泡背景和 AI 的纯文本区分；两边都不带头像。

    为什么不用 st.columns([1, 2]) 分栏：它在窄窗口下会把两列堆叠成一列，
    右对齐直接消失（实测 480px 宽的面板就是这样）。这里用 margin-left:auto
    加最大宽度把气泡推到右边，多窄的窗口都成立。

    颜色按明暗模式各写一套：Streamlit 1.64 没在 :root 上暴露主题色变量
    （实测 --primary-color、--secondary-background-color 都取不到），所以跟着
    .streamlit/config.toml 里的主题色手工配。改主题时这里要一起改。

    外边距是必需的：气泡里没有下边距，markdown 块的上边距也是 0，两者会贴成
    一块，看起来像同一条消息。
    """
    is_dark = st.context.theme.type == "dark"
    bubble = "#2B3040" if is_dark else "#EDF1FE"
    body = html.escape(text).replace("\n", "<br>")
    st.markdown(
        '<div style="width:fit-content; max-width:72%; '
        f"background:{bubble}; border-radius:16px; padding:0.6rem 1rem;"
        # 左外边距必须写在 margin 简写里（第四位 = auto）：简写会覆盖前面单独写的
        # margin-left，写成两条会让气泡掉回左边
        f' margin:0.5rem 0 1.2rem auto;">{body}</div>',
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict]) -> None:
    """来源卡片：页码、相似度、片段预览。

    page_start/page_end 是后端存的"块在该页文本里的字符区间"，前端拿到它就能在
    原页里定位到那一段；这里先把它显示出来，等后端加上 PDF 文件接口就能跳转高亮。
    """
    if not sources:
        return

    with st.expander(f"引用来源 · {len(sources)} 处"):
        for source in sources:
            page = source.get("page", -1)
            where = f"第 {page} 页" if page and page > 0 else "页码未知"
            span = ""
            if source.get("page_start", -1) >= 0:
                span = f" · 字符 {source['page_start']}-{source['page_end']}"

            st.markdown(
                f"**[{source.get('index', '-')}]** "
                f"`{source['source']}#{source['chunk_id']}` "
                f"{where} · 相似度 {source.get('score', 0):.3f}{span}"
            )
            text = source.get("text", "")
            st.caption(text[:280] + ("…" if len(text) > 280 else ""))


def render_thinking(data: dict, *, live: bool = False) -> None:
    """答案上方的折叠块，显示 agent 实际检索了什么。

    live=True 表示这是刚跑完的那一轮，额外带上 token 用量。
    """
    rounds = data.get("search_rounds") or []
    sources = data.get("sources") or []
    usage = data.get("usage") or {}
    if not rounds and not sources:
        return

    label = f"检索 {len(rounds)} 轮 · 引用 {len(sources)} 处"
    if live and usage.get("total_tokens"):
        label += f" · {usage['total_tokens']:,} tokens"

    with st.status(label, type="compact") as status:
        for index, item in enumerate(rounds, start=1):
            note = f" —— {item['note']}" if item.get("note") else ""
            st.markdown(f"{index}. `{item['query']}` · 命中 {item['hit_count']} 条{note}")
        status.update(label=label, state="complete")


def wait_until_indexed(paper_id: str, timeout: float = 600.0) -> tuple[bool, str]:
    """轮询论文状态直到入库完成，返回 (是否成功, 说明)。

    解析一篇论文要几十秒，期间把状态显示给用户，比让界面卡着强。
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            papers = {p["paper_id"]: p for p in api.list_papers()}
        except Exception as exc:
            return False, f"查询状态失败：{exc}"

        paper = papers.get(paper_id)
        if paper is None:
            return False, "论文不见了，可能已被删除"
        if paper["status"] == "indexed":
            return True, f"{paper['page_count']} 页 / {paper['chunk_count']} 块"
        if paper["status"] == "failed":
            return False, paper.get("error") or "解析失败"
        time.sleep(2)

    return False, "等待超时"


def render_paper_list() -> None:
    """侧边栏的论文列表：只列用户自己加的论文。

    区分靠 created_by 字段，不靠 paper_id 前缀：用户贴 arXiv 链接加进来的论文
    也是 arxiv: 开头，同样要显示。created_by == "system" 的是随应用预置的知识库
    论文，不占用户的列表，但仍然参与检索。

    上传入口在对话栏里，这里只负责看和删。
    """
    st.markdown("#### 论文库")

    try:
        papers = [p for p in api.list_papers() if p.get("created_by") != "system"]
    except Exception as exc:
        st.caption(f":red[读取论文列表失败：{exc}]")
        return

    if not papers:
        st.caption("还没有论文。把 PDF 拖进下面的输入框，或直接粘贴 arXiv 链接。")
        return

    for paper in papers:
        label = STATUS_LABEL.get(paper["status"], paper["status"])
        detail = f"{label} · {paper['page_count']} 页" if paper["status"] == "indexed" else label

        # 用水平容器而不是 st.columns：侧边栏只有 300 多像素宽，比例分栏会被
        # Streamlit 判成太窄而堆叠，删除按钮就掉到下一行去了
        with st.container(horizontal=True, vertical_alignment="center"):
            with st.container():
                st.markdown(f"**{paper_title(paper)}**")
                st.caption(detail)
            if st.button(":material/delete:", key=f"del-{paper['paper_id']}",
                         help="删除这篇论文（向量和文件一起清）"):
                try:
                    api.delete_paper(paper["paper_id"])
                except Exception as exc:
                    st.error(f"删除失败：{exc}")
                else:
                    st.rerun()


def render_conversation_panel(on_new, on_open) -> None:
    """侧边栏的会话列表：只有标题，新的在上。"""
    st.markdown("#### 历史对话")

    if st.button("新对话", icon=":material/add_comment:", width="stretch",
                 key="new_chat"):
        on_new()

    try:
        conversations = api.list_conversations()
    except Exception as exc:
        st.caption(f":red[读取会话失败：{exc}]")
        return

    if not conversations:
        st.caption("还没有历史对话。")
        return

    for conversation in conversations:
        title = conversation.get("title") or "未命名对话"
        # 同上：窄侧边栏里用水平容器，两个按钮才留在同一行
        with st.container(horizontal=True, vertical_alignment="center"):
            # 标题要截短、也不能设 width="stretch"：侧边栏内容区只有 ~290px，
            # 长标题加删除按钮正好超出，按钮就会被挤到下一行。完整标题放在 help 里。
            label = title if len(title) <= 16 else title[:16] + "…"
            if st.button(label, key=f"open-{conversation['conversation_id']}", help=title):
                on_open(conversation["conversation_id"])
            if st.button(":material/delete:", key=f"delc-{conversation['conversation_id']}",
                         help="删除这个会话"):
                try:
                    api.delete_conversation(conversation["conversation_id"])
                except Exception as exc:
                    st.error(f"删除失败：{exc}")
                else:
                    st.rerun()
