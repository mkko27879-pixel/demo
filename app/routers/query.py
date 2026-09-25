import logging
import json
import os
from contextvars import ContextVar

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.services import chat_store, paper_registry
from app.services.page_renderer import render_page_data_url
from app.services.vector_store import search_similar_chunks

load_dotenv()

router = APIRouter()

logger = logging.getLogger(__name__)

# DeepSeek 的 OpenAI 兼容接口，地址和模型名都能在 .env 里覆盖
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# 当前 key 可用的模型只有两个：deepseek-flash（多模态，支持图片输入）和
# deepseek-v4-pro。注意接口对不认识的名字不报错，而是悄悄回退到
# deepseek-flash，所以名字写错了往往看起来"能跑但效果不对"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

NOT_FOUND_ANSWER = "未找到相关论文内容，请先上传论文。"
# 模型只发了工具调用、没写正文时的兜底文案，避免返回 None 让前端崩溃
EMPTY_ANSWER_HINT = "模型没有返回文字答案，请重试或换个问法。"

# 一次问答允许的最大检索次数。这个上限是防呆用的：不给上限时模型偶尔会
# 一直换关键词搜下去，把一次问答拖成十几轮。次数本身仍由模型自主决定。
MAX_SEARCH_ROUNDS = 5
DEFAULT_TOP_K = 3
MAX_TOP_K = 10
# 一次问答最多接受多少条历史消息。长对话会把 token 吃光，也容易让模型跑偏
MAX_HISTORY_MESSAGES = 20
# 一次问答最多渲染几页图。实测看一次图的开销能顶几十次纯文本问答
# （249k vs 6k tokens），不设上限的话模型可能一页页翻过去，账单很难看。
MAX_PAGES_PER_QUERY = 2


# ---- 检索记录 --------------------------------------------------------------
# 工具每被调用一次，就把这一轮的命中和元信息写进当前请求的 state。
# 用 ContextVar 而不是模块级全局变量，是为了并发请求之间不串数据；
# 记录的是同一个可变 dict/list 对象，所以子上下文里的 append 也能被本请求读到。
_search_state: ContextVar[dict | None] = ContextVar("_search_state", default=None)


def _new_state(default_top_k: int) -> dict:
    return {"default_top_k": default_top_k, "hits": [], "rounds": [], "pages_viewed": 0}


def _reserve_round(query: str, top_k: int) -> dict | None:
    """先占住一个检索名额，返回可以后续回填的占位记录；超过上限时返回 None。

    为什么不是"检查完再检索"：模型会在同一批里并行发多个 tool_call，
    每个调用各自做一次"当前轮次 < 上限"的检查都会通过，然后一起跑完、一起记录，
    实测 5 轮的上限被跑成了 6 轮。先 append 占位再判断，并发的调用里就只有
    名额够的那个能留下，其余立刻被拒。
    """
    state = _search_state.get()
    if state is None:
        return {}

    slot = {"query": query, "top_k": top_k, "hit_count": 0, "note": "检索中"}
    state["rounds"].append(slot)
    if len(state["rounds"]) > MAX_SEARCH_ROUNDS:
        state["rounds"].remove(slot)
        return None
    return slot


def _format_hits(hits: list[dict]) -> str:
    """把检索结果拼成带来源标记的文本，模型照抄标记即可引用到具体片段。

    这里只给来源标记和页码，不给相似度：这段文字是喂给模型看的，里面每个字符它
    都可能原样抄进答案——实测"（第 2 页，相似度 0.7546）"就这么出现在了正文里。
    页码留着有用（用户能照着翻），相似度对用户没有任何意义。
    """
    lines = ["以下是参考资料。只用于核对事实，回答时提炼要点概括，禁止逐段转述或复述这些资料："]
    for index, hit in enumerate(hits, start=1):
        page = int(hit.get("page", -1))
        head = f"[{hit['source']}#{hit['chunk_id']}]"
        if page > 0:
            head += f" 第 {page} 页"
        # 用醒目的分隔线把每段资料框起来：之前片段头和正文长得一样，模型容易
        # 把它当成"该输出的内容"而直接复述
        lines.append(f"\n--- 资料 {index}：{head} ---\n{hit['text']}")
    return "\n".join(lines)


@tool
def search_papers(question: str, paper_id: str = "", top_k: int = 0) -> str:
    """在已上传的论文库里做一次语义检索，返回最相关的原文片段。

    需要用的情况：问题涉及论文的方法、实验、数据、结论、公式等具体内容时，
    必须先调用本工具拿到原文片段，不能凭记忆作答。
    不需要用的情况：寒暄、或明显与库内论文无关的问题，直接回答即可。

    可以调用多次：第一次没拿到能回答问题的片段时，换个关键词、换个说法再查，
    不同问句会召回不同片段。

    追问要先把指代补全：用户问"那它的实验结果呢"，不能直接拿这句话去检索，
    必须先把"它"换成上文里具体的论文、方法或实验名。

    Args:
        question: 本次检索的问句原文，写成信息完整的一句话。
        paper_id: 只在某一篇论文里检索，如 "arxiv:2601.00597"、"upload:3f2a9c1b"；
                  不知道论文 id、或者想跨全库找，就留空。
        top_k: 本次返回几条片段，0 表示用服务端默认值。
    """
    state = _search_state.get()
    k = top_k if top_k and top_k > 0 else (state or {}).get("default_top_k", DEFAULT_TOP_K)
    k = max(1, min(int(k), MAX_TOP_K))

    # 上限超了就明确告诉模型停手，让它基于已有片段收尾
    slot = _reserve_round(question, k)
    if slot is None:
        return (
            f"已达到本次问答的检索次数上限（{MAX_SEARCH_ROUNDS} 次），不要继续调用本工具。"
            "请基于已经拿到的片段作答；片段不足就直说资料不够。"
        )

    try:
        hits = search_similar_chunks(question, top_k=k, paper_id=paper_id or None)
    except Exception as e:
        # 检索失败必须如实回传给模型，否则它会把"没搜到"当成"库里没有"，
        # 直接回一句"未找到相关论文内容"
        slot["note"] = f"检索失败：{type(e).__name__}: {e}"
        return f"{slot['note']}。可以换个问法再试一次。"

    if not hits:
        slot["note"] = "没有检索到相关片段，可能是库内没有收录相关内容，或问法差异太大"
        return f"{slot['note']}。可以换个关键词再试一次。"

    slot["hit_count"] = len(hits)
    slot["note"] = None
    if state is not None:
        state["hits"].extend(hits)
    return _format_hits(hits)


@tool
def list_library_papers() -> str:
    """列出知识库里已经入库的论文（paper_id、标题或文件名、页数、块数）。

    用来回答"库里有哪些论文""有没有讲 XX 的论文"这类问题，
    也可以在你不确定某篇论文的 paper_id 时先调用它确认。
    """
    records = [r for r in paper_registry.list_papers() if r["status"] == "indexed"]
    if not records:
        return "知识库里还没有已入库的论文。"

    lines = []
    for r in records:
        title = r["title"] or r["original_filename"] or "（无标题）"
        lines.append(f"- {r['paper_id']}｜{title}｜{r['page_count']} 页 / {r['chunk_count']} 块")
    return "\n".join(lines)


@tool
def view_page_image(paper_id: str, page: int) -> list:
    """把论文的某一页渲染成图片直接看。

    什么时候用：问公式、表格、图里的内容，而文本检索拿到的片段明显残缺时——
    PDF 里的公式和表格经常提取成乱码或串行，靠文本永远读不对，看一眼原页最省事。
    什么时候不用：普通正文用 search_papers 就够了，图片会显著增加成本。

    Args:
        paper_id: 论文 id，如 "arxiv:2601.00597"、"upload:3f2a9c1b"。
        page: 页码，从 1 开始。search_papers 返回的片段里带 page 字段，直接用它。
    """
    record = paper_registry.get_paper(paper_id)
    if record is None:
        return f"找不到这篇论文：{paper_id}"
    if not os.path.isfile(record["file_path"]):
        return f"磁盘上找不到 {paper_id} 的 PDF，可能要重新上传。"

    # 成本闸门：看一页图的开销能顶几十次文本检索，必须限制次数
    state = _search_state.get()
    if state is not None:
        if state.get("pages_viewed", 0) >= MAX_PAGES_PER_QUERY:
            return (f"本次问答已经看过 {MAX_PAGES_PER_QUERY} 页图了，不要再调用本工具。"
                    "请基于已有的图像和文本作答；确实看不清就如实说明看不清。")
        state["pages_viewed"] = state.get("pages_viewed", 0) + 1

    try:
        data_url = render_page_data_url(record["file_path"], int(page))
    except Exception as e:
        return f"渲染第 {page} 页失败：{type(e).__name__}: {e}"

    return [{"type": "image_url", "image_url": {"url": data_url}}]


# 系统提示词决定 agent 的自主行为：先判断要不要查库，片段不足时换问法再查，
# 每处结论都要带 [paper_id#chunk_id] 形式的来源标记。
SYSTEM_PROMPT = (
    "你是一个严谨的论文问答助手，回答范围是知识库里已经上传的论文。\n"
    "第一步先判断这个问题要不要查库：\n"
    "当用户问题不涉及论文数据来源时，不必输出数据来源标识，直接回答问题。\n"
    "1) 只要涉及论文的方法、实验、数据、结论、公式等具体内容，就必须调用 search_papers；\n"
    "2) 问“库里有哪些论文”这类清单问题，调用 list_library_papers；\n"
    "3) 纯寒暄、或明显与库内论文无关的问题，直接回答，不必检索。\n"
    "问题里提到某一篇具体的论文时，先用 list_library_papers 确认它的 paper_id，"
    "再用 search_papers 的 paper_id 参数把检索限定在那篇里。\n"
    "4) 问公式、表格、图里的内容，而文本片段明显残缺（公式是乱码、表格串了行）时，"
    "先用 search_papers 定位到页码，再调 view_page_image 看那一页的原始图像来回答。"
    f"图片的成本极高，一次问答最多看 {MAX_PAGES_PER_QUERY} 页，只在文本确实读不出来时才用，"
    "优先把文本片段用足再考虑看图。\n"
    "注意对话历史：用户追问时（“它”“那篇论文”“实验结果呢”），要结合上文把指代补全"
    "再组织检索问句。\n"
    "4) 如果本轮问题开头带【当前聚焦论文：xxx】标记，说明用户刚选定了那一篇，"
    "“这篇”“它”默认就指它：直接用 search_papers 的 paper_id 参数把检索限定在那篇，"
    "不要反问用户指的是哪篇。用户明确提到别的论文时，以用户说的为准。\n"
    "检索次数由你自己决定：第一次没拿到能支撑回答的片段时，换关键词或换角度再调用，"
    f"最多 {MAX_SEARCH_ROUNDS} 次，够用就停。\n"
    "只使用资料里的信息，资料里没有的就直说不知道。\n"
    "最重要的一条：绝对不要输出资料原文。资料是给你读的素材，不是答案的一部分——"
    "不管问的是概括还是细节，答案里都不能出现资料里的原句、段落、"
    "“--- 资料 N ---”这类分隔标记，也不能把资料开头照搬过来。"
    "答案只能用你自己组织的语言写。\n"
    "回答要求：1) 结论先行，第一句话就概括主旨；"
    "2) 总结类问题（“主要讲了什么”“介绍一下这篇论文”）用 3~5 个要点，"
    "每个要点一两句话；"
    "3) 来源标记 [paper_id#chunk_id] 跟在结论后面就行，同一句话最多一个，"
    "不要编造片段号；"
    "4) 不要用“根据资料”“资料 X 提到”这类话开头。\n"
    "每个要点独占一行并以 - 开头，要点之间不要空行。\n"
    f"如果始终没有检索到任何片段，只回答“{NOT_FOUND_ANSWER}”"
)

# 初始化模型
model = init_chat_model(
    model=DEEPSEEK_MODEL,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=DEEPSEEK_BASE_URL,
)

# agent 只建一次复用，每次请求的检索记录靠 _search_state 隔离
agent = create_agent(
    model=model,
    tools=[search_papers, list_library_papers, view_page_image],
    system_prompt=SYSTEM_PROMPT,
)


class ChatMessage(BaseModel):
    role: str = Field(..., description="user 或 assistant")
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户原始问句")
    history: list[ChatMessage] = Field(default_factory=list,
                                       description="之前的对话，用来支持追问")
    conversation_id: str | None = Field(None, description="要续接的会话 id；不传则新建会话")
    focus_paper_id: str | None = Field(
        None, description="当前聚焦的论文 id（比如刚上传的那篇），会收窄 agent 的检索范围")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K, description="每次检索返回的片段条数")


class SourceChunk(BaseModel):
    text: str
    source: str             # paper_id，如 arxiv:2601.00597 或 upload:3f2a9c...(uuid)
    chunk_id: int
    page: int               # 页码，从 1 开始；-1 表示未知
    page_start: int = -1    # 在本页文本里的起始字符下标，前端据此回原文高亮
    page_end: int = -1      # 结束下标（不含）
    score: float


class SearchRound(BaseModel):
    query: str              # 这一轮实际用的检索问句
    top_k: int
    hit_count: int
    note: str | None = None  # 出错或空结果时的说明


class QueryResponse(BaseModel):
    conversation_id: str     # 会话 id，下一轮追问带上它
    question: str            # 用户原问句
    answer: str              # LLM 最终回答
    sources: list[SourceChunk]
    search_rounds: list[SearchRound]  # agent 实际查了几次库、每轮命中多少
    usage: dict              # 本次问答的 token 用量，用来算钱


def extract_sources(hits: list[dict]) -> list[SourceChunk]:
    """把各轮检索命中的片段去重成来源列表。

    同一个 chunk 常被多个不同问句命中，这里按 (source, chunk_id) 去重并保留
    相似度最高的一次，最后按相似度从高到低排序。来源直接取工具的真实返回值，
    不解析模型写出来的引用标记，这样模型漏标或标错时来源依然可信。
    """
    best: dict[tuple[str, int], SourceChunk] = {}
    for hit in hits:
        key = (str(hit["source"]), int(hit["chunk_id"]))
        chunk = SourceChunk(
            text=hit["text"],
            source=key[0],
            chunk_id=key[1],
            page=int(hit.get("page", -1)),
            page_start=int(hit.get("page_start", -1)),
            page_end=int(hit.get("page_end", -1)),
            score=float(hit["score"]),
        )
        if key not in best or chunk.score > best[key].score:
            best[key] = chunk
    return sorted(best.values(), key=lambda c: c.score, reverse=True)


def _message_text(message) -> str:
    """取消息里的纯文本。多模态模型可能把 content 返回成内容块列表。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text") or "")
        else:
            parts.append(getattr(block, "text", "") or "")
    return "\n".join(p for p in parts if p).strip()


def _final_answer(messages) -> str:
    """取最后一条有正文的 AI 消息。只调工具、没写正文的消息要跳过。"""
    for message in reversed(messages):
        if getattr(message, "type", "") != "ai":
            continue
        text = _message_text(message)
        if text:
            return text
    return ""


def _sum_usage(messages) -> dict:
    """累加本次问答的 token 用量。

    一次问答会产生多条 AIMessage（每调用一轮工具就多一条），每条各自带
    usage_metadata，要全部加起来才是真实成本。多轮检索很费钱，这笔账得让
    调用方能看见。
    """
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for message in messages:
        usage = getattr(message, "usage_metadata", None) or {}
        for key in total:
            total[key] += usage.get(key) or 0
    return total


def _sse(payload: dict) -> str:
    """把一条消息打包成 SSE 帧。用 type 字段区分事件类型，客户端解析简单。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _prepare_context(request: QueryRequest, question: str) -> tuple[str, list[dict]]:
    """确定本次问答用哪个会话、带哪些历史消息。

    给了 conversation_id 就以服务端存的历史为准（客户端不用自己维护上下文）；
    没给就新开一个会话，并把请求里带的 history 当前置上下文（无状态用法）。
    """
    if request.conversation_id:
        if chat_store.get_conversation(request.conversation_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in chat_store.get_messages(request.conversation_id,
                                             limit=MAX_HISTORY_MESSAGES)
        ]
        return request.conversation_id, history

    conversation_id = chat_store.create_conversation(title=question[:60])
    # role 必须走白名单：客户端如果能塞 system 消息进来，就等于可以直接顶掉
    # 系统提示词（最典型的 prompt injection 入口）。
    history = [
        {"role": m.role, "content": m.content}
        for m in request.history[-MAX_HISTORY_MESSAGES:]
        if m.role in ("user", "assistant")
    ]
    return conversation_id, history


def _save_turn(conversation_id: str, question: str, answer: str,
               sources: list[SourceChunk]) -> None:
    """把这一轮问答写进会话历史，下一次追问就能用上。"""
    chat_store.add_message(conversation_id, "user", question)
    chat_store.add_message(conversation_id, "assistant", answer,
                           sources=[s.model_dump() for s in sources])


def _with_focus(question: str, focus_paper_id: str | None) -> str:
    """给问题加上"当前聚焦论文"的标记。

    只发给模型，不改用户的问题本身：会话历史里存的、接口返回的都是用户原话。
    前端在上传完一篇论文后会把它的 id 带过来，这样用户接着问"这篇讲了什么"，
    agent 知道"这篇"是谁，不会反问"你指的是哪篇"。
    """
    if not focus_paper_id:
        return question
    return f"【当前聚焦论文：{focus_paper_id}】\n{question}"


@router.post("/paper", response_model=QueryResponse)
def query_paper(request: QueryRequest):
    """非流式问答：答案全部生成完再一次性返回。"""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    conversation_id, history = _prepare_context(request, question)
    messages = history + [
        {"role": "user", "content": _with_focus(question, request.focus_paper_id)}
    ]

    state = _new_state(request.top_k)
    token = _search_state.set(state)
    try:
        result = agent.invoke({"messages": messages})
    except Exception as e:
        # 这里是模型/服务端的故障，不该按客户端参数错误返回 400
        raise HTTPException(status_code=500, detail=f"调用模型失败：{type(e).__name__}: {e}")
    finally:
        # state 对象我们已经持有引用，reset 只是把 ContextVar 还给外层
        _search_state.reset(token)

    sources = extract_sources(state["hits"])
    answer = _final_answer(result["messages"])
    if not answer:
        answer = NOT_FOUND_ANSWER if not sources else EMPTY_ANSWER_HINT

    usage = _sum_usage(result["messages"])
    logger.info("问答完成：会话 %s，引用 %d 条来源，检索 %d 轮，token %s",
                conversation_id, len(sources), len(state["rounds"]), usage)

    _save_turn(conversation_id, question, answer, sources)
    return QueryResponse(
        conversation_id=conversation_id,
        question=question,
        answer=answer,
        sources=sources,
        search_rounds=[SearchRound(**r) for r in state["rounds"]],
        usage=usage,
    )


@router.post("/paper/stream")
async def query_paper_stream(request: QueryRequest):
    """流式问答（SSE）：边检索边推事件，前端不用干等十几秒。

    事件类型：
      status  开始检索
      sources 检索到的来源（第一段正文出现时就已经确定）
      token   答案增量，客户端拼起来就是完整答案
      error   出错
      done    收尾，带 conversation_id / usage / search_rounds

    用 POST 而不是 GET：问题可能很长，塞进 URL 不合适。浏览器端用
    fetch + ReadableStream 读即可，不依赖 EventSource。
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    conversation_id, history = _prepare_context(request, question)
    messages = history + [
        {"role": "user", "content": _with_focus(question, request.focus_paper_id)}
    ]

    async def event_stream():
        state = _new_state(request.top_k)
        context_token = _search_state.set(state)
        parts: list[str] = []
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        sent_sources = False
        error = None

        try:
            yield _sse({"type": "status", "text": "正在检索论文库…"})
            async for chunk, _meta in agent.astream({"messages": messages},
                                                    stream_mode="messages"):
                # stream_mode="messages" 不只会推模型的 token，还会把工具返回的
                # ToolMessage 一起推出来——而工具返回的正是检索到的资料原文。
                # 不过滤的话，资料会被当成答案的一部分流出去，用户看到的就是
                # "开头一大段原文、最后一句才是总结"。改多少遍提示词都没用，
                # 因为问题根本不在模型身上。
                if isinstance(chunk, ToolMessage):
                    continue
                text = _message_text(chunk)
                if text:
                    if not sent_sources and state["hits"]:
                        yield _sse({
                            "type": "sources",
                            "sources": [s.model_dump() for s in extract_sources(state["hits"])],
                        })
                        sent_sources = True
                    parts.append(text)
                    yield _sse({"type": "token", "text": text})

                # 流式返回时 usage 只挂在部分块上，取最后一个非空的就是本轮总量
                chunk_usage = getattr(chunk, "usage_metadata", None)
                if chunk_usage:
                    usage = {k: chunk_usage.get(k) or 0 for k in usage}
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        finally:
            _search_state.reset(context_token)

        if error:
            yield _sse({"type": "error", "text": error})
            return

        sources = extract_sources(state["hits"])
        answer = "".join(parts).strip()
        if not answer:
            answer = NOT_FOUND_ANSWER if not sources else EMPTY_ANSWER_HINT

        _save_turn(conversation_id, question, answer, sources)
        logger.info("流式问答完成：会话 %s，引用 %d 条来源，检索 %d 轮，token %s",
                    conversation_id, len(sources), len(state["rounds"]), usage)

        yield _sse({
            "type": "done",
            "conversation_id": conversation_id,
            "answer": answer,
            "sources": [s.model_dump() for s in sources],
            "search_rounds": state["rounds"],
            "usage": usage,
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")
