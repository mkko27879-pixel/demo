"""query 路由里的纯函数测试。

这个模块 import 时会连 Milvus 并构造 agent，所以跑之前要确保 Milvus 在运行。
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.services import paper_registry
from app.routers.query import (
    MAX_SEARCH_ROUNDS,
    MAX_PAGES_PER_QUERY,
    _final_answer,
    _message_text,
    _reserve_round,
    _search_state,
    extract_sources,
    view_page_image,
)

# 这几条测试要用一篇真实入库的论文（本地跑过 papers 接口之后才有）
SAMPLE_PAPER = "arxiv:2601.00597"


@pytest.mark.skipif(paper_registry.get_paper(SAMPLE_PAPER) is None,
                    reason="本地还没有入库这篇样例论文")
def test_view_page_is_limited():
    """看图必须限次：实测看一次图的 token 能顶几十次纯文本问答。"""
    state = {"default_top_k": 3, "hits": [], "rounds": [], "pages_viewed": 0}
    token = _search_state.set(state)
    try:
        for _ in range(MAX_PAGES_PER_QUERY):
            result = view_page_image.invoke({"paper_id": SAMPLE_PAPER, "page": 1})
            assert isinstance(result, list)
            assert result[0]["type"] == "image_url"
            assert result[0]["image_url"]["url"].startswith("data:image/png;base64,")

        blocked = view_page_image.invoke({"paper_id": SAMPLE_PAPER, "page": 2})
        assert isinstance(blocked, str) and "已经看过" in blocked
    finally:
        _search_state.reset(token)


def test_reserve_round_blocks_over_limit():
    """并行 tool_call 时，名额是先占用后判断的，超出的调用必须被拒。"""
    state = {"default_top_k": 3, "hits": [], "rounds": []}
    token = _search_state.set(state)
    try:
        slots = [_reserve_round(f"q{i}", 3) for i in range(MAX_SEARCH_ROUNDS)]
        assert all(s is not None for s in slots)

        assert _reserve_round("超出的那一次", 3) is None
        assert len(state["rounds"]) == MAX_SEARCH_ROUNDS   # 被拒的那次不留下痕迹
    finally:
        _search_state.reset(token)


def test_reserve_round_works_without_state():
    """工具被单独调用（没有请求上下文）时不该崩，只是不记录。"""
    assert _reserve_round("随便问问", 3) == {}


def test_extract_sources_dedupes_and_sorts_by_score():
    hits = [
        {"text": "旧的", "source": "arxiv:1", "chunk_id": 3, "page": 2, "score": 0.40},
        {"text": "新的", "source": "arxiv:1", "chunk_id": 3, "page": 2, "score": 0.62},
        {"text": "另一篇", "source": "upload:x", "chunk_id": 1, "page": 1, "score": 0.55},
    ]
    sources = extract_sources(hits)

    assert [s.source for s in sources] == ["arxiv:1", "upload:x"]
    assert sources[0].score == 0.62            # 同一块保留最高分那次
    assert sources[0].text == "新的"


def test_final_answer_skips_empty_ai_message():
    messages = [HumanMessage("问题"), AIMessage(content=""), AIMessage(content="真正的答案")]
    assert _final_answer(messages) == "真正的答案"


def test_final_answer_returns_empty_when_no_text():
    messages = [HumanMessage("问题"), AIMessage(content="")]
    assert _final_answer(messages) == ""


def test_message_text_handles_content_blocks():
    message = AIMessage(content=[{"type": "text", "text": "第一段"},
                                 {"type": "text", "text": "第二段"}])
    assert _message_text(message) == "第一段\n第二段"


def test_message_text_handles_plain_string():
    assert _message_text(AIMessage(content="  普通文本  ")) == "普通文本"
