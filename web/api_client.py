"""后端 API 封装。

所有网络调用集中在这里，界面层不直接碰 httpx——这样换后端地址、加超时、
处理错误都只有一处要改。
"""

import json
import os
from collections.abc import Iterator

import httpx

# 后端默认跑在本机 8000 端口；用环境变量可以指到别的地方
BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# 读超时给得很大：一次问答要跑多轮检索加模型生成，实测能到 20-30 秒；
# 上传 PDF 后的解析更慢，所以这里不能按普通接口的 5-10 秒来设。
_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=120.0, pool=10.0)
_SHORT_TIMEOUT = httpx.Timeout(connect=3.0, read=10.0, write=10.0, pool=5.0)


def _url(path: str) -> str:
    return f"{BASE_URL}{path}"


def backend_ready() -> bool:
    """后端是否可达。侧边栏用它决定要不要给用户提示。"""
    try:
        return httpx.get(_url("/api/health"), timeout=_SHORT_TIMEOUT).status_code == 200
    except Exception:
        return False


def list_papers() -> list[dict]:
    resp = httpx.get(_url("/api/papers"), timeout=_SHORT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["papers"]


def add_arxiv_paper(url: str) -> dict:
    resp = httpx.post(_url("/api/papers"), json={"url": url}, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def upload_pdf(filename: str, content: bytes) -> dict:
    resp = httpx.post(
        _url("/api/papers/upload"),
        files={"file": (filename, content, "application/pdf")},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def delete_paper(paper_id: str) -> None:
    resp = httpx.delete(_url(f"/api/papers/{paper_id}"), timeout=_TIMEOUT)
    resp.raise_for_status()


def list_conversations() -> list[dict]:
    resp = httpx.get(_url("/api/conversations"), timeout=_SHORT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["conversations"]


def get_conversation(conversation_id: str) -> dict:
    resp = httpx.get(_url(f"/api/conversations/{conversation_id}"), timeout=_SHORT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def delete_conversation(conversation_id: str) -> None:
    resp = httpx.delete(_url(f"/api/conversations/{conversation_id}"), timeout=_SHORT_TIMEOUT)
    resp.raise_for_status()


def stream_tokens(question: str, conversation_id: str | None,
                  top_k: int, sink: dict,
                  focus_paper_id: str | None = None) -> Iterator[str]:
    """提问并把答案逐字吐出来。

    后端的 SSE 事件有 status / sources / token / error / done 五种。这里只把
    token 转成字符串给 st.write_stream 渲染，其余事件统统塞进 sink，由界面在
    流结束后补渲染——write_stream 执行期间没法更新别的元素。

    sink 是调用方传进来的 dict，流结束后里面会有 sources / search_rounds /
    usage / conversation_id / error。

    focus_paper_id 是"当前聚焦的论文"（比如刚上传的那篇）：后端会把这个问题
    限定在那篇里检索，用户问"这篇讲了什么"时 agent 就不会反问指哪一篇。
    """
    payload: dict = {"question": question, "top_k": top_k}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    if focus_paper_id:
        payload["focus_paper_id"] = focus_paper_id

    with httpx.stream("POST", _url("/api/query/paper/stream"),
                      json=payload, timeout=_TIMEOUT) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                continue

            kind = event.get("type")
            if kind == "token":
                yield event.get("text", "")
            elif kind == "error":
                sink["error"] = event.get("text", "未知错误")
                yield f"\n\n> 出错了：{sink['error']}"
            elif kind in ("sources", "done"):
                sink.update(event)
