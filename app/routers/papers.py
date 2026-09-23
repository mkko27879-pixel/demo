import os
import re
import time

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.pdf_loader import load_pdf
from app.services.text_splitter import split_documents  # 导入切分函数
from app.services.vector_store import insert_chunks, list_papers

router = APIRouter()

# arxiv.org 会解析到多个 Fastly IP，实测建连耗时能从 0.1s 抖到 7s，
# 赶上慢的那个 IP，TLS 握手就会以 "_ssl.c:993: The handshake operation timed out" 告终。
# 所以失败后要换域名重试，一次不成不代表链接有问题。
ARXIV_HOSTS = ["arxiv.org", "export.arxiv.org"]
MAX_ATTEMPTS = 3
# 超时分档：连接阶段短一点快速失败，读取阶段给足时间。
# 原来拍平的 30 秒等于无论卡在哪一步都要等满，还容易正好卡在握手超时上。
DOWNLOAD_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)
# arXiv 建议带上可识别的 User-Agent，默认 UA 容易被限流
ARXIV_USER_AGENT = "llm-paper-agent/0.1 (local RAG demo)"


class PaperRequest(BaseModel):
    url: str


def parse_arxiv_id(url: str) -> str:
    """从各种形式的 arXiv 链接里提取论文 ID。

    支持 /abs/ /pdf/ /html/ 三种路径、带 .pdf 后缀、带版本号 v1/v2，
    以及 hep-th/9901001 这种带分类前缀的旧式编号。
    """
    cleaned = url.strip().split("?")[0].split("#")[0].rstrip("/")
    parts = [p for p in cleaned.split("/") if p]
    if not parts:
        raise ValueError(f"认不出这是个 arXiv 链接：{url}")

    # 末尾的 abs / pdf / html 只是路径标记，真正的位置在它们后面
    if len(parts) >= 2 and parts[-2] in {"abs", "pdf", "html"}:
        paper_id = parts[-1]
    elif len(parts) >= 3 and parts[-3] in {"abs", "pdf", "html"}:
        paper_id = f"{parts[-2]}/{parts[-1]}"  # 旧式编号，分类前缀要保留
    else:
        paper_id = parts[-1]

    if paper_id.endswith(".pdf"):
        paper_id = paper_id[: -len(".pdf")]
    paper_id = re.sub(r"v\d+$", "", paper_id)  # 去掉版本号

    if not re.fullmatch(r"(\d{4}\.\d{4,5}|[a-z-]+(\.[A-Z]{2})?/\d{7})", paper_id):
        raise ValueError(f"认不出这是个 arXiv 链接：{url}")
    return paper_id


def download_pdf(arxiv_id: str, save_path: str) -> int:
    """按域名轮换重试下载 PDF，返回文件字节数。"""
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        host = ARXIV_HOSTS[(attempt - 1) % len(ARXIV_HOSTS)]
        pdf_url = f"https://{host}/pdf/{arxiv_id}.pdf"
        try:
            resp = httpx.get(
                pdf_url,
                follow_redirects=True,
                timeout=DOWNLOAD_TIMEOUT,
                headers={"User-Agent": ARXIV_USER_AGENT},
            )
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(resp.content)
            return len(resp.content)
        except Exception as e:
            last_error = e
            if attempt < MAX_ATTEMPTS:
                time.sleep(attempt)  # 递增等待，顺便让 DNS 重新轮询一次 IP

    raise RuntimeError(
        f"下载 PDF 失败（已重试 {MAX_ATTEMPTS} 次，域名轮换 {ARXIV_HOSTS}）："
        f"{type(last_error).__name__}: {last_error}。"
        f"这种握手超时一般是到 arxiv.org 的链路抖动，稍后重试即可。"
    )


# 路由路径要写成 "/papers"：main.py 里 include_router 带了 prefix="/api"，
# 最终地址才是 POST /api/papers（写成 "" 的话会变成 POST /api，Swagger 里看着就不对）
@router.post("/papers")
def add_paper(request: PaperRequest):
    try:
        arxiv_id = parse_arxiv_id(request.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    save_dir = "data/papers"
    os.makedirs(save_dir, exist_ok=True)
    # 旧式编号里带 "/"，直接当文件名会建出子目录，换成下划线
    save_path = f"{save_dir}/{arxiv_id.replace('/', '_')}.pdf"

    try:
        # 1. 下载 PDF（内部按域名轮换重试）
        download_pdf(arxiv_id, save_path)

        # 2. 加载 PDF 为 Document 列表
        docs = load_pdf(save_path)

        # 3. 切分 Document
        chunks = split_documents(docs)
        # 将切分后的块向量化并存入 Milvus
        vector_count = insert_chunks(chunks, arxiv_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"arxiv_id": arxiv_id, "chunk_count": len(chunks), "vector_count": vector_count}

@router.get("/papers")
def get_papers():
    return {"papers": list_papers()}
