import logging
import os
import re
import time
import uuid

import httpx
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.services import paper_registry
from app.services.paper_indexer import index_paper
from app.services.vector_store import delete_paper_vectors

logger = logging.getLogger(__name__)

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

# 论文标识统一成带前缀的字符串，一眼能看出论文从哪来：
#   arxiv:2601.00597      从 arXiv 下载的
#   upload:3f2a9c...e1    用户本地上传的
# 前缀还有个用处：以后想只在某类论文里检索，可以直接按 "arxiv:%" 前缀过滤。
ARXIV_PREFIX = "arxiv:"
UPLOAD_PREFIX = "upload:"

DOWNLOAD_DIR = "data/papers"
UPLOAD_DIR = "data/papers/uploads"
# 上传大小上限：论文 PDF 极少超过 50MB，不设限等于让人有机会拖垮服务
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
# PDF 文件头魔数，用来挡住"把 docx 改成 .pdf 后缀"这类误传
PDF_MAGIC = b"%PDF-"


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
        number = parts[-1]
    elif len(parts) >= 3 and parts[-3] in {"abs", "pdf", "html"}:
        number = f"{parts[-2]}/{parts[-1]}"  # 旧式编号，分类前缀要保留
    else:
        number = parts[-1]

    if number.endswith(".pdf"):
        number = number[: -len(".pdf")]
    number = re.sub(r"v\d+$", "", number)  # 去掉版本号

    if not re.fullmatch(r"(\d{4}\.\d{4,5}|[a-z-]+(\.[A-Z]{2})?/\d{7})", number):
        raise ValueError(f"认不出这是个 arXiv 链接：{url}")
    return number


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


# ---- 后台任务 --------------------------------------------------------------
# 下载和解析都很慢（下载十几秒、一篇 80 块的论文向量化要几十秒），堵在请求里
# 前端必然超时。所以这两件事都丢进后台，接口立刻返回，进度靠元数据表的 status 查。


def _download_and_index(paper_id: str, arxiv_id: str, save_path: str) -> None:
    """后台任务：先下载再入库。异常只写状态，不再往外抛——响应早就返回了。"""
    logger.info("开始下载 %s（%s）", paper_id, arxiv_id)
    try:
        download_pdf(arxiv_id, save_path)
    except Exception as e:
        logger.warning("下载失败 %s：%s", paper_id, e)
        paper_registry.update_status(paper_id, "failed",
                                     error=f"下载失败：{type(e).__name__}: {e}")
        return
    logger.info("下载完成 %s，开始解析入库", paper_id)
    try:
        index_paper(paper_id, save_path)
    except Exception:
        pass  # index_paper 内部已经把失败原因写进状态了，这里只负责别让线程崩掉


def _index_only(paper_id: str, pdf_path: str) -> None:
    """后台任务：解析本地上传的 PDF。"""
    try:
        index_paper(paper_id, pdf_path)
    except Exception:
        pass


# ---- arXiv 入口 -----------------------------------------------------------


# 路由路径要写成 "/papers"：main.py 里 include_router 带了 prefix="/api"，
# 最终地址才是 POST /api/papers（写成 "" 的话会变成 POST /api，Swagger 里看着就不对）
@router.post("/papers")
def add_paper(request: PaperRequest, background: BackgroundTasks):
    """提交一篇 arXiv 论文：立刻返回 paper_id，下载和入库在后台跑。"""
    try:
        arxiv_id = parse_arxiv_id(request.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    paper_id = f"{ARXIV_PREFIX}{arxiv_id}"
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    # 旧式编号里带 "/"，直接当文件名会建出子目录，换成下划线
    save_path = os.path.join(DOWNLOAD_DIR, f"{arxiv_id.replace('/', '_')}.pdf")

    paper_registry.save_paper(paper_id, "arxiv", save_path,
                              original_filename=f"{arxiv_id}.pdf")
    paper_registry.update_status(paper_id, "downloading")
    background.add_task(_download_and_index, paper_id, arxiv_id, save_path)

    return {"paper_id": paper_id, "arxiv_id": arxiv_id, "status": "downloading"}


# ---- 管理接口（阶段 4/7） --------------------------------------------------


@router.get("/papers")
def get_papers():
    """论文列表。数据来自元数据表，不再靠向量库里的 chunk_id == 0 反推。"""
    return {"papers": paper_registry.list_papers()}


@router.delete("/papers/{paper_id}")
def delete_paper_detail(paper_id: str):
    """删除论文：向量、元数据、磁盘文件三处一起清。"""
    record = paper_registry.get_paper(paper_id)
    if record is None:
        raise HTTPException(status_code=404, detail="没有这篇论文")

    # 顺序有讲究：先删向量再删元数据。反过来的话，向量没删干净就成了
    # "列表里查不到、却还能被检索命中"的孤儿数据，比直接报错更难查。
    delete_paper_vectors(paper_id)
    paper_registry.delete_paper(paper_id)

    # 文件删不掉不算删除失败（比如被预览器占着），如实返回即可
    file_removed = True
    if record["file_path"] and os.path.isfile(record["file_path"]):
        try:
            os.remove(record["file_path"])
        except OSError:
            file_removed = False

    logger.info("已删除 %s（文件删除成功=%s）", paper_id, file_removed)
    return {"paper_id": paper_id, "deleted": True, "file_removed": file_removed}


# ---- 本地上传（阶段 1/2/5） -----------------------------------------------


@router.post("/papers/upload")
def upload_paper(background: BackgroundTasks, file: UploadFile = File(...)):
    """接收本地 PDF：落盘后直接丢给后台解析入库。

    上传和解析合成一个接口。文件都已经在服务端了，还让调用方再手动触发一次
    解析是没有意义的步骤；进度统一看 GET /api/papers 里的 status。
    """
    # 原始文件名只用于展示：先剥掉客户端塞进来的目录部分，再截断长度，
    # 免得 "C:\\Users\\...\\很长的名字.pdf" 原样进数据库
    raw_name = os.path.basename((file.filename or "").replace("\\", "/")).strip()
    display_name = raw_name[:120] or "unnamed.pdf"
    if not display_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    # 硬盘文件名用纯 uuid，paper_id 再带一层来源前缀
    file_stem = uuid.uuid4().hex
    paper_id = f"{UPLOAD_PREFIX}{file_stem}"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, f"{file_stem}.pdf")

    size = 0
    try:
        with open(save_path, "wb") as f:
            # 先验文件头：单纯改扩展名冒充 PDF 的，这一步就被挡住
            head = file.file.read(8)
            if not head.startswith(PDF_MAGIC):
                raise HTTPException(status_code=400, detail="这不是 PDF（文件头不是 %PDF-）")
            f.write(head)
            size = len(head)

            # 分块写盘，每块 1MB，内存占用恒定；边写边累计大小，超限立刻停
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过上限 {MAX_UPLOAD_BYTES // 1024 // 1024} MB",
                    )
                f.write(chunk)
    except HTTPException:
        # 校验失败时删掉写了一半的残文件，别留垃圾
        if os.path.isfile(save_path):
            os.remove(save_path)
        raise

    paper_registry.save_paper(paper_id, "upload", save_path, original_filename=display_name)
    paper_registry.update_status(paper_id, "indexing")
    background.add_task(_index_only, paper_id, save_path)
    logger.info("收到上传 %s（%s，%.1f MB），已提交后台解析",
                paper_id, display_name, size / 1024 / 1024)

    return {
        "paper_id": paper_id,
        "filename": display_name,
        "size": size,
        "status": "indexing",
    }
