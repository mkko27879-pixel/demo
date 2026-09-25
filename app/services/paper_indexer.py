import logging

from app.services import paper_registry
from app.services.pdf_loader import load_pdf
from app.services.text_cleaner import clean_documents
from app.services.text_splitter import split_documents
from app.services.vector_store import insert_chunks

logger = logging.getLogger(__name__)


def index_paper(paper_id: str, pdf_path: str) -> dict:
    """把 PDF 解析、清洗、切分、向量化后入库，并把状态写回元数据表。

    arXiv 下载和本地上传共用这一条流水线，保证两种来源在库里形态完全一致：
    清洗程度、切分参数、入库字段都一样，检索质量才可预期。

    失败时会把原因写进元数据表的状态里再抛出去，方便后台任务和同步调用各自处理。
    """
    try:
        docs = clean_documents(load_pdf(pdf_path))
        chunks = split_documents(docs)
        vector_count = insert_chunks(chunks, paper_id)
    except Exception as e:
        # exception 级别的日志会带上完整堆栈，排障时不用再复现一次
        logger.exception("解析入库失败 %s", paper_id)
        paper_registry.update_status(paper_id, "failed", error=f"{type(e).__name__}: {e}")
        raise

    logger.info("入库完成 %s：%d 页 / %d 块", paper_id, len(docs), len(chunks))
    paper_registry.update_status(paper_id, "indexed",
                                 page_count=len(docs), chunk_count=len(chunks))
    return {
        "paper_id": paper_id,
        "page_count": len(docs),
        "chunk_count": len(chunks),
        "vector_count": vector_count,
    }
