import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.getenv("PAPER_DB_PATH", "data/papers.db")

# 向量库里只有 text/vector/source/chunk_id/page 这些"检索用"的字段，
# 标题、作者、上传时间这类结构化信息塞进 Milvus 查起来很别扭，也没法排序分页。
# 这里用 SQLite 单文件存元数据：标准库自带、有事务、单文件方便备份和查看。
_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id          TEXT PRIMARY KEY,
    source_type       TEXT NOT NULL,              -- arxiv / upload
    title             TEXT,
    authors           TEXT,
    original_filename TEXT,
    file_path         TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    page_count        INTEGER DEFAULT 0,
    chunk_count       INTEGER DEFAULT 0,
    status            TEXT NOT NULL DEFAULT 'uploaded',
    -- status 取值：uploaded 已收文件 / downloading 下载中 / indexing 解析中
    --              indexed 已入库 / failed 失败（原因在 error 字段）
    error             TEXT,
    -- 谁加的：user = 用户自己在界面上传或贴链接加的；system = 随应用预置的
    -- 知识库论文。界面上的论文库只列 user 的，system 的对用户不可见但仍参与检索。
    created_by        TEXT NOT NULL DEFAULT 'user'
)
"""

# 老库建表时没有 created_by，而 CREATE TABLE IF NOT EXISTS 不会补列，得手动迁移
_ADD_CREATED_BY = "ALTER TABLE papers ADD COLUMN created_by TEXT NOT NULL DEFAULT 'user'"


@contextmanager
def _db():
    """每次操作开一个短连接。

    注意 sqlite3 的 `with conn:` 管的是事务（自动 commit/rollback），
    它并不会关闭连接——连接必须自己 close，否则会一直占着文件句柄。
    """
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    with _db() as conn:
        conn.execute(_SCHEMA)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(papers)")}
        if "created_by" not in columns:
            conn.execute(_ADD_CREATED_BY)


def save_paper(paper_id: str, source_type: str, file_path: str,
               original_filename: str | None = None,
               title: str | None = None,
               created_by: str = "user") -> None:
    """登记一篇论文。重复登记时只刷新路径和展示名，不动已解析出来的统计信息。

    created_by 在重复登记时不会被覆盖：预置论文不会因为重新解析就变成用户的。
    """
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _db() as conn:
        conn.execute(
            """
            INSERT INTO papers (paper_id, source_type, title, original_filename,
                                file_path, created_at, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                file_path = excluded.file_path,
                original_filename = COALESCE(excluded.original_filename, papers.original_filename),
                title = COALESCE(excluded.title, papers.title)
            """,
            (paper_id, source_type, title, original_filename, file_path, created_at,
             created_by),
        )


def update_status(paper_id: str, status: str, *,
                  page_count: int | None = None,
                  chunk_count: int | None = None,
                  error: str | None = None) -> None:
    """更新状态。COALESCE 是为了让没传的统计字段保持原值，不被 None 覆盖。"""
    with _db() as conn:
        conn.execute(
            """
            UPDATE papers
               SET status = ?, error = ?,
                   page_count = COALESCE(?, page_count),
                   chunk_count = COALESCE(?, chunk_count)
             WHERE paper_id = ?
            """,
            (status, error, page_count, chunk_count, paper_id),
        )


def get_paper(paper_id: str) -> dict | None:
    with _db() as conn:
        row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    return dict(row) if row else None


def list_papers() -> list[dict]:
    with _db() as conn:
        rows = conn.execute("SELECT * FROM papers ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def delete_paper(paper_id: str) -> bool:
    with _db() as conn:
        cur = conn.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))
    return cur.rowcount > 0


init_db()
