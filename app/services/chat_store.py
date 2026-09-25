import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime

# 和论文元数据共用同一个 SQLite 文件：都是"本地运行状态"，放一起便于备份和查看
DB_PATH = os.getenv("PAPER_DB_PATH", "data/papers.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    title           TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    sources         TEXT,          -- 助手回答引用到的来源，JSON 数组
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
"""


@contextmanager
def _db():
    """短连接。sqlite3 的 `with conn:` 管的是事务，不会关连接，得自己 close。"""
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
        conn.executescript(_SCHEMA)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_conversation(title: str | None = None) -> str:
    conversation_id = uuid.uuid4().hex
    now = _now()
    with _db() as conn:
        conn.execute(
            """INSERT INTO conversations (conversation_id, title, created_at, updated_at)
               VALUES (?, ?, ?, ?)""",
            (conversation_id, title, now, now),
        )
    return conversation_id


def add_message(conversation_id: str, role: str, content: str,
                sources: list[dict] | None = None) -> None:
    now = _now()
    with _db() as conn:
        conn.execute(
            """INSERT INTO messages (conversation_id, role, content, sources, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (conversation_id, role, content,
             json.dumps(sources, ensure_ascii=False) if sources else None, now),
        )
        conn.execute("UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                     (now, conversation_id))


def get_messages(conversation_id: str, limit: int | None = None) -> list[dict]:
    """取会话消息，按时间正序返回。limit 表示只取最近 N 条（先倒序取再翻回来）。"""
    with _db() as conn:
        if limit:
            rows = conn.execute(
                """SELECT role, content, sources, created_at FROM messages
                    WHERE conversation_id = ? ORDER BY id DESC LIMIT ?""",
                (conversation_id, limit),
            ).fetchall()
            rows = list(reversed(rows))
        else:
            rows = conn.execute(
                """SELECT role, content, sources, created_at FROM messages
                    WHERE conversation_id = ? ORDER BY id""",
                (conversation_id,),
            ).fetchall()

    return [
        {**dict(row), "sources": json.loads(row["sources"]) if row["sources"] else []}
        for row in rows
    ]


def get_conversation(conversation_id: str) -> dict | None:
    with _db() as conn:
        row = conn.execute("SELECT * FROM conversations WHERE conversation_id = ?",
                           (conversation_id,)).fetchone()
    return dict(row) if row else None


def list_conversations() -> list[dict]:
    with _db() as conn:
        rows = conn.execute(
            """SELECT c.*, COUNT(m.id) AS message_count
                 FROM conversations c
                 LEFT JOIN messages m ON m.conversation_id = c.conversation_id
                GROUP BY c.conversation_id
                -- rowid 兜底：updated_at 只精确到秒，同一秒建的两个会话光靠它排不
                -- 出先后，列表顺序就会随机跳。rowid 单调递增，能补上这个精度。
                ORDER BY c.updated_at DESC, c.rowid DESC""",
        ).fetchall()
    return [dict(row) for row in rows]


def delete_conversation(conversation_id: str) -> bool:
    with _db() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cur = conn.execute("DELETE FROM conversations WHERE conversation_id = ?",
                           (conversation_id,))
    return cur.rowcount > 0


init_db()
