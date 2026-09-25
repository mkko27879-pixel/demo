import logging

from fastapi import FastAPI

# 日志要在 import 业务模块之前配好，模块初始化时打的日志才会走同一套格式。
# 出问题时至少要能从终端看出"哪一步、哪篇论文、什么错"。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from app.routers import conversations, health, papers, query  # noqa: E402

app = FastAPI()

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(papers.router, prefix="/api", tags=["papers"])
app.include_router(query.router, prefix="/api/query", tags=["query"])
app.include_router(conversations.router, prefix="/api", tags=["conversations"])
