from fastapi import FastAPI
from app.routers import health, papers, query
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(papers.router, prefix="/api", tags=["papers"])
app.include_router(query.router, prefix="/api/query", tags=["query"])