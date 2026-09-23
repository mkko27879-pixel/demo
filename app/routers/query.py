import os

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.vector_store import search_similar_chunks

load_dotenv()

router = APIRouter()

# DeepSeek 的 OpenAI 兼容接口，地址和模型名都能在 .env 里覆盖
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# 当前 key 可用的模型只有两个：deepseek-flash（多模态，支持图片输入）和
# deepseek-v4-pro。注意接口对不认识的名字不报错，而是悄悄回退到
# deepseek-flash，所以名字写错了往往看起来"能跑但效果不对"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


def call_deepseek(prompt: str) -> str:
    """直接用 httpx 调 DeepSeek 的 /chat/completions，返回回答文本。

    这里没用 langchain 的 init_chat_model：当前环境只有 langchain-core /
    langchain-community，没有装 langchain（也没有 langchain-deepseek /
    langchain-openai），`from langchain_community.chat_models import init_chat_model`
    会直接 ImportError。httpx 本来就是项目依赖，用它更省事。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("环境变量 DEEPSEEK_API_KEY 未设置，请检查 .env")

    resp = httpx.post(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": "你是一个严谨的论文问答助手，只根据给定的上下文回答问题。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "stream": False,
        },
        timeout=60,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek 调用失败：HTTP {resp.status_code} - {resp.text[:300]}")

    return resp.json()["choices"][0]["message"]["content"]


@router.post("")
def query_paper(request: QueryRequest):
    try:
        # 1. 检索相关片段
        results = search_similar_chunks(request.question, request.top_k)
        if not results:
            return {"answer": "未找到相关论文内容，请先上传论文。", "sources": []}

        # 2. 拼接上下文（带来源标注）
        context = "\n\n".join([f"来源: {r['source']}\n{r['text']}" for r in results])

        # 3. 构造 Prompt（严格限制仅根据上下文回答）
        prompt = f"你是一个严谨的论文问答助手。请仅根据以下上下文回答问题，不要编造，用中文回答问题。\n\n上下文：\n{context}\n\n问题：{request.question}\n\n回答："

        # 4. 调用 LLM 生成答案
        answer = call_deepseek(prompt)

        # 5. 提取去重后的来源
        sources = list(set([r["source"] for r in results]))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"question": request.question, "answer": answer, "sources": sources}
