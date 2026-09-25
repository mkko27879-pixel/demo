import logging

from fastapi import APIRouter, HTTPException

from app.services import chat_store

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/conversations")
def get_conversations():
    """会话列表，按最近更新排序。前端用它渲染历史对话侧边栏。"""
    return {"conversations": chat_store.list_conversations()}


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    """一个会话的完整问答记录。前端刷新页面后用这个把上下文恢复出来。"""
    conversation = chat_store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    conversation["messages"] = chat_store.get_messages(conversation_id)
    return conversation


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    if not chat_store.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    logger.info("已删除会话 %s", conversation_id)
    return {"conversation_id": conversation_id, "deleted": True}
