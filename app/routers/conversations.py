# 会话相关的接口：会话列表、某个会话的消息、删除会话。
# 和文档接口一样，只能操作自己的会话。

# Annotated：给查询参数加校验规则
from typing import Annotated

# APIRouter：路由分组；HTTPException：返回错误；Query：查询参数的规则；status：状态码常量
from fastapi import APIRouter, HTTPException, Query, status

# 查询函数
from app import queries
# 依赖：当前用户、数据库会话
from app.deps import CurrentUser, DbSession
# 消息模型
from app.models import Message
# 响应数据结构
from app.schemas import ConversationOut, MessageOut

# 所有路径以 /conversations 开头
router = APIRouter(prefix="/conversations", tags=["conversations"])


# GET /conversations：会话列表，每个会话带上最后一条消息，最近活跃的在前
@router.get("", response_model=list[ConversationOut])
def list_conversations(
    user: CurrentUser, db: DbSession, limit: Annotated[int, Query(ge=1, le=100)] = 20
) -> list[ConversationOut]:
    # 阶段 1 写好的查询：用 LATERAL 子查询，一条 SQL 取出所有会话和它们的最后一条消息，没有 N+1 问题
    summaries = queries.list_conversations_with_last_message(db, user.id, limit=limit)
    return [
        ConversationOut(
            id=s.conversation.id,
            title=s.conversation.title,
            created_at=s.conversation.created_at,
            updated_at=s.conversation.updated_at,
            # model_validate：把数据库对象转换成 schema 对象；没有消息时是 None
            last_message=MessageOut.model_validate(s.last_message) if s.last_message else None,
        )
        for s in summaries
    ]


# GET /conversations/{conversation_id}/messages：某个会话的全部消息，最早的在前
@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
def list_messages(conversation_id: int, user: CurrentUser, db: DbSession) -> list[Message]:
    # 先确认会话存在且属于自己。只调用 list_messages 的话，"不是自己的会话"和"空会话"都返回空列表，
    # 前端分不清；这里先查一次，不存在就明确返回 404
    if queries.get_conversation(db, user.id, conversation_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    return queries.list_messages(db, user.id, conversation_id)


# DELETE /conversations/{conversation_id}：删除会话，它的消息会被数据库级联删除
@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: int, user: CurrentUser, db: DbSession) -> None:
    conversation = queries.get_conversation(db, user.id, conversation_id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    db.delete(conversation)
    db.commit()
