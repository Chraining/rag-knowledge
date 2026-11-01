# 问答接口：POST /chat。整个问答流程在这里串起来：
#   1. 找到会话（或准备新建）
#   2. 扣配额（Redis 计数器，超了返回 429）
#   3. 交给 Agent：大模型自己决定调用哪些工具（检索、列出文档、读取文档），再生成回答（见 agent.py）
#   4. 在一个事务里保存：会话、提问、回答、使用记录
# 外部服务失败时（embedding 或 LLM 挂了）：退还配额，什么都不保存，返回 503 友好提示，而不是 500

# logging：记录日志
import logging

# APIRouter：路由分组；HTTPException：返回错误；status：状态码常量
from fastapi import APIRouter, HTTPException, status
# func：调用 SQL 函数（clock_timestamp）
from sqlalchemy import func

# 各个模块
from app import agent, llm, queries, quota
# 依赖：当前用户、数据库会话
from app.deps import CurrentUser, DbSession
# 模型
from app.models import Conversation, Message, MessageRole, UsageAction, UsageLog
# 请求和响应的数据结构
from app.schemas import ChatRequest, ChatResponse, SourceOut

# 日志对象
logger = logging.getLogger(__name__)

# 路由分组，Swagger 里归到 "chat" 组
router = APIRouter(tags=["chat"])

# 会话标题最多取问题的前 50 个字符
TITLE_LENGTH = 50
# 来源里展示段落开头的多少个字符
SNIPPET_LENGTH = 100


# POST /chat：提问
@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, user: CurrentUser, db: DbSession) -> ChatResponse:
    # ---------- 第 1 步：找到会话 ----------
    conversation = None
    if body.conversation_id is not None:
        # 和文档一样：按"当前用户"查，别人的会话查不到，返回 404
        conversation = queries.get_conversation(db, user.id, body.conversation_id)
        if conversation is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")

    # ---------- 第 2 步：扣配额 ----------
    # 放在第 1 步之后：会话 id 写错（404）不应该消耗配额
    try:
        quota.consume(db, user)
    except quota.QuotaExceeded:
        # 429 Too Many Requests：请求太多，超出限制
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Daily question quota exceeded")

    # ---------- 第 3 步：Agent 检索 + 生成 ----------
    # answered 记录是否成功拿到了回答。用 try/finally 保证：只要中途失败（不管什么原因），都退还配额
    answered = False
    try:
        # 历史消息（新会话没有历史）
        history = queries.list_messages(db, user.id, conversation.id) if conversation else []
        # 结束当前的只读事务，把数据库连接还给连接池。
        # 原因：下面调用 LLM 可能要十几秒，这期间如果一直占着连接（"idle in transaction"），
        # 并发一高，连接池很快就被占满，其他所有接口都要排队等连接
        db.commit()
        result = agent.Agent(db, user.id).run(body.question, history)
        answered = True
    except llm.LLMUnavailable:
        # 外部服务失败（超时、重试后仍失败、没配 key 等）：记录详细原因，给用户一个友好的提示
        logger.exception("chat failed for user %s", user.id)
        # 503 Service Unavailable：服务暂时不可用，告诉前端"稍后重试"是有意义的
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "The AI service is temporarily unavailable. Please try again later."
        )
    finally:
        # finally 里的代码无论成功还是抛异常都会执行
        if not answered:
            quota.refund(user.id)

    # ---------- 第 4 步：保存（一个事务） ----------
    if conversation is None:
        # 新会话，标题取问题的开头
        conversation = Conversation(user_id=user.id, title=body.question[:TITLE_LENGTH])
        db.add(conversation)
    # 更新会话的"最后活跃时间"，会话列表按它排序。
    # 只往会话里加消息，不会修改 conversations 这一行，onupdate 不会触发，所以要手动设置。
    # 用 clock_timestamp() 而不是 now()：now() 是"事务开始的时间"，同一个事务里永远不变；
    # clock_timestamp() 是真实的当前时间（和 messages.created_at 用的一样，原因见 models.py）
    conversation.updated_at = func.clock_timestamp()
    # 提问
    db.add(Message(conversation=conversation, role=MessageRole.USER, content=body.question))
    # 回答，带上引用来源（存 JSON）
    sources = [{"chunk_id": c.chunk_id, "document_id": c.document_id, "score": _round(c.score)} for c in result.sources]
    db.add(
        Message(
            conversation=conversation,
            role=MessageRole.ASSISTANT,
            content=result.answer,
            sources=sources,
            tokens_used=result.tokens,
            cost=result.cost,
        )
    )
    # 使用记录：Redis 不可用时，配额就靠数这张表；同时也是统计用量和费用的依据
    db.add(UsageLog(user_id=user.id, action=UsageAction.QUERY, tokens_used=result.tokens, cost=result.cost))
    # 四种写入在同一个事务里，要么全部成功，要么全部不生效
    db.commit()

    return ChatResponse(
        conversation_id=conversation.id,
        answer=result.answer,
        sources=[
            SourceOut(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                filename=c.filename,
                score=_round(c.score),
                snippet=c.content[:SNIPPET_LENGTH],
            )
            for c in result.sources
        ],
        tokens=result.tokens,
        cost=result.cost,
    )


# 相似度保留 4 位小数。通读（read_document）得到的段落没有相似度，保持 None
def _round(score: float | None) -> float | None:
    return None if score is None else round(score, 4)
