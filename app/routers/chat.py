# 问答接口：POST /chat。整个 RAG 流程在这里串起来：
#   1. 找到会话（或准备新建）
#   2. 扣配额（Redis 计数器，超了返回 429）
#   3. 问题 → 向量（带缓存）→ 在自己的文档里检索最相关的几段
#   4. 段落 + 历史对话 + 问题 → 调 LLM 生成回答
#   5. 在一个事务里保存：会话、提问、回答、使用记录
# 外部服务失败时（embedding 或 LLM 挂了）：退还配额，什么都不保存，返回 503 友好提示，而不是 500

# logging：记录日志
import logging

# APIRouter：路由分组；HTTPException：返回错误；status：状态码常量
from fastapi import APIRouter, HTTPException, status
# func：调用 SQL 函数（clock_timestamp）
from sqlalchemy import func

# 各个模块
from app import llm, queries, quota, rag
# 读取配置（检索段数）
from app.config import settings
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

# 检索不到任何相关段落时的固定回答。这种情况下不调用 LLM：反正它也只能说"没找到"，还要花钱
NO_CONTEXT_ANSWER = "在你的文档里没有找到相关内容。如果刚上传了文档，请等它处理完成（状态变为 ready）后再提问。"
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

    # ---------- 第 3、4 步：检索 + 生成 ----------
    # answered 记录是否成功拿到了回答。用 try/finally 保证：只要中途失败（不管什么原因），都退还配额
    answered = False
    try:
        # 历史消息（新会话没有历史）
        history = queries.list_messages(db, user.id, conversation.id) if conversation else []
        # 问题 → 向量 → 检索
        chunks = queries.search_chunks(db, user.id, rag.embed_query(body.question), settings.retrieval_top_k)
        # 去掉相似度太低的段落。向量检索总会返回 top_k 个"最近的"，哪怕它们其实毫不相关
        chunks = [c for c in chunks if c.score >= settings.retrieval_min_score]
        # 结束当前的只读事务，把数据库连接还给连接池。
        # 原因：下面调用 LLM 可能要十几秒，这期间如果一直占着连接（"idle in transaction"），
        # 并发一高，连接池很快就被占满，其他所有接口都要排队等连接
        db.commit()
        if chunks:
            answer, tokens = llm.chat(rag.build_messages(body.question, chunks, history))
        else:
            answer, tokens = NO_CONTEXT_ANSWER, 0
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

    # ---------- 第 5 步：保存（一个事务） ----------
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
    sources = [{"chunk_id": c.chunk_id, "document_id": c.document_id, "score": round(c.score, 4)} for c in chunks]
    db.add(
        Message(
            conversation=conversation, role=MessageRole.ASSISTANT, content=answer, sources=sources, tokens_used=tokens
        )
    )
    # 使用记录：Redis 不可用时，配额就靠数这张表；同时也是统计用量和费用的依据
    db.add(UsageLog(user_id=user.id, action=UsageAction.QUERY, tokens_used=tokens))
    # 四种写入在同一个事务里，要么全部成功，要么全部不生效
    db.commit()

    return ChatResponse(
        conversation_id=conversation.id,
        answer=answer,
        sources=[
            SourceOut(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                filename=c.filename,
                score=round(c.score, 4),  # 保留 4 位小数
                snippet=c.content[:SNIPPET_LENGTH],
            )
            for c in chunks
        ],
    )
