# 接口的"数据结构"：定义请求体长什么样、响应体长什么样。
# 用 Pydantic 定义，FastAPI 会自动：
#   1. 校验请求数据（格式不对直接返回 422，接口函数根本不会执行）
#   2. 把返回的对象转换成 JSON，并且只包含这里定义的字段
#   3. 生成 Swagger 文档（/docs 页面）
#
# 为什么不直接返回数据库模型（如 User）？
#   数据库模型里有 password_hash 等不该给前端看的字段。schema 相当于一个"白名单"，
#   只有列在这里的字段才会出现在响应里，以后给模型加字段也不会意外泄露。

# datetime：日期时间类型
from datetime import datetime

# BaseModel：所有 schema 的基类
# ConfigDict：配置 schema 的行为
# EmailStr：自动校验邮箱格式的字符串类型（依赖 email-validator 库）
# Field：给字段加额外的校验规则（长度、正则等）
# field_validator：自定义字段的校验/转换逻辑
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

# 文档状态枚举，复用数据库模型里的定义
from app.models import DocumentStatus, MessageRole


# ============================ 用户 ============================


# 注册时前端发来的数据
class UserCreate(BaseModel):
    # 用户名：3～50 个字符，只能是字母、数字、下划线
    # pattern 是正则表达式：^ 开头，$ 结尾，[A-Za-z0-9_] 允许的字符，{3,50} 长度范围
    username: str = Field(pattern=r"^[A-Za-z0-9_]{3,50}$")
    # 邮箱：EmailStr 会检查格式是否合法
    email: EmailStr
    # 密码：8～128 个字符。设上限是为了防止有人提交超长密码，让哈希计算占用大量 CPU
    password: str = Field(min_length=8, max_length=128)

    # 自定义处理：用户名和邮箱统一转成小写
    # 这样 "Alice" 和 "alice" 会被当成同一个用户名，防止注册出两个看起来一样的账号
    # mode="after"：在类型校验之后执行（此时已经确认它是合法的字符串）
    @field_validator("username", "email", mode="after")
    @classmethod
    def lowercase(cls, value: str) -> str:
        return value.lower()


# 返回给前端的用户信息。注意：没有 password_hash
class UserOut(BaseModel):
    # from_attributes=True：允许直接从数据库模型对象（如 User）读取属性来构造
    # 没有这个设置的话，只能从字典构造
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    daily_quota: int
    created_at: datetime


# 登录成功后返回的 token
class Token(BaseModel):
    # 字段名必须叫 access_token 和 token_type，这是 OAuth2 标准规定的，
    # Swagger 页面的 "Authorize" 按钮就是按这个格式读取 token 的
    access_token: str
    token_type: str = "bearer"


# ============================ 文档 ============================


# 返回给前端的文档信息
class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    file_size: int
    content_hash: str
    status: DocumentStatus
    chunk_count: int
    # str | None：可以为空
    error_message: str | None
    created_at: datetime
    processed_at: datetime | None
    # 注意：没有 storage_path。服务器上的文件路径是内部细节，暴露给前端没有好处，
    # 反而会泄露服务器的目录结构


# ============================ 问答 ============================


# 请求体：POST /chat
class ChatRequest(BaseModel):
    # 问题，1 到 2000 个字符。限制长度：太长的问题会让提示词变得很长，费用和延迟都增加
    question: str = Field(min_length=1, max_length=2000)
    # 接着哪个会话继续问。不传（None）就新建一个会话
    conversation_id: int | None = None


# 回答引用的一个来源
class SourceOut(BaseModel):
    chunk_id: int  # 段落 id
    document_id: int  # 文档 id
    filename: str  # 文件名
    score: float | None  # 相似度，越接近 1 越相关；通读文档得到的段落没有相似度，为 null
    snippet: str  # 段落开头的一部分，方便用户看出引用的是哪里


# 响应体：POST /chat
class ChatResponse(BaseModel):
    conversation_id: int  # 会话 id，追问时传回来
    answer: str  # 回答
    sources: list[SourceOut]  # 引用来源


# 一条消息
class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: MessageRole  # user 或 assistant
    content: str
    # 引用来源（只有 assistant 消息有）。格式：[{"chunk_id": 1, "document_id": 2, "score": 0.83}, ...]
    sources: list[dict] | None
    created_at: datetime


# 会话列表中的一项：会话信息 + 最后一条消息
class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    last_message: MessageOut | None  # 空会话没有消息

