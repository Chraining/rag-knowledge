# 配置模块：集中管理所有"可能因环境而变"的值（数据库地址、配额等）。
# 好处：开发、测试、生产只需改 .env 文件，不用改代码。

# pydantic_settings 是一个库，能自动从环境变量和 .env 文件里读取配置
from pydantic_settings import BaseSettings, SettingsConfigDict


# 继承 BaseSettings 后，下面每个字段都会自动去找同名的环境变量（不区分大小写）
# 例如字段 database_url 会读取 .env 里的 DATABASE_URL
class Settings(BaseSettings):
    # model_config 是这个类的"元配置"：
    #   env_file=".env"   -> 从项目根目录的 .env 文件读取
    #   extra="ignore"    -> .env 里出现这个类没定义的变量时，忽略而不是报错
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 开发用数据库的连接地址。格式：方言+驱动://用户名:密码@主机:端口/数据库名
    # postgresql+psycopg 表示"PostgreSQL 数据库，用 psycopg 驱动连接"
    # 等号后面是默认值：.env 里没写时才用它
    database_url: str = "postgresql+psycopg://rag:rag@127.0.0.1:5433/rag"
    # 测试专用数据库，和开发库分开，跑测试时不会弄乱开发数据
    test_database_url: str = "postgresql+psycopg://rag:rag@127.0.0.1:5433/rag_test"
    # 新用户默认每天能提问的次数（对应 users.daily_quota 的默认值）
    default_daily_quota: int = 50

    # ---------- 认证（JWT） ----------
    # 签名 JWT 用的密钥。谁拿到它，谁就能伪造任意用户的 token，所以：
    #   1. 生产环境必须在 .env 里换成一个很长的随机字符串
    #   2. 绝不能提交到 git（.env 已在 .gitignore 里）
    # 生成方法：python -c "import secrets; print(secrets.token_urlsafe(48))"
    jwt_secret_key: str = "dev-only-insecure-secret-change-me"
    # 签名算法。HS256 = 用同一个密钥签名和验证（对称加密），单个服务用它就够了
    jwt_algorithm: str = "HS256"
    # token 有效期（分钟）。过期后要重新登录
    access_token_expire_minutes: int = 60

    # ---------- 文件上传 ----------
    # 上传文件保存的目录（相对于项目根目录）
    storage_dir: str = "storage"
    # 单个文件的大小上限：10MB。10 * 1024 * 1024 = 10485760 字节
    max_upload_bytes: int = 10 * 1024 * 1024

    # ---------- 后台任务（Celery） ----------
    # 消息队列（broker）的地址：上传接口把"请处理文档 X"的消息放进 Redis，worker 从这里取
    # 格式：redis://主机:端口/库编号。Redis 默认有 16 个库（0-15），这里用 0 号
    # 写 127.0.0.1 而不是 localhost：Windows 会把 localhost 先解析成 IPv6 的 ::1，
    # 而 Docker 里的 Redis 只监听 IPv4，每次新建连接都要白等约 2 秒才退回到 IPv4（实测）
    celery_broker_url: str = "redis://127.0.0.1:6380/0"

    # ---------- Redis（配额计数、缓存） ----------
    # 和 Celery 用同一个 Redis，但用 1 号库，和任务队列的数据分开，互不干扰
    redis_url: str = "redis://127.0.0.1:6380/1"

    # ---------- LLM 和 embedding ----------
    # 两家服务都提供"OpenAI 兼容"接口，所以都用 openai 这个 Python 库调用，只是地址、key、模型名不同。
    # API key 都写进 .env，绝不能提交到 git

    # 生成回答：DeepSeek。key 在 platform.deepseek.com 创建，写进 .env 的 DEEPSEEK_API_KEY
    deepseek_api_key: str = ""
    chat_base_url: str = "https://api.deepseek.com"
    # deepseek-chat：DeepSeek 的普通对话模型（不带"深度思考"，回答快、便宜）
    chat_model: str = "deepseek-chat"

    # 文字转向量：阿里云百炼 DashScope（DeepSeek 没有 embedding 接口）。
    # key 在百炼控制台创建，写进 .env 的 DASHSCOPE_API_KEY
    dashscope_api_key: str = ""
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v4"
    # 一次调用最多等多少秒。超时就当作失败，不让用户的请求一直卡着
    llm_timeout_seconds: float = 30
    # 每次提问检索多少段最相关的文字交给 LLM
    retrieval_top_k: int = 5
    # 最低相似度：低于它的段落当作"不相关"丢掉，不发给 LLM，也不显示为来源。
    # 0.3 是根据 text-embedding-v4 的实测定的：相关段落约 0.55-0.67，无关段落约 0.12-0.19。
    # 换了 embedding 模型，这个值要重新测
    retrieval_min_score: float = 0.3


# 在模块加载时创建一个全局唯一的配置对象
# 其他文件用 `from app.config import settings` 拿到它，不用每次都重新读 .env
settings = Settings()
