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
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    # 测试专用数据库，和开发库分开，跑测试时不会弄乱开发数据
    test_database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag_test"
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
    celery_broker_url: str = "redis://localhost:6379/0"


# 在模块加载时创建一个全局唯一的配置对象
# 其他文件用 `from app.config import settings` 拿到它，不用每次都重新读 .env
settings = Settings()
