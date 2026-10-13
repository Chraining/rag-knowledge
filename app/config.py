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


# 在模块加载时创建一个全局唯一的配置对象
# 其他文件用 `from app.config import settings` 拿到它，不用每次都重新读 .env
settings = Settings()
