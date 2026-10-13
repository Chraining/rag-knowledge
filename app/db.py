# 数据库连接模块：创建"引擎"和"会话工厂"，整个程序共用。

# create_engine：创建数据库引擎（负责管理到数据库的连接池）
from sqlalchemy import create_engine
# sessionmaker：一个"制造 Session 的工厂"
from sqlalchemy.orm import sessionmaker

# 导入上面配置模块里的全局配置对象
from app.config import settings

# engine（引擎）= 数据库连接池。它不会马上连数据库，第一次用到时才连。
# 连接池的意思是：提前建好一些连接反复使用，避免每次查询都重新建立连接（建连接很慢）。
# pool_pre_ping=True：每次从池里取连接前先"ping"一下，
#   如果数据库重启过、连接已断开，就自动换一个新连接，而不是直接报错。
engine = create_engine(settings.database_url, pool_pre_ping=True)

# SessionLocal 是一个工厂，调用 SessionLocal() 就得到一个新的 Session。
# Session（会话）是你和数据库交互的"工作区"：增删改查都通过它，最后 commit 提交。
# bind=engine：这个工厂造出来的 Session 都用上面的引擎连数据库。
# expire_on_commit=False：commit 之后，Python 对象上的属性值仍然保留可读。
#   默认是 True，commit 后再读属性会重新查一次数据库；设成 False 更直观，也避免多余查询。
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
