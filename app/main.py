# 应用入口：创建 FastAPI 应用，把各个路由分组挂上去。
# 启动命令：uv run uvicorn app.main:app --reload
#   app.main:app 的意思是"app/main.py 文件里的 app 变量"
#   --reload：代码改动后自动重启，只在开发时用
# 启动后打开 http://127.0.0.1:8000/docs 可以看到 Swagger 接口文档，并直接在页面上调用接口

# FastAPI：Web 框架
from fastapi import FastAPI

# 导入各个路由模块
from app.routers import auth, chat, conversations, documents

# 创建应用。title 会显示在 Swagger 页面顶部
app = FastAPI(title="RAG Knowledge Base")

# 把路由挂到应用上。之后 /auth/... 和 /documents/... 的请求就会交给对应的函数处理
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(conversations.router)
