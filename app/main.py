# 应用入口：创建 FastAPI 应用，把各个路由分组挂上去。
# 启动命令：uv run uvicorn app.main:app --reload
#   app.main:app 的意思是"app/main.py 文件里的 app 变量"
#   --reload：代码改动后自动重启，只在开发时用
# 启动后打开 http://127.0.0.1:8000/docs 可以看到 Swagger 接口文档，并直接在页面上调用接口

# Path：处理文件路径
from pathlib import Path

# FastAPI：Web 框架
from fastapi import FastAPI
# StaticFiles：把一个文件夹里的文件（HTML、图片等）原样提供给浏览器
from fastapi.staticfiles import StaticFiles

# 导入各个路由模块
from app.routers import auth, chat, conversations, documents, health

# 前端文件夹的位置：项目根目录下的 frontend/。
# 用 __file__（本文件的路径）推算，而不是写相对路径，这样不管从哪个目录启动都能找到
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# 创建应用。title 会显示在 Swagger 页面顶部
app = FastAPI(title="RAG Knowledge Base")

# 把路由挂到应用上。之后 /auth/... 和 /documents/... 的请求就会交给对应的函数处理
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(health.router)

# 前端页面：打开 http://127.0.0.1:8000/ 就会返回 frontend/index.html。
# 必须放在所有接口的后面：挂在 "/" 上会匹配所有路径，FastAPI 按注册顺序匹配，
# 放在后面才能保证 /auth、/documents 等请求先交给接口处理。
# html=True：访问目录（如 "/"）时自动返回其中的 index.html。
# 前端和接口来自同一个地址（同源），所以浏览器不会拦截请求，不需要配置 CORS
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
