# 应用镜像的"配方"：应用（API）和 worker 共用这一个镜像，只是启动命令不同（见 docker-compose.yml）。
# 构建：docker compose build      Docker 会从上到下执行每一步，每一步的结果都会被缓存

# 基础镜像：官方的 Python 3.13，slim 是精简版的 Debian Linux（体积小，只带运行 Python 必需的东西）
FROM python:3.13-slim

# 从 uv 官方镜像里只复制 uv 这一个可执行文件过来，不用另外安装。版本和本机一致
COPY --from=ghcr.io/astral-sh/uv:0.11.12 /uv /usr/local/bin/uv

# 环境变量：
#   UV_COMPILE_BYTECODE=1：安装时就把 .py 编译成 .pyc，容器启动更快
#   UV_LINK_MODE=copy：直接复制文件（容器里用不了 uv 默认的硬链接方式，会报警告）
#   UV_PYTHON_DOWNLOADS=never：用镜像自带的 Python，不让 uv 另外下载
#   PYTHONUNBUFFERED=1：print 和日志立刻输出，不缓存。否则 docker logs 里会看到日志延迟甚至丢失
#   PATH：把虚拟环境的 bin 目录放到最前面，这样直接敲 uvicorn、celery、alembic 就能运行
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# 工作目录：之后的命令都在 /app 里执行，相当于 cd /app
WORKDIR /app

# 第 1 步：只复制依赖清单，安装依赖。
# 为什么先装依赖、后复制代码：Docker 按步骤缓存，某一步的输入没变就直接用缓存。
# 依赖清单很少变，代码经常变。这样改代码后重新构建，这一步直接用缓存，几秒就好；
# 如果先复制代码，每次改代码都要重新装一遍所有依赖
#   --frozen：严格按 uv.lock 里锁定的版本安装，和本机测试过的版本完全一致
#   --no-dev：不装开发依赖（pytest 等），镜像里用不到
#   --no-install-project：只装依赖，不装项目本身（代码还没复制进来）
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 第 2 步：复制运行需要的代码。测试、文档等不复制（见 .dockerignore）
COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
COPY frontend ./frontend
COPY scripts ./scripts

# 第 3 步：创建一个普通用户来运行程序，而不是用 root（管理员）。
# 万一程序有漏洞被攻破，攻击者拿到的也只是普通用户的权限，改不了系统文件。
# 同时创建上传文件的存储目录，并交给这个用户（否则普通用户没有权限写入）
RUN useradd --create-home appuser \
    && mkdir -p /data/storage \
    && chown appuser /data/storage
# 之后的命令和容器里运行的程序，都用这个普通用户的身份
USER appuser

# 上传的文件存到 /data/storage。docker-compose.yml 会在这里挂载一个卷，应用和 worker 共用
ENV STORAGE_DIR=/data/storage

# 说明这个容器监听 8000 端口（只是文档作用，真正对外开放要在 docker-compose.yml 的 ports 里写）
EXPOSE 8000

# 默认启动命令（API）：先执行数据库迁移，把表结构升级到最新，再启动 uvicorn。
#   --host 0.0.0.0：监听所有网卡。默认的 127.0.0.1 在容器里只代表容器自己，外面连不进来
#   exec：让 uvicorn 替换掉 sh 进程。这样 docker stop 发出的停止信号能直接送到 uvicorn，它才能优雅退出
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
