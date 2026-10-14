# 调用外部 AI 服务：把文字转成向量（embedding）、让大模型生成回答（chat）。
# 用的是阿里云百炼 DashScope 的"OpenAI 兼容"接口，所以可以直接用 openai 这个库，
# 以后想换成别家（DeepSeek、硅基流动等兼容 OpenAI 接口的服务），只需要改 .env 里的地址和模型名。
#
# 外部服务一定会出问题（超时、限流、宕机），这里的应对：
#   1. 超时：每次调用最多等 llm_timeout_seconds 秒
#   2. 重试：openai 库自带重试，遇到网络错误、429（限流）、5xx（服务端错误）时
#      自动等一会儿再试，最多 2 次，等待时间越来越长（指数退避）
#   3. 降级：重试完还是失败，统一抛出 LLMUnavailable，由调用方决定怎么"体面地失败"
#      （问答接口返回 503 友好提示；后台任务交给 Celery 再重试）

# OpenAI：客户端类；OpenAIError：openai 库所有错误的基类
from openai import OpenAI, OpenAIError

# 读取配置
from app.config import settings
# 向量维度
from app.models import EMBEDDING_DIM

# DashScope 的 embedding 接口一次最多接收 10 段文字，多了要分批发送
EMBED_BATCH_SIZE = 10


# 外部 AI 服务不可用（没配 key、超时、重试后仍然失败等）
class LLMUnavailable(Exception):
    pass


# 创建客户端。每次调用时创建，而不是在模块加载时创建一个全局的：
#   没配 API key 时，导入这个模块不会报错（测试和 worker 启动都不受影响），只有真正调用时才报错
def _client() -> OpenAI:
    if not settings.dashscope_api_key:
        raise LLMUnavailable("DASHSCOPE_API_KEY is not set in .env")
    return OpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,  # 超时秒数
        max_retries=2,  # 自动重试次数
    )


# 把多段文字转换成向量，返回的顺序和输入的顺序一一对应
def embed_texts(texts: list[str]) -> list[list[float]]:
    client = _client()
    vectors: list[list[float]] = []
    try:
        # 每 10 段发一次请求
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            response = client.embeddings.create(
                model=settings.embedding_model,
                input=texts[start : start + EMBED_BATCH_SIZE],
                dimensions=EMBEDDING_DIM,  # 指定输出 1024 维，必须和数据库的向量列一致
            )
            # 按 index 排序后再取出向量，保证顺序和输入一致（不依赖服务端返回的顺序）
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    except OpenAIError as e:
        # 把 openai 库的各种错误统一转换成我们自己的错误类型，调用方不用关心底层用的是哪个库
        raise LLMUnavailable(f"embedding failed: {e}") from e
    return vectors


# 让大模型根据对话消息生成回答。返回 (回答内容, 消耗的 token 数)
# messages 的格式：[{"role": "system"/"user"/"assistant", "content": "..."}, ...]
def chat(messages: list[dict[str, str]]) -> tuple[str, int]:
    client = _client()
    try:
        response = client.chat.completions.create(model=settings.chat_model, messages=messages)
    except OpenAIError as e:
        raise LLMUnavailable(f"chat failed: {e}") from e
    # 取第一个候选回答。content 理论上可能是 None，用 or "" 兜底
    answer = response.choices[0].message.content or ""
    # usage 里记录了这次调用消耗的 token 数（输入 + 输出），用于统计用量和费用
    tokens = response.usage.total_tokens if response.usage else 0
    return answer, tokens
