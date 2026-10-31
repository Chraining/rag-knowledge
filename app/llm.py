# 调用外部 AI 服务：把文字转成向量（embedding，阿里云百炼）、让大模型生成回答（chat，DeepSeek）。
# 两家都提供"OpenAI 兼容"接口，所以都直接用 openai 这个库，
# 以后想换成别家（通义千问、硅基流动等兼容 OpenAI 接口的服务），只需要改 .env 里的地址、key 和模型名。
#
# 外部服务一定会出问题（超时、限流、宕机），这里的应对：
#   1. 超时：每次调用最多等 llm_timeout_seconds 秒
#   2. 重试：openai 库自带重试，遇到网络错误、429（限流）、5xx（服务端错误）时
#      自动等一会儿再试，最多 2 次，等待时间越来越长（指数退避）
#   3. 降级：重试完还是失败，统一抛出 LLMUnavailable，由调用方决定怎么"体面地失败"
#      （问答接口返回 503 友好提示；后台任务交给 Celery 再重试）

# dataclass：自动生成 __init__ 的装饰器，用来定义只装数据的小类
from dataclasses import dataclass

# httpx2：发 HTTP 请求的库（openai 库底层用的就是它）。重排序接口不兼容 OpenAI，直接用它调用
import httpx2
# OpenAI：客户端类；OpenAIError：openai 库所有错误的基类
from openai import OpenAI, OpenAIError
# Vector：pgvector 库的向量类型，内部用紧凑的 4 字节浮点数存储，可以直接写进数据库
from pgvector import Vector

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
# 参数：api_key 是密钥；base_url 是服务地址；key_name 是 .env 里的变量名，只用于报错信息
def _client(api_key: str, base_url: str, key_name: str) -> OpenAI:
    if not api_key:
        raise LLMUnavailable(f"{key_name} is not set in .env")
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=settings.llm_timeout_seconds,  # 超时秒数
        max_retries=2,  # 自动重试次数
    )


# 把多段文字转换成向量，返回的顺序和输入的顺序一一对应。
# 返回 pgvector 的 Vector 而不是 Python 的 float 列表，是为了省内存：
#   Python 列表里的每个数都是一个独立的对象，约 32 字节，一个 1024 维的向量约 36KB；
#   Vector 内部是连续的 4 字节浮点数，只要 4KB。处理一本长小说要上万个向量，差别是几百 MB 和几十 MB。
#   数据库里的 vector 列本来就是 4 字节浮点数，所以不损失精度
def embed_texts(texts: list[str]) -> list[Vector]:
    # embedding 用阿里云百炼
    client = _client(settings.dashscope_api_key, settings.embedding_base_url, "DASHSCOPE_API_KEY")
    vectors: list[Vector] = []
    try:
        # 每 10 段发一次请求
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            response = client.embeddings.create(
                model=settings.embedding_model,
                input=texts[start : start + EMBED_BATCH_SIZE],
                dimensions=EMBEDDING_DIM,  # 指定输出 1024 维，必须和数据库的向量列一致
            )
            # 按 index 排序后再取出向量，保证顺序和输入一致（不依赖服务端返回的顺序）
            # 每批的结果马上转成紧凑的 Vector，原来的 float 列表随即被释放
            vectors.extend(Vector(item.embedding) for item in sorted(response.data, key=lambda d: d.index))
    except OpenAIError as e:
        # 把 openai 库的各种错误统一转换成我们自己的错误类型，调用方不用关心底层用的是哪个库
        raise LLMUnavailable(f"embedding failed: {e}") from e
    return vectors


# 模型要调用的一个工具
@dataclass
class ToolCall:
    id: str  # 这次调用的编号，返回工具结果时要带上，模型才知道是哪次调用的结果
    name: str  # 工具名，例如 "search_documents"
    arguments: str  # 参数，是一个 JSON 字符串（模型生成的，可能格式不对），由调用方解析


# 大模型的一次回复：要么是回答（content），要么是要调用的工具（tool_calls 不为空）
@dataclass
class ChatReply:
    content: str  # 回答内容；调用工具时通常为空
    tool_calls: list[ToolCall]  # 要调用的工具，可以一次调用多个
    tokens: int  # 这次调用消耗的 token 数（输入 + 输出），用于统计用量和费用


# 让大模型根据对话消息生成回复。
# messages 的格式：[{"role": "system"/"user"/"assistant"/"tool", "content": "..."}, ...]
# tools：可供模型调用的工具说明（OpenAI 的 function calling 格式），不传就是普通对话。
# allow_tools=False：工具说明照样发送（对话里已经有工具调用记录，去掉说明有的服务会报错），
#   但要求模型这次必须直接回答，不能再调用工具
def chat(messages: list[dict], tools: list[dict] | None = None, allow_tools: bool = True) -> ChatReply:
    # 回答用 DeepSeek
    client = _client(settings.deepseek_api_key, settings.chat_base_url, "DEEPSEEK_API_KEY")
    # tool_choice="auto"：模型自己决定调不调用工具；"none"：不许调用
    extra = {"tools": tools, "tool_choice": "auto" if allow_tools else "none"} if tools else {}
    try:
        response = client.chat.completions.create(model=settings.chat_model, messages=messages, **extra)
    except OpenAIError as e:
        raise LLMUnavailable(f"chat failed: {e}") from e
    # 取第一个候选回复
    message = response.choices[0].message
    # 模型要调用的工具。没有调用工具时 tool_calls 是 None，用 or [] 兜底
    calls = [ToolCall(c.id, c.function.name, c.function.arguments) for c in message.tool_calls or []]
    # usage 里记录了这次调用消耗的 token 数
    tokens = response.usage.total_tokens if response.usage else 0
    # content 可能是 None（只调用工具时），用 or "" 兜底
    return ChatReply(message.content or "", calls, tokens)


# 重排序：给每段文字和问题的相关程度打分（0-1），返回的分数和输入的顺序一一对应。
# 和向量检索的区别：向量检索是问题和段落各自转成向量再比较，没有"一起读"过；
# 重排序模型把问题和段落放在一起读，判断得更准，但每次都要调用模型，又慢又花钱，
# 所以只用来给向量检索选出的少量候选重新排序
def rerank(query: str, documents: list[str]) -> list[float]:
    if not documents:
        return []
    if not settings.dashscope_api_key:
        raise LLMUnavailable("DASHSCOPE_API_KEY is not set in .env")
    try:
        response = httpx2.post(
            settings.rerank_url,
            headers={"Authorization": f"Bearer {settings.dashscope_api_key}"},
            # top_n：返回几段的分数。这里要全部
            json={"model": settings.rerank_model, "input": {"query": query, "documents": documents}, "parameters": {"top_n": len(documents)}},
            timeout=settings.llm_timeout_seconds,
        )
        # 4xx、5xx 状态码抛出 HTTPStatusError
        response.raise_for_status()
        # 返回的结果按分数从高到低排，每项带着它在输入里的位置 index
        results = response.json()["output"]["results"]
    # HTTPError：网络错误、超时、错误状态码；KeyError、ValueError：返回的内容格式不对
    except (httpx2.HTTPError, KeyError, ValueError) as e:
        raise LLMUnavailable(f"rerank failed: {e}") from e
    # 按 index 放回输入的顺序
    scores = [0.0] * len(documents)
    for item in results:
        scores[item["index"]] = item["relevance_score"]
    return scores
