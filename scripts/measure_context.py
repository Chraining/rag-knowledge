"""Measure where the prompt tokens go in a long conversation.

    uv run python -m scripts.evaluate              # 先建好评测库（自建评测集），已经建过可以跳过
    uv run python -m scripts.measure_context       # 回放一段多轮对话，打印每轮提示词的构成（调用付费的 DeepSeek 接口）

测量多轮对话里，提示词（输入）的 token 都花在了哪里。

在评测库 rag_eval 的自建评测集上，用线上的 Agent 回放一段写好的多轮对话（CONVERSATION），记录每次调用大模型时：
  - 接口返回的输入 token 数（prompt_tokens），这是真实计费的数字
  - 提示词里各部分的字数：系统提示词、工具说明、历史对话、本次问题、本次提问中的工具调用和结果
接口只告诉我们总的 token 数，不会按部分拆分，所以各部分按字数统计，再用"总 token ÷ 总字数"换算成大约的 token 数。

第 1 轮给回答提了一个格式要求，之后每轮检查回答是否还遵守，最后一轮直接问"我一开始提了什么要求"：
看历史预算（agent.HISTORY_TOKEN_BUDGET）把第 1 轮挤出去之后，模型还记不记得。
对话历史只放在内存里，不写数据库。
"""

# json：把工具说明转成文字，统计字数
import json

# select：查询评测用户和文档
from sqlalchemy import select
# Session：数据库会话
from sqlalchemy.orm import Session

# agent：线上的 Agent；llm：要记录的大模型调用
from app import agent, llm
# 模型
from app.models import Document, Message, MessageRole, User
# 连接评测库（不存在会创建并建表）
from scripts.evaluate import prepare_database

# 回放的对话。包含：追问（"那……呢""它"）、需要通读文件的问题（read_document）、对比两个文件（多次调用工具）
CONVERSATION = [
    "接下来我问的问题，回答都在最后加上“（来自知识库）”。",
    "我是 P7，去上海出差住酒店一晚能报多少？",
    "那吃饭呢？",
    "三个多月前的发票还能报销吗？",
    "路由器报 E171 怎么处理？",
    "那 E117 呢？",
    "XR-500 有几个光口？",
    "它最多能连多少台设备？",
    "帮我总结一下员工手册。",
    "工作满 5 年了，一年能休几天年假？",
    "去年没休完的年假今年还能用吗？",
    "第三季度经营会议上定了哪些事？",
    "和第二季度比有什么变化？",
    "SW-24P 的零售价是多少？",
    "它的 PoE 总功率是多少？",
    "我一开始对你的回答提了什么要求？",
]
# 第 1 轮要求回答末尾加上的标记
MARKER = "来自知识库"
# 提示词的几个部分，按在提示词里的顺序
SECTIONS = ["系统", "工具说明", "历史", "问题", "工具"]
# 工具说明也是输入的一部分，每次调用都会发送（接口会把它转成文字放进提示词）
TOOLS_CHARS = len(json.dumps(agent.TOOLS, ensure_ascii=False))


# 一组消息的字数：消息内容 + 工具调用的参数
def size(messages: list[dict]) -> int:
    return sum(
        len(m.get("content") or "") + sum(len(c["function"]["arguments"]) for c in m.get("tool_calls", []))
        for m in messages
    )


# 一次调用里各部分的字数。Agent 组装的消息顺序是：[系统提示词] + 历史 + [本次问题] + 工具调用和结果
def section_chars(messages: list[dict], history_len: int) -> dict[str, int]:
    question = 1 + history_len
    return {
        "系统": size(messages[:1]),
        "工具说明": TOOLS_CHARS,
        "历史": size(messages[1:question]),
        "问题": size(messages[question : question + 1]),
        "工具": size(messages[question + 1 :]),
    }


def main() -> None:
    engine = prepare_database()
    # 每次调用大模型的记录：(各部分字数, 输入 token 数)
    calls: list[tuple[dict[str, int], int]] = []
    # 本轮的历史条数，记录时用来划分"历史"和"本次问题"
    history_len = 0
    real_chat = llm.chat

    # 包一层：照常调用真实的接口，顺便记录发送的消息和输入 token 数。
    # Agent 里写的是 llm.chat(...)，每次调用时才去 llm 模块里取这个函数，所以替换模块上的属性就能生效
    def recording_chat(messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto") -> llm.ChatReply:
        reply = real_chat(messages, tools, tool_choice)
        calls.append((section_chars(messages, history_len), reply.prompt_tokens))
        return reply

    llm.chat = recording_chat
    with Session(engine) as db:
        user_id = db.scalar(select(User.id).where(User.username == "eval"))
        filenames = set(db.scalars(select(Document.filename).where(Document.user_id == user_id)))
        # 评测库里可能是公开数据集（最后一次运行 evaluate 用的是哪份，里面就是哪份）
        if "员工手册.md" not in filenames:
            raise SystemExit("评测库里没有自建评测集，先运行：uv run python -m scripts.evaluate")

        history: list[Message] = []
        print(f"历史预算：{agent.HISTORY_TOKEN_BUDGET} token；表中字数是本轮最后一次调用（提示词最长的一次）的")
        print(f"{'轮次':>4} {'调用':>4} {'输入token':>9} " + " ".join(f"{s:>6}" for s in SECTIONS) + f" {'窗口外':>6}  遵守  回答")
        for turn, question in enumerate(CONVERSATION, start=1):
            first_call = len(calls)
            history_len = len(agent.recent_history(history))
            result = agent.Agent(db, user_id).run(question, history)
            turn_calls = calls[first_call:]
            chars, _ = turn_calls[-1]
            # 被窗口挤掉、模型已经看不到的历史字数
            dropped = size([{"content": m.content} for m in history[: len(history) - history_len]])
            # 第 1 轮是提要求本身，不检查
            followed = "" if turn == 1 else ("✓" if MARKER in result.answer[-20:] else "✗")
            print(
                f"{turn:>4} {len(turn_calls):>4} {max(t for _, t in turn_calls):>9} "
                + " ".join(f"{chars[s]:>6}" for s in SECTIONS)
                + f" {dropped:>6}  {followed:^4}  {result.answer[:40].replace(chr(10), ' ')}"
            )
            # 和线上一样，历史里只存问题和最终回答
            history += [Message(role=MessageRole.USER, content=question), Message(role=MessageRole.ASSISTANT, content=result.answer)]

    # 整段对话所有调用加起来：每次调用都会重新发送整个提示词，所以这就是实际付费的输入
    total_tokens = sum(t for _, t in calls)
    totals = {s: sum(chars[s] for chars, _ in calls) for s in SECTIONS}
    total_chars = sum(totals.values())
    # 中文、英文、JSON 混在一起，一个比例只是近似值
    ratio = total_tokens / total_chars
    print(f"\n共 {len(calls)} 次调用，输入 {total_tokens} token、{total_chars} 字，平均每字 ≈ {ratio:.2f} token")
    print("整段对话的输入里各部分的占比（token 按上面的比例换算）：")
    for s in SECTIONS:
        print(f"  {s:<6} {totals[s]:>7} 字  ≈ {totals[s] * ratio:>7.0f} token  {totals[s] / total_chars:>5.1%}")
    engine.dispose()


if __name__ == "__main__":
    main()
