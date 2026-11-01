"""Measure answer correctness and where the prompt tokens go in a long conversation.

    uv run python -m scripts.evaluate                        # 先建好评测库（自建评测集），已经建过可以跳过
    uv run python -m scripts.measure_context                 # 回放一段多轮对话一次（调用付费的 DeepSeek 接口）
    uv run python -m scripts.measure_context --runs 3        # 回放 3 次，看平均值和哪些轮次不稳定
    uv run python -m scripts.measure_context --budget 1500   # 调小历史预算，让这段对话触发摘要（app/summary.py）

测量多轮对话里回答对不对，以及提示词（输入）的 token 都花在了哪里。

在评测库 rag_eval 的自建评测集上，用线上的 Agent 回放一段写好的多轮对话（CONVERSATION），每轮记录：
  - 正确：回答里是否包含期望的关键内容（取自文档原文，例如"512"）。去掉空格和 markdown 的 ** 再比较。
    只是关键词检查，不是逐字判断：答错但碰巧提到了这个词也算对（例如列出所有档位的年假），所以是偏宽松的上限
  - 检索：这一轮是否调用了 search_documents 或 read_document。没检索也可能答对（答案在历史里），但更可能是编的
  - 格式：第 1 轮给回答提了一个格式要求，之后每轮检查回答里有没有这个标记，看历史里的要求有没有被记住
  - 输入 token 数（接口返回的真实数字）和费用（按 app/config.py 里的价格计算）
  - 提示词里各部分的字数：系统提示词、摘要、工具说明、历史对话、本次问题、本次提问中的工具调用和结果
  - 历史超出预算时，和线上一样做摘要（app/summary.py），打印压缩了几条消息；摘要调用的 token 和费用计入每次的合计
    接口只告诉我们总的 token 数，不会按部分拆分，所以各部分按字数统计，再用"总 token ÷ 总字数"换算
大模型每次的输出都不一样，只跑一次的结果可能是碰巧的，比较两种方案时用 --runs 多跑几次。
对话历史只放在内存里，不写数据库。
"""

# argparse：解析命令行参数
import argparse
# json：把工具说明转成文字，统计字数
import json
# dataclass：定义只装数据的小类
from dataclasses import dataclass
# Decimal：费用是精确的十进制小数
from decimal import Decimal

# select：查询评测用户和文档
from sqlalchemy import select
# Session：数据库会话
from sqlalchemy.orm import Session

# agent：线上的 Agent；llm：要记录的大模型调用；summary：历史超出预算时的摘要
from app import agent, llm, summary
# 读取配置（模型名，用来提醒费用按哪个模型的价格算）
from app.config import settings
# 模型
from app.models import Document, Message, MessageRole, User
# 连接评测库（不存在会创建并建表）
from scripts.evaluate import prepare_database

# 回放的对话：(问题, 期望的关键内容)。关键内容全部出现在回答里才算对；None 表示这一轮不判断对错。
# 包含：追问（"那……呢""它"）、需要通读文件的问题（read_document）、对比两个文件（要分别检索两次）
CONVERSATION: list[tuple[str, list[str] | None]] = [
    ("接下来我问的问题，回答都在最后加上“（来自知识库）”。", None),
    ("我是 P7，去上海出差住酒店一晚能报多少？", ["700"]),
    ("那吃饭呢？", ["120"]),
    ("三个多月前的发票还能报销吗？", ["90"]),
    ("路由器报 E171 怎么处理？", ["地址池"]),
    ("那 E117 呢？", ["冲突"]),
    ("XR-500 有几个光口？", ["2个10G"]),
    ("它最多能连多少台设备？", ["512"]),
    ("帮我总结一下员工手册。", ["年假"]),
    ("工作满 5 年了，一年能休几天年假？", ["5天"]),
    ("去年没休完的年假今年还能用吗？", ["3月31日"]),
    ("第三季度经营会议上定了哪些事？", ["2.1亿", "成都"]),
    # 要同时说出两个季度的营收：只凭上一轮第三季度的内容答不出来，必须去看第二季度的纪要
    ("和第二季度比有什么变化？", ["1.82亿", "1.56亿"]),
    ("SW-24P 的零售价是多少？", ["2499"]),
    ("它的 PoE 总功率是多少？", ["370"]),
    ("我一开始对你的回答提了什么要求？", ["来自知识库"]),
]
# 第 1 轮要求回答里加上的标记
MARKER = "来自知识库"
# 算作"检索了"的工具。list_documents 只列文件名，看不到内容
SEARCH_TOOLS = {"search_documents", "read_document"}
# 提示词的几个部分，按在提示词里的顺序
SECTIONS = ["系统", "摘要", "工具说明", "历史", "问题", "工具"]
# 工具说明也是输入的一部分，每次调用都会发送（接口会把它转成文字放进提示词）
TOOLS_CHARS = len(json.dumps(agent.TOOLS, ensure_ascii=False))


# 一次调用大模型的记录
@dataclass
class Call:
    chars: dict[str, int]  # 提示词各部分的字数
    prompt_tokens: int  # 输入 token 数
    cost: Decimal  # 费用
    tools: list[str]  # 这次调用要求执行的工具名


# 一轮问答的结果
@dataclass
class Turn:
    correct: bool | None  # 回答是否包含期望的关键内容；None：这一轮不判断
    searched: bool  # 是否检索了
    followed: bool | None  # 是否遵守了第 1 轮的格式要求；None：第 1 轮本身
    calls: list[Call]
    # 这一轮之后做的摘要：(压缩了几条消息, 大模型的回复)；没有做摘要就是 None
    summary: tuple[int, llm.ChatReply] | None = None


# 记录每次调用大模型：照常调用真实的接口，顺便记录发送的消息、token 数、费用和要求执行的工具。
# 用法：llm.chat = recorder.chat。Agent 里写的是 llm.chat(...)，每次调用时才去 llm 模块里取这个函数，所以替换模块上的属性就能生效
class Recorder:
    def __init__(self):
        self.calls: list[Call] = []
        # 本轮有没有摘要、有几条历史，用来划分提示词的各部分，每轮开始前设置
        self.has_summary = False
        self.history_len = 0
        self.real_chat = llm.chat

    def chat(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto") -> llm.ChatReply:
        reply = self.real_chat(messages, tools, tool_choice)
        chars = section_chars(messages, self.has_summary, self.history_len)
        self.calls.append(Call(chars, reply.prompt_tokens, reply.cost, [c.name for c in reply.tool_calls]))
        return reply


# 一组消息的字数：消息内容 + 工具调用的参数
def size(messages: list[dict]) -> int:
    return sum(
        len(m.get("content") or "") + sum(len(c["function"]["arguments"]) for c in m.get("tool_calls", []))
        for m in messages
    )


# 一次调用里各部分的字数。Agent 组装的消息顺序是：[系统提示词] + [摘要]（有的话）+ 历史 + [本次问题] + 工具调用和结果
def section_chars(messages: list[dict], has_summary: bool, history_len: int) -> dict[str, int]:
    history = 2 if has_summary else 1
    question = history + history_len
    return {
        "系统": size(messages[:1]),
        "摘要": size(messages[1:history]),
        "工具说明": TOOLS_CHARS,
        "历史": size(messages[history:question]),
        "问题": size(messages[question : question + 1]),
        "工具": size(messages[question + 1 :]),
    }


# 比较前去掉空格和 markdown 的加粗符号：模型可能写 "2 个 10G"、"**512** 台"
def normalize(text: str) -> str:
    return "".join(text.split()).replace("*", "")


# 回放一次整段对话，打印每轮的结果
def run_conversation(db: Session, user_id: int, recorder: Recorder) -> list[Turn]:
    # 和线上一样：还没被摘要的消息，以及更早的消息的摘要
    history: list[Message] = []
    conversation_summary: str | None = None
    turns: list[Turn] = []
    print(f"{'轮次':>4} {'调用':>4} {'输入token':>9} " + " ".join(f"{s:>6}" for s in SECTIONS) + f" {'窗口外':>6}  检索 格式 正确  回答")
    for number, (question, expected) in enumerate(CONVERSATION, start=1):
        first_call = len(recorder.calls)
        recorder.has_summary = bool(conversation_summary)
        history_len = recorder.history_len = len(agent.recent_history(history))
        result = agent.Agent(db, user_id).run(question, history, conversation_summary)
        turn_calls = recorder.calls[first_call:]
        answer = normalize(result.answer)
        turn = Turn(
            correct=None if expected is None else all(normalize(e) in answer for e in expected),
            searched=any(name in SEARCH_TOOLS for c in turn_calls for name in c.tools),
            followed=None if number == 1 else MARKER in answer,
            calls=turn_calls,
        )
        turns.append(turn)
        # 被预算挤掉、又还没被摘要，模型完全看不到的历史字数
        dropped = size([{"content": m.content} for m in history[: len(history) - history_len]])
        # 表中的字数是本轮最后一次调用（提示词最长的一次）的
        chars = turn_calls[-1].chars
        print(
            f"{number:>4} {len(turn_calls):>4} {max(c.prompt_tokens for c in turn_calls):>9} "
            + " ".join(f"{chars[s]:>6}" for s in SECTIONS)
            + f" {dropped:>6}  {mark(turn.searched):^4} {mark(turn.followed):^4} {mark(turn.correct):^4}  "
            + result.answer[:30].replace("\n", " ")
        )
        # 和线上一样，历史里只存问题和最终回答
        history += [Message(role=MessageRole.USER, content=question), Message(role=MessageRole.ASSISTANT, content=result.answer)]
        # 和线上一样，历史超出预算就摘要。线上是后台任务，这里直接执行，相当于下一个问题来之前任务已经做完了（正常情况）。
        # 摘要的调用不经过 recorder：它不是回答问题的提示词，不计入上面各部分的占比，单独统计
        fold = summary.messages_to_fold(history)
        if fold:
            llm.chat = recorder.real_chat
            try:
                reply = summary.summarize(conversation_summary, history[:fold])
            finally:
                llm.chat = recorder.chat
            conversation_summary, history = reply.content, history[fold:]
            turn.summary = (fold, reply)
            print(f"        ↳ 摘要：压缩了最早的 {fold} 条消息，摘要 {len(reply.content)} 字，输入 {reply.prompt_tokens} token")
    return turns


# ✓ / ✗ / 空白（不判断）
def mark(value: bool | None) -> str:
    return "" if value is None else ("✓" if value else "✗")


# 一组结果里为 True 的个数 / 参与判断的个数，例如 "14/15"
def ratio(values: list[bool | None]) -> str:
    judged = [v for v in values if v is not None]
    return f"{sum(judged)}/{len(judged)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="测量多轮对话的正确率和提示词构成")
    parser.add_argument("--runs", type=int, default=1, help="回放几次整段对话")
    parser.add_argument(
        "--budget", type=int, default=agent.HISTORY_TOKEN_BUDGET, help="历史预算（token）。调小可以让这段对话触发摘要"
    )
    args = parser.parse_args()
    # 摘要逻辑（app/summary.py）每次都读 agent.HISTORY_TOKEN_BUDGET，改这一个值就够了
    agent.HISTORY_TOKEN_BUDGET = args.budget

    engine = prepare_database()
    recorder = Recorder()
    llm.chat = recorder.chat
    with Session(engine) as db:
        user_id = db.scalar(select(User.id).where(User.username == "eval"))
        filenames = set(db.scalars(select(Document.filename).where(Document.user_id == user_id)))
        # 评测库里可能是公开数据集（最后一次运行 evaluate 用的是哪份，里面就是哪份）
        if "员工手册.md" not in filenames:
            raise SystemExit("评测库里没有自建评测集，先运行：uv run python -m scripts.evaluate")

        print(f"模型：{settings.chat_model}；历史预算：{agent.HISTORY_TOKEN_BUDGET} token")
        runs: list[list[Turn]] = []
        for run in range(1, args.runs + 1):
            print(f"\n===== 第 {run} 次 =====")
            runs.append(run_conversation(db, user_id, recorder))

    print("\n每次的结果：")
    print(f"  {'次':>2}  {'正确':>5}  {'检索':>5}  {'格式':>5}  {'输入token':>9}  {'摘要次数':>4}  {'费用（元）':>10}")
    for run, turns in enumerate(runs, start=1):
        run_calls = [c for t in turns for c in t.calls]
        summaries = [t.summary[1] for t in turns if t.summary]
        # 输入 token 和费用都包括摘要的调用
        tokens = sum(c.prompt_tokens for c in run_calls) + sum(r.prompt_tokens for r in summaries)
        cost = sum(c.cost for c in run_calls) + sum(r.cost for r in summaries)
        print(
            f"  {run:>2}  {ratio([t.correct for t in turns]):>5}  {ratio([t.searched for t in turns[1:]]):>5}  "
            f"{ratio([t.followed for t in turns]):>5}  {tokens:>9}  {len(summaries):>8}  {cost:>10.4f}"
        )
    # 配置里的价格是 deepseek-flash 的（deepseek-chat 是它的旧名字），换了模型价格就不对了
    if settings.chat_model not in ("deepseek-chat", "deepseek-flash"):
        print(f"  注意：费用按 app/config.py 里 deepseek-flash 的价格算，{settings.chat_model} 的价格不同，要先改配置，否则不准")

    # 每一轮在几次回放里答对了，只列出不是每次都对的，看哪些问题不稳定
    unstable = [
        (number, question, sum(bool(run[number - 1].correct) for run in runs))
        for number, (question, expected) in enumerate(CONVERSATION, start=1)
        if expected is not None and not all(run[number - 1].correct for run in runs)
    ]
    print(f"\n不是每次都答对的轮次（{len(unstable)} 个）：")
    for number, question, right in unstable:
        print(f"  第 {number} 轮 {question}  答对 {right}/{len(runs)}")

    # 所有调用加起来：每次调用都会重新发送整个提示词，所以这就是实际付费的输入
    calls = recorder.calls
    total_tokens = sum(c.prompt_tokens for c in calls)
    totals = {s: sum(c.chars[s] for c in calls) for s in SECTIONS}
    total_chars = sum(totals.values())
    # 中文、英文、JSON 混在一起，一个比例只是近似值
    per_char = total_tokens / total_chars
    print(f"\n共 {len(calls)} 次调用，输入 {total_tokens} token、{total_chars} 字，平均每字 ≈ {per_char:.2f} token")
    print("输入里各部分的占比（token 按上面的比例换算）：")
    for s in SECTIONS:
        print(f"  {s:<6} {totals[s]:>7} 字  ≈ {totals[s] * per_char:>7.0f} token  {totals[s] / total_chars:>5.1%}")
    engine.dispose()


if __name__ == "__main__":
    main()
