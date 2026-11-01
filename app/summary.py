# 对话摘要：历史太长时，把最早的一部分消息压缩成一段摘要，代替原来的消息发给模型。
#
# 什么时候摘要：还没被摘要的消息估算超过 agent.HISTORY_TOKEN_BUDGET 时。
# 摘要多少：把最早的消息压缩掉，只留下最近一半预算（KEEP_RATIO）以内的原始消息。
#   留一半预算，而不是刚好压到预算以内：否则下一轮又超出，几乎每轮都要摘要一次（每次都要多调用一次大模型）。
#   压到一半后，要再聊几轮才会再次超出
# 怎么摘要：增量合并，新摘要 = 大模型(旧摘要 + 这次要压缩的消息)。不从头重新摘要全部历史，
#   因为全部历史会越来越长，重新摘要越来越贵；而且早就被摘要的消息，原文已经不再发给模型了
# 在哪里执行：Celery 后台任务（app/tasks.py 的 summarize_conversation），用户不用等。
#   摘要只在下一次提问时才用到；还没做完时下一个问题就来了，就先用旧的摘要
# 原始消息一直留在数据库里，不会因为摘要被删掉

# 大模型调用；agent：估算 token 数、取最近的历史、历史预算
from app import agent, llm
# 消息模型
from app.models import Message, MessageRole

# 摘要后留下的原始消息占历史预算的比例。用比例而不是固定的数：历史预算改了，这里跟着变
KEEP_RATIO = 0.5

# 摘要的长度上限占历史预算的比例：摘要必须比它代替的历史小得多，压缩才有意义。
# 1/8：历史预算 6000 token 时上限约 750 token（约 1200 字），压缩至少省下被代替部分的 7/8。
# 依据（scripts/measure_context.py --budget 1500 实测）：600-1000 字的摘要仍然保住了用户的要求，15 个问题全对；
# 必须保留的"用户的要求和偏好"不到 100 字，其余是可以重新检索的事实。
# 没有测过更小的比例行不行，这个值是有依据的粗略选择，不是测出来的最优值
SUMMARY_RATIO = 1 / 8

# 摘要的提示词。固定的三部分，最重要的是第一部分：用户的要求一字不改地保留，
# 这样即使提出要求的那条消息被压缩掉了，模型也不会忘记（以前固定只带最近 6 条消息时，第 5 轮就忘了）。
# 不记录"文档里没有找到某内容"：实测模型会根据历史里的这类结论，断定知识库里没有某个文档，不再检索就直接拒答。
# 长度上限（{max_chars} 在调用时填入，见 max_chars）：不限制时，"已确认的信息"每次合并都在变多，
# 实测一次合并后从 1163 字涨到 2383 字，和整个历史预算差不多大。可以删掉旧的事实，
# 因为每个问题都会重新检索（agent 第 1 轮强制调用工具），文档里的信息随时能查回来。
# 这只是对模型的要求，不是保证：实测第二次合并时模型还是会写到 1700 多字。
# 试过超长时再调用一次让模型压缩，仍然有一次压缩后 1810 字，还多花约 20% 的钱，所以没有采用。
# 摘要比上下文窗口小得多（最长约 2400 字，约 1500 token），超出一些只是多花一点钱，不会撑爆上下文
SUMMARY_PROMPT = """你负责压缩一段对话的历史。下面给出"之前的摘要"（可能为空）和"新的对话"，请把它们合并成一份新的摘要，在后续对话中代替这些原始消息。
按下面三个部分输出，没有内容的部分写"无"：

## 用户的要求和偏好
用户对回答方式、格式、范围等提出的要求，一字不改地照抄原话。之前摘要里的要求要保留，除非用户明确取消了。

## 已确认的信息
对话中根据文档确认的事实，每条注明来自哪个文件。只记录回答里给出的具体信息，不要记录"文档里没有找到某内容"这类结论（可能只是当时没检索到）。

## 对话进展
用户在问什么话题，还有哪些问题没有解决。

整份摘要不超过 {max_chars} 字。超出时按这个顺序删减：先缩短"对话进展"，再删掉"已确认的信息"里较早、和当前话题无关的条目（需要时可以重新检索文档）。"用户的要求和偏好"永远完整保留。
只输出摘要本身，不要加其他说明。"""


# 摘要最多多少字：历史预算 × SUMMARY_RATIO，换算成字数。每次调用时算，历史预算改了跟着变
def max_chars() -> int:
    return round(agent.HISTORY_TOKEN_BUDGET * SUMMARY_RATIO / agent.TOKENS_PER_CHAR)


# 还没被摘要的消息里，最早的多少条应该被压缩。0 表示还不需要摘要
def messages_to_fold(unsummarized: list[Message]) -> int:
    if agent.estimate_tokens(unsummarized) <= agent.HISTORY_TOKEN_BUDGET:
        return 0
    keep = agent.recent_history(unsummarized, agent.HISTORY_TOKEN_BUDGET * KEEP_RATIO)
    return len(unsummarized) - len(keep)


# 把旧摘要和要压缩的消息合并成新摘要。返回大模型的回复（content 是新摘要，还带着 token 数和费用）
def summarize(old_summary: str | None, messages: list[Message]) -> llm.ChatReply:
    dialogue = "\n\n".join(f"{'用户' if m.role == MessageRole.USER else '助手'}：{m.content}" for m in messages)
    return llm.chat(
        [
            {"role": "system", "content": SUMMARY_PROMPT.format(max_chars=max_chars())},
            {"role": "user", "content": f"之前的摘要：\n{old_summary or '无'}\n\n新的对话：\n{dialogue}"},
        ]
    )
