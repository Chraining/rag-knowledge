# 中文分词：把一段文字切成一个个词，给关键词检索用。
# 为什么需要：英文单词之间有空格，PostgreSQL 的全文检索按空格就能切词；中文没有空格，
# 不先分词的话，一整句会被当成一个"词"，几乎什么都匹配不上。
# 段落入库时（processing.save_chunks）和检索时（queries.keyword_search_chunks）都用这个函数，
# 两边切法必须完全一样，词才对得上

# logging：关掉 jieba 加载词典时打印的提示
import logging
# re：判断一个词是不是由字母、数字、汉字组成
import re

# jieba：最常用的中文分词库，用内置词典 + 统计模型（HMM）猜出词典里没有的新词
import jieba

# jieba 第一次分词时会打印 "Building prefix dict..." 等日志，没有用处
jieba.setLogLevel(logging.WARNING)

# 只保留由字母、数字、汉字组成的词，去掉空格和标点（jieba 会把它们也当成"词"输出）
WORD = re.compile(r"\w+")


# 分词，例如 "路由器报E171怎么处理" → ["路由", "路由器", "报", "e171", "怎么", "处理"]
def segment(text: str) -> list[str]:
    # lcut_for_search：搜索引擎模式，长词里的短词也会切出来（"路由器" → "路由"、"路由器"），
    # 这样搜"路由"也能匹配到写着"路由器"的段落。
    # lower()：统一成小写，搜 "xr-500" 也能匹配 "XR-500"
    return [word.lower() for word in jieba.lcut_for_search(text) if WORD.fullmatch(word)]
