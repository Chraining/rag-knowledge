"""Build a small evaluation subset of the public C-MTEB DuRetrieval dataset.

    uv run --with pyarrow python -m scripts.build_duretrieval

生成公开数据集 DuRetrieval 的评测子集。用法见上面的命令（--with pyarrow：临时装上读取 parquet 文件的库，不写进项目依赖）。

DuRetrieval：百度发布的中文检索数据集，收录在中文检索基准 C-MTEB 里。
内容是真实的百度搜索查询和网页段落，每个查询人工标注了哪些段落和它相关（平均约 5 段）。
全量有 10 万段、2000 个查询，全部做 embedding 又慢又贵，所以抽取一个子集：
  随机 300 个查询 + 它们的全部相关段落 + 随机的其他段落作为干扰项，凑够 1 万段。
子集里干扰项少了，分数会比全量偏高；但不同检索方式在同一个子集上比较，结论仍然有效。
随机种子固定，每次生成的子集完全一样。

输出（不提交到 git，需要时用上面的命令重新生成）：
  eval/duretrieval/corpus.jsonl     每行一个段落：{"id": 段落 id, "text": 内容}
  eval/duretrieval/questions.jsonl  每行一个问题：{"question": 查询, "relevant": [相关段落 id], "type": "duretrieval"}
下载的原始文件缓存在 eval/duretrieval/raw/，第二次运行不再下载。
"""

# json：写 jsonl 文件
import json
# random：随机抽样
import random
# urlretrieve：下载文件
from urllib.request import urlretrieve
# Path：处理文件路径
from pathlib import Path

# pyarrow.parquet：读取 parquet 格式（一种按列存储的数据文件，Hugging Face 上的数据集多用这种格式）
import pyarrow.parquet as pq

# 输出目录
OUT_DIR = Path(__file__).resolve().parent.parent / "eval" / "duretrieval"
# 原始文件的缓存目录
RAW_DIR = OUT_DIR / "raw"
# 三个原始文件：段落、查询、相关性标注（qrels）。地址带具体文件名，数据集更新后也不会悄悄变成别的内容
BASE = "https://huggingface.co/datasets/C-MTEB"
FILES = {
    "corpus": f"{BASE}/DuRetrieval/resolve/main/data/corpus-00000-of-00001-19b9e924cb33e4d5.parquet",
    "queries": f"{BASE}/DuRetrieval/resolve/main/data/queries-00000-of-00001-7c7edb40be6b560c.parquet",
    "qrels": f"{BASE}/DuRetrieval-qrels/resolve/main/data/dev-00000-of-00001-d3c385852a7c0c9d.parquet",
}
# 抽多少个查询
N_QUESTIONS = 300
# 子集一共多少段（相关段落 + 干扰项）
N_PASSAGES = 10_000
# 随机种子：固定后每次抽到的都一样，结果可以复现
SEED = 42


# 下载（已经下载过就跳过），读成字典列表，例如 [{"id": ..., "text": ...}, ...]
def load(name: str) -> list[dict]:
    path = RAW_DIR / f"{name}.parquet"
    if not path.exists():
        print(f"下载 {name} ...")
        urlretrieve(FILES[name], path)
    return pq.read_table(path).to_pylist()


# 写 jsonl：每行一个 JSON。ensure_ascii=False：中文按原样写，不转成 \uXXXX
def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    # parents=True：上级目录不存在就一起创建；exist_ok=True：已存在不报错
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # {段落 id: 内容}
    passages = {row["id"]: row["text"] for row in load("corpus")}
    # {查询 id: 查询内容}
    queries = {row["id"]: row["text"] for row in load("queries")}
    # {查询 id: [相关段落 id, ...]}
    relevant: dict[str, list[str]] = {}
    for row in load("qrels"):
        relevant.setdefault(row["qid"], []).append(row["pid"])

    rnd = random.Random(SEED)
    # sorted 之后再抽：保证抽样结果不依赖字典的顺序
    chosen = rnd.sample(sorted(relevant), N_QUESTIONS)
    # 这些查询的全部相关段落（用集合去重：一段可能同时和好几个查询相关）
    needed = {pid for qid in chosen for pid in relevant[qid]}
    # 从剩下的段落里随机抽干扰项，凑够 N_PASSAGES 段
    others = sorted(set(passages) - needed)
    picked = sorted(needed) + rnd.sample(others, N_PASSAGES - len(needed))

    OUT_DIR.mkdir(exist_ok=True)
    write_jsonl(OUT_DIR / "corpus.jsonl", [{"id": pid, "text": passages[pid]} for pid in picked])
    write_jsonl(
        OUT_DIR / "questions.jsonl",
        [{"question": queries[qid], "relevant": relevant[qid], "type": "duretrieval"} for qid in chosen],
    )
    chars = sum(len(passages[pid]) for pid in picked)
    print(f"完成：{N_QUESTIONS} 个查询，{len(needed)} 段相关段落 + {N_PASSAGES - len(needed)} 段干扰项，共 {chars} 字")


if __name__ == "__main__":
    main()
