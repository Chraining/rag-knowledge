# 测量处理一个超长文档时 worker 的内存峰值。只把"调用百炼 embedding 的网络请求"换成假的（返回随机向量），
# 其他都走真实代码：读文件、提取、切分、embed_texts 里的处理、写数据库。所以测出来的内存和真实情况一致，而且不花钱。
#
# 在和 worker 一样有 400MB 内存上限的新容器里运行（会往开发数据库写入，结束后删掉）：
#   docker compose run --rm -T worker python -m scripts.measure_memory 5000000 < /dev/null
# 参数：文档的字数（默认 500 万，约等于 10MB 的 GBK 编码小说）

# random：生成随机文字和假向量
import random
# resource：读取当前进程的内存峰值（只在 Linux 上有，所以要在容器里运行）
import resource
# sys：读取命令行参数
import sys
# time：计时
import time
# SimpleNamespace：随手造一个有属性的对象，用来假装成 openai 库返回的结果
from types import SimpleNamespace

# select：查询测试用户；func：统计段数
from sqlalchemy import func, select

# 被测的代码
from app import llm, processing, storage
# 数据库会话
from app.db import SessionLocal
# 模型
from app.models import Chunk, Document, User
# 向量维度
from app.models import EMBEDDING_DIM


# 假的 embedding 客户端：和 openai 库的接口长得一样，但不发网络请求
class FakeEmbeddings:
    def create(self, model, input, dimensions):
        # 和真实接口一样：每段文字返回一个 1024 维的向量（Python 的 float 列表），带着序号 index
        data = [SimpleNamespace(index=i, embedding=[random.random() for _ in range(dimensions)]) for i in range(len(input))]
        return SimpleNamespace(data=data)


# 读取整个容器（cgroup）用过的内存峰值。比进程峰值更接近"会不会被系统杀掉"
def cgroup_peak_mb() -> str:
    try:
        with open("/sys/fs/cgroup/memory.peak") as f:
            return f"{int(f.read()) / 1024 / 1024:.0f} MB"
    except OSError:
        return "unknown"


def main() -> None:
    chars = int(sys.argv[1]) if len(sys.argv) > 1 else 5_000_000
    # 把 llm 模块里创建客户端的函数换掉，embed_texts 本身的代码照常执行
    llm._client = lambda *args: SimpleNamespace(embeddings=FakeEmbeddings())
    # 生成一篇随机的中文"小说"，用 GBK 编码保存（每个汉字 2 字节）
    common = "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"
    # 先把每个汉字做成一个字符串对象（list(common)），再从这些对象里随机挑。
    # 不能写 random.choice(common)：每次都会新建一个单字的字符串对象（Python 只缓存拉丁字母的单字），
    # 500 万个对象要占约 380MB，测出来的就成了这个脚本自己的内存，而不是处理文档的内存
    text = "".join(random.choices(list(common), k=chars))
    content = text.encode("gbk")
    del text

    with SessionLocal() as db:
        user = db.scalar(select(User).order_by(User.id))
        doc = Document(
            user_id=user.id,
            filename="memory-test.txt",
            storage_path=storage.save_file(user.id, ".txt", content),
            file_size=len(content),
            content_hash=__import__("hashlib").sha256(content).hexdigest(),
        )
        db.add(doc)
        db.commit()
        del content
        start = time.monotonic()
        try:
            processing.process_document(db, doc.id)
            chunks = db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == doc.id))
            # ru_maxrss：这个进程用过的最大内存，Linux 上单位是 KB
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"chars={chars:,} chunks={chunks:,} dim={EMBEDDING_DIM} "
                  f"time={time.monotonic() - start:.0f}s process_peak={peak:.0f} MB container_peak={cgroup_peak_mb()}")
        finally:
            # 清理：删掉测试文档（段落会跟着级联删除）和文件
            db.rollback()
            db.delete(db.get(Document, doc.id))
            db.commit()
            storage.delete_file(doc.storage_path)


if __name__ == "__main__":
    main()
