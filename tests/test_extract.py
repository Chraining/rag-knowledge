# 测试文件格式：上传时的检查、各种格式的文字提取、压缩炸弹防护、表格按行切分

# io：在内存里生成测试文件
import io
# zipfile：手工做各种"假的" Office 文件
import zipfile

# docx、openpyxl、pptx：生成真实的 Word、Excel、PowerPoint 测试文件
import docx
import openpyxl
import pptx
# pytest：测试框架
import pytest
# Inches：PowerPoint 里的长度单位，添加表格时要指定位置和大小
from pptx.util import Inches
# select：直接查数据库做验证
from sqlalchemy import select

# 被测试的模块
from app import extract, processing
# 模型
from app.models import Chunk, DocumentStatus
# 用文档接口的上传辅助函数
from tests.test_documents import upload
# 复用处理测试里的 fixture：把内容存成一个 pending 状态的文档。导入之后 pytest 就能在这个文件里用它
from tests.test_processing import make_document  # noqa: F401


# ============================ 辅助函数：生成测试文件 ============================


# 把一个"保存到文件"的对象保存到内存里，返回文件内容
def to_bytes(obj) -> bytes:
    buffer = io.BytesIO()
    obj.save(buffer)
    return buffer.getvalue()


# Word：一个段落、一个表格、再一个段落（用来检查顺序）
def make_docx() -> bytes:
    document = docx.Document()
    document.add_paragraph("第一段：报销流程")
    table = document.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "项目"
    table.rows[0].cells[1].text = "上限"
    table.rows[1].cells[0].text = "打车"
    table.rows[1].cells[1].text = "100元"
    document.add_paragraph("最后一段")
    return to_bytes(document)


# PowerPoint：第 1 页有标题和备注，第 2 页有一个表格
def make_pptx() -> bytes:
    presentation = pptx.Presentation()
    # slide_layouts[1]：带标题和正文的版式；[5]：只有标题的版式
    first = presentation.slides.add_slide(presentation.slide_layouts[1])
    first.shapes.title.text = "季度目标"
    first.notes_slide.notes_text_frame.text = "强调增长 20%"
    second = presentation.slides.add_slide(presentation.slide_layouts[5])
    table = second.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(4), Inches(1)).table
    table.cell(0, 0).text = "地区"
    table.cell(0, 1).text = "销量"
    table.cell(1, 0).text = "华东"
    table.cell(1, 1).text = "300"
    return to_bytes(presentation)


# Excel：两个工作表，中间有空行
def make_xlsx() -> bytes:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "员工"
    sheet.append(["姓名", "部门"])
    sheet.append(["张三", "销售"])
    sheet.append([None, None])  # 空行，应该被跳过
    sheet.append(["李四", "研发"])
    second = workbook.create_sheet("预算")
    second.append(["项目", "金额"])
    second.append(["服务器", 5000])
    return to_bytes(workbook)


# 手工做一个 zip：files 是 {文件名: 内容}
def make_zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    # ZIP_DEFLATED：压缩存储（默认是不压缩）。压缩炸弹就是靠高压缩率
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


# ============================ 文字提取 ============================


# 测试：Word 的段落和表格按原文顺序提取，表格的每一行带着列名
def test_extract_docx_keeps_order_and_table_headers():
    text = extract.extract_text(make_docx(), ".docx")
    assert text.splitlines() == ["第一段：报销流程", "项目: 打车, 上限: 100元", "最后一段"]


# 测试：PowerPoint 提取每页的文字、表格和备注，并标上页码
def test_extract_pptx_with_notes_and_table():
    text = extract.extract_text(make_pptx(), ".pptx")
    assert "第 1 页\n季度目标" in text
    assert "备注：强调增长 20%" in text
    assert "第 2 页" in text and "地区: 华东, 销量: 300" in text


# 测试：Excel 每一行带着工作表名和列名，空行被跳过，多个工作表都提取
def test_extract_xlsx_rows_with_headers():
    text = extract.extract_text(make_xlsx(), ".xlsx")
    assert text.splitlines() == [
        "员工 | 姓名: 张三, 部门: 销售",
        "员工 | 姓名: 李四, 部门: 研发",
        "预算 | 项目: 服务器, 金额: 5000",
    ]


# 测试：CSV 和 Excel 一样处理；GBK 编码的 CSV 也能读（Excel 在中文 Windows 上默认存成 GBK）
def test_extract_csv_gbk():
    content = "城市,人口\n上海,2487万\n".encode("gbk")
    assert extract.extract_text(content, ".csv") == "城市: 上海, 人口: 2487万"


# 测试：表格里比表头多出来的列用"第 N 列"代替列名，空单元格不写
def test_rows_to_lines_extra_and_empty_cells():
    lines = extract.rows_to_lines([["名称", "备注"], ["A", "", "额外"]])
    assert lines == ["名称: A, 第3列: 额外"]


# 测试：HTML 去掉标签、跳过 <script> 和 <style>、还原 &amp; 这类转义字符、块级标签之间换行
def test_extract_html():
    html = (
        "<html><head><title>标题</title><style>p{color:red}</style></head>"
        "<body><p>你好&amp;欢迎</p><script>alert(1)</script><div>第二   段</div></body></html>"
    )
    assert extract.extract_text(html.encode(), ".html") == "标题\n你好&欢迎\n第二 段"


# 测试：损坏的 Office 文件（是 zip、有主文件，但内容不对）标记为永久性错误，而不是让 worker 崩溃
def test_extract_corrupt_docx_is_permanent_error():
    content = make_zip({"word/document.xml": b"not xml at all"})
    with pytest.raises(extract.PermanentError):
        extract.extract_text(content, ".docx")


# ============================ 上传时的检查 ============================


# 测试：真实的 Word、PowerPoint、Excel、CSV、HTML 文件都能上传
@pytest.mark.parametrize(
    "filename, content",
    [
        ("a.docx", make_docx()),
        ("a.pptx", make_pptx()),
        ("a.xlsx", make_xlsx()),
        ("a.csv", b"a,b\n1,2\n"),
        ("a.html", b"<p>hi</p>"),
    ],
    # ids：给每组参数起个短名字。不写的话 pytest 会用文件的原始字节当名字，在 Windows 上会因为太长而报错
    ids=["docx", "pptx", "xlsx", "csv", "html"],
)
def test_upload_new_formats(client, register_and_login, filename, content):
    headers = register_and_login("alice")
    assert upload(client, headers, filename, content, "application/octet-stream").status_code == 201


# 测试：内容和扩展名不符的 Office 文件，返回 415
@pytest.mark.parametrize(
    "filename, content",
    [
        ("text.docx", b"just some text"),  # 根本不是 zip
        ("plain.docx", make_zip({"hello.txt": b"hi"})),  # 是 zip，但不是 Word 文件
        ("sheet.docx", make_xlsx()),  # 是 Excel 文件，被改成了 .docx
    ],
    ids=["not-zip", "plain-zip", "renamed-xlsx"],
)
def test_upload_rejects_fake_office_files(client, register_and_login, filename, content):
    headers = register_and_login("alice")
    r = upload(client, headers, filename, content, "application/octet-stream")
    assert r.status_code == 415
    assert "Not a valid .docx file" in r.json()["detail"]


# 测试：不支持的格式返回 415，错误信息里列出了支持哪些格式
def test_upload_rejects_unknown_format_with_list(client, register_and_login):
    headers = register_and_login("alice")
    r = upload(client, headers, "old.doc", b"whatever", "application/msword")
    assert r.status_code == 415
    assert ".docx" in r.json()["detail"]


# 测试（安全）：压缩炸弹。压缩后很小、解压后超过上限的文件，返回 415。
# 为了让测试跑得快，把上限临时调小到 10KB，再做一个解压后 100KB 的文件（全是 0，压缩后只有几百字节）
def test_upload_rejects_zip_bomb(client, register_and_login, monkeypatch):
    monkeypatch.setattr(extract, "MAX_UNZIPPED_BYTES", 10_000)
    content = make_zip({"word/document.xml": b"<w/>", "word/padding.bin": b"\x00" * 100_000})
    assert len(content) < 2_000  # 确认它确实"压缩后很小"
    headers = register_and_login("alice")
    r = upload(client, headers, "bomb.docx", content, "application/octet-stream")
    assert r.status_code == 415
    assert "too much data" in r.json()["detail"]


# 测试（安全）：单个 XML 文件太大也拒绝（解析成树结构后很占内存）
def test_rejects_oversized_xml_part(monkeypatch):
    monkeypatch.setattr(extract, "MAX_XML_PART_BYTES", 1_000)
    content = make_zip({"word/document.xml": b"<w>" + b"a" * 5_000 + b"</w>"})
    with pytest.raises(extract.UnsupportedFile):
        extract.validate(".docx", content)


# ============================ 表格按行切分 ============================


# 测试：表格按整行装进每一段，不会把一行从中间切断
def test_split_lines_keeps_rows_whole():
    assert processing.split_lines("aaaa\nbbbb\ncccc", size=10) == ["aaaa\nbbbb", "cccc"]


# 测试：单独一行就超过上限时，这一行按普通方式切，前后的行不受影响
def test_split_lines_long_row():
    chunks = processing.split_lines("aa\n" + "x" * 25 + "\nbb", size=10)
    assert chunks[0] == "aa" and chunks[-1] == "bb"
    assert all(len(c) <= 10 for c in chunks)


# 测试：Excel 文档走完整流程，每一段都是完整的行，并且带着列名
def test_process_xlsx_document(db, make_document):
    doc = make_document(make_xlsx(), ".xlsx")
    processing.process_document(db, doc.id)
    db.refresh(doc)
    assert doc.status == DocumentStatus.READY
    content = db.scalar(select(Chunk.content).where(Chunk.document_id == doc.id))
    assert "员工 | 姓名: 张三, 部门: 销售" in content
