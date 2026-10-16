# 文件格式相关的所有逻辑：上传时检查文件是否合法、处理时从文件里提取纯文本。
# 上传接口（routers/documents.py）和后台处理（processing.py）都用这里的函数，规则只写在一个地方。
#
# 支持的格式：
#   纯文本类：.md .txt .csv .html .htm（UTF-8 或 GBK 编码）
#   .pdf：有文字层的 PDF（扫描件提取不出文字）
#   Office：.docx（Word）.pptx（PowerPoint）.xlsx（Excel）
#
# Office 文件本质上是 zip 压缩包，里面装着很多 XML 文件。要防范"压缩炸弹"：
# 一个 10MB 的恶意文件，解压后可能有好几 GB，会把 worker 的内存撑爆。
# 所以打开之前先看压缩包目录里登记的"解压后大小"，太大就拒绝

# csv：解析 CSV 表格
import csv
# io：把 bytes / str 包装成"像文件一样可以读取的对象"，各个解析库都需要这种对象
import io
# zipfile：读取 zip 压缩包（Office 文件就是 zip）
import zipfile
# HTMLParser：Python 自带的 HTML 解析器
from html.parser import HTMLParser
# zip_longest：把两个长度不同的列表配对，短的那个用默认值补齐
from itertools import zip_longest

# docx：读取 Word 文件（python-docx 库）
import docx
# openpyxl：读取 Excel 文件
import openpyxl
# pptx：读取 PowerPoint 文件（python-pptx 库）
import pptx
# Table：Word 里的表格类型，用来区分"段落"和"表格"
from docx.table import Table
# PdfReader：读取 PDF；PyPdfError：pypdf 所有错误的基类（文件损坏等）
from pypdf import PdfReader
from pypdf.errors import PyPdfError

# Office 格式 → 压缩包里一定存在的"主文件"。用它判断这个 zip 是不是真的是对应格式的文件
OFFICE_MAIN_PART = {
    ".docx": "word/document.xml",
    ".pptx": "ppt/presentation.xml",
    ".xlsx": "xl/workbook.xml",
}
# 纯文本类的格式
TEXT_EXTENSIONS = {".md", ".txt", ".csv", ".html", ".htm"}
# 所有允许上传的扩展名
ALLOWED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf"} | set(OFFICE_MAIN_PART)
# 表格类的格式：按"整行"切块，而不是按字数硬切（见 processing.split_lines）
TABLE_EXTENSIONS = {".csv", ".xlsx"}

# 压缩包解压后的总大小上限：50MB
MAX_UNZIPPED_BYTES = 50 * 1024 * 1024
# 压缩包里单个 XML 文件的大小上限：20MB。
# XML 被解析成内存里的树结构后，占用的内存是文件大小的好几倍，worker 的内存上限只有 400MB
MAX_XML_PART_BYTES = 20 * 1024 * 1024


# 上传的文件不被支持（类型不对、内容和扩展名不符、编码不对、压缩包太大等）。上传接口把它转成 415
class UnsupportedFile(Exception):
    pass


# "永久性错误"：文件内容本身有问题（损坏的文件、扫描件提取不出文字等）。
# 这类错误重试多少次结果都一样，所以不重试，直接标记为 failed。
# 和它相对的是"临时性错误"（数据库短暂断开、磁盘忙等），那种值得重试，见 tasks.py
class PermanentError(Exception):
    pass  # 不需要额外内容，只是用类名来区分错误类型


# ============================ 上传时的检查 ============================


# 检查上传的文件：扩展名在白名单里，并且内容和扩展名相符。不合法就抛出 UnsupportedFile。
# 只看扩展名不可靠：任何文件都可以被改名成 .pdf 或 .docx
def validate(extension: str, content: bytes) -> None:
    if extension not in ALLOWED_EXTENSIONS:
        # sorted：按字母排好顺序，错误信息每次都一样
        raise UnsupportedFile("Supported file types: " + " ".join(sorted(ALLOWED_EXTENSIONS)))
    if extension == ".pdf":
        # PDF 文件的前 5 个字节一定是 "%PDF-"（叫"魔数"，magic number）
        if not content.startswith(b"%PDF-"):
            raise UnsupportedFile("Not a valid .pdf file")
    elif extension in OFFICE_MAIN_PART:
        _check_office_zip(extension, content)
    else:
        # 纯文本：能用 UTF-8 或 GBK 解码就行
        decode_text(content)


# 把文本文件的内容解码成字符串。先试 UTF-8，不行再试 GB18030（GBK 的超集，老的中文 Windows 程序常用）
def decode_text(content: bytes) -> str:
    unsupported = UnsupportedFile("Text files must be UTF-8 or GBK encoded")
    try:
        # utf-8-sig：和 utf-8 一样，只是会自动去掉 Windows 记事本有时加在开头的 BOM 标记。
        # 二进制文件（图片、压缩包等）几乎不可能恰好是合法的 UTF-8，所以解得开就当作文本
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        text = content.decode("gb18030")
    except UnicodeDecodeError as e:
        raise unsupported from e
    # GB18030 能解开大多数字节组合，很多二进制文件也能"解码成功"（变成一堆乱码）。
    # 二进制文件里几乎一定有空字符（\x00），正常的 GBK 文本里没有，用它来区分
    if "\x00" in text:
        raise unsupported
    return text


# 检查 Office 文件：必须是 zip、里面有对应格式的主文件、解压后不会太大
def _check_office_zip(extension: str, content: bytes) -> None:
    # zip 文件的魔数是 "PK\x03\x04"（PK 是 zip 格式发明者名字的缩写）
    if not content.startswith(b"PK\x03\x04"):
        raise UnsupportedFile(f"Not a valid {extension} file")
    try:
        # 这一步只读取压缩包末尾的"目录"，不解压任何内容，所以很快，也不占内存
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as e:
        raise UnsupportedFile(f"Not a valid {extension} file") from e
    # with：用完自动关闭压缩包
    with archive:
        # infolist()：压缩包里每个文件的信息（文件名、解压后的大小等）
        entries = archive.infolist()
        # 主文件不存在：可能是一个普通的 zip 被改成了 .docx，或者是 .docx 被改成了 .xlsx
        if OFFICE_MAIN_PART[extension] not in {e.filename for e in entries}:
            raise UnsupportedFile(f"Not a valid {extension} file")
        # 压缩炸弹检查。file_size 是目录里登记的"解压后大小"。
        # 就算有人在目录里谎报了一个很小的值也没用：Python 的 zipfile 解压时最多只输出登记的那么多字节
        too_big = sum(e.file_size for e in entries) > MAX_UNZIPPED_BYTES or any(
            e.filename.endswith(".xml") and e.file_size > MAX_XML_PART_BYTES for e in entries
        )
        if too_big:
            raise UnsupportedFile("File contains too much data once unzipped")


# ============================ 处理时提取文字 ============================


# 从文件内容里提取纯文本。文件有问题时抛出 PermanentError
def extract_text(content: bytes, extension: str) -> str:
    try:
        if extension == ".pdf":
            text = _pdf_text(content)
        elif extension in OFFICE_MAIN_PART:
            # 处理时再检查一次压缩炸弹：多一道保险，而且这一步很便宜
            _check_office_zip(extension, content)
            text = _office_text(extension, content)
        elif extension == ".csv":
            text = _csv_text(decode_text(content))
        elif extension in (".html", ".htm"):
            text = _html_text(decode_text(content))
        else:
            # .md 和 .txt：直接解码
            text = decode_text(content)
    except UnsupportedFile as e:
        # 上传时检查过，正常不会走到这里；万一走到了，说明文件本身有问题，重试也没用
        raise PermanentError(str(e)) from e
    # PostgreSQL 的 text 类型不能存空字符（\x00），而 PDF 提取出的文字里偶尔会有，必须去掉
    return text.replace("\x00", "")


# PDF：逐页提取文字，页与页之间用空行隔开
def _pdf_text(content: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(content))
        # 有密码的 PDF 读不出内容
        if reader.is_encrypted:
            raise PermanentError("PDF is encrypted")
        # extract_text() 在某些页上可能返回 None，用 "or ''" 换成空字符串
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except PyPdfError as e:
        # 文件头是 %PDF-，但内容损坏，pypdf 解析失败。from e：保留原始错误，方便排查
        raise PermanentError(f"Cannot parse PDF: {e}") from e


# Office 文件：交给对应的函数处理
def _office_text(extension: str, content: bytes) -> str:
    # 扩展名 → 处理函数
    readers = {".docx": _docx_text, ".pptx": _pptx_text, ".xlsx": _xlsx_text}
    try:
        return readers[extension](io.BytesIO(content))
    except Exception as e:
        # 文件损坏时，三个库会抛出各种不同的错误（zip 错误、XML 错误、KeyError 等），
        # 对我们来说都一样：这个文件读不了，重试也没用
        raise PermanentError(f"Cannot parse {extension} file: {e}") from e


# Word：按原文顺序取出段落和表格
def _docx_text(file: io.BytesIO) -> str:
    document = docx.Document(file)
    parts: list[str] = []
    # iter_inner_content()：按文档里的先后顺序依次给出段落和表格。
    # 如果分别取 document.paragraphs 和 document.tables，就不知道表格原来在哪两段之间了
    for block in document.iter_inner_content():
        if isinstance(block, Table):
            # 表格：每一行转成"列名: 值"的形式
            parts.extend(rows_to_lines([cell.text for cell in row.cells] for row in block.rows))
        elif block.text.strip():
            # 段落：跳过空段落
            parts.append(block.text)
    return "\n".join(parts)


# PowerPoint：每页的文字框、表格、演讲者备注。每页前面标上页码，回答时能看出出自第几页
def _pptx_text(file: io.BytesIO) -> str:
    presentation = pptx.Presentation(file)
    pages: list[str] = []
    # enumerate(..., 1)：页码从 1 开始数
    for number, slide in enumerate(presentation.slides, 1):
        texts: list[str] = []
        # shapes：这一页上的所有元素（文字框、图片、表格等）
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                texts.append(shape.text_frame.text)
            elif shape.has_table:
                texts.extend(rows_to_lines([cell.text for cell in row.cells] for row in shape.table.rows))
        # 演讲者备注：经常写着幻灯片上没有的解释，很有价值。没有备注区时 notes_text_frame 是 None
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame
            if notes is not None and notes.text.strip():
                texts.append("备注：" + notes.text)
        # 只有图片、没有文字的页跳过
        if texts:
            pages.append(f"第 {number} 页\n" + "\n".join(texts))
    return "\n\n".join(pages)


# Excel：每个工作表的每一行转成一行文字，带上工作表名和列名
def _xlsx_text(file: io.BytesIO) -> str:
    # read_only=True：边读边处理，不把整个表格一次性装进内存（大表格时省很多内存）
    # data_only=True：公式单元格取"上次保存时算出的值"，而不是公式本身（"=SUM(A1:A3)" 对检索没有意义）
    workbook = openpyxl.load_workbook(file, read_only=True, data_only=True)
    try:
        lines: list[str] = []
        for sheet in workbook.worksheets:
            # values_only=True：只要单元格的值，不要格式等其他信息
            lines.extend(rows_to_lines(sheet.iter_rows(values_only=True), title=sheet.title))
        return "\n".join(lines)
    finally:
        # read_only 模式会一直打开着文件，用完要手动关闭
        workbook.close()


# CSV：和 Excel 一样，每一行转成一行文字
def _csv_text(text: str) -> str:
    return "\n".join(rows_to_lines(csv.reader(io.StringIO(text))))


# 把表格的行转成文字。第一个非空行当作表头（列名），后面每一行变成 "列名: 值, 列名: 值"。
# 为什么不直接把单元格用逗号连起来：表格被切成很多段之后，中间的段落看不到表头，
# 只有一串数字，AI 不知道它们是什么意思。每一行都带上列名，任何一段单独拿出来都看得懂
# 例如表头 ["姓名", "部门"]，行 ["张三", "销售"] → "姓名: 张三, 部门: 销售"
def rows_to_lines(rows, title: str = "") -> list[str]:
    header: list[str] | None = None
    lines: list[str] = []
    for row in rows:
        # 每个单元格转成去掉首尾空白的字符串，空单元格（None）变成 ""
        cells = ["" if value is None else str(value).strip() for value in row]
        # 整行都是空的，跳过
        if not any(cells):
            continue
        # 第一个非空行是表头
        if header is None:
            header = cells
            continue
        # zip_longest：这一行比表头长时，多出来的列没有列名，用 "第 N 列" 代替；空的单元格不写
        pairs = [
            f"{name or f'第{i + 1}列'}: {value}"
            for i, (name, value) in enumerate(zip_longest(header, cells, fillvalue=""))
            if value
        ]
        # 有工作表名时加在前面，例如 "销售数据 | 姓名: 张三, 部门: 销售"
        lines.append((f"{title} | " if title else "") + ", ".join(pairs))
    return lines


# HTML：去掉标签，只留下文字。<script> 和 <style> 里的内容是代码，不是给人看的，整个跳过
class _HTMLText(HTMLParser):
    # 里面的内容要跳过的标签
    SKIP_TAGS = {"script", "style", "noscript", "template"}
    # "块级"标签：在它们的前后换行，否则相邻两段的文字会粘在一起
    BLOCK_TAGS = {
        "p", "div", "br", "li", "tr", "td", "th", "table", "ul", "ol", "pre", "blockquote",
        "section", "article", "header", "footer", "title", "h1", "h2", "h3", "h4", "h5", "h6",
    }

    def __init__(self) -> None:
        # 父类默认 convert_charrefs=True：会自动把 &amp; &lt; 等转义字符还原成 & < 等
        super().__init__()
        # 收集到的文字片段
        self.parts: list[str] = []
        # 当前在几层"要跳过的标签"里面。用计数而不是 True/False，是为了处理嵌套的情况
        self.skip_depth = 0

    # 遇到开始标签，例如 <p>
    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    # 遇到结束标签，例如 </p>
    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS:
            # max(…, 0)：多出来的结束标签（写错的 HTML）不会让计数变成负数
            self.skip_depth = max(self.skip_depth - 1, 0)
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    # 遇到标签之间的文字
    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)


def _html_text(text: str) -> str:
    parser = _HTMLText()
    parser.feed(text)
    parser.close()
    # 每一行里连续的空白（空格、制表符）合并成一个空格；去掉空行
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)
