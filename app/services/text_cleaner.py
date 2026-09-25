import re

from langchain_core.documents import Document

# arXiv 每页顶部都有的标识行，例如 "arXiv:2601.01674v1 [hep-ex] 4 Jan 2026"
_ARXIV_HEADER = re.compile(r"^.*arXiv:\s*\d{4}\.\d{4,5}v\d+\s*\[[^\]]*\].*$", re.MULTILINE)
# 单独成行的页码
_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
# 行尾连字符断词：experi-\nment -> experiment
_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
# 段内换行只是排版换行（前后都不是空行），合并成空格；空行保留为段落分隔
_SOFT_WRAP = re.compile(r"(?<!\n)\n(?!\s*\n)")
# 连续空白
_SPACES = re.compile(r"[ \t\u00a0\u3000]{2,}")
# 参考文献段的起始标题（单独成行）
_REFERENCES_HEAD = re.compile(r"^\s*(References|REFERENCES|Bibliography)\s*$", re.MULTILINE)
# 单条文献，形如 "[12] A. Author et al., Phys. Rev. D 100, 012345 (2019)"
_REF_ENTRY = re.compile(r"^\s*\[\d{1,3}\]")
# PDF 字体映射的产物：Unicode 连字。"diﬀerent" 里的 ﬀ 是一个单独的字符，
# 不还原成 ff 的话，这个词和 "different" 在向量空间里完全对不上。
_LIGATURES = str.maketrans({
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "ft", "\ufb06": "st",
})


def _is_reference_page(text: str) -> bool:
    """判断这页是不是整页参考文献。

    参考文献是纯噪声源：术语密集、期刊缩写满天飞，特别容易和"统计方法比较模型"
    这种抽象问句匹配上——实测一条 Zelevinsky 1995 的文献条目能排到检索第一，
    而它跟问题毫无关系。它又几乎不可能是答案来源，所以整页丢弃是划算的。
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 5:
        return False
    return sum(1 for ln in lines if _REF_ENTRY.match(ln)) / len(lines) > 0.5


def clean_pdf_text(text: str) -> str:
    """清洗 PyPDFLoader 抽出来的原始文本。

    PDF 抽出来的文本是按"视觉行"排列的，直接向量化会损失语义，实测有三类问题：
      - "experi-\\nment" 这种连字符断词会把单词切两半，embedding 直接跑偏
      - 页眉页脚是纯噪声，实测检索时 "arXiv:2601.01674v1 [hep-ex]" 能排到第一
      - 每行末尾都带换行，一个段落被切得很碎，语义完整性变差
      - "diﬀerent" 这种连字（ﬀ 是单个 Unicode 字符）和 "different" 匹配不上
    这里只做上述几类确定性清洗，不重排词序、不改公式。

    已知取舍：跨行连字符（如 well-\\nknown）会被合并成 wellknown。PDF 里断词换行
    远多于复合词跨行，所以选择这一侧。
    """
    # 软连字符、零宽字符先删掉，它们不可见但会混进 token；
    # 再把连字还原成普通字母
    text = text.replace("\u00ad", "").replace("\u200b", "").replace("\ufeff", "")
    text = text.translate(_LIGATURES)
    text = _ARXIV_HEADER.sub("", text)
    text = _PAGE_NUMBER.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    # 参考文献必须在合并换行之前处理：这两条规则按"视觉行"判断
    # （References 标题单独成行、每条文献以 [12] 开头），一旦先把段内换行
    # 折成空格，标题和条目就都不在行首了，规则全部失效。
    # 第一版就是栽在这个顺序上——改完指标一点没动，块数也没少。
    match = _REFERENCES_HEAD.search(text)
    if match:
        text = text[: match.start()]
    if _is_reference_page(text):
        return ""

    text = _SOFT_WRAP.sub(" ", text)
    text = _SPACES.sub(" ", text)
    return text.strip()


def clean_documents(docs: list[Document]) -> list[Document]:
    """逐页清洗正文，metadata（页码等）原样保留。

    整页洗完是空的时候直接丢掉：那种页通常是纯封面或纯页眉页脚，留着只会生成
    空块污染检索。
    """
    cleaned = []
    for doc in docs:
        text = clean_pdf_text(doc.page_content)
        if text:
            cleaned.append(Document(page_content=text, metadata=doc.metadata))
    return cleaned
