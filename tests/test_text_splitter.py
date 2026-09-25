from langchain_core.documents import Document

from app.services.text_splitter import split_documents


def _doc(text: str) -> Document:
    return Document(page_content=text, metadata={"page": 0})


def test_english_uses_large_chunks():
    chunks = split_documents([_doc("word " * 400)])          # 2000 字符英文
    assert chunks, "不该切出空结果"
    assert all(len(c.page_content) <= 900 for c in chunks)
    assert len(chunks) < 5


def test_chinese_uses_small_chunks():
    chunks = split_documents([_doc("核物理中的熵与复杂度分析，" * 60)])  # 纯中文
    assert all(len(c.page_content) <= 400 for c in chunks)
    assert len(chunks) >= 3


def test_metadata_survives_splitting():
    chunks = split_documents([_doc("hello world " * 100)])
    assert all(c.metadata["page"] == 0 for c in chunks)


def test_offsets_point_back_to_source():
    """偏移必须可用：拿 page_start/page_end 去切原页文本，应当还原出这个块。"""
    text = "熵与复杂度的定义如下。" * 120
    chunks = split_documents([_doc(text)])

    assert len(chunks) > 1, "文本不够长，切不出多块就测不出偏移"
    for chunk in chunks:
        start, end = chunk.metadata["page_start"], chunk.metadata["page_end"]
        assert text[start:end] == chunk.page_content
