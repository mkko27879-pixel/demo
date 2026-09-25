from langchain_core.documents import Document

from app.services.text_cleaner import clean_documents, clean_pdf_text


def test_removes_arxiv_header_and_page_number():
    raw = "arXiv:2601.01674v1 [hep-ex] 4 Jan 2026\nSome content here.\n\n42\n"
    cleaned = clean_pdf_text(raw)
    assert "arXiv:" not in cleaned
    assert cleaned == "Some content here."


def test_joins_hyphen_break():
    assert clean_pdf_text("experi-\nment shows") == "experiment shows"


def test_folds_ligature():
    # ﬀ 是单独的 Unicode 字符，不还原就和 different 匹配不上
    assert "different" in clean_pdf_text("di\ufb00erent ways")
    assert "flow" in clean_pdf_text("\ufb02ow")


def test_merges_soft_wrap_but_keeps_paragraphs():
    assert clean_pdf_text("line one\nline two") == "line one line two"
    assert clean_pdf_text("para one\n\npara two") == "para one\n\npara two"


def test_drops_page_that_is_only_noise():
    docs = [Document(page_content="arXiv:2601.01674v1 [hep-ex] 4 Jan 2026",
                     metadata={"page": 0})]
    assert clean_documents(docs) == []


def test_keeps_metadata():
    docs = [Document(page_content="real content " * 5, metadata={"page": 7})]
    cleaned = clean_documents(docs)
    assert cleaned[0].metadata["page"] == 7


def test_strips_references_section():
    raw = ("正文第一段，讲的是方法。\n"
           "\n"
           "References\n"
           "\n"
           "[1] A. Author, Phys. Rev. D 100, 012345 (2019)\n"
           "[2] B. Author, ibid. 101, 054321 (2020)\n")
    cleaned = clean_pdf_text(raw)
    assert "正文第一段" in cleaned
    assert "References" not in cleaned
    assert "012345" not in cleaned


def test_drops_reference_only_page():
    raw = "\n".join(f"[{i}] A. Author et al., Phys. Rev. D {i}, 012345 (2019)"
                    for i in range(1, 12))
    assert clean_pdf_text(raw) == ""


def test_keeps_normal_academic_text_with_brackets():
    """正文里的方括号引用不能被误当文献条目丢掉。"""
    raw = ("我们改进了此前的方法 [4]，并结合 ARIS [5] 与 SCENE [6] 的数据，"
           "把能量下限推到 7 keV，结论见 Fig. 2。")
    cleaned = clean_pdf_text(raw)
    assert "ARIS" in cleaned and "7 keV" in cleaned
