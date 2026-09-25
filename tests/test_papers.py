import pytest

from app.routers.papers import parse_arxiv_id


@pytest.mark.parametrize("url, expected", [
    ("https://arxiv.org/abs/2601.00597", "2601.00597"),
    ("https://arxiv.org/pdf/2601.00597.pdf", "2601.00597"),
    ("https://arxiv.org/pdf/2601.00597v3", "2601.00597"),
    ("https://arxiv.org/html/2601.00597v1", "2601.00597"),
    # 旧式编号带分类前缀，斜杠必须保留
    ("https://arxiv.org/abs/hep-th/9901001", "hep-th/9901001"),
    # 带查询串和结尾斜杠
    ("http://export.arxiv.org/abs/2601.01674?context=cs.CL", "2601.01674"),
    ("https://arxiv.org/abs/2601.00597/", "2601.00597"),
    # 直接贴编号
    ("2601.00597", "2601.00597"),
])
def test_parse_arxiv_id_ok(url, expected):
    assert parse_arxiv_id(url) == expected


@pytest.mark.parametrize("url", [
    "",
    "https://example.com/paper.pdf",
    "https://arxiv.org/abs/abc",
    "https://arxiv.org/abs/2601.005",       # 小数位不足 4 位
])
def test_parse_arxiv_id_rejects(url):
    with pytest.raises(ValueError):
        parse_arxiv_id(url)
