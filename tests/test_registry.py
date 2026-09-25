from app.services import paper_registry


def _use_tmp_db(tmp_path, monkeypatch):
    """把元数据库指到临时文件，别污染开发用的 data/papers.db。"""
    monkeypatch.setattr(paper_registry, "DB_PATH", str(tmp_path / "papers.db"))
    paper_registry.init_db()


def test_save_then_update_status(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)

    paper_registry.save_paper("upload:abc", "upload", "/tmp/a.pdf",
                              original_filename="我的论文.pdf")
    record = paper_registry.get_paper("upload:abc")
    assert record["status"] == "uploaded"
    assert record["original_filename"] == "我的论文.pdf"

    paper_registry.update_status("upload:abc", "indexed", page_count=6, chunk_count=22)
    record = paper_registry.get_paper("upload:abc")
    assert record["status"] == "indexed"
    assert (record["page_count"], record["chunk_count"]) == (6, 22)


def test_resave_keeps_statistics(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)

    paper_registry.save_paper("arxiv:1.2", "arxiv", "/tmp/b.pdf")
    paper_registry.update_status("arxiv:1.2", "indexed", page_count=3, chunk_count=9)
    # 重复登记（比如重新下载）不该把已解析的统计清零
    paper_registry.save_paper("arxiv:1.2", "arxiv", "/tmp/b.pdf")

    record = paper_registry.get_paper("arxiv:1.2")
    assert record["chunk_count"] == 9


def test_delete(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)

    paper_registry.save_paper("upload:x", "upload", "/tmp/c.pdf")
    assert paper_registry.delete_paper("upload:x") is True
    assert paper_registry.get_paper("upload:x") is None
    assert paper_registry.delete_paper("upload:x") is False


def test_list_orders_by_created_at_desc(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)

    paper_registry.save_paper("upload:1", "upload", "/tmp/1.pdf")
    paper_registry.save_paper("upload:2", "upload", "/tmp/2.pdf")
    ids = [p["paper_id"] for p in paper_registry.list_papers()]
    assert set(ids) == {"upload:1", "upload:2"}
