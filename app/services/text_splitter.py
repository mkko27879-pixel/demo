from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def attach_offsets(chunks: list[Document], docs: list[Document]) -> list[Document]:
    """给每个块标上它在所属页文本里的字符区间 [page_start, page_end)。

    切分器本身不返回偏移，这里用顺序查找倒推：每页从上次命中的位置往后找下一块。
    游标只 +1、不跳到块尾，因为相邻块有 overlap，同一段文字会出现两次，跳过去
    就会把后面那块定位到错误的位置。

    定位不到就不标——宁缺毋滥，前端拿不到区间时按页码跳转也能用。
    """
    page_texts = {doc.metadata.get("page"): doc.page_content
                  for doc in docs if doc.metadata.get("page") is not None}

    cursor: dict[int, int] = {}
    for chunk in chunks:
        page = chunk.metadata.get("page")
        page_text = page_texts.get(page)
        if not page_text:
            continue

        start = cursor.get(page, 0)
        pos = page_text.find(chunk.page_content, start)
        if pos < 0:
            pos = page_text.find(chunk.page_content)  # 游标跳过头了，从头再试一次
        if pos < 0:
            continue

        chunk.metadata["page_start"] = pos
        chunk.metadata["page_end"] = pos + len(chunk.page_content)
        cursor[page] = pos + 1
    return chunks


def split_documents(docs: list[Document]) -> list[Document]:
    """按中英文选择切分粒度。

    chunk_size 是字符数，同一个数字对中英文的信息量完全不是一回事：
    英文 900 字符约 150 个词，中文 900 字符就是 900 个字，塞进一个块里
    主题会被稀释，检索命中率明显下降。所以先看整篇文档的中文占比，再定参数。

    注意改这里会改变切分结果，已有数据要重新解析入库，否则新旧块粒度不一致。
    """
    text = "".join(d.page_content for d in docs)
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    cjk_ratio = cjk / len(text) if text else 0

    if cjk_ratio > 0.3:
        # 中文：字的信息密度高，块要小一些，重叠按块大小的 15% 左右
        chunk_size, chunk_overlap = 400, 60
    else:
        # 英文：原来 500 字符切得太碎（一个段落被拆成好几块），放大到 900
        chunk_size, chunk_overlap = 900, 150

    # 初始化递归字符切分器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,          # 每块最大字符数
        chunk_overlap=chunk_overlap,    # 相邻块重叠字符数，保持上下文连贯
        length_function=len
    )
    # 使用 split_documents 保留原文档的 metadata（如页码）
    chunks = splitter.split_documents(docs)
    # 再补上每块在本页文本里的字符区间，前端据此回原文高亮
    return attach_offsets(chunks, docs)
