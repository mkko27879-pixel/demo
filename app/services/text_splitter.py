from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents(docs: list[Document]) -> list[Document]:
    # 初始化递归字符切分器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # 每块最大字符数
        chunk_overlap=50,      # 相邻块重叠字符数，保持上下文连贯
        length_function=len
    )
    # 使用 split_documents 保留原文档的 metadata（如页码）
    return splitter.split_documents(docs)