from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path: str):
    # 初始化 PDF 加载器，传入本地绝对/相对路径
    loader = PyPDFLoader(file_path=pdf_path)
    # 返回的是 List[Document]，每个 Document 代表一页，含 page_content 和 metadata
    return loader.load()

