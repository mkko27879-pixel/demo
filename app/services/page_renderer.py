import base64

# 用 pymupdf 而不是 fitz 别名：1.28 起 fitz 这个入口被标记为弃用
import pymupdf


def render_page_png(pdf_path: str, page_number: int, dpi: int = 120) -> bytes:
    """把 PDF 的某一页渲染成 PNG 字节。page_number 从 1 开始，和入库的页码一致。"""
    with pymupdf.open(pdf_path) as doc:
        if not 1 <= page_number <= doc.page_count:
            raise ValueError(f"页码超出范围：{page_number}（这篇共 {doc.page_count} 页）")
        page = doc[page_number - 1]
        # 120 DPI：公式里的小字号还能看清，像素数只有 144 DPI 的七成。
        # 图片 token 是按像素算的，实测看一次图能顶几十次纯文本问答，
        # 所以清晰度够用就行，别再往上加。
        return page.get_pixmap(dpi=dpi).tobytes("png")


def render_page_data_url(pdf_path: str, page_number: int, dpi: int = 120) -> str:
    """渲染成 data URL，可以直接塞进 OpenAI 兼容格式的 image_url 字段。"""
    png = render_page_png(pdf_path, page_number, dpi)
    return "data:image/png;base64," + base64.b64encode(png).decode()
