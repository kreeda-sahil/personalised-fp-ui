"""PDF rasterization helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pymupdf
from PIL import Image


def pdf_bytes_to_png(
    pdf_bytes: bytes,
    *,
    page_index: int = 0,
    dpi: int = 200,
) -> Image.Image:
    """Render a PDF page to a PIL RGB image."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        if doc.page_count == 0:
            raise ValueError("PDF has no pages.")
        idx = min(max(page_index, 0), doc.page_count - 1)
        page = doc.load_page(idx)
        zoom = dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def save_png(image: Image.Image, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path
