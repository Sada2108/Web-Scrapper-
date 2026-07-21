"""
pdf_images.py
-------------
Extract figures/images from PDF datasheets using PyMuPDF (fitz).

Two strategies:
  1. Embedded pass:  pull real embedded bitmaps via page.get_images()
  2. Caption-region: find "Figure N" captions and render the region above each one

Heuristic note on the caption-region pass:
  We scan text blocks for /fig(?:ure)?\\.?\\s*\\d+/i, get the bounding box, and
  render a cropped pixmap from (caption_top - 40% page height) to caption_top,
  clamped to page bounds.  This is a reasonable heuristic for datasheets where
  figure captions sit directly below the corresponding graph, but it is NOT
  pixel-perfect — it may over-crop (include unrelated text above the figure)
  or under-crop (if the figure extends further up than 40 % of the page).
"""

import hashlib
import os
import re
from typing import List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


class PdfFigure:
    """One extracted figure from a PDF."""

    def __init__(
        self,
        image_bytes: bytes,
        page_num: int,
        method: str,
        caption: str = "",
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.image_bytes = image_bytes
        self.page_num = page_num
        self.method = method  # "embedded" or "rendered_region"
        self.caption = caption
        self.bbox = bbox

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.image_bytes).hexdigest()[:16]


def extract_pdf_figures(pdf_bytes: bytes) -> List[PdfFigure]:
    """Run both extraction passes and return a deduplicated list of figures."""
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) required: pip install PyMuPDF")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    figures: List[PdfFigure] = []
    seen_hashes: set = set()

    for page_num in range(len(doc)):
        page = doc[page_num]
        for fig in _embedded_pass(doc, page, page_num):
            h = fig.content_hash
            if h not in seen_hashes:
                seen_hashes.add(h)
                figures.append(fig)
        for fig in _caption_region_pass(page, page_num, doc):
            h = fig.content_hash
            if h not in seen_hashes:
                seen_hashes.add(h)
                figures.append(fig)

    doc.close()
    return figures


# -- embedded pass -----------------------------------------------------------

def _embedded_pass(doc, page, page_num: int) -> List[PdfFigure]:
    """Pull real embedded bitmap images via ``page.get_images(full=True)``."""
    result: List[PdfFigure] = []
    image_list = page.get_images(full=True)
    seen_xrefs: set = set()

    for img_info in image_list:
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        try:
            raw = doc.extract_image(xref)
            # PyMuPDF < 1.26 used dict with "image" key; 1.26+ changed API.
            if isinstance(raw, dict):
                img_bytes = raw["image"]
            else:
                img_bytes = raw
            result.append(
                PdfFigure(
                    image_bytes=img_bytes,
                    page_num=page_num,
                    method="embedded",
                    caption=f"Embedded image (xref={xref})",
                )
            )
        except Exception:
            continue

    return result


# -- caption-region pass -----------------------------------------------------

def _caption_region_pass(page, page_num: int, doc) -> List[PdfFigure]:
    """
    Find "Figure N" captions and render the page region above each one.

    Heuristic: for each caption we crop from ``max(0, caption_top - 40% page
    height)`` to ``caption_top``.  This works well for typical datasheets
    where the graph sits immediately above its label, but is not guaranteed
    to capture every figure cleanly.
    """
    result: List[PdfFigure] = []
    blocks = page.get_text("dict").get("blocks", [])

    page_height = page.rect.height
    page_width = page.rect.width

    captions: List[Tuple[float, str]] = []  # (y0, caption_text)

    for block in blocks:
        if block.get("type") != 0:  # 0 = text block
            continue
        text = _block_text(block)
        if re.search(r"fig(?:ure)?\.?\s*\d+", text, re.IGNORECASE):
            bbox = block.get("bbox")
            if bbox:
                captions.append((bbox[1], text.strip()))

    for caption_top, caption_text in captions:
        # Region extends upward by ~40% of page height from the caption baseline
        region_top = max(0, caption_top - 0.40 * page_height)
        # Include the caption line itself plus a small tail for context
        region_bottom = min(page_height, caption_top + page_height * 0.02)
        clip = fitz.Rect(0, region_top, page_width, region_bottom)

        try:
            pix = page.get_pixmap(clip=clip, dpi=150)
            img_bytes = pix.tobytes("png")
            result.append(
                PdfFigure(
                    image_bytes=img_bytes,
                    page_num=page_num,
                    method="rendered_region",
                    caption=caption_text,
                    bbox=tuple(clip),
                )
            )
        except Exception:
            continue

    return result


def _block_text(block: dict) -> str:
    """Concatenate text spans in a text block."""
    parts: List[str] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            parts.append(span.get("text", ""))
    return " ".join(parts)


# -- saving ------------------------------------------------------------------

def save_figures(figures: List[PdfFigure], output_dir: str) -> List[dict]:
    """Save each figure as a PNG file.  Returns metadata dicts."""
    os.makedirs(output_dir, exist_ok=True)
    saved: List[dict] = []

    for fig in figures:
        filename = f"p{fig.page_num+1:02d}_{fig.method}_{fig.content_hash}.png"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(fig.image_bytes)
        saved.append(
            {
                "filepath": filepath,
                "page_num": fig.page_num,
                "method": fig.method,
                "caption": fig.caption,
                "size_bytes": len(fig.image_bytes),
            }
        )

    return saved
