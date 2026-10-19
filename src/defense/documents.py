"""
Document adapters: turn an uploaded file into what a real tool would hand the agent, plus a
visual preview for the demo console.

  PDF    -> the text of every page (what a PDF-reader tool returns) + rendered page images
            + text a person cannot see: white / transparent / tiny / off-page text
  HTML   -> the page source (what a web tool returns); the console renders it as a person sees it
  image  -> the screenshot itself (ViT on the pixels, OCR for its text) + OCR text boxes
  DOCX / TXT / MD -> plain text

This is a first tool adapter in the sense of the semester-8 plan ("tool adapters").
"""

from __future__ import annotations

import base64
import io
import re
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

KINDS = ("pdf", "web", "image", "text", "docx")
_IMAGE_MAGIC = (b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"RIFF", b"BM")
_HTML_HINT = re.compile(rb"<\s*(?:!doctype\s+html|html|body|head|div|p|span|h[1-6])\b", re.I)


def sniff_kind(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if data[:5] == b"%PDF-" or name.endswith(".pdf"):
        return "pdf"
    if any(data.startswith(m) for m in _IMAGE_MAGIC) or name.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")):
        return "image"
    if name.endswith(".docx") or (data[:2] == b"PK" and b"word/" in data[:4000]):
        return "docx"
    if name.endswith((".html", ".htm")) or _HTML_HINT.search(data[:4000]):
        return "web"
    return "text"


def data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
@dataclass
class PDFDocument:
    text: str
    page_count: int
    pages: List[Dict[str, Any]] = field(default_factory=list)     # preview: {image, width, height}
    hidden: List[Dict[str, Any]] = field(default_factory=list)    # {page, text, reason, rect:[x0,y0,x1,y1] 0-1}


def _norm_rect(r, page_rect) -> List[float]:
    w, h = page_rect.width or 1, page_rect.height or 1
    x0, y0, x1, y1 = r
    return [round(max(0.0, min(1.0, v)), 4) for v in ((x0 - page_rect.x0) / w, (y0 - page_rect.y0) / h,
                                                      (x1 - page_rect.x0) / w, (y1 - page_rect.y0) / h)]


def _paragraphs(blocks: List[Dict[str, Any]]) -> List[str]:
    """Text blocks -> paragraphs, wrapped lines joined (a line break is layout, not a sentence end).
    Some PDF writers put every line in its own block: a block whose first line starts right below the
    previous block's last line (gap < 30 % of a line), with the same line height and left edge, continues it."""
    paras: List[str] = []
    last = None                                              # bbox of the previous block's last line
    for block in blocks:
        lines = [ln for ln in block.get("lines", []) if "".join(s.get("text", "") for s in ln.get("spans", [])).strip()]
        if block.get("type") != 0 or not lines:
            continue
        text = " ".join(" ".join("".join(s.get("text", "") for s in ln["spans"]) for ln in lines).split())
        first = lines[0]["bbox"]
        if last is not None:
            h = last[3] - last[1]
            if (-0.5 * h < first[1] - last[3] < 0.3 * h and abs((first[3] - first[1]) - h) < 1.5
                    and abs(first[0] - last[0]) < 3):
                paras[-1] += ("" if paras[-1].endswith("-") and paras[-1][-2:-1].isalpha() else " ") + text
                last = lines[-1]["bbox"]
                continue
        paras.append(text)
        last = lines[-1]["bbox"]
    return paras


def read_pdf(data: bytes, max_preview_pages: int = 6, zoom: float = 1.5) -> PDFDocument:
    import pymupdf
    doc = pymupdf.open(stream=data, filetype="pdf")
    texts, pages, hidden = [], [], []
    for pno, page in enumerate(doc):
        blocks = page.get_text("dict").get("blocks", [])
        texts.append("\n\n".join(_paragraphs(blocks)))
        if pno < max_preview_pages:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            pages.append({"image": data_uri(pix.tobytes("jpeg", jpg_quality=70), "image/jpeg"),
                          "width": pix.width, "height": pix.height})
        # filled, non-white shapes: white text drawn on them is visible (e.g. a coloured letterhead)
        dark_fills = [pymupdf.Rect(d["rect"]) for d in page.get_drawings()
                      if d.get("fill") is not None and min(d["fill"]) < 0.8]
        for bno, block in enumerate(blocks):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    if len(txt) < 2:
                        continue
                    c = span.get("color", 0)
                    r, g, b = (c >> 16) & 255, (c >> 8) & 255, c & 255
                    reason = None
                    if span.get("alpha", 255) == 0:
                        reason = "transparent text"
                    elif min(r, g, b) >= 235 and not any(f.contains(pymupdf.Rect(span["bbox"])) for f in dark_fills):
                        reason = "white text on a white page"
                    elif span.get("size", 12) < 2.5:
                        reason = "microscopic text"
                    elif not page.rect.intersects(pymupdf.Rect(span["bbox"])):
                        reason = "text placed outside the page"
                    if reason:
                        last = hidden[-1] if hidden else None
                        if last and (last["page"], last["block"], last["reason"]) == (pno, bno, reason):
                            last["text"] += " " + txt                        # same text block: one finding
                            last["rects"].append(_norm_rect(span["bbox"], page.rect))
                        else:
                            hidden.append({"page": pno, "block": bno, "text": txt, "reason": reason,
                                           "rects": [_norm_rect(span["bbox"], page.rect)]})
    return PDFDocument("\n\n".join(t for t in texts if t).strip(), len(doc), pages, hidden)


def pdf_highlights(data: bytes, snippets: Sequence[Dict[str, str]], max_pages: int = 6) -> List[Dict[str, Any]]:
    """Locate sentences on the rendered pages. snippets: [{text, action}] -> [{page, action, rects}]."""
    import pymupdf
    doc = pymupdf.open(stream=data, filetype="pdf")
    out = []
    for sn in snippets:
        words = " ".join(sn["text"].split())
        for probe in dict.fromkeys((words, words[:70], words[:35])):   # whole sentence (it may wrap), then its start
            if len(probe) < 8:
                continue
            found = False
            for pno, page in enumerate(doc):
                if pno >= max_pages:
                    break
                rects = page.search_for(probe)
                if rects:
                    out.append({"page": pno, "action": sn["action"], "text": words[:160],
                                "rects": [_norm_rect(r, page.rect) for r in rects]})
                    found = True
                    break
            if found:
                break
    return out


# --------------------------------------------------------------------------- #
# Other formats
# --------------------------------------------------------------------------- #
def read_docx(data: bytes) -> str:
    import docx
    d = docx.Document(io.BytesIO(data))
    return "\n\n".join(p.text for p in d.paragraphs if p.text.strip())


def decode_text(data: bytes) -> str:
    for enc in ("utf-8", "utf-16"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def image_preview(path: str, max_side: int = 1100) -> Optional[Dict[str, Any]]:
    try:
        from PIL import Image
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=84)
            return {"image": data_uri(buf.getvalue(), "image/jpeg"), "width": img.width, "height": img.height}
    except Exception:
        return None


def ocr_boxes(predictor, path: str) -> List[Dict[str, Any]]:
    """OCR text boxes (normalised 0-1) on the same downscaled image the runtime reads."""
    try:
        engine = predictor.ocr_engine
        if not engine:
            return []
        arr = predictor._ocr_input(path)
        if isinstance(arr, str):
            return []
        h, w = arr.shape[:2]
        results, _ = engine(arr)
        boxes = []
        for item in results or []:
            pts, txt = item[0], item[1]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            boxes.append({"text": txt, "rect": [round(min(xs) / w, 4), round(min(ys) / h, 4),
                                                round(max(xs) / w, 4), round(max(ys) / h, 4)]})
        return boxes
    except Exception:
        return []


def is_docx(data: bytes) -> bool:
    try:
        return "word/document.xml" in zipfile.ZipFile(io.BytesIO(data)).namelist()
    except zipfile.BadZipFile:
        return False
