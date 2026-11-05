"""Step 2: parse any supported file into pages of elements with bounding boxes, plus a page image per page.

PDF and images go through Docling (layout model, TableFormer, OCR when there is no text layer) and yield page images.
DOCX and PPTX are parsed natively by Docling; page images need LibreOffice (rendered to PDF first) and are skipped
without it, so those files get the text path only. Markdown and text are chunked from their own structure.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from pipeline.chunking import Element
from pipeline.config import settings


@dataclass
class ParsedPage:
    page_no: int
    width: int
    height: int
    image: Image.Image | None
    elements: list[Element]
    text: str
    parser: str


@dataclass
class ParsedDocument:
    title: str
    pages: list[ParsedPage] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


_converter = None


def _docling():
    """One converter per process. Models load on first use."""
    global _converter
    if _converter is not None:
        return _converter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions, EasyOcrOptions, PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

    s = settings()
    dev = {"cuda": AcceleratorDevice.CUDA, "mps": AcceleratorDevice.MPS, "cpu": AcceleratorDevice.CPU}.get(resolve_device(), AcceleratorDevice.AUTO)
    opts = PdfPipelineOptions(do_ocr=True, do_table_structure=True, generate_page_images=True, images_scale=1.0)
    opts.ocr_options = EasyOcrOptions(lang=list(s.ocr_languages))        # OCR only where a page has no text layer
    opts.table_structure_options.mode = TableFormerMode.FAST
    opts.accelerator_options = AcceleratorOptions(device=dev)
    img_opts = PdfPipelineOptions(do_ocr=True, do_table_structure=True, generate_page_images=True, images_scale=1.0)
    img_opts.ocr_options = EasyOcrOptions(force_full_page_ocr=True, lang=list(s.ocr_languages))
    img_opts.table_structure_options.mode = TableFormerMode.FAST
    img_opts.accelerator_options = AcceleratorOptions(device=dev)
    _converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts), InputFormat.IMAGE: ImageFormatOption(pipeline_options=img_opts)})
    return _converter


def resolve_device() -> str:
    s = settings()
    if s.device != "auto":
        return s.device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _soffice() -> str | None:
    for c in (shutil.which("soffice"), "/Applications/LibreOffice.app/Contents/MacOS/soffice"):
        if c and Path(c).exists():
            return c
    return None


def office_to_pdf(name: str, data: bytes) -> bytes | None:
    """Render DOCX/PPTX to PDF with LibreOffice so the visual path can embed pages. None when LibreOffice is absent."""
    exe = _soffice()
    if not exe:
        return None
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / name
        src.write_bytes(data)
        subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", td, str(src)], check=True, capture_output=True, timeout=180)
        out = src.with_suffix(".pdf")
        return out.read_bytes() if out.exists() else None


def _elements_from_docling(doc, page_no: int, W: float, H: float) -> list[Element]:
    from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem

    heading_path: list[str] = []
    els: list[Element] = []
    for item, _level in doc.iterate_items():
        if not getattr(item, "prov", None):
            continue
        prov = item.prov[0]
        if prov.page_no != page_no:
            continue
        b = prov.bbox.to_top_left_origin(H)
        bbox = [round(b.l / W, 4), round(b.t / H, 4), round(b.r / W, 4), round(b.b / H, 4)] if W and H else None
        label = item.label.value if hasattr(item.label, "value") else str(item.label)
        el = Element(id=f"{page_no}-e{len(els):03d}", type=label, bbox=bbox, heading_path=list(heading_path))
        if isinstance(item, SectionHeaderItem):
            lvl = int(getattr(item, "level", 1) or 1)
            heading_path = heading_path[: max(0, lvl - 1)] + [item.text.strip()]
            el.text, el.level = item.text, lvl
        elif isinstance(item, TableItem):
            el.markdown = item.export_to_markdown(doc)
            el.caption = item.caption_text(doc)
        elif isinstance(item, PictureItem):
            el.caption = item.caption_text(doc)
        else:
            el.text = getattr(item, "text", "") or ""
        els.append(el)
    return els


def parse_with_docling(name: str, data: bytes, title: str) -> ParsedDocument:
    from docling.datamodel.base_models import DocumentStream

    result = _docling().convert(DocumentStream(name=name, stream=io.BytesIO(data)))
    doc = result.document
    parsed = ParsedDocument(title=title)
    page_nos = sorted(doc.pages.keys()) or [1]
    for pno in page_nos:
        page = doc.pages.get(pno)
        W = float(page.size.width) if page and page.size else 0.0
        H = float(page.size.height) if page and page.size else 0.0
        img = page.image.pil_image if page and page.image is not None else None
        els = _elements_from_docling(doc, pno, W, H)
        text = "\n".join((e.text or e.markdown or e.caption or "") for e in els).strip()
        w, h = (img.width, img.height) if img else (int(W), int(H))
        parsed.pages.append(ParsedPage(page_no=pno, width=w, height=h, image=img, elements=els, text=text, parser="docling"))
    return parsed


def parse_office(name: str, kind: str, data: bytes, title: str) -> ParsedDocument:
    """DOCX/PPTX: render to PDF when LibreOffice exists (gives page images + boxes); otherwise Docling's native reader, text only."""
    pdf = office_to_pdf(name, data)
    if pdf:
        return parse_with_docling(Path(name).stem + ".pdf", pdf, title)
    from docling.datamodel.base_models import DocumentStream

    result = _docling().convert(DocumentStream(name=name, stream=io.BytesIO(data)))
    doc = result.document
    parsed = ParsedDocument(title=title, notes=["LibreOffice not available: parsed natively, no page images, visual path skipped"])
    by_page: dict[int, list] = {}
    from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem

    heading_path: list[str] = []
    for item, _ in doc.iterate_items():
        pno = item.prov[0].page_no if getattr(item, "prov", None) else 1
        els = by_page.setdefault(pno, [])
        label = item.label.value if hasattr(item.label, "value") else str(item.label)
        el = Element(id=f"{pno}-e{len(els):03d}", type=label, heading_path=list(heading_path))
        if isinstance(item, SectionHeaderItem):
            lvl = int(getattr(item, "level", 1) or 1); heading_path = heading_path[: max(0, lvl - 1)] + [item.text.strip()]; el.text = item.text
        elif isinstance(item, TableItem):
            el.markdown = item.export_to_markdown(doc); el.caption = item.caption_text(doc)
        elif isinstance(item, PictureItem):
            el.caption = item.caption_text(doc)
        else:
            el.text = getattr(item, "text", "") or ""
        els.append(el)
    for pno in sorted(by_page):
        els = by_page[pno]
        parsed.pages.append(ParsedPage(page_no=pno, width=0, height=0, image=None, elements=els, text="\n".join((e.text or e.markdown or e.caption or "") for e in els), parser=f"docling-{kind}"))
    return parsed


def parse_text(name: str, data: bytes, title: str, kind: str) -> ParsedDocument:
    """Markdown / plain text: one logical page per ~1,500 words so evidence still points somewhere."""
    text = data.decode("utf-8", errors="replace")
    words = text.split()
    size = 1500
    parsed = ParsedDocument(title=title)
    if not words:
        return parsed
    # keep markdown structure: split on word count but at line boundaries
    lines = text.splitlines()
    pages: list[list[str]] = [[]]
    count = 0
    for line in lines:
        pages[-1].append(line)
        count += len(line.split())
        if count >= size:
            pages.append([]); count = 0
    for i, chunk_lines in enumerate([p for p in pages if any(l.strip() for l in p)], start=1):
        body = "\n".join(chunk_lines)
        parsed.pages.append(ParsedPage(page_no=i, width=0, height=0, image=None, elements=[], text=body, parser=kind))
    return parsed


def to_display_copy(img: Image.Image, width: int) -> bytes:
    im = img.convert("RGB")
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80, optimize=True)
    return buf.getvalue()
