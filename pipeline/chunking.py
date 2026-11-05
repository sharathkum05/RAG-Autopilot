"""Step 3: structural chunking. The page is the evidence unit; chunks exist to give the text embedders the right size.

Rules: paragraphs packed to ~chunk_words without splitting one; a table is one chunk, split by rows with the header
repeated when long; a figure chunk is its caption; the heading path is prepended for embedding only.
Every chunk keeps document, page, element ids and the union bounding box.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Element:
    id: str
    type: str                    # section_header | title | text | paragraph | list_item | table | picture | caption | ...
    text: str = ""
    markdown: str | None = None  # tables
    caption: str | None = None   # tables / pictures
    bbox: list[float] | None = None   # normalised [x1,y1,x2,y2]
    heading_path: list[str] = field(default_factory=list)
    level: int | None = None


@dataclass
class Chunk:
    id: str
    document_id: str
    page_no: int
    kind: str
    text: str
    embed_text: str
    heading_path: list[str]
    element_ids: list[str]
    bbox: list[float] | None
    n_tokens: int
    part: int = 0


def _union(boxes: list[list[float] | None]) -> list[float] | None:
    bs = [b for b in boxes if b]
    if not bs:
        return None
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def _split_table(markdown: str, rows_per_part: int) -> list[str]:
    lines = [l for l in markdown.splitlines() if l.strip()]
    if len(lines) <= 2 + rows_per_part:
        return [markdown]
    header, body = lines[:2], lines[2:]
    return ["\n".join(header + body[i:i + rows_per_part]) for i in range(0, len(body), rows_per_part)]


def chunk_page(document_id: str, doc_title: str, page_no: int, elements: list[Element],
               chunk_words: int = 300, table_rows_per_part: int = 25) -> list[Chunk]:
    out: list[dict] = []
    buf: list[str] = []; ids: list[str] = []; boxes: list[list[float] | None] = []; hp: list[str] = []; carry = ""

    def flush() -> None:
        nonlocal buf, ids, boxes, carry
        if buf:
            text = " ".join(buf).strip()
            out.append({"kind": "text", "text": text, "element_ids": ids, "bbox": _union(boxes), "heading_path": hp})
            sents = _SENT.split(text)
            carry = sents[-1] if len(sents) > 1 and len(sents[-1].split()) < 60 else ""   # one-sentence overlap
        buf, ids, boxes = [], [], []

    for el in elements:
        t = el.type
        if t == "table" and el.markdown:
            flush()
            lead = (el.caption or "").strip()
            for i, part in enumerate(_split_table(el.markdown, table_rows_per_part)):
                out.append({"kind": "table", "text": (lead + "\n" if lead else "") + part, "element_ids": [el.id], "bbox": el.bbox, "heading_path": el.heading_path, "part": i})
        elif t == "picture":
            flush()
            if (el.caption or "").strip():
                out.append({"kind": "figure", "text": el.caption.strip(), "element_ids": [el.id], "bbox": el.bbox, "heading_path": el.heading_path})
        else:
            text = (el.text or "").strip()
            if not text:
                continue
            if t in ("section_header", "title"):
                flush(); hp = el.heading_path + [text]
            words = len(text.split())
            if buf and sum(len(b.split()) for b in buf) + words > chunk_words:
                flush()
            if not buf:
                hp = el.heading_path if t not in ("section_header", "title") else hp
                if carry:
                    buf.append(carry); carry = ""
            buf.append(text); ids.append(el.id); boxes.append(el.bbox)
    flush()

    chunks: list[Chunk] = []
    for i, c in enumerate(out):
        path = c["heading_path"]
        embed = doc_title + (" › " + " › ".join(path) if path else "") + "\n" + c["text"]
        chunks.append(Chunk(id=f"{document_id}:{page_no}:c{i:02d}", document_id=document_id, page_no=page_no, kind=c["kind"], text=c["text"],
                            embed_text=embed, heading_path=list(path), element_ids=list(c["element_ids"]), bbox=c["bbox"],
                            n_tokens=int(len(c["text"].split()) * 1.3), part=c.get("part", 0)))
    return chunks


def chunk_markdown(document_id: str, doc_title: str, page_no: int, markdown: str, chunk_words: int = 300, table_rows_per_part: int = 25) -> list[Chunk]:
    """Chunk shipped OCR markdown (ViDoRe V3) or a Markdown upload: headings and tables are recognised from the text."""
    head = re.compile(r"^(#+\s+.+|\d+(\.\d+)*\s+[A-Z][^|]{2,60})$")
    elements: list[Element] = []
    table: list[str] = []
    hp: list[str] = []          # current heading stack, parents first
    n = 0

    def level_of(s: str) -> int:
        if s.startswith("#"):
            return len(s) - len(s.lstrip("#"))
        m = re.match(r"^(\d+(?:\.\d+)*)\s", s)
        return m.group(1).count(".") + 1 if m else 1

    def flush_table() -> None:
        nonlocal table, n
        if len(table) >= 2:
            elements.append(Element(id=f"{page_no}-e{n:03d}", type="table", markdown="\n".join(table), heading_path=list(hp))); n += 1
        table = []

    for line in markdown.splitlines():
        s = line.strip()
        if "|" in s and s.count("|") >= 2:
            table.append(s); continue
        if table:
            flush_table()
        if not s:
            continue
        if head.match(s) and len(s.split()) <= 10:
            title = s.lstrip("# ").strip()
            parents = hp[: max(0, level_of(s) - 1)]
            elements.append(Element(id=f"{page_no}-e{n:03d}", type="section_header", text=title, heading_path=list(parents), level=level_of(s))); n += 1
            hp = parents + [title]
            continue
        elements.append(Element(id=f"{page_no}-e{n:03d}", type="text", text=s, heading_path=list(hp))); n += 1
    flush_table()
    return chunk_page(document_id, doc_title, page_no, elements, chunk_words, table_rows_per_part)
