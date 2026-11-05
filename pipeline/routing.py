"""Step 1: detect what a file is from its bytes, not its extension."""
from __future__ import annotations

from dataclasses import dataclass

import filetype

# Office files are zip containers; filetype reports them by the inner marker.
_OFFICE = {"application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
           "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx"}


@dataclass(frozen=True)
class FileKind:
    kind: str            # pdf | docx | pptx | image | markdown | text | unsupported
    mime: str
    visual: bool         # can produce page images for the ColQwen path without extra tooling


def detect(name: str, data: bytes) -> FileKind:
    guess = filetype.guess(data)
    mime = guess.mime if guess else ""
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if mime == "application/pdf" or ext == "pdf":
        return FileKind("pdf", "application/pdf", True)
    if mime in _OFFICE or ext in ("docx", "pptx"):
        k = _OFFICE.get(mime, ext)
        return FileKind(k, mime or f"application/{k}", False)   # rendered to pages only when LibreOffice is present
    if (mime and mime.startswith("image/")) or ext in ("png", "jpg", "jpeg", "webp", "tiff"):
        return FileKind("image", mime or "image/*", True)
    if ext in ("md", "markdown"):
        return FileKind("markdown", "text/markdown", False)
    if ext in ("txt",) or (not mime and _looks_text(data)):
        return FileKind("text", "text/plain", False)
    return FileKind("unsupported", mime or "application/octet-stream", False)


def _looks_text(data: bytes) -> bool:
    sample = data[:4096]
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return b"\x00" not in sample
