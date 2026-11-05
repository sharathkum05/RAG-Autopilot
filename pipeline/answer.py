"""Step 9: Claude answers only from the evidence pages, with native citations, streamed. Each page is sent as a
citable text document plus its image, so the model can read charts and tables while citations stay structured."""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Iterator

import anthropic
import httpx

from pipeline.config import settings
from pipeline.retrieval import Evidence
from pipeline.tracing import child, end

SYSTEM = (
    "You answer questions about documents the user uploaded. Each document block is one page of evidence, with its page image "
    "alongside. Answer only from these pages and cite the page for every claim. If the pages do not contain the answer, say so "
    "plainly instead of guessing. Keep answers concise."
)

# USD per million tokens (input, output). Used for the per-turn cost shown in the UI; verify against current pricing.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50), "claude-opus-4-8": (5.0, 25.0)}
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}


@dataclass
class AnswerResult:
    text: str = ""
    cited: list[int] = field(default_factory=list)       # evidence indices (0-based) that were cited
    stop_reason: str | None = None
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    ttr_ms: int = 0
    note: str | None = None


def _image_block(url: str, max_bytes: int = 4_000_000) -> dict | None:
    try:
        r = httpx.get(url, timeout=20)
        r.raise_for_status()
        if len(r.content) > max_bytes:
            return None
        return {"type": "image", "source": {"type": "base64", "media_type": r.headers.get("content-type", "image/jpeg").split(";")[0], "data": base64.b64encode(r.content).decode()}}
    except Exception:
        return None


def build_messages(question: str, evidence: list[Evidence], with_images: bool = True) -> list[dict]:
    content: list[dict] = []
    for e in evidence:
        content.append({"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": (e.page_text or "(no text extracted)")[:20000]},
                        "title": f"{e.doc_name} p.{e.page_no}", "context": f"Page {e.page_no} of {e.doc_name}.", "citations": {"enabled": True}})
        if with_images and e.image_url:
            img = _image_block(e.image_url)
            if img:
                content.append(img)
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def stream_answer(question: str, evidence: list[Evidence], model: str | None = None, api_key: str | None = None, n_pages: int = 3, trace=None) -> Iterator[tuple[str, str | AnswerResult]]:
    """Yields ("token", text) as the answer streams, with " [n]" markers appended after cited passages, then ("final", AnswerResult)."""
    s = settings()
    model = model or s.answer_model
    client = anthropic.Anthropic(api_key=api_key or s.anthropic_api_key or None)
    ev = evidence[:n_pages]
    messages = build_messages(question, ev)
    extra: dict = {}
    if model in FALLBACK_MODELS:
        extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    t0 = time.perf_counter()
    result = AnswerResult(model=model)
    cited: list[int] = []
    block_cites: list[int] = []
    text_parts: list[str] = []
    obs = child(trace, "Answering", as_type="generation", model=model, input={"question": question, "pages": [f"{e.doc_name} p.{e.page_no}" for e in ev]})
    try:
        with client.beta.messages.stream(model=model, max_tokens=4096, system=SYSTEM, messages=messages, **extra) as stream:
            for event in stream:
                et = event.type
                if et == "content_block_start":
                    block_cites = []
                elif et == "content_block_delta":
                    d = event.delta
                    if d.type == "text_delta":
                        text_parts.append(d.text)
                        yield "token", d.text
                    elif d.type == "citations_delta":
                        di = getattr(d.citation, "document_index", None)
                        if di is not None and di < len(ev):
                            if di not in cited:
                                cited.append(di)
                            n = cited.index(di) + 1
                            if n not in block_cites:
                                block_cites.append(n)
                elif et == "content_block_stop" and block_cites:
                    marker = "".join(f" [{n}]" for n in block_cites)
                    text_parts.append(marker)
                    yield "token", marker
                    block_cites = []
            final = stream.get_final_message()
        result.text = "".join(text_parts).strip()
        result.cited = cited
        result.stop_reason = final.stop_reason
        result.model = final.model
        u = final.usage
        result.tokens_in = int(getattr(u, "input_tokens", 0) or 0) + int(getattr(u, "cache_read_input_tokens", 0) or 0)
        result.tokens_out = int(getattr(u, "output_tokens", 0) or 0)
        pin, pout = PRICES.get(final.model, PRICES.get(model, (4.0, 20.0)))
        result.cost_usd = round(result.tokens_in / 1e6 * pin + result.tokens_out / 1e6 * pout, 5)
        result.ttr_ms = int((time.perf_counter() - t0) * 1000)
        if final.stop_reason == "refusal":
            det = getattr(final, "stop_details", None)
            result.note = f"The model declined this request ({getattr(det, 'explanation', None) or getattr(det, 'category', None) or 'no reason given'})."
        elif final.stop_reason == "max_tokens":
            result.note = "The answer was cut off at the token limit."
    finally:
        end(obs, output=result.text[:2000] if result.text else None, usage_details={"input": result.tokens_in, "output": result.tokens_out}, cost_details={"total": result.cost_usd}, metadata={"stop_reason": result.stop_reason, "cited": cited})
    yield "final", result
