"""Backtracking for one answered turn: why its confidence came out as it did, which chunks it used, and what looks wrong.

Works from what the turn row stores. Turns from Nov 5 on store every retrieved chunk with a `cited` flag (and its final
score); older turns stored only the cited chunks, so their diagnosis says what it cannot see. The confidence rule mirrored
here is the one in server/app.py: high = Claude cited a page AND the #1 page was found by both image and text search or
led #2 by more than 0.05; medium = cited, but neither; low = nothing cited.
"""
from __future__ import annotations

import re

MARGIN = 0.05            # the margin in the confidence rule
LOW_RERANK = 0.2         # bge-reranker-v2-m3 relevance below this: the reranker barely thinks the page answers it
SLOW_ENCODE_MS = 800     # question encoding normally takes 20-70 ms; far above that means a cold or busy GPU

TEXT_ONLY = {"bm25", "text_hybrid", "text_hybrid_rerank"}
RERANKED = {"text_hybrid_rerank", "fused_rerank"}
DECLINE = re.compile(r"\b(?:(?:do|does|did)(?: not|n't)|never) (?:mention|cover|say|name|discuss|describe|state|define|include|use)"
                     r"|\bnot (?:mentioned|covered|found|stated|described)|\bno (?:mention|information)", re.I)


def _p(c: dict) -> str:
    return f"p.{c.get('pno') or c.get('page')}"


def diagnose(turn: dict, mode: str | None) -> dict:
    chunks: list[dict] = turn.get("evidence") or []
    full = any("cited" in c for c in chunks)                     # newer rows carry the whole ranked list
    cited = [c for c in chunks if c.get("cited", True)]
    uncited = sorted((c for c in chunks if c.get("cited") is False), key=lambda c: (c.get("scores") or {}).get("fusedRank") or 99)
    ranked = sorted(chunks, key=lambda c: (c.get("scores") or {}).get("fusedRank") or 99) if full else []
    reranked, fused = mode in RERANKED, mode not in TEXT_ONLY and mode is not None
    conf = turn.get("confidence")
    findings: list[dict] = []

    def add(level: str, title: str, detail: str) -> None:
        findings.append({"level": level, "title": title, "detail": detail})

    # ---- why the confidence is what it is
    top = ranked[0] if ranked else next((c for c in cited if (c.get("scores") or {}).get("fusedRank") == 1), None)
    second = ranked[1] if len(ranked) > 1 else None
    ts = (top or {}).get("scores") or {}
    agree = ts.get("visualRank") is not None and ts.get("textRank") is not None
    margin = (ts.get("final") - (second.get("scores") or {}).get("final")) if (top and second and ts.get("final") is not None
                                                                                and (second.get("scores") or {}).get("final") is not None) else None
    if conf == "low" or not cited:
        why = "Low: Claude cited no page. It answered without evidence or said the retrieved pages don't cover the question."
    elif conf == "high":
        why = ("High: Claude cited a page and the #1 page was found by both image and text search." if agree else
               f"High: Claude cited a page and the #1 page led #2 by {margin:.3f}, a clear win." if margin is not None else
               "High: Claude cited a page and the #1 page was clearly ahead (found by both searches or led #2 by more than 0.05).")
    else:
        parts = []
        if top is not None:
            parts.append(f"the #1 page ({_p(top)}) was found only by {'text' if ts.get('textRank') is not None else 'image'} search"
                         if fused else "this config has no image search, so the two searches can never agree")
        else:
            parts.append("the #1 page was not found by both image and text search")
        parts.append(f"it led #2 by only {margin:.3f} (needs more than {MARGIN})" if margin is not None else f"it did not lead #2 by more than {MARGIN}")
        why = "Medium: Claude cited a page, but " + " and ".join(parts) + ". The ranking had no clear winner."

    # ---- the answer itself
    answer = turn.get("answer") or ""
    if not cited:
        add("bad", "No page cited", "Nothing in the answer is tied to a page, so it cannot be checked against the document.")
    if DECLINE.search(answer[:600]):
        add("bad", "Answer says the pages don't cover it",
            "Usually a retrieval miss: the page with the answer was not among the pages Claude was given. "
            "Explain this retrieval shows whether it ranked further down or was never found.")
    if turn.get("status") == "warn":
        add("warn", "Answer was cut off or declined", "The model stopped early (token limit or refusal); the stored answer may be incomplete.")

    # ---- the chunks it used
    for c in cited:
        s = c.get("scores") or {}
        r, rr, vr, tr = s.get("fusedRank"), s.get("rerank"), s.get("visualRank"), s.get("textRank")
        best_search = min(x for x in (vr, tr) if x is not None) if (vr is not None or tr is not None) else None
        if reranked and rr is not None and best_search == 1 and r and r > 1:
            add("warn", f"Reranker demoted {_p(c)}",
                f"{'Image and text search both' if vr == 1 and tr == 1 else 'A search path'} ranked {_p(c)} first, but the reranker scored it "
                f"{rr:.2f} and moved it to #{r}. Claude still cited it, so the page was right and the reranker was wrong.")
        if reranked and rr is not None and rr < LOW_RERANK:
            add("warn", f"Low reranker score on {_p(c)} ({rr:.2f})",
                "The reranker reads only the page text. " + ("This chunk is a table, which comes through as pipe-separated text "
                                                             "the reranker reads poorly, even when it holds the answer." if c.get("kind") == "table"
                                                             else "Low scores mean it found little overlap between the question and the text."))
        if fused and (vr is None) != (tr is None):
            add("info", f"{_p(c)} found by {'text' if tr is not None else 'image'} search only",
                "Only one of the two searches found it, so the ranking leaned on a single signal.")
    if top is not None and top not in cited and cited:
        add("warn", f"#1 page {_p(top)} was not cited",
            f"The top-ranked page did not help; Claude answered from #{', #'.join(str((c.get('scores') or {}).get('fusedRank')) for c in cited)} instead.")
    if not full and cited and all(((c.get("scores") or {}).get("fusedRank") or 1) > 1 for c in cited):
        add("info", "#1 page not stored", "This turn predates full ranked-list storage, so the uncited pages are not shown. "
                                          "Explain this retrieval re-runs every stage for this question.")

    # ---- speed
    enc = (turn.get("timings_ms") or {}).get("Encoding question")
    if enc and enc > SLOW_ENCODE_MS:
        add("info", f"Slow question encoding ({enc / 1000:.1f} s)", "Normally 20 to 70 ms; this much means the GPU container was cold or busy.")

    worst = "bad" if any(f["level"] == "bad" for f in findings) else "warn" if any(f["level"] == "warn" for f in findings) else "ok"
    headline = next((f["title"] for f in findings if f["level"] == worst), None) if worst != "ok" else (
        "Healthy: cited pages ranked at the top and both searches agreed" if conf == "high" else "No problems found in the stored scores")
    return {"verdict": worst, "headline": headline, "confidence_why": why, "findings": findings, "full_ranking": full,
            "uncited": uncited}
