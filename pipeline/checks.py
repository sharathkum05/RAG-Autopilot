"""Deterministic answer checks against the benchmark's reference answers and page labels. No model in the loop.

Each check is a pure function of (generated answer, cited pages, pages shown to the model, reference answers, labelled
pages, query types) and returns a score in [0, 1], or None when it does not apply to the query. The query score is the
weighted mean of the applicable checks; an answer that declines while labelled pages exist scores 0.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

WEIGHTS: dict[str, float] = {"grounded": 0.30, "valid_citations": 0.10, "anchors": 0.30, "content": 0.15, "type": 0.15}
PASS_THRESHOLD = 0.6
DESCRIPTIONS: dict[str, str] = {
    "grounded": "at least one cited page is a page the benchmark labels as relevant",
    "valid_citations": "every cited page was among the pages shown to the model (no invented citations)",
    "anchors": "share of the reference answer's anchors (code identifiers, quoted literals, numbers, named terms) present in the answer",
    "content": "share of the reference answer's content words present in the answer, best over the reference variants",
    "type": "boolean questions: yes/no polarity matches the reference; numerical questions: the reference's numbers appear",
    "refusal": "gate: an answer that declines while labelled pages exist scores 0",
    "rouge_l": "ROUGE-L F1 against the best reference variant (reported, not weighted)",
}

STOP = set("""a an the of to in on for and or is are was were be been being it its this that these those as at by with from
which who whom whose what when where why how do does did can could should would may might must will shall not no yes if
then than so such into onto over under about between among through during before after above below up down out off
again further once here there all any both each few more most other some own same too very just also used use using
one two like e g i you we they he she your our their""".split())
COMMON_TERMS = {"python"}                      # capitalised in every reference of this corpus; carries no signal

TOKEN = re.compile(r"[a-z0-9_']+")
CODE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*\([^()\n]{0,40}\)|`[^`\n]{1,60}`|'[^'\n]{1,60}'|\"[^\"\n]{1,60}\"|\b\w+_\w+\b|\b[a-z]+\.[a-z_]+\b")
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
CAPS = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-zA-Z]{2,}\b")
MARKDOWN = re.compile(r"\*\*|`|\s\[\d+\]")
REFUSAL = re.compile(r"(couldn't|could not|can't|cannot|unable to|did not|didn't|don't|do not) (find|locate|see)|not (mentioned|covered|contained|present|found|addressed|included) in (the|these|this|any)|no information"
                     r"|(does|do|doesn't|don't) not? ?(mention|contain|cover|address|discuss|say|answer|define|compare|provide|give|list|explain)\b"
                     r"|(pages|documents?|text|material|excerpts?) (don't|do not|doesn't|does not) (say|answer|define|compare|mention|contain|cover|include|discuss|provide|give|list|explain|show)"
                     r"|not in the (indexed |provided |given )?(document|pages|text|material)|contain no (complete )?(code )?examples?", re.I)
NEGATION = re.compile(r"\b(not|no|cannot|never|neither|nor)\b|n't\b", re.I)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", MARKDOWN.sub(" ", text or "").replace("’", "'")).strip()


def stem(w: str) -> str:
    if len(w) <= 4:
        return w
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + rep
    return w


def tokens(text: str) -> list[str]:
    return TOKEN.findall(clean(text).lower())


def content_words(text: str) -> set[str]:
    return {stem(t.strip("'")) for t in tokens(text) if t.strip("'") not in STOP and len(t.strip("'")) > 2 and not t.isdigit()}


def anchors(reference: str) -> list[str]:
    """Canonical anchor keys from a reference answer: `name(` for calls, bare literals, dotted names, numbers, named terms."""
    keys: list[str] = []
    for m in CODE.findall(reference):
        k = m.strip("`'\"").strip()
        if "(" in k:
            k = k.split("(")[0].strip() + "("
        k = k.lower()
        if k and k.rstrip("(") not in STOP and len(k.rstrip("(")) > 1 and k not in keys:
            keys.append(k)
    for m in NUM.findall(reference):
        if m not in keys:
            keys.append(m)
    for m in CAPS.findall(reference):
        k = m.lower()
        if k not in COMMON_TERMS and k not in STOP and k not in keys:
            keys.append(k)
    return keys


def _whole(key: str, text: str) -> bool:
    """Whole-token match for plain words ('err' must not match 'error'); substring match for code-like keys."""
    if re.fullmatch(r"[a-z0-9]+", key):
        return re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", text) is not None
    return key in text


def has_anchor(answer_lc: str, answer_stems: set[str], key: str) -> bool:
    if _whole(key, answer_lc):
        return True
    if key.endswith("(") and _whole(key[:-1], answer_lc):          # `tail(` also matches "tail method"
        return True
    return stem(key) in answer_stems


def lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(answer: str, reference: str) -> float:
    a, b = tokens(answer), tokens(reference)
    l = lcs(a, b)
    if not l:
        return 0.0
    p, r = l / len(a), l / len(b)
    return 2 * p * r / (p + r)


def polarity(text: str) -> str | None:
    t = clean(text).strip().lower()
    m = re.match(r"^(yes|no)\b", t)
    if m:
        return m.group(1)
    first = re.split(r"(?<=[.!?])\s", t, maxsplit=1)[0]
    if not first:
        return None
    return "no" if NEGATION.search(first) else "yes"


@dataclass
class CheckResult:
    scores: dict[str, float | None]
    score: float
    passed: bool
    refused: bool
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"checks": self.scores, "score": round(self.score, 4), "passed": self.passed, "refused": self.refused, "detail": self.detail}


def run_checks(answer: str, cited_pages: list[int], shown_pages: list[int], relevant_pages: list[int], references: list[str], query_types: list[str]) -> CheckResult:
    refs = [r for r in references if r and r.strip()] or [""]
    ans_lc = clean(answer).lower()
    ans_stems = content_words(answer)
    rel, shown = set(relevant_pages), set(shown_pages)
    scores: dict[str, float | None] = {}
    detail: dict = {}

    scores["grounded"] = 1.0 if any(p in rel for p in cited_pages) else 0.0
    scores["valid_citations"] = (1.0 if all(p in shown for p in cited_pages) else 0.0) if cited_pages else 0.0
    detail["cited"], detail["labelled_cited"] = cited_pages, sorted(p for p in cited_pages if p in rel)

    # anchors: best coverage over the reference variants
    best, best_detail = None, {}
    for ref in refs:
        keys = anchors(ref)
        if not keys:
            continue
        found = [k for k in keys if has_anchor(ans_lc, ans_stems, k)]
        cov = len(found) / len(keys)
        if best is None or cov > best:
            best, best_detail = cov, {"anchors": keys, "found": found, "missing": [k for k in keys if k not in found]}
    scores["anchors"] = best
    detail.update(best_detail)

    # content words: best recall over the reference variants
    recalls = []
    for ref in refs:
        cw = content_words(ref)
        if cw:
            recalls.append(len(cw & ans_stems) / len(cw))
    scores["content"] = max(recalls) if recalls else None

    # type-specific
    qt = set(query_types or [])
    tscore: float | None = None
    if "boolean" in qt:
        pr, pa = polarity(refs[0]), polarity(answer)
        tscore = 1.0 if (pr and pa and pr == pa) else 0.0
        detail["polarity"] = {"reference": pr, "answer": pa}
    elif "numerical" in qt:
        nums = [n for n in NUM.findall(refs[0])]
        if nums:
            found = [n for n in nums if n in ans_lc]
            tscore = len(found) / len(nums)
            detail["numbers"] = {"reference": nums, "found": found}
    scores["type"] = tscore

    scores["rouge_l"] = max(rouge_l(answer, r) for r in refs)

    # a decline is a short answer that matches, or any answer whose opening sentence declines ("These pages don't say…")
    opening = re.split(r"(?<=[.!?])\s", clean(answer), maxsplit=1)[0]
    # ...but an answer that cites a labelled page is an answer, however much it hedges; the other checks score it
    refused = (bool(REFUSAL.search(opening)) or (bool(REFUSAL.search(clean(answer))) and len(tokens(answer)) < 80)) and not scores["grounded"]
    detail["refused"] = refused
    if refused and rel:
        return CheckResult(scores, 0.0, False, True, detail)
    applicable = {k: v for k, v in scores.items() if k in WEIGHTS and v is not None}
    wsum = sum(WEIGHTS[k] for k in applicable)
    score = sum(WEIGHTS[k] * v for k, v in applicable.items()) / wsum if wsum else 0.0
    return CheckResult(scores, score, score >= PASS_THRESHOLD, False, detail)


def rubric() -> dict:
    return {"weights": WEIGHTS, "threshold": PASS_THRESHOLD, "descriptions": DESCRIPTIONS}
