"""Model-graded second opinion next to the deterministic checks: a frozen four-dimension rubric (relevance, groundedness,
correctness, coherence) on a fixed OpenAI model with JSON output, so a rerun grades the same answer the same way.
Off when OPENAI_API_KEY is not set."""
from __future__ import annotations

import json
import time

from pipeline.config import settings

DIMENSIONS = ("relevance", "groundedness", "correctness", "coherence")
DIMENSION_INFO = {
    "relevance": "5 answers exactly the question asked · 3 a related or broader question · 1 off-topic or declines",
    "groundedness": "5 every claim is supported by the page excerpts shown to the model · 3 some claims go beyond them · 1 mostly unsupported or invented",
    "correctness": "5 same facts as the reference answer, nothing contradicts it · 3 partly right, a key fact missing or hedged · 1 wrong, contradicts it, or declines",
    "coherence": "5 clear, well organised, no contradictions · 3 understandable but muddled or repetitive · 1 incoherent",
}
PRICES = {"gpt-5-mini": (0.25, 2.0), "gpt-5": (1.25, 10.0), "gpt-5-nano": (0.05, 0.40)}   # $ per 1M tokens (input, output)


def price_of(model: str | None) -> tuple[float, float]:
    m = model or ""
    return next((p for name, p in sorted(PRICES.items(), key=lambda kv: -len(kv[0])) if m.startswith(name)), PRICES["gpt-5-mini"])


def rubric() -> dict:
    """What the judge sees and how it scores, for the Answer Evals tab."""
    return {"model": settings().judge_model, "scale": "integer 1 to 5 per dimension; overall = mean of the four",
            "sees": "the question, the benchmark reference answer, the page excerpts shown to the answering model, the answer",
            "dimensions": DIMENSION_INFO, "frozen": "fixed model, fixed prompt, JSON output: the same answer grades the same way on a rerun",
            "price_per_1m_tokens": dict(zip(("input", "output"), price_of(settings().judge_model)))}

RUBRIC = """You grade one answer from a document question-answering system. You get the question, the benchmark's reference
answer, the page excerpts the system showed its model, and the system's answer. Cited page markers like [1] are fine.
Score each dimension with an integer from 1 to 5:

relevance     5 answers exactly the question asked · 3 answers a related or broader question · 1 off-topic or declines
groundedness  5 every claim is supported by the excerpts · 3 some claims go beyond the excerpts · 1 mostly unsupported or invented
correctness   5 states the same facts as the reference, nothing contradicts it · 3 partly right, a key fact missing or hedged · 1 wrong, contradicts the reference, or declines
coherence     5 clear, well organised, no contradictions · 3 understandable but muddled or repetitive · 1 incoherent

Judge correctness only by factual agreement with the reference; wording, length and style do not matter.
If no excerpts are given, score groundedness by agreement with the reference instead.
Respond with JSON only: {"relevance": n, "groundedness": n, "correctness": n, "coherence": n, "reason": "<one sentence>"}"""


def _score(v) -> int:
    try:
        return min(5, max(1, int(round(float(v)))))
    except (TypeError, ValueError):
        return 1


def judge_answer(question: str, reference: str, answer: str, context: str = "", model: str | None = None, api_key: str | None = None) -> dict | None:
    """Four 1-5 scores and their mean (`overall`). `correctness` is also kept on the old 0-1 scale so earlier runs and the
    agreement rule (the judge agrees with the checks when correctness >= 0.5, i.e. 3 of 5) keep working."""
    s = settings()
    key = api_key or s.openai_api_key
    if not key:
        return None
    from openai import OpenAI

    model = model or s.judge_model
    t0 = time.perf_counter()
    payload = {"question": question, "reference_answer": reference, "excerpts_shown_to_the_model": context or "(none)", "answer": answer}
    resp = OpenAI(api_key=key).chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": RUBRIC}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    scores = {d: _score(data.get(d)) for d in DIMENSIONS}
    overall = round(sum(scores.values()) / len(scores), 2)
    u = getattr(resp, "usage", None)
    return {"model": getattr(resp, "model", model), "scores": scores, "overall": overall, "correctness": (scores["correctness"] - 1) / 4,
            "reason": str(data.get("reason", ""))[:300] if data else "judge returned no JSON",
            "tokens_in": int(getattr(u, "prompt_tokens", 0) or 0), "tokens_out": int(getattr(u, "completion_tokens", 0) or 0),
            "ms": int((time.perf_counter() - t0) * 1000)}
