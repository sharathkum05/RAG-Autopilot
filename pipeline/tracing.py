"""Langfuse tracing: one trace per turn, a span per stage, a generation for the model call. No-ops when keys are absent.

Observations are created explicitly from their parent (thread-safe) rather than through the ambient OpenTelemetry
context, because retrieval runs in a worker thread while the HTTP response streams from another.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from pipeline.config import settings


@lru_cache
def langfuse():
    s = settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return None
    from langfuse import Langfuse

    return Langfuse(public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, host=s.langfuse_host)


def start_trace(name: str, **kw: Any):
    """Root observation for a turn. Returns the observation or None."""
    lf = langfuse()
    if lf is None:
        return None
    try:
        return lf.start_observation(name=name, as_type="span", **kw)
    except Exception:
        return None


def child(parent, name: str, as_type: str = "span", **kw: Any):
    """Child observation under `parent`; safe to call from any thread. Returns None when tracing is off."""
    if parent is None:
        return None
    try:
        return parent.start_observation(name=name, as_type=as_type, **kw)
    except Exception:
        return None


def end(obs, **update: Any) -> None:
    if obs is None:
        return
    try:
        if update:
            obs.update(**update)
        obs.end()
    except Exception:
        pass


def trace_id_of(obs) -> str | None:
    try:
        return getattr(obs, "trace_id", None)
    except Exception:
        return None


def flush() -> None:
    lf = langfuse()
    if lf is not None:
        try:
            lf.flush()
        except Exception:
            pass


def langfuse_ping() -> dict:
    lf = langfuse()
    if lf is None:
        raise RuntimeError("LANGFUSE keys are not set")
    return {"authenticated": bool(lf.auth_check())}
