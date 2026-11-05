"""Runtime secrets: apply new values immediately (no restart) and persist them to .env when the file is writable
(local runs). Hosted instances keep the override in memory and tell the user to set the platform secret."""
from __future__ import annotations

import os
import re
from pathlib import Path

from pipeline.config import settings

ENV_PATH = Path(os.environ.get("MATT_ENV_FILE", ".env"))
SERVICES: dict[str, list[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "qdrant": ["QDRANT_URL", "QDRANT_API_KEY"],
    "supabase": ["SUPABASE_URL", "SUPABASE_SERVICE_KEY"],
    "langfuse": ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"],
    "huggingface": ["HF_TOKEN"],
    "openai": ["OPENAI_API_KEY"],
}
ALLOWED = {k for ks in SERVICES.values() for k in ks}


def env_writable() -> bool:
    if os.environ.get("MODAL_TASK_ID"):          # on Modal, keys come from the Modal secret; nothing to write
        return False
    try:
        if ENV_PATH.exists():
            return os.access(ENV_PATH, os.W_OK)
        return os.access(ENV_PATH.parent, os.W_OK)
    except OSError:
        return False


def _write_env(values: dict[str, str]) -> None:
    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        m = re.match(r"^\s*([A-Z0-9_]+)\s*=", line)
        if m and m.group(1) in values:
            out.append(f"{m.group(1)}={values[m.group(1)]}")
            seen.add(m.group(1))
        else:
            out.append(line)
    for k, v in values.items():
        if k not in seen:
            out.append(f"{k}={v}")
    ENV_PATH.write_text("\n".join(out).rstrip("\n") + "\n")
    try:
        os.chmod(ENV_PATH, 0o600)
    except OSError:
        pass


def _reset_clients() -> None:
    """Drop cached service clients so the next call uses the new values."""
    from pipeline import stores, tracing
    settings.cache_clear()
    stores.qdrant.cache_clear()
    stores.supabase.cache_clear()
    tracing.langfuse.cache_clear()


def apply(values: dict[str, str]) -> dict:
    """Validates names, applies to the running process, persists to .env when possible. Returns what happened."""
    clean = {k: v.strip() for k, v in values.items() if k in ALLOWED and v is not None and v.strip()}
    if not clean:
        raise ValueError("no recognised keys")
    for k, v in clean.items():
        os.environ[k] = v
    _reset_clients()
    persisted = False
    if env_writable():
        _write_env(clean)
        persisted = True
    return {"applied": sorted(clean), "persisted": persisted, "env_file": str(ENV_PATH.resolve()) if persisted else None}
