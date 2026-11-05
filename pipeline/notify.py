"""Phone alerts when someone uses the demo: a push to an ntfy topic (https://ntfy.sh) on page visits, questions and
uploads. Off unless NTFY_TOPIC is set. Fire-and-forget on a daemon thread, so a slow or failed push never delays or
breaks a request; visits are throttled so one person browsing sends one alert, not one per page."""
from __future__ import annotations

import logging
import threading
import time
import urllib.request

from pipeline.config import settings

log = logging.getLogger("matt")
DASHBOARD = "https://rag-autopilot.vercel.app"
VISIT_EVERY_S = 30 * 60
_last_visit = 0.0
_lock = threading.Lock()


def notify(title: str, message: str, tags: str = "", click: str = DASHBOARD) -> None:
    topic = settings().ntfy_topic
    if not topic:
        return

    def send() -> None:
        try:
            req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=message.encode("utf-8"), method="POST",
                                         headers={"Title": title.encode("utf-8").decode("latin-1", "ignore"), "Tags": tags, "Click": click})
            urllib.request.urlopen(req, timeout=8).read()
        except Exception as e:  # noqa: BLE001
            log.warning("ntfy push failed: %s", e)

    threading.Thread(target=send, daemon=True).start()


def visit() -> None:
    """Someone opened a dashboard page (the sidebar asks for /api/system on every load). At most one alert per 30 min."""
    global _last_visit
    with _lock:
        now = time.time()
        if now - _last_visit < VISIT_EVERY_S:
            return
        _last_visit = now
    notify("RAG demo opened", "Someone just opened the dashboard.", tags="eyes")
