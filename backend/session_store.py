"""Persists a finished session's conversation, report, and evaluation
metrics to disk as JSON.

Session data otherwise lives only in ConversationManager.sessions (an
in-memory dict) and is lost on process restart. This module writes a
snapshot when a participant downloads their report - the point in this
app's study protocol where a conversation is considered finished - so
results can be re-collected later even after the backend restarts.
"""

import glob
import json
import os
from datetime import datetime, timezone

EXPORT_DIR = os.path.join(os.path.dirname(__file__), "session_exports")


def save_session(session) -> str:
    """Writes a JSON snapshot of the session to EXPORT_DIR. Returns the file path.

    Call session.wait_for_pending_metrics() first (the download route does)
    so the final turn's background evaluation isn't still in flight when
    this snapshot is taken.
    """
    os.makedirs(EXPORT_DIR, exist_ok=True)
    now_utc = datetime.now(timezone.utc)
    payload = {
        "session_id": session.id,
        "saved_at": now_utc.isoformat(),
        "conversation_history": session.conversation_history,
        "report_data": session.get_report_summary(),
        "completion": session.get_completion_ratio(),
        "metrics_history": session.metrics_history,
        "aggregated_metrics": session.get_aggregated_metrics(),
    }
    # Local time in the filename so files sort/read naturally for whoever is
    # running the study; saved_at above stays UTC for unambiguous analysis.
    timestamp = now_utc.astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    short_id = session.id[:8]  # matches report_generator's Report ID / download filename convention
    filename = f"{timestamp}_{short_id}.json"
    path = os.path.join(EXPORT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_all_sessions(export_dir: str = EXPORT_DIR) -> list[dict]:
    """Reads every exported session JSON from export_dir, sorted by filename
    (which sorts chronologically, since filenames start with the timestamp).
    """
    sessions = []
    for path in sorted(glob.glob(os.path.join(export_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            sessions.append(json.load(f))
    return sessions
