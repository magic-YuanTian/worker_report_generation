"""Persist completed sessions and reload them for later study analysis."""

import glob
import json
import os
from datetime import datetime, timezone

EXPORT_DIR = os.path.join(os.path.dirname(__file__), "session_exports")
METRICS_SCHEMA_VERSION = "4.0"


def save_session(session, collection_config: dict | None = None) -> str:
    """Write one session snapshot to EXPORT_DIR and return the file path."""
    os.makedirs(EXPORT_DIR, exist_ok=True)
    now_utc = datetime.now(timezone.utc)
    metrics_history = list(getattr(session, "metrics_history", []))
    aggregated_metrics = (
        session.get_aggregated_metrics() if hasattr(session, "get_aggregated_metrics") else {}
    )
    evaluation_state = (
        session.get_evaluation_state()
        if hasattr(session, "get_evaluation_state")
        else {
            "status": "complete" if metrics_history else "not_run",
            "evaluated_at": None,
            "config": None,
            "turns_evaluated": len(metrics_history),
            "artifact_source": None,
        }
    )
    payload = {
        "session_id": session.id,
        "saved_at": now_utc.isoformat(),
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "collection_config": collection_config or {},
        "evaluation": evaluation_state,
        "evaluation_config": evaluation_state.get("config"),
        "conversation_history": session.conversation_history,
        "turn_records": session.get_ordered_turn_records(),
        "report_data": session.get_report_summary(),
        "completion": session.get_completion_ratio(),
        "metrics_history": metrics_history,
        "aggregated_metrics": aggregated_metrics,
    }

    timestamp = now_utc.astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    short_id = session.id[:8]
    filename = f"{timestamp}_{short_id}.json"
    path = os.path.join(EXPORT_DIR, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def load_all_sessions(export_dir: str = EXPORT_DIR) -> list[dict]:
    """Read every exported session JSON from export_dir in filename order."""
    sessions = []
    for path in sorted(glob.glob(os.path.join(export_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as handle:
            sessions.append(json.load(handle))
    return sessions
