"""Run offline LLM-as-a-judge on saved session exports.

This script is intentionally separate from live chat collection. It reads the
saved turn records in backend/session_exports/, computes per-turn metrics, and
rewrites each export JSON in place with:

- metrics_history
- aggregated_metrics
- per-turn evaluation_status / metrics payloads
- top-level evaluation metadata

For schema 4.0 exports, evaluation uses the saved retrieval snapshot already
stored in each turn record. Older exports may lack that snapshot; in that case
the script can rebuild retrieval from the saved user message as a compatibility
fallback and then persist the snapshot for future reruns.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from conversation_manager import get_collection_config
import evaluator
from rag_engine import init_rag, query_rag
from sessions import EXPORT_DIR, METRICS_SCHEMA_VERSION


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate saved session export JSON files in place."
    )
    parser.add_argument(
        "--dir",
        default=EXPORT_DIR,
        help="Directory containing session export JSON files.",
    )
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        help="Specific export filename to evaluate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="Only evaluate files whose top-level evaluation.status is not complete.",
    )
    parser.add_argument(
        "--no-rebuild-missing-snapshots",
        action="store_true",
        help="Fail instead of rebuilding retrieval for older exports missing saved snapshots.",
    )
    return parser


def _export_paths(export_dir: Path, selected_files: list[str] | None) -> list[Path]:
    if selected_files:
        paths = [export_dir / name for name in selected_files]
        missing = [path.name for path in paths if not path.exists()]
        if missing:
            raise SystemExit(f"Missing export file(s): {', '.join(missing)}")
        return paths
    return sorted(export_dir.glob("*.json"))


def _should_evaluate(payload: dict, pending_only: bool) -> bool:
    if not pending_only:
        return True
    status = (
        payload.get("evaluation", {}).get("status")
        if isinstance(payload.get("evaluation"), dict)
        else None
    )
    return status != "complete"


def _serialize_retrieval_items(items: list[dict]) -> list[dict]:
    return [
        {
            "file_name": item.get("file_name"),
            "page_label": item.get("page_label"),
            "score": item.get("score"),
            "text": item.get("text", ""),
        }
        for item in items
    ]


def _attach_retrieval_snapshot(turn_record: dict, rag_result: dict) -> None:
    """Persist a self-contained retrieval snapshot for future judge reruns."""
    retrieval = turn_record.setdefault("retrieval", {})
    sources = _serialize_retrieval_items(rag_result.get("sources", []))
    retrieval.update(
        {
            "query": rag_result.get("query", turn_record.get("user_message", "")),
            "candidate_count": len(rag_result.get("candidates", [])),
            "accepted_source_count": len(rag_result.get("sources", [])),
            "source_file_names": [source.get("file_name") for source in rag_result.get("sources", [])],
            "sources": sources,
            "evaluation_snapshot": {
                "context": rag_result.get("context", ""),
                "candidates": _serialize_retrieval_items(
                    rag_result.get("candidates", [])
                ),
                "relevance": [bool(flag) for flag in rag_result.get("relevance", [])],
            },
        }
    )


def _rag_result_from_snapshot(turn_record: dict) -> dict | None:
    retrieval = turn_record.get("retrieval", {})
    snapshot = retrieval.get("evaluation_snapshot", {})
    candidates = snapshot.get("candidates")
    relevance = snapshot.get("relevance")
    context = snapshot.get("context")
    if not isinstance(candidates, list) or not isinstance(relevance, list):
        return None
    if context is None:
        return None
    return {
        "query": retrieval.get("query", turn_record.get("user_message", "")),
        "context": context,
        "sources": _serialize_retrieval_items(retrieval.get("sources", [])),
        "candidates": _serialize_retrieval_items(candidates),
        "relevance": [bool(flag) for flag in relevance],
    }


def _evaluate_turn_record(
    turn_record: dict,
    retriever,
    *,
    allow_rebuild_missing_snapshots: bool,
) -> tuple[dict, str]:
    """Return one metric payload plus the artifact source used."""
    rag_result = _rag_result_from_snapshot(turn_record)
    artifact_source = "saved_turn_snapshot"

    if rag_result is None:
        if not allow_rebuild_missing_snapshots:
            round_id = turn_record.get("round_id", "unknown_round")
            raise RuntimeError(
                f"Missing saved retrieval snapshot for {round_id}. "
                "Re-run without --no-rebuild-missing-snapshots to backfill it."
            )
        artifact_source = "reconstructed_retrieval"
        rag_result = query_rag(retriever, turn_record.get("user_message", ""))
        rag_result["query"] = turn_record.get("user_message", "")
        _attach_retrieval_snapshot(turn_record, rag_result)

    metrics = evaluator._evaluate_turn(
        turn_record.get("user_message", ""),
        turn_record.get("assistant_response", "") or "",
        rag_result,
        {
            "turn_index": turn_record.get("turn_index"),
            "round_id": turn_record.get("round_id"),
            "user_timestamp": turn_record.get("user_timestamp"),
            "assistant_timestamp": turn_record.get("assistant_timestamp"),
        },
    )
    return metrics, artifact_source


def _artifact_source_label(artifact_sources: list[str]) -> str | None:
    if not artifact_sources:
        return None
    unique = sorted(set(artifact_sources))
    if len(unique) == 1:
        return unique[0]
    return "mixed"


def _evaluate_payload(
    payload: dict,
    retriever,
    *,
    allow_rebuild_missing_snapshots: bool,
) -> tuple[dict, int]:
    turn_records = sorted(
        payload.get("turn_records", []),
        key=lambda record: record.get("turn_index", 0),
    )

    metrics_history = []
    artifact_sources = []
    for turn_record in turn_records:
        metrics, artifact_source = _evaluate_turn_record(
            turn_record,
            retriever,
            allow_rebuild_missing_snapshots=allow_rebuild_missing_snapshots,
        )
        artifact_sources.append(artifact_source)
        turn_record["metrics"] = metrics
        turn_record["evaluation_status"] = "complete"
        turn_record["evaluation_completed_at"] = metrics.get("turn", {}).get(
            "evaluated_at"
        )
        turn_record["evaluation_failed_reason"] = None
        metrics_history.append(metrics)

    evaluated_at = _now_utc_iso()
    payload["metrics_schema_version"] = METRICS_SCHEMA_VERSION
    if not isinstance(payload.get("collection_config"), dict) or not payload.get(
        "collection_config"
    ):
        payload["collection_config"] = get_collection_config()
    payload["metrics_history"] = metrics_history
    payload["aggregated_metrics"] = evaluator.aggregate(metrics_history)
    payload["evaluation"] = {
        "status": "complete",
        "evaluated_at": evaluated_at,
        "config": evaluator.get_evaluation_config(),
        "turns_evaluated": len(metrics_history),
        "artifact_source": _artifact_source_label(artifact_sources),
    }
    # Keep the legacy top-level field for existing readers.
    payload["evaluation_config"] = payload["evaluation"]["config"]
    return payload, len(metrics_history)


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    parser = _build_parser()
    args = parser.parse_args()

    export_dir = Path(args.dir)
    paths = _export_paths(export_dir, args.files)
    selected_paths = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if _should_evaluate(payload, args.pending_only):
            selected_paths.append(path)

    print(f"Found {len(paths)} export file(s).")
    print(f"Selected {len(selected_paths)} file(s) for evaluation.")
    if not selected_paths:
        return

    retriever = None
    allow_rebuild = not args.no_rebuild_missing_snapshots
    total_turns = 0
    for index, path in enumerate(selected_paths, start=1):
        payload = json.loads(path.read_text(encoding="utf-8"))
        needs_rebuild = any(
            _rag_result_from_snapshot(turn_record) is None
            for turn_record in payload.get("turn_records", [])
        )
        if needs_rebuild and retriever is None:
            retriever = init_rag()

        payload, turn_count = _evaluate_payload(
            payload,
            retriever,
            allow_rebuild_missing_snapshots=allow_rebuild,
        )
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        total_turns += turn_count
        artifact_source = payload.get("evaluation", {}).get("artifact_source")
        print(
            f"[{index}/{len(selected_paths)}] Evaluated {path.name} "
            f"({turn_count} turns, source={artifact_source})"
        )

    print(f"Done. Evaluated {total_turns} turn metric record(s).")


if __name__ == "__main__":
    main()
