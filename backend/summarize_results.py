"""Summarizes evaluation metrics across all exported session JSON files.

Run after a study session (or the whole study) is done to get pooled
RAG quality metrics - Precision@K, Faithfulness, Answer Relevance, and
inter-model agreement - across every participant, not just per-session.

Pooling reuses evaluator.aggregate() by concatenating every session's
metrics_history before aggregating, so the math (means, Cohen's kappa)
is identical to what a single session already computes - just over a
combined turn/claim set instead of one session's.

Usage:
    python summarize_results.py
    python summarize_results.py --dir path/to/session_exports
    python summarize_results.py --out study_summary.json
"""

import argparse
import json

import evaluator
from session_store import EXPORT_DIR, load_all_sessions


def summarize(export_dir: str = EXPORT_DIR) -> dict:
    sessions = load_all_sessions(export_dir)

    combined_metrics_history = []
    for s in sessions:
        combined_metrics_history.extend(s.get("metrics_history", []))

    completions = [s["completion"] for s in sessions if s.get("completion") is not None]

    return {
        "n_sessions": len(sessions),
        "n_turns_total": len(combined_metrics_history),
        "mean_report_completion": sum(completions) / len(completions) if completions else None,
        "pooled_metrics": evaluator.aggregate(combined_metrics_history),
        "per_session": [
            {
                "session_id": s.get("session_id"),
                "saved_at": s.get("saved_at"),
                "completion": s.get("completion"),
                "n_turns": len(s.get("metrics_history", [])),
                "aggregated_metrics": s.get("aggregated_metrics"),
            }
            for s in sessions
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=EXPORT_DIR, help="Directory of session export JSON files")
    parser.add_argument("--out", default=None, help="Optional path to write the full summary as JSON")
    args = parser.parse_args()

    result = summarize(args.dir)

    print(f"Sessions found: {result['n_sessions']}")
    print(f"Total turns pooled: {result['n_turns_total']}")
    print(f"Mean report completion: {result['mean_report_completion']}")
    print()
    print("Pooled RAG metrics (across all participants):")
    print(json.dumps(result["pooled_metrics"], indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull summary (including per-session breakdown) written to {args.out}")


if __name__ == "__main__":
    main()
