"""Summarize evaluated session exports and render study figures.

Usage:
    python summarize.py
    python summarize.py --dir path/to/session_exports
    python summarize.py --out res/study_summary.json
"""

import argparse
import json
import os
from statistics import mean

import evaluator
from sessions import EXPORT_DIR, load_all_sessions

# Figure settings stay local to keep this script self-contained and easy to
# adjust for paper-ready output without adding another plotting module.
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULT_DIR = os.path.join(PROJECT_ROOT, "res")
FIGURE_DIRNAME = "figures"
DEFAULT_SUMMARY_FILENAME = "study_summary.json"
FIGURE_DPI = 300
PANEL_FIGURE_SIZE = (18, 10)
POOLED_FIGURE_SIZE = (14, 8)
TREND_FIGURE_SIZE = (16, 9)
RETRIEVAL_COLOR = "#6B86A3"
FAITHFULNESS_COLOR = "#7E95A6"
ANSWER_COLOR = "#8A92AE"
MAJORITY_COLOR = "#626C79"
GRID_COLOR = "#E3E6EA"
SPINE_COLOR = "#C7CDD4"
TEXT_COLOR = "#2F3640"
BAR_EDGE_COLOR = "#75808C"
TITLE_FONT_SIZE = 24
AXIS_LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 18
LEGEND_FONT_SIZE = 16
VALUE_LABEL_FONT_SIZE = 16


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _model_display_name(model_name: str | None, fallback: str) -> str:
    """Exact model label, wrapped slightly so long slugs fit in figures."""
    if not model_name:
        return fallback
    text = str(model_name)
    return text.replace("/", "/\n", 1) if "/" in text else text


def _count_schema_versions(sessions: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for session in sessions:
        version = str(session.get("metrics_schema_version", "legacy"))
        counts[version] = counts.get(version, 0) + 1
    return counts


def _study_eval_config(sessions: list[dict]) -> dict:
    """Use the first available saved config, else fall back to current defaults."""
    for session in sessions:
        evaluation = session.get("evaluation")
        if isinstance(evaluation, dict):
            config = evaluation.get("config")
            if isinstance(config, dict) and config:
                return config
        config = session.get("evaluation_config")
        if isinstance(config, dict) and config:
            return config
    return evaluator.get_evaluation_config()


def _sorted_turn_metrics(session: dict) -> list[dict]:
    """Return per-turn metric records in stable round order."""
    metrics = list(session.get("metrics_history", []))
    return sorted(metrics, key=lambda item: item.get("turn", {}).get("turn_index", 10**9))


def _judge_models_for_summary(summary: dict) -> list[str]:
    config = summary.get("evaluation_config", {})
    models = config.get("judge_models")
    if isinstance(models, list) and models:
        return list(models)
    pooled = summary.get("pooled_metrics", {})
    for metric_name in ("faithfulness", "answer_relevance"):
        metric = pooled.get(metric_name, {})
        models = metric.get("judge_models")
        if isinstance(models, list) and models:
            return list(models)
    return evaluator.get_evaluation_config().get("judge_models", [])


def _extract_turn_series(sessions: list[dict], key: str) -> list[tuple[int, float]]:
    """Collect (turn_index, score) pairs across all sessions for one metric line."""
    points = []
    for session in sessions:
        for metric in _sorted_turn_metrics(session):
            turn_index = metric.get("turn", {}).get("turn_index")
            if turn_index is None:
                continue

            if key == "retrieval_precision_proxy":
                entry = metric.get("retrieval_precision_proxy")
                if not entry and metric.get("precision_at_k") is not None:
                    entry = {"scored": True, "score": metric["precision_at_k"]}
                score = entry.get("score") if entry and entry.get("scored") else None
            elif key == "faithfulness_majority":
                entry = metric.get("faithfulness", {})
                score = entry.get("majority_vote_score")
            else:
                entry = metric.get("answer_relevance", {})
                majority_relevant = entry.get("majority_relevant")
                score = float(majority_relevant) if majority_relevant is not None else None

            if score is not None:
                points.append((turn_index, score))
    return points


def _turn_level_means(points: list[tuple[int, float]]) -> tuple[list[int], list[float]]:
    grouped: dict[int, list[float]] = {}
    for turn_index, score in points:
        grouped.setdefault(turn_index, []).append(score)

    ordered_turns = sorted(grouped)
    return ordered_turns, [_safe_mean(grouped[turn]) for turn in ordered_turns]


def _bar_value(values: list[float | None]) -> list[float]:
    return [value if value is not None else 0 for value in values]


def _apply_paper_style(ax) -> None:
    """Muted figure styling closer to standard CS-paper plots."""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=TICK_FONT_SIZE)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.xaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.title.set_color(TEXT_COLOR)
    ax.title.set_size(TITLE_FONT_SIZE)


def _tight_save(fig, path: str, *, rect=None) -> None:
    """Apply consistent padding before saving larger paper-ready figures."""
    fig.tight_layout(rect=rect, pad=1.1)
    fig.savefig(path, dpi=FIGURE_DPI)


def _render_figures(summary: dict, sessions: list[dict], figure_dir: str) -> list[str]:
    """Generate compact CS-style figures from exported records."""
    if not sessions:
        return []
    if not any(session.get("metrics_history") for session in sessions):
        summary["figure_generation_status"] = "skipped_no_evaluated_metrics"
        return []

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        summary["figure_generation_status"] = "skipped_missing_matplotlib"
        return []

    os.makedirs(figure_dir, exist_ok=True)
    obsolete_paths = [os.path.join(figure_dir, "judge_agreement.png")]
    for stale_path in obsolete_paths:
        if os.path.exists(stale_path):
            os.remove(stale_path)
    figure_paths = []
    pooled = summary["pooled_metrics"]
    judge_models = _judge_models_for_summary(summary)
    judge_labels = {
        model: _model_display_name(model, model)
        for model in judge_models
    }
    retrieval_k = pooled["retrieval_precision_proxy"].get("k")

    metric_labels = [
        f"Retrieval\nProxy@{retrieval_k}",
        "Faithfulness\n(Majority)",
        "Answer Rel.\n(Majority)",
    ]
    metric_values = [
        pooled["retrieval_precision_proxy"]["mean"],
        pooled["faithfulness"]["majority_vote_mean"],
        pooled["answer_relevance"]["majority_vote_mean"],
    ]
    metric_colors = [RETRIEVAL_COLOR, FAITHFULNESS_COLOR, ANSWER_COLOR]

    fig, ax = plt.subplots(figsize=POOLED_FIGURE_SIZE)
    positions = list(range(len(metric_labels)))
    bars = ax.bar(
        positions,
        _bar_value(metric_values),
        color=metric_colors,
        width=0.72,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.6,
    )
    ax.set_xticks(positions, metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Pooled RAG Evaluation Overview")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color=GRID_COLOR, alpha=0.8)
    ax.set_axisbelow(True)
    _apply_paper_style(ax)
    for bar, value in zip(bars, metric_values):
        if value is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=VALUE_LABEL_FONT_SIZE,
                color=TEXT_COLOR,
            )
    metric_path = os.path.join(figure_dir, "pooled_metric_means.png")
    _tight_save(fig, metric_path)
    plt.close(fig)
    figure_paths.append(metric_path)

    faith_labels = [judge_labels[model] for model in judge_models] + ["Majority\nvote"]
    faith_values = [
        pooled["faithfulness"]["per_judge_mean"].get(model) for model in judge_models
    ] + [pooled["faithfulness"]["majority_vote_mean"]]
    faith_colors = [FAITHFULNESS_COLOR for _ in judge_models] + [MAJORITY_COLOR]

    fig, ax = plt.subplots(figsize=PANEL_FIGURE_SIZE)
    positions = list(range(len(faith_labels)))
    bars = ax.bar(
        positions,
        _bar_value(faith_values),
        color=faith_colors,
        width=0.72,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.6,
    )
    ax.set_xticks(positions, faith_labels)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Faithfulness score")
    ax.set_title("Faithfulness by Judge and Majority Vote")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color=GRID_COLOR, alpha=0.8)
    ax.set_axisbelow(True)
    _apply_paper_style(ax)
    ax.margins(x=0.05)
    for bar, value in zip(bars, faith_values):
        if value is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=VALUE_LABEL_FONT_SIZE,
                color=TEXT_COLOR,
            )
    faith_path = os.path.join(figure_dir, "faithfulness_panel_means.png")
    _tight_save(fig, faith_path)
    plt.close(fig)
    figure_paths.append(faith_path)

    answer_labels = [judge_labels[model] for model in judge_models] + ["Majority\nvote"]
    answer_values = [
        pooled["answer_relevance"]["per_judge_mean"].get(model) for model in judge_models
    ] + [pooled["answer_relevance"]["majority_vote_mean"]]
    answer_colors = [ANSWER_COLOR for _ in judge_models] + [MAJORITY_COLOR]

    fig, ax = plt.subplots(figsize=PANEL_FIGURE_SIZE)
    positions = list(range(len(answer_labels)))
    bars = ax.bar(
        positions,
        _bar_value(answer_values),
        color=answer_colors,
        width=0.72,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.6,
    )
    ax.set_xticks(positions, answer_labels)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Answer relevance score")
    ax.set_title("Answer Relevance by Judge and Majority Vote")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color=GRID_COLOR, alpha=0.8)
    ax.set_axisbelow(True)
    _apply_paper_style(ax)
    ax.margins(x=0.05)
    for bar, value in zip(bars, answer_values):
        if value is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=VALUE_LABEL_FONT_SIZE,
                color=TEXT_COLOR,
            )
    answer_path = os.path.join(figure_dir, "answer_relevance_panel_means.png")
    _tight_save(fig, answer_path)
    plt.close(fig)
    figure_paths.append(answer_path)

    retrieval_turns, retrieval_means = _turn_level_means(
        _extract_turn_series(sessions, "retrieval_precision_proxy")
    )
    faith_turns, faith_means = _turn_level_means(
        _extract_turn_series(sessions, "faithfulness_majority")
    )
    answer_turns, answer_means = _turn_level_means(
        _extract_turn_series(sessions, "answer_relevance_majority")
    )

    fig, ax = plt.subplots(figsize=TREND_FIGURE_SIZE)
    if retrieval_turns:
        ax.plot(
            retrieval_turns,
            retrieval_means,
            marker="o",
            linewidth=2.4,
            markersize=10,
            label=f"Retrieval proxy@{retrieval_k}",
            color=RETRIEVAL_COLOR,
        )
    if faith_turns:
        ax.plot(
            faith_turns,
            faith_means,
            marker="s",
            linewidth=2.4,
            markersize=10,
            label="Faithfulness (majority vote)",
            color=FAITHFULNESS_COLOR,
        )
    if answer_turns:
        ax.plot(
            answer_turns,
            answer_means,
            marker="^",
            linewidth=2.4,
            markersize=10,
            label="Answer relevance (majority vote)",
            color=ANSWER_COLOR,
        )
    all_turns = sorted(set(retrieval_turns + faith_turns + answer_turns))
    if all_turns:
        ax.set_xticks(all_turns)
        ax.set_xlim(min(all_turns) - 0.15, max(all_turns) + 0.15)
    ax.set_ylim(0, 1.10)
    ax.set_xlabel("Turn index")
    ax.set_ylabel("Mean score across sessions")
    ax.set_title("Metric Trends by Conversation Round")
    ax.grid(True, linestyle="--", linewidth=0.7, color=GRID_COLOR, alpha=0.8)
    _apply_paper_style(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        columnspacing=1.6,
        handlelength=2.8,
    )
    trend_path = os.path.join(figure_dir, "metric_trends_by_turn.png")
    _tight_save(fig, trend_path, rect=(0, 0.08, 1, 1))
    plt.close(fig)
    figure_paths.append(trend_path)

    return figure_paths


def summarize(export_dir: str = EXPORT_DIR) -> dict:
    sessions = load_all_sessions(export_dir)

    combined_metrics_history = []
    total_rounds = 0
    for session in sessions:
        combined_metrics_history.extend(session.get("metrics_history", []))
        total_rounds += len(session.get("turn_records", []))

    completions = [
        session["completion"]
        for session in sessions
        if session.get("completion") is not None
    ]
    pooled_metrics = evaluator.aggregate(combined_metrics_history)
    evaluation_config = _study_eval_config(sessions)
    os.makedirs(RESULT_DIR, exist_ok=True)
    figure_dir = os.path.join(RESULT_DIR, FIGURE_DIRNAME)

    summary = {
        "n_sessions": len(sessions),
        "n_rounds_total": total_rounds,
        "n_turns_total": len(combined_metrics_history),
        "mean_report_completion": (
            sum(completions) / len(completions) if completions else None
        ),
        "pooled_metrics": pooled_metrics,
        "evaluation_config": evaluation_config,
        "schema_versions": _count_schema_versions(sessions),
        "result_dir": RESULT_DIR,
        "figure_generation_status": "generated",
        "figure_paths": [],
        "per_session": [
            {
                "session_id": session.get("session_id"),
                "saved_at": session.get("saved_at"),
                "completion": session.get("completion"),
                "n_turns": len(session.get("metrics_history", [])),
                "n_rounds": len(session.get("turn_records", [])),
                "metrics_schema_version": session.get("metrics_schema_version", "legacy"),
                "evaluation": session.get("evaluation", {}),
                "aggregated_metrics": evaluator.aggregate(
                    session.get("metrics_history", [])
                ),
            }
            for session in sessions
        ],
    }
    summary["figure_paths"] = _render_figures(summary, sessions, figure_dir)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        default=EXPORT_DIR,
        help="Directory of session export JSON files",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(RESULT_DIR, DEFAULT_SUMMARY_FILENAME),
        help="Path to write the full summary as JSON",
    )
    args = parser.parse_args()

    result = summarize(args.dir)

    print(f"Sessions found: {result['n_sessions']}")
    print(f"Conversation rounds saved: {result['n_rounds_total']}")
    print(f"Evaluated turns pooled: {result['n_turns_total']}")
    print(f"Mean report completion: {result['mean_report_completion']}")
    print()
    print("Pooled RAG metrics (across all participants):")
    print(json.dumps(result["pooled_metrics"], indent=2))
    print(f"\nSchema versions: {json.dumps(result['schema_versions'])}")
    print(f"Result directory: {result['result_dir']}")
    print(f"Figure generation status: {result['figure_generation_status']}")

    if result["figure_paths"]:
        print("\nFigures written:")
        for path in result["figure_paths"]:
            print(f"- {path}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nFull summary written to {args.out}")


if __name__ == "__main__":
    main()
