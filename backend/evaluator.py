"""Multi-judge evaluation metrics for RAG chat turns.

Independent from the chat pipeline: computes a retrieval precision proxy from
the saved retrieval snapshot, Faithfulness, and Answer Relevance. Faithfulness
and Answer Relevance are judged by three separate model families routed through
OpenRouter, then aggregated with majority-vote and pairwise agreement
statistics.

The normal workflow is offline: eval_exports.py reads saved session exports and
rewrites them with per-turn metrics plus aggregate summaries. The async helper
is retained for compatibility, but live chat collection does not call judges.
"""

import json
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import combinations

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Claim decomposition is a separate knob from the panel itself. Keep it
# materially cheaper than the panel while still strong on structured
# instruction following because every Faithfulness turn depends on it.
CLAIM_DECOMPOSITION_MODEL = "openai/gpt-4.1-mini"

# Judge panel. Use exact OpenRouter slugs so exports and figures match the
# actual routed models. Keep these on chat-completions-compatible models that
# return explicit text answers for the binary judge path. Order matters for
# stable reporting and pair labels.
JUDGE_MODELS = [
    "openai/gpt-4.1",
    "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-4.5",
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Runtime concurrency knobs. Keep them explicit because they affect cost,
# latency, and provider-side rate-limit pressure.
_EXECUTOR_MAX_WORKERS = int(os.environ.get("EVAL_EXECUTOR_WORKERS", "4"))
_VERIFY_POOL_MAX_WORKERS = int(os.environ.get("EVAL_VERIFY_POOL_WORKERS", "6"))

# Keep the retrieval proxy window aligned with the runtime context window by
# default. Override only for explicit study-only retrieval analyses.
_RETRIEVAL_PRECISION_K = None

_openrouter_client: OpenAI | None = None
_executor: ThreadPoolExecutor | None = None


def _judge_models() -> list[str]:
    """Return the configured judge model slugs in stable reporting order."""
    return list(JUDGE_MODELS)


def _judge_pairs(judge_models: list[str]) -> list[tuple[str, str]]:
    """Stable pair ordering for pairwise agreement reporting."""
    return list(combinations(judge_models, 2))


def _pair_label(left: str, right: str) -> str:
    return f"{left} vs {right}"


def _zero_map(judge_models: list[str]) -> dict[str, int]:
    return {model: 0 for model in judge_models}


def _none_map(judge_models: list[str]) -> dict[str, float | None]:
    return {model: None for model in judge_models}


def _get_openrouter_client() -> OpenAI:
    """Shared OpenRouter client for claim decomposition and all judges."""
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
    return _openrouter_client


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=_EXECUTOR_MAX_WORKERS,
            thread_name_prefix="evaluator",
        )
    return _executor


def _message_text(content) -> str:
    """Normalize OpenRouter/OpenAI message content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return str(content)


def get_evaluation_config() -> dict:
    """Expose the active evaluation knobs for export, plotting, and audit."""
    return {
        "claim_decomposition_model": CLAIM_DECOMPOSITION_MODEL,
        "judge_models": _judge_models(),
        "openrouter_base_url": OPENROUTER_BASE_URL,
        "retrieval_precision_k": _resolved_retrieval_k(),
        "executor_max_workers": _EXECUTOR_MAX_WORKERS,
        "verify_pool_max_workers": _VERIFY_POOL_MAX_WORKERS,
    }


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolved_retrieval_k(k: int | None = None) -> int:
    """Resolve the retrieval cutoff used by the proxy metric."""
    if k is not None:
        return k
    if _RETRIEVAL_PRECISION_K is not None:
        return _RETRIEVAL_PRECISION_K

    from rag_engine import MAX_RESULTS

    return MAX_RESULTS


def compute_retrieval_precision_proxy(
    candidates: list[dict], relevance: list[bool], k: int | None = None
) -> dict:
    """Fraction of top-k runtime candidates judged relevant by the same gate."""
    k_value = _resolved_retrieval_k(k)

    if not candidates:
        return {
            "metric_name": f"retrieval_precision_proxy@{k_value}",
            "scored": False,
            "score": None,
            "k": k_value,
            "candidate_count": 0,
            "top_k_count": 0,
            "relevant_count_in_top_k": 0,
            "skip_reason": "no_candidates",
        }

    top_k_relevance = relevance[:k_value]
    if not top_k_relevance:
        return {
            "metric_name": f"retrieval_precision_proxy@{k_value}",
            "scored": False,
            "score": None,
            "k": k_value,
            "candidate_count": len(candidates),
            "top_k_count": 0,
            "relevant_count_in_top_k": 0,
            "skip_reason": "empty_top_k",
        }

    relevant_count = sum(1 for item in top_k_relevance if item)
    return {
        "metric_name": f"retrieval_precision_proxy@{k_value}",
        "scored": True,
        "score": relevant_count / len(top_k_relevance),
        "k": k_value,
        "candidate_count": len(candidates),
        "top_k_count": len(top_k_relevance),
        "relevant_count_in_top_k": relevant_count,
        "skip_reason": None,
    }


def compute_precision_at_k(
    candidates: list[dict], relevance: list[bool], k: int | None = None
) -> float | None:
    """Backward-compatible scalar wrapper around the retrieval proxy."""
    return compute_retrieval_precision_proxy(candidates, relevance, k)["score"]


DECOMPOSE_CLAIMS_PROMPT = """\
Extract the factual claims from this assistant response that are presented \
as facts (e.g. safety guidance, code requirements, technical statements).

Do NOT extract:
- Questions the assistant is asking the user
- Acknowledgments or restatements of what the user just said
- Greetings, filler, or meta-commentary about the conversation

If the response contains no such factual claims, return an empty array.

Return ONLY a JSON array of strings, nothing else.

Response:
{response}"""

VERIFY_CLAIM_PROMPT = """\
Is this claim supported by the context below?

Claim: {claim}

Context:
{context}

Reply with ONLY "yes" or "no".""" 

ANSWER_RELEVANCE_PROMPT = """\
This is one turn from a conversational assistant that helps someone fill \
out a daily work report through natural dialogue.

Judge ONLY whether the response engages with the user's message in good \
faith. Ignore anything the response says afterward about an unrelated \
report field (e.g. pivoting from a safety question to asking about \
materials used) - that trailing pivot is normal for this assistant and \
must NOT count against relevance.

Answer "yes" if the response does any of the following for the user's \
message: answers it directly, acknowledges it accurately, or asks a \
directly related follow-up question about it. Answer "no" only if the \
response ignores, misreads, or is substantively unrelated to what the \
user said.

User message: {user_message}

Assistant response: {response}

Reply with ONLY "yes" or "no".""" 


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(3))
def _decompose_claims(response: str) -> list[str]:
    client = _get_openrouter_client()
    result = client.chat.completions.create(
        model=CLAIM_DECOMPOSITION_MODEL,
        messages=[
            {
                "role": "user",
                "content": DECOMPOSE_CLAIMS_PROMPT.format(response=response),
            }
        ],
        temperature=0,
        max_tokens=500,
    )
    text = _message_text(result.choices[0].message.content).strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        claims = json.loads(text)
    except json.JSONDecodeError:
        return []
    return [claim.strip() for claim in claims if isinstance(claim, str) and claim.strip()]


def _majority_vote(values: list[bool]) -> bool | None:
    """Return the strict majority over available boolean votes."""
    if not values:
        return None
    true_count = sum(1 for value in values if value)
    return true_count > (len(values) / 2)


def _available_votes(vote_map: dict[str, bool], error_map: dict[str, bool]) -> list[bool]:
    """Return only votes from judges that did not error on the item."""
    return [
        vote_map[model]
        for model in vote_map
        if not error_map.get(model, False)
    ]


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(3))
def _judge_yes_no(model: str, prompt: str) -> bool:
    """Generic binary judge call routed through OpenRouter."""
    client = _get_openrouter_client()
    result = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=4,
    )
    answer = _message_text(result.choices[0].message.content).strip().lower()
    match = re.match(r"^(yes|no)\b", answer)
    if not match:
        finish_reason = getattr(result.choices[0], "finish_reason", None)
        raise ValueError(
            f"{model} returned no parseable yes/no answer "
            f"(finish_reason={finish_reason!r}, content={answer!r})"
        )
    return match.group(1) == "yes"


def compute_faithfulness(response: str, context: str) -> dict:
    """Claim-level support against retrieved context, judged by a model panel."""
    judge_models = _judge_models()
    base_payload = {
        "metric_name": "faithfulness",
        "scored": False,
        "skip_reason": None,
        "claims": [],
        "claim_count": 0,
        "per_judge_scores": _none_map(judge_models),
        "per_judge_supported_counts": _zero_map(judge_models),
        "per_judge_evaluable_counts": _zero_map(judge_models),
        "per_judge_error_counts": _zero_map(judge_models),
        "majority_vote_score": None,
        "majority_supported_count": 0,
        "majority_evaluable_claim_count": 0,
    }

    if not context:
        return {**base_payload, "skip_reason": "no_context"}

    try:
        claims = _decompose_claims(response)
    except Exception:
        return {**base_payload, "skip_reason": "judge_error"}

    if not claims:
        return {**base_payload, "skip_reason": "no_claims"}

    judged = [
        {
            "claim": claim,
            "judge_votes": {model: False for model in judge_models},
            "judge_errors": {model: False for model in judge_models},
            "majority_supported": None,
        }
        for claim in claims
    ]

    max_workers = min(len(claims) * len(judge_models), _VERIFY_POOL_MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for index, claim in enumerate(claims):
            prompt = VERIFY_CLAIM_PROMPT.format(claim=claim, context=context)
            for model in judge_models:
                futures[pool.submit(_judge_yes_no, model, prompt)] = (index, model)

        for future in as_completed(futures):
            index, model = futures[future]
            try:
                judged[index]["judge_votes"][model] = future.result()
            except Exception:
                judged[index]["judge_errors"][model] = True

    per_judge_supported_counts = _zero_map(judge_models)
    per_judge_evaluable_counts = _zero_map(judge_models)
    per_judge_error_counts = _zero_map(judge_models)
    majority_supported_count = 0
    majority_evaluable_claim_count = 0

    for claim in judged:
        available = _available_votes(claim["judge_votes"], claim["judge_errors"])
        majority_supported = _majority_vote(available)
        claim["majority_supported"] = majority_supported
        if majority_supported is not None:
            majority_evaluable_claim_count += 1
            if majority_supported:
                majority_supported_count += 1

        for model in judge_models:
            if claim["judge_errors"][model]:
                per_judge_error_counts[model] += 1
                continue
            per_judge_evaluable_counts[model] += 1
            if claim["judge_votes"][model]:
                per_judge_supported_counts[model] += 1

    per_judge_scores = {
        model: (
            per_judge_supported_counts[model] / per_judge_evaluable_counts[model]
            if per_judge_evaluable_counts[model]
            else None
        )
        for model in judge_models
    }

    return {
        "metric_name": "faithfulness",
        "scored": True,
        "skip_reason": None,
        "claims": judged,
        "claim_count": len(judged),
        "per_judge_scores": per_judge_scores,
        "per_judge_supported_counts": per_judge_supported_counts,
        "per_judge_evaluable_counts": per_judge_evaluable_counts,
        "per_judge_error_counts": per_judge_error_counts,
        "majority_vote_score": (
            majority_supported_count / majority_evaluable_claim_count
            if majority_evaluable_claim_count
            else None
        ),
        "majority_supported_count": majority_supported_count,
        "majority_evaluable_claim_count": majority_evaluable_claim_count,
    }


def compute_answer_relevance(user_message: str, response: str) -> dict:
    """Turn-level response relevance judged by a three-model panel."""
    judge_models = _judge_models()
    judge_votes = {model: False for model in judge_models}
    judge_errors = {model: False for model in judge_models}

    with ThreadPoolExecutor(max_workers=len(judge_models)) as pool:
        futures = {
            pool.submit(
                _judge_yes_no,
                model,
                ANSWER_RELEVANCE_PROMPT.format(
                    user_message=user_message,
                    response=response,
                ),
            ): model
            for model in judge_models
        }
        for future in as_completed(futures):
            model = futures[future]
            try:
                judge_votes[model] = future.result()
            except Exception:
                judge_errors[model] = True

    available = _available_votes(judge_votes, judge_errors)

    return {
        "metric_name": "answer_relevance",
        "scored": True,
        "skip_reason": None,
        "judge_votes": judge_votes,
        "judge_errors": judge_errors,
        "majority_relevant": _majority_vote(available),
        "majority_evaluable": bool(available),
    }


def _evaluate_turn(
    user_message: str,
    response: str,
    rag_result: dict,
    turn_meta: dict | None = None,
) -> dict:
    candidates = rag_result.get("candidates", [])
    relevance = rag_result.get("relevance", [])
    context = rag_result.get("context", "")
    sources = rag_result.get("sources", [])
    retrieval_precision_proxy = compute_retrieval_precision_proxy(candidates, relevance)

    return {
        "turn": {
            "turn_index": turn_meta.get("turn_index") if turn_meta else None,
            "round_id": turn_meta.get("round_id") if turn_meta else None,
            "user_timestamp": turn_meta.get("user_timestamp") if turn_meta else None,
            "assistant_timestamp": (
                turn_meta.get("assistant_timestamp") if turn_meta else None
            ),
            "evaluated_at": _now_utc_iso(),
            "user_message": user_message,
            "assistant_response": response,
            "candidate_count": len(candidates),
            "accepted_source_count": len(sources),
            "source_refs": [
                {
                    "file_name": source.get("file_name"),
                    "page_label": source.get("page_label"),
                    "score": source.get("score"),
                }
                for source in sources
            ],
        },
        "retrieval_precision_proxy": retrieval_precision_proxy,
        "precision_at_k": retrieval_precision_proxy["score"],
        "faithfulness": compute_faithfulness(response, context),
        "answer_relevance": compute_answer_relevance(user_message, response),
    }


def evaluate_turn_async(
    user_message: str,
    response: str,
    rag_result: dict,
    callback=None,
    turn_meta: dict | None = None,
) -> Future:
    """Submit evaluation to a background pool and return immediately."""
    executor = _get_executor()
    future = executor.submit(
        _evaluate_turn,
        user_message,
        response,
        rag_result,
        turn_meta,
    )
    if callback is not None:
        future.add_done_callback(callback)
    return future


def _raw_agreement(rater_a: list[bool], rater_b: list[bool]) -> float | None:
    """Observed agreement rate for paired binary judgments."""
    n = len(rater_a)
    if n == 0 or n != len(rater_b):
        return None
    return sum(1 for a_val, b_val in zip(rater_a, rater_b) if a_val == b_val) / n


def _pairwise_boolean_stats(
    judge_models: list[str],
    items: list[dict],
) -> tuple[dict[str, float | None], dict[str, int]]:
    """Pairwise raw agreement over item-level judge votes."""
    raw_map: dict[str, float | None] = {}
    compared_counts: dict[str, int] = {}

    for left, right in _judge_pairs(judge_models):
        left_votes: list[bool] = []
        right_votes: list[bool] = []
        for item in items:
            vote_map = item.get("judge_votes", {})
            error_map = item.get("judge_errors", {})
            if error_map.get(left) or error_map.get(right):
                continue
            if left not in vote_map or right not in vote_map:
                continue
            left_votes.append(vote_map[left])
            right_votes.append(vote_map[right])

        label = _pair_label(left, right)
        raw_map[label] = _raw_agreement(left_votes, right_votes)
        compared_counts[label] = len(left_votes)

    return raw_map, compared_counts


def _normalize_retrieval_entry(metric: dict) -> dict | None:
    entry = metric.get("retrieval_precision_proxy")
    if entry:
        return entry

    score = metric.get("precision_at_k")
    if score is None:
        return None

    return {
        "metric_name": f"retrieval_precision_proxy@{_resolved_retrieval_k()}",
        "scored": True,
        "score": score,
        "k": _resolved_retrieval_k(),
        "candidate_count": None,
        "top_k_count": None,
        "relevant_count_in_top_k": None,
        "skip_reason": None,
    }


def _normalize_faithfulness_entry(entry: dict | None) -> dict | None:
    if not entry:
        return None
    if "per_judge_scores" in entry:
        return entry

    # Legacy two-judge schema.
    judge_models = ["openai/gpt-4o-mini", "google/gemini-2.5-flash"]
    claims = []
    for claim in entry.get("claims", []):
        claims.append(
            {
                "claim": claim.get("claim"),
                "judge_votes": {
                    judge_models[0]: claim.get("openai_supported", False),
                    judge_models[1]: claim.get("google_supported", False),
                },
                "judge_errors": {
                    judge_models[0]: claim.get("openai_error", False),
                    judge_models[1]: claim.get("google_error", False),
                },
                "majority_supported": _majority_vote(
                    [
                        claim.get("openai_supported", False),
                        claim.get("google_supported", False),
                    ]
                ),
            }
        )

    per_judge_scores = {
        judge_models[0]: entry.get("openai_score"),
        judge_models[1]: entry.get("google_score"),
    }
    per_judge_supported_counts = {
        judge_models[0]: entry.get("openai_supported_count", 0),
        judge_models[1]: entry.get("google_supported_count", 0),
    }
    per_judge_error_counts = {
        judge_models[0]: entry.get("openai_error_count", 0),
        judge_models[1]: entry.get("google_error_count", 0),
    }
    per_judge_evaluable_counts = {
        model: max(len(claims) - per_judge_error_counts.get(model, 0), 0)
        for model in judge_models
    }
    majority_supported_count = sum(1 for claim in claims if claim["majority_supported"])
    majority_evaluable_claim_count = len(
        [claim for claim in claims if claim["majority_supported"] is not None]
    )

    return {
        "metric_name": "faithfulness",
        "scored": entry.get("scored", bool(claims)),
        "skip_reason": entry.get("skip_reason"),
        "claims": claims,
        "claim_count": entry.get("claim_count", len(claims)),
        "per_judge_scores": per_judge_scores,
        "per_judge_supported_counts": per_judge_supported_counts,
        "per_judge_evaluable_counts": per_judge_evaluable_counts,
        "per_judge_error_counts": per_judge_error_counts,
        "majority_vote_score": (
            majority_supported_count / majority_evaluable_claim_count
            if majority_evaluable_claim_count
            else None
        ),
        "majority_supported_count": majority_supported_count,
        "majority_evaluable_claim_count": majority_evaluable_claim_count,
    }


def _normalize_answer_relevance_entry(entry: dict | None) -> dict | None:
    if not entry:
        return None
    if "judge_votes" in entry:
        return entry

    # Legacy two-judge schema.
    judge_models = ["openai/gpt-4o-mini", "google/gemini-2.5-flash"]
    judge_votes = {
        judge_models[0]: entry.get("openai_relevant", False),
        judge_models[1]: entry.get("google_relevant", False),
    }
    judge_errors = {
        judge_models[0]: entry.get("openai_error", False),
        judge_models[1]: entry.get("google_error", False),
    }
    available = _available_votes(judge_votes, judge_errors)
    return {
        "metric_name": "answer_relevance",
        "scored": entry.get("scored", True),
        "skip_reason": entry.get("skip_reason"),
        "judge_votes": judge_votes,
        "judge_errors": judge_errors,
        "majority_relevant": _majority_vote(available),
        "majority_evaluable": bool(available),
    }


def _collect_judge_models(
    faithfulness_entries: list[dict],
    answer_relevance_entries: list[dict],
) -> list[str]:
    """Union of judge models observed in the records, preserving panel order."""
    observed: set[str] = set()
    ordered: list[str] = []

    for model in _judge_models():
        ordered.append(model)
        observed.add(model)

    for entry in faithfulness_entries:
        for model in entry.get("per_judge_scores", {}):
            if model not in observed:
                ordered.append(model)
                observed.add(model)
        for claim in entry.get("claims", []):
            for model in claim.get("judge_votes", {}):
                if model not in observed:
                    ordered.append(model)
                    observed.add(model)

    for entry in answer_relevance_entries:
        for model in entry.get("judge_votes", {}):
            if model not in observed:
                ordered.append(model)
                observed.add(model)

    return ordered


def aggregate(metrics_history: list[dict]) -> dict:
    """Per-metric means plus majority-vote and pairwise raw-agreement stats."""
    retrieval_entries = [
        _normalize_retrieval_entry(metric)
        for metric in metrics_history
        if _normalize_retrieval_entry(metric) is not None
    ]
    retrieval_scores = [
        entry["score"]
        for entry in retrieval_entries
        if entry.get("scored") and entry.get("score") is not None
    ]
    retrieval_skipped = sum(1 for entry in retrieval_entries if not entry.get("scored"))

    faithfulness_entries = [
        _normalize_faithfulness_entry(metric.get("faithfulness"))
        for metric in metrics_history
        if metric.get("faithfulness") is not None
    ]
    faithfulness_entries = [entry for entry in faithfulness_entries if entry is not None]

    answer_relevance_entries = [
        _normalize_answer_relevance_entry(metric.get("answer_relevance"))
        for metric in metrics_history
        if metric.get("answer_relevance") is not None
    ]
    answer_relevance_entries = [
        entry for entry in answer_relevance_entries if entry is not None
    ]

    judge_models = _collect_judge_models(faithfulness_entries, answer_relevance_entries)

    scored_faithfulness_entries = [
        entry for entry in faithfulness_entries if entry.get("scored")
    ]
    faithfulness_per_judge_values: dict[str, list[float]] = {model: [] for model in judge_models}
    faithfulness_per_judge_error_counts: dict[str, int] = {model: 0 for model in judge_models}
    faithfulness_majority_values: list[float] = []
    faithfulness_claim_items: list[dict] = []
    for entry in scored_faithfulness_entries:
        for model in judge_models:
            score = entry.get("per_judge_scores", {}).get(model)
            if score is not None:
                faithfulness_per_judge_values[model].append(score)
            faithfulness_per_judge_error_counts[model] += entry.get(
                "per_judge_error_counts", {}
            ).get(model, 0)
        majority_score = entry.get("majority_vote_score")
        if majority_score is not None:
            faithfulness_majority_values.append(majority_score)
        faithfulness_claim_items.extend(entry.get("claims", []))

    answer_relevance_per_judge_flags: dict[str, list[bool]] = {
        model: [] for model in judge_models
    }
    answer_relevance_per_judge_error_counts: dict[str, int] = {
        model: 0 for model in judge_models
    }
    answer_relevance_majority_flags: list[bool] = []
    for entry in answer_relevance_entries:
        for model in judge_models:
            if entry.get("judge_errors", {}).get(model):
                answer_relevance_per_judge_error_counts[model] += 1
                continue
            if model in entry.get("judge_votes", {}):
                answer_relevance_per_judge_flags[model].append(
                    entry["judge_votes"][model]
                )
        if entry.get("majority_relevant") is not None:
            answer_relevance_majority_flags.append(entry["majority_relevant"])

    retrieval_mean = (
        sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else None
    )
    retrieval_skip_reason_counts: dict[str, int] = {}
    for entry in retrieval_entries:
        reason = entry.get("skip_reason")
        if reason:
            retrieval_skip_reason_counts[reason] = (
                retrieval_skip_reason_counts.get(reason, 0) + 1
            )

    faithfulness_skip_reason_counts: dict[str, int] = {}
    for entry in faithfulness_entries:
        if entry.get("scored"):
            continue
        reason = entry.get("skip_reason") or "unspecified"
        faithfulness_skip_reason_counts[reason] = (
            faithfulness_skip_reason_counts.get(reason, 0) + 1
        )

    faithfulness_pairwise_raw, faithfulness_pairwise_n = _pairwise_boolean_stats(
        judge_models, faithfulness_claim_items
    )
    answer_pairwise_raw, answer_pairwise_n = _pairwise_boolean_stats(
        judge_models, answer_relevance_entries
    )

    return {
        "retrieval_precision_proxy": {
            "mean": retrieval_mean,
            "n_scored": len(retrieval_scores),
            "n_skipped": retrieval_skipped,
            "k": _resolved_retrieval_k(),
            "skip_reason_counts": retrieval_skip_reason_counts,
        },
        "faithfulness": {
            "judge_models": judge_models,
            "per_judge_mean": {
                model: (
                    sum(faithfulness_per_judge_values[model])
                    / len(faithfulness_per_judge_values[model])
                    if faithfulness_per_judge_values[model]
                    else None
                )
                for model in judge_models
            },
            "majority_vote_mean": (
                sum(faithfulness_majority_values) / len(faithfulness_majority_values)
                if faithfulness_majority_values
                else None
            ),
            "pairwise_raw_agreement": faithfulness_pairwise_raw,
            "pairwise_compared_count": faithfulness_pairwise_n,
            "n_turns": len(scored_faithfulness_entries),
            "n_skipped": len(faithfulness_entries) - len(scored_faithfulness_entries),
            "n_claims": len(faithfulness_claim_items),
            "per_judge_error_count": faithfulness_per_judge_error_counts,
            "skip_reason_counts": faithfulness_skip_reason_counts,
        },
        "answer_relevance": {
            "judge_models": judge_models,
            "per_judge_mean": {
                model: (
                    sum(1 for flag in answer_relevance_per_judge_flags[model] if flag)
                    / len(answer_relevance_per_judge_flags[model])
                    if answer_relevance_per_judge_flags[model]
                    else None
                )
                for model in judge_models
            },
            "majority_vote_mean": (
                sum(1 for flag in answer_relevance_majority_flags if flag)
                / len(answer_relevance_majority_flags)
                if answer_relevance_majority_flags
                else None
            ),
            "pairwise_raw_agreement": answer_pairwise_raw,
            "pairwise_compared_count": answer_pairwise_n,
            "n_turns": len(answer_relevance_entries),
            "per_judge_error_count": answer_relevance_per_judge_error_counts,
        },
        # Compatibility alias while older code still refers to the legacy key.
        "precision_at_k": {
            "mean": retrieval_mean,
            "n": len(retrieval_scores),
        },
    }
