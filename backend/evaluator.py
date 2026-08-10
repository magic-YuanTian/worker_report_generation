"""Dual-judge evaluation metrics for RAG chat turns.

Independent from the chat pipeline: computes Precision@K (pure, reuses
existing retrieval judgments from rag_engine.query_rag), Faithfulness, and
Answer Relevance. The latter two use two independent judge models - one
OpenAI, one Google (via OpenRouter) - so a single model's self-preference
bias doesn't skew the score; inter-model agreement is reported via Cohen's
kappa.

Runs fire-and-forget in a background thread pool (evaluate_turn_async) so
it never adds latency to the live chat response.
"""

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

EVAL_MODEL_OPENAI = "gpt-4o-mini"
EVAL_MODEL_GOOGLE = "google/gemini-2.5-flash"  # verify slug at openrouter.ai/models before running the study
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_EXECUTOR_MAX_WORKERS = int(os.environ.get("EVAL_EXECUTOR_WORKERS", "4"))
_VERIFY_POOL_MAX_WORKERS = 4  # caps per-turn parallel claim-verification calls

# ---------------------------------------------------------------------------
# Lazy singleton clients (mirrors rag_engine._get_judge_client) and executor
# ---------------------------------------------------------------------------

_openai_client: OpenAI | None = None
_google_client: OpenAI | None = None
_executor: ThreadPoolExecutor | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _get_google_client() -> OpenAI:
    global _google_client
    if _google_client is None:
        _google_client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
    return _google_client


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=_EXECUTOR_MAX_WORKERS, thread_name_prefix="evaluator"
        )
    return _executor


# ---------------------------------------------------------------------------
# Precision@K - pure computation, no LLM calls. Reuses the existing
# retrieval-stage relevance judgments already computed by query_rag().
# ---------------------------------------------------------------------------

def compute_precision_at_k(
    candidates: list[dict], relevance: list[bool], k: int | None = None
) -> float | None:
    """Fraction of the top-k candidates judged relevant. None if nothing was retrieved."""
    if not candidates:
        return None
    if k is None:
        from rag_engine import MAX_RESULTS
        k = MAX_RESULTS
    top_k_relevance = relevance[:k]
    if not top_k_relevance:
        return None
    return sum(1 for r in top_k_relevance if r) / len(top_k_relevance)


# ---------------------------------------------------------------------------
# Faithfulness - one shared claim decomposition, then two independent
# models verify each claim against the retrieved context. Sharing the
# claim set gives paired per-claim ratings, which is what Cohen's kappa
# needs (two raters judging the same items).
# ---------------------------------------------------------------------------

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


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(3))
def _decompose_claims(response: str) -> list[str]:
    client = _get_openai_client()
    result = client.chat.completions.create(
        model=EVAL_MODEL_OPENAI,
        messages=[
            {"role": "user", "content": DECOMPOSE_CLAIMS_PROMPT.format(response=response)}
        ],
        temperature=0,
        max_tokens=500,
    )
    text = (result.choices[0].message.content or "").strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        claims = json.loads(text)
    except json.JSONDecodeError:
        return []
    return [c.strip() for c in claims if isinstance(c, str) and c.strip()]


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(3))
def _verify_claim_openai(claim: str, context: str) -> bool:
    client = _get_openai_client()
    result = client.chat.completions.create(
        model=EVAL_MODEL_OPENAI,
        messages=[
            {"role": "user", "content": VERIFY_CLAIM_PROMPT.format(claim=claim, context=context)}
        ],
        temperature=0,
        max_tokens=3,
    )
    answer = (result.choices[0].message.content or "").strip().lower()
    return answer.startswith("yes")


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(3))
def _verify_claim_google(claim: str, context: str) -> bool:
    client = _get_google_client()
    result = client.chat.completions.create(
        model=EVAL_MODEL_GOOGLE,
        messages=[
            {"role": "user", "content": VERIFY_CLAIM_PROMPT.format(claim=claim, context=context)}
        ],
        temperature=0,
        max_tokens=3,
    )
    answer = (result.choices[0].message.content or "").strip().lower()
    return answer.startswith("yes")


def compute_faithfulness(response: str, context: str) -> dict | None:
    """None if context is empty or the response has no verifiable claims - not every
    turn is RAG-eligible, and scoring those as 0 would artificially deflate the aggregate.
    """
    if not context:
        return None
    try:
        claims = _decompose_claims(response)
    except Exception:
        return None
    if not claims:
        return None

    judged = [
        {"claim": c, "openai_supported": False, "google_supported": False} for c in claims
    ]

    with ThreadPoolExecutor(max_workers=min(len(claims) * 2, _VERIFY_POOL_MAX_WORKERS)) as pool:
        futures = {}
        for i, claim in enumerate(claims):
            futures[pool.submit(_verify_claim_openai, claim, context)] = (i, "openai_supported")
            futures[pool.submit(_verify_claim_google, claim, context)] = (i, "google_supported")
        for fut in as_completed(futures):
            i, key = futures[fut]
            try:
                judged[i][key] = fut.result()
            except Exception:
                pass  # treat as unsupported, consistent with rag_engine's judge_one handling

    openai_score = sum(1 for c in judged if c["openai_supported"]) / len(judged)
    google_score = sum(1 for c in judged if c["google_supported"]) / len(judged)

    return {"claims": judged, "openai_score": openai_score, "google_score": google_score}


# ---------------------------------------------------------------------------
# Answer Relevance - dual single-shot judgment per turn. Framed for a
# conversational, multi-turn slot-filling agent: a response that is only
# a follow-up question is valid and should score "yes".
# ---------------------------------------------------------------------------

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
def _answer_relevance_openai(user_message: str, response: str) -> bool:
    client = _get_openai_client()
    result = client.chat.completions.create(
        model=EVAL_MODEL_OPENAI,
        messages=[
            {
                "role": "user",
                "content": ANSWER_RELEVANCE_PROMPT.format(
                    user_message=user_message, response=response
                ),
            }
        ],
        temperature=0,
        max_tokens=3,
    )
    answer = (result.choices[0].message.content or "").strip().lower()
    return answer.startswith("yes")


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(3))
def _answer_relevance_google(user_message: str, response: str) -> bool:
    client = _get_google_client()
    result = client.chat.completions.create(
        model=EVAL_MODEL_GOOGLE,
        messages=[
            {
                "role": "user",
                "content": ANSWER_RELEVANCE_PROMPT.format(
                    user_message=user_message, response=response
                ),
            }
        ],
        temperature=0,
        max_tokens=3,
    )
    answer = (result.choices[0].message.content or "").strip().lower()
    return answer.startswith("yes")


def compute_answer_relevance(user_message: str, response: str) -> dict:
    """Always computed - every turn has a user message and a response."""
    openai_relevant = False
    google_relevant = False
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_answer_relevance_openai, user_message, response): "openai",
            pool.submit(_answer_relevance_google, user_message, response): "google",
        }
        for fut in as_completed(futures):
            provider = futures[fut]
            try:
                result = fut.result()
            except Exception:
                result = False
            if provider == "openai":
                openai_relevant = result
            else:
                google_relevant = result
    return {"openai_relevant": openai_relevant, "google_relevant": google_relevant}


# ---------------------------------------------------------------------------
# Orchestration - fire-and-forget entry point for conversation_manager.py
# ---------------------------------------------------------------------------

def _evaluate_turn(user_message: str, response: str, rag_result: dict) -> dict:
    candidates = rag_result.get("candidates", [])
    relevance = rag_result.get("relevance", [])
    context = rag_result.get("context", "")

    return {
        "precision_at_k": compute_precision_at_k(candidates, relevance),
        "faithfulness": compute_faithfulness(response, context),
        "answer_relevance": compute_answer_relevance(user_message, response),
    }


def evaluate_turn_async(
    user_message: str, response: str, rag_result: dict, callback=None
) -> Future:
    """Submits evaluation to a background thread pool and returns immediately.

    rag_result is exactly the dict returned by rag_engine.query_rag().
    """
    executor = _get_executor()
    future = executor.submit(_evaluate_turn, user_message, response, rag_result)
    if callback is not None:
        future.add_done_callback(callback)
    return future


# ---------------------------------------------------------------------------
# Aggregation - Cohen's kappa + means, computed on demand from a session's
# accumulated metrics_history (see conversation_manager.Session).
# ---------------------------------------------------------------------------

def _cohens_kappa(rater_a: list[bool], rater_b: list[bool]) -> float | None:
    """Standard binary Cohen's kappa. None if there's nothing to compare."""
    n = len(rater_a)
    if n == 0 or n != len(rater_b):
        return None
    po = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n
    p_a_true = sum(1 for a in rater_a if a) / n
    p_b_true = sum(1 for b in rater_b if b) / n
    pe = p_a_true * p_b_true + (1 - p_a_true) * (1 - p_b_true)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def aggregate(metrics_history: list[dict]) -> dict:
    """Per-metric means (ignoring skipped/None turns) plus pooled Cohen's kappa
    for the two dual-judged metrics.
    """
    precision_scores = [
        m["precision_at_k"] for m in metrics_history if m.get("precision_at_k") is not None
    ]

    faithfulness_entries = [m["faithfulness"] for m in metrics_history if m.get("faithfulness")]
    openai_claim_flags: list[bool] = []
    google_claim_flags: list[bool] = []
    for entry in faithfulness_entries:
        for claim in entry.get("claims", []):
            openai_claim_flags.append(claim["openai_supported"])
            google_claim_flags.append(claim["google_supported"])
    faithfulness_openai_scores = [e["openai_score"] for e in faithfulness_entries]
    faithfulness_google_scores = [e["google_score"] for e in faithfulness_entries]

    answer_relevance_entries = [
        m["answer_relevance"] for m in metrics_history if m.get("answer_relevance") is not None
    ]
    ar_openai_flags = [e["openai_relevant"] for e in answer_relevance_entries]
    ar_google_flags = [e["google_relevant"] for e in answer_relevance_entries]

    return {
        "precision_at_k": {
            "mean": sum(precision_scores) / len(precision_scores) if precision_scores else None,
            "n": len(precision_scores),
        },
        "faithfulness": {
            "openai_mean": (
                sum(faithfulness_openai_scores) / len(faithfulness_openai_scores)
                if faithfulness_openai_scores
                else None
            ),
            "google_mean": (
                sum(faithfulness_google_scores) / len(faithfulness_google_scores)
                if faithfulness_google_scores
                else None
            ),
            "agreement_kappa": _cohens_kappa(openai_claim_flags, google_claim_flags),
            "n_turns": len(faithfulness_entries),
            "n_claims": len(openai_claim_flags),
        },
        "answer_relevance": {
            "openai_mean": sum(ar_openai_flags) / len(ar_openai_flags) if ar_openai_flags else None,
            "google_mean": sum(ar_google_flags) / len(ar_google_flags) if ar_google_flags else None,
            "agreement_kappa": _cohens_kappa(ar_openai_flags, ar_google_flags),
            "n_turns": len(answer_relevance_entries),
        },
    }
