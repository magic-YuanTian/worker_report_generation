# RAG Evaluation Setup

Automated evaluation of the conversational assistant's RAG quality: Precision@K, Faithfulness, and Answer Relevance, each computed per chat turn and persisted per session. Faithfulness and Answer Relevance are dual-judged by two independent models (OpenAI + Google via OpenRouter) with Cohen's kappa reported between them.

---

## Part 1 — Guide: How to Run an Evaluation

### One-time setup

1. In `backend/.env`, set both keys — `OPENROUTER_API_KEY` is required because Faithfulness and Answer Relevance are judged by two different models (OpenAI + Google):
   ```
   OPENAI_API_KEY=sk-...
   OPENROUTER_API_KEY=sk-or-...
   ```

### Running a session

2. Chat normally, then download the report when done. Evaluation runs automatically in the background on every turn; downloading persists the full session (conversation, report, metrics) to `backend/session_exports/`.

### After the study (or after a batch of sessions)

3. Run the summarizer:
   ```
   cd backend
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   python summarize_results.py
   ```
   Further analysis can be added directly in `summarize_results.py`.

---

## Part 2 — Details: What Was Built

### The three metrics

| Metric | Judge(s) | Skipped when |
|---|---|---|
| **Precision@K** | Existing retrieval-stage judge (`rag_engine._judge_relevance`, unchanged) | Nothing retrieved → `None` |
| **Faithfulness** | New, dual (OpenAI + Google) | Empty context or no factual claims → `None` |
| **Answer Relevance** | New, dual (OpenAI + Google) | Never |

### Core functions

- **`evaluator.py`** — computes all three metrics; dual-judges Faithfulness (shared claim decomposition, independent verification) and Answer Relevance; runs evaluation asynchronously so chat latency is unaffected; aggregates results (means + Cohen's kappa) per session or pooled across sessions.
- **`session_store.py`** — persists a session to `backend/session_exports/` on download, and reads all exports back for summarization.
- **`summarize_results.py`** — CLI that pools every export's metrics through `evaluator.aggregate()` and prints/exports the result.
- **`conversation_manager.py`** — kicks off evaluation each turn, tracks in-flight evaluations, and waits for them to finish before a session is persisted.
- **`rag_engine.py`** — additionally exposes the retrieval judge's raw candidate/relevance data so Precision@K can reuse it without new LLM calls.

---

## Part 3 — Outlook: What's Still Manual

The original measurement plan (from the paper's methodology) has three layers. Only the first is now automated.

### ✅ Automated (this work)
**RAG technical quality** — Precision@K, Faithfulness, Answer Relevance.

### ❌ Still fully manual

- User survey (SUS + regulatory awareness, including a pre-interaction baseline)
- Report quality rubric (instructor scoring + human inter-rater agreement)
- Study logistics (IRB, recruitment, scheduling)

### Known limitations

- OpenRouter model slug needs reverifying against the live catalog
- Prompt validation only covered a handful of manual test turns
- Cost was estimated, not measured at scale
- Kappa's degenerate case (all-agree → reported as `1.0`) needs more volume before being meaningful
