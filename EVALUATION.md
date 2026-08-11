# Evaluation User Guide

This project now uses a simple two-step evaluation workflow:

1. collect chats first
2. run LLM-as-a-judge later on the saved records

That separation is intentional. It means you can:

- chat and save records without waiting for judges
- rerun evaluation later
- change judge-side parameters without recollecting chats

This guide describes evaluation on saved real session exports. The synthetic
session generator has been removed from the repo.

The three evaluation metrics are:

- Retrieval Precision Proxy
- Faithfulness
- Answer Relevance

Faithfulness and Answer Relevance use a three-model judge panel through OpenRouter:

- `openai/gpt-4.1`
- `google/gemini-2.5-flash`
- `anthropic/claude-sonnet-4.5`

Faithfulness first decomposes assistant responses into factual claims with
`openai/gpt-4.1-mini`, then asks each judge whether each claim is supported by
the saved retrieved context.

## Quick start

If you just want the shortest working flow:

1. set `OPENAI_API_KEY` in `backend/.env`
2. chat normally and download reports
3. set `OPENROUTER_API_KEY` when you are ready to evaluate
4. run `python eval_exports.py`
5. run `python summarize.py`

## Step 1 - Set up the keys

For normal chat collection, you only need the OpenAI key:

```text
OPENAI_API_KEY=sk-...
```

If you want to run LLM-as-a-judge later, also add:

```text
OPENROUTER_API_KEY=sk-or-...
```

The OpenRouter key is not required for ordinary chatting. It is only needed for offline evaluation.

## Step 2 - Collect chats

Use the app normally. When a session is done, download the report.

That download step saves the session to:

```text
backend/session_exports/
```

Important:

- judge-based evaluation does not run during chat
- the saved export is a raw collection artifact first
- its top-level evaluation status will be `not_run` until you evaluate it later

Each saved session export includes:

- full conversation history
- per-round turn records
- report data
- retrieval metadata and accepted sources
- a saved retrieval snapshot per round
- collection configuration
- evaluation status

The saved retrieval snapshot is what makes later reruns possible. It stores:

- the retrieved context used for that turn
- the candidate pool
- the relevance mask used for the retrieval proxy

## Step 3 - Run evaluation later

When you are ready to score the saved chats, run:

```text
cd backend
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python eval_exports.py
```

This script rewrites the export files in place and fills:

- `metrics_history`
- `aggregated_metrics`
- per-turn `evaluation_status`
- per-turn `metrics`
- top-level `evaluation` metadata
- top-level `evaluation_config` for compatibility with older readers

Current exports use schema `4.0`, so evaluation runs directly from the saved
retrieval snapshot in each session export.

Useful options:

| Option | Use |
|---|---|
| `--pending-only` | Only evaluate exports whose top-level `evaluation.status` is not `complete`. |
| `--file <name>.json` | Evaluate one named export. Pass this option multiple times for a small batch. |
| `--no-rebuild-missing-snapshots` | Fail on older exports that do not contain saved retrieval snapshots instead of reconstructing retrieval. |

## Step 4 - Build the summary and figures

After evaluation, run:

```text
cd backend
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python summarize.py
```

By default the summary is written to `res/study_summary.json`, and figures are
written to `res/figures/`:

| Figure | Content |
|---|---|
| `pooled_metric_means.png` | Pooled mean scores for Retrieval Precision Proxy, Faithfulness majority vote, and Answer Relevance majority vote. |
| `faithfulness_panel_means.png` | Faithfulness mean score for each judge model plus the majority-vote mean. |
| `answer_relevance_panel_means.png` | Answer Relevance mean score for each judge model plus the majority-vote mean. |
| `metric_trends_by_turn.png` | Mean metric scores by conversation turn index across sessions, with hollow `N/A` markers where Faithfulness was skipped rather than scored as zero. |

This script:

- pools all saved exports in the folder
- reports both saved conversation rounds and evaluated turns
- computes aggregated study metrics
- writes a summary JSON
- writes figure files

If the exports have not been evaluated yet, the summarizer will still run, but it will report zero evaluated turns and skip figure generation.

## What each metric means

| Metric | What it measures | Notes |
|---|---|---|
| Retrieval Precision Proxy@4 | How many top retrieved candidates are judged relevant near the top of the ranking | Proxy metric, not a human-labeled benchmark |
| Faithfulness | Whether factual claims in the assistant response are supported by the saved retrieved context | Claim-level three-judge panel, reported with per-judge means, majority-vote mean, pairwise raw agreement, skipped-turn counts, and judge error counts |
| Answer Relevance | Whether the assistant responds appropriately to the user's immediately preceding message | Turn-level three-judge panel, reported with per-judge means, majority-vote mean, pairwise raw agreement, and judge error counts |

## How majority vote is calculated

Faithfulness and Answer Relevance use the same strict-majority rule over the
available non-error judge votes. A majority result is `true` only when more than
half of the usable judge votes are `yes`; otherwise it is `false`. If all judges
fail for an item, the majority result is `null` and that item is excluded from
majority-mean calculations.

For Answer Relevance, each user-assistant turn gets one panel vote. For example,
`yes, yes, no` becomes majority relevant, while `yes, no, no` becomes not
majority relevant. The reported majority-vote mean is the fraction of evaluated
turns whose majority result is relevant.

For Faithfulness, majority vote is applied separately to each extracted factual
claim. The turn-level majority Faithfulness score is:

```text
claims supported by majority / claims with a usable majority vote
```

For example, if a response has 4 factual claims and the panel majority says 3
are supported by the retrieved context, that turn's majority Faithfulness score
is `0.75`.

In `metric_trends_by_turn.png`, `turn_index` means the user-assistant round
number inside a saved session. `1` is the first user message and assistant
response after the initial greeting. Faithfulness may start later than
retrieval or answer relevance because it is skipped when a turn has no saved
retrieved context or no factual claims to verify.

## Known limitations

- Retrieval Precision Proxy is still a proxy metric.
- Faithfulness depends on model-based claim extraction and claim verification.
- Answer Relevance is tuned to this specific conversational assistant, not a fully general RAG setting.
- Majority vote improves robustness, but it is still model-based judgment, not human annotation.
- OpenRouter model availability, pricing, and slugs should be rechecked before formal study runs.
- Prompt validation is still limited.
- Study-scale cost has not been fully measured yet.

## Main files

- [backend/app.py](C:/dev/auto-report/worker_report_generation/backend/app.py)  
  exposes the chat, report, and export endpoints

- [backend/conversation_manager.py](C:/dev/auto-report/worker_report_generation/backend/conversation_manager.py)  
  collects conversation rounds and saves the retrieval snapshot needed for later evaluation

- [backend/eval_exports.py](C:/dev/auto-report/worker_report_generation/backend/eval_exports.py)  
  runs the offline LLM-as-a-judge pass on saved exports

- [backend/evaluator.py](C:/dev/auto-report/worker_report_generation/backend/evaluator.py)  
  computes the metric payloads and aggregates them

- [backend/sessions.py](C:/dev/auto-report/worker_report_generation/backend/sessions.py)  
  saves session exports to `backend/session_exports/`

- [backend/summarize.py](C:/dev/auto-report/worker_report_generation/backend/summarize.py)  
  pools evaluated exports and produces the summary JSON and figures

- [backend/rag_engine.py](C:/dev/auto-report/worker_report_generation/backend/rag_engine.py)  
  provides retrieval results and the saved retrieval artifacts used by the proxy metric

- [backend/report_generator.py](C:/dev/auto-report/worker_report_generation/backend/report_generator.py)  
  builds the downloadable PDF report from the collected session data

## What is still manual

This automation only covers the collection-plus-evaluation layer.

Still manual:

- user surveys
- report-quality scoring by human raters
- IRB / recruitment / scheduling / study operations
