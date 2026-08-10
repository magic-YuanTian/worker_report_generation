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

- `openai/gpt-4o-mini`
- `google/gemini-2.5-flash`
- `anthropic/claude-3-haiku`

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

Current exports use schema `4.0`, so evaluation runs directly from the saved
retrieval snapshot in each session export.

## Step 4 - Build the summary and figures

After evaluation, run:

```text
cd backend
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python summarize.py
```

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
| Faithfulness | Whether factual claims in the assistant response are supported by the saved retrieved context | Three-judge panel, reported with per-judge and majority-vote summaries |
| Answer Relevance | Whether the assistant responds appropriately to the user's immediately preceding message | Three-judge panel, reported with per-judge and majority-vote summaries |

## Known limitations

- Retrieval Precision Proxy is still a proxy metric.
- Faithfulness depends on model-based claim extraction and claim verification.
- Answer Relevance is tuned to this specific conversational assistant, not a fully general RAG setting.
- Majority vote improves robustness, but it is still model-based judgment, not human annotation.
- OpenRouter model availability and slugs should be rechecked before formal study runs.
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
