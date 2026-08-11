# AGENTS.md

## Project

Worker daily report generator:

- `backend/`: Flask API, chat/session logic, RAG, PDF export, evaluation.
- `frontend/`: React + Vite chat UI.
- `backend/documents/`: PDF sources for RAG.
- `backend/session_exports/`: generated session exports.
- `res/`: generated summaries and figures.

User flow: create session, chat, fill report fields, download PDF, save export for later evaluation.

## Run Commands

Backend:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Runtime notes:

- Backend: `http://127.0.0.1:5001`.
- Frontend dev server: port `3000`, proxying `/api` to backend.
- Normal chat needs `OPENAI_API_KEY` in `backend/.env`.
- Offline evaluation also needs `OPENROUTER_API_KEY`.
- First backend start builds `backend/rag_storage/` from PDFs.

## Files To Know

- `backend/conversation_manager.py`: chat prompts, session state, report extraction.
- `backend/rag_engine.py`: document indexing, retrieval, retrieval relevance gate.
- `backend/sessions.py`: session export writer, schema version `4.0`.
- `backend/evaluator.py`: retrieval proxy, faithfulness, answer relevance.
- `backend/eval_exports.py`: offline evaluator for saved exports.
- `backend/summarize.py`: summary JSON and figure generation.
- `frontend/src/App.jsx`: main UI.

## Contracts

- Preserve API routes used by the frontend: `POST /api/session`, `POST /api/chat`, `GET /api/report/<session_id>`, `GET /api/report/<session_id>/metrics`, `GET /api/report/<session_id>/download`.
- If `REPORT_TEMPLATE` changes, update extraction, UI rendering, PDF output, exports, and evaluation consumers together.
- Keep each turn's saved retrieval snapshot compatible with `eval_exports.py`; it is the offline evaluation contract.
- Chat model is `gpt-4o`; embedding model is `text-embedding-3-large`; evaluator models are documented in `backend/models.yaml`.

## Evaluation

Exports are created when users download reports.

```powershell
cd backend
.venv\Scripts\activate
python eval_exports.py
python summarize.py
```

Notes:

- `eval_exports.py` rewrites export JSON files in place.
- `summarize.py` writes `res/study_summary.json` and figures under `res/figures/`.
- Current judges: `openai/gpt-4.1`, `google/gemini-2.5-flash`, `anthropic/claude-sonnet-4.5`.
- Faithfulness uses `openai/gpt-4.1-mini` for claim decomposition before judge voting.
- Do not regenerate exports or `res/` unless the task explicitly asks for it.

## Synthetic Conversations

When regenerating or creating synthetic conversations, keep users aligned with the assistant's current question.

Avoid synthetic turns that jump ahead to future fields, such as answering workforce when asked about task progress, or answering safety when asked about materials.

Required mitigation step:

- Review assistant/user turn pairs in order.
- Reject or rewrite user turns that bypass the active question.
- Preserve scenario facts, but move each fact to the point where the assistant asks for it.
- Do not optimize only for report-field coverage; prioritize believable question-answer flow.

## Validation

Frontend changes:

```powershell
cd frontend
npm run lint
npm run build
```

Backend changes:

- There is no committed backend test suite.
- At minimum, run the touched backend script or start `python app.py`.
- If export or evaluation logic changed, run the relevant evaluation script.

## Working Rules

- Treat `backend/session_exports/*.json`, `res/`, `backend/rag_storage/`, `.env`, `.venv`, and `node_modules/` as generated or local artifacts.
- Do not commit secrets.
- Keep prompt edits concise and plain-language.
- Prefer small, contract-aware changes.
