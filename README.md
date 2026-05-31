# Worker Daily Report Generator

```
report_generation_github/
├── backend/    Flask API + RAG engine (Python)
└── frontend/   React + Vite chat UI
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- An **OpenAI API key** ([create one here](https://platform.openai.com/api-keys))

## 1. Clone

```bash
git clone https://github.com/magic-YuanTian/worker_report_generation.git
cd worker_report_generation
```

## 2. Add your OpenAI key

```bash
cp .env.example backend/.env
# then open backend/.env and replace sk-... with your real key
```

That's it — only `OPENAI_API_KEY` is needed. The chat model defaults to `gpt-4o` and embeddings to `text-embedding-3-large`; both are hard-coded in `backend/conversation_manager.py` and `backend/rag_engine.py` if you want to change them.

## 3. Start the backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The server runs on `http://127.0.0.1:5001`.

> **First run:** the RAG index is not committed. The backend reads every PDF in `backend/documents/`, generates embeddings, and persists them to `backend/rag_storage/`. This takes a minute or two and is only done once — subsequent starts load the saved index instantly.
>
> To use your own reference documents, drop PDFs into `backend/documents/` (and delete `backend/rag_storage/` to force a rebuild).

## 4. Start the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:3000`). The Vite dev server proxies `/api/*` to the backend, so no extra config is needed.

## 5. Use it

1. Answer the assistant's questions about the day's work.
2. Watch the **Report Progress** panel on the right fill in.
3. Click an AI message to inspect which reference passages it grounded on.
4. When enough fields are filled, click **Download Report (PDF)**.

## Production build

To serve the compiled frontend yourself:

```bash
cd frontend
npm run build       # outputs to frontend/dist/
```

Then serve `frontend/dist/` from any static host and point its `/api` calls at the Flask backend.

## Troubleshooting

- **`AuthenticationError`** — Check `OPENAI_API_KEY` in `backend/.env` is a valid key.
- **First request takes a long time** — Expected on first run while the RAG index is built. Watch the backend logs.
- **Frontend can't reach backend** — Confirm `python app.py` is running on port `5001`; the proxy in `frontend/vite.config.js` expects that port.
