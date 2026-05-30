# Worker Daily Report Generator

A conversational web app that helps construction workers and building-lab students put together a daily activity report. The assistant walks the user through a short chat, fills in a structured report (activity, materials, equipment, safety hazards), pulls in relevant context from OSHA guidance and building codes via RAG, and exports the result as a PDF.

```
report_generation_github/
├── backend/          Flask API + RAG engine (Python)
├── frontend/         React + Vite chat UI
└── pdf-vectorizer/   Standalone CLI for chunking & embedding PDFs (optional)
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- An **Azure OpenAI** resource with two deployments:
  - a chat model (default name: `gpt-4o`)
  - an embedding model (default name: `text-embedding-3-large`)

## 1. Clone

```bash
git clone https://github.com/magic-YuanTian/worker_report_generation.git
cd worker_report_generation
```

## 2. Configure credentials

Copy the example file and fill in your Azure OpenAI endpoint and key:

```bash
cp .env.example backend/.env
# then edit backend/.env
```

The backend reads these variables on startup:

| Variable | Description |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | e.g. `https://your-resource.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | Azure key for the resource above |
| `AZURE_OPENAI_API_VERSION` | Chat API version (default `2024-10-01-preview`) |
| `AZURE_OPENAI_LLM_MODEL` | Chat deployment name (default `gpt-4o`) |
| `AZURE_OPENAI_EMBEDDING_API_VERSION` | Embedding API version (default `2024-02-15-preview`) |
| `AZURE_OPENAI_EMBEDDING_MODEL` | Embedding deployment name (default `text-embedding-3-large`) |

## 3. Start the backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The server runs on `http://127.0.0.1:5001`.

> **First run:** the RAG index is not committed. The backend will read every PDF in `backend/documents/`, generate embeddings, and persist them to `backend/rag_storage/`. This takes a minute or two and is only done once — subsequent starts load the saved index instantly.
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

## Optional: pdf-vectorizer

`pdf-vectorizer/` is a standalone CLI for extracting, chunking, and embedding PDFs outside the main app — useful if you want to build embeddings with a local `sentence-transformers` model instead of Azure. See `pdf-vectorizer/README.md` for usage.

## Troubleshooting

- **`AuthenticationError` from Azure** — Check `AZURE_OPENAI_ENDPOINT` ends with a trailing `/` and the deployment names match what's in your Azure resource.
- **First request takes a long time** — Expected on first run while the RAG index is built. Watch the backend logs.
- **Frontend can't reach backend** — Confirm `python app.py` is running on port `5001`; the proxy in `frontend/vite.config.js` expects that port.
