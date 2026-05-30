# PDF Vectorization Pipeline

Extract text from PDFs, chunk it, and generate embeddings for semantic search and RAG.

**Stack:** PyMuPDF / pdfplumber · LangChain splitter · sentence-transformers / OpenAI · NumPy `.npz`

## Setup

```bash
uv sync
```

## Usage

```bash
# Single file
uv run python main.py extract data/raw/1.pdf
uv run python main.py chunk   data/raw/1.pdf
uv run python main.py embed   data/raw/1.pdf

# Batch (add -r for recursive)
uv run python main.py extract data/raw/ -r
uv run python main.py chunk   data/raw/ -r
uv run python main.py embed   data/raw/ -r

# Full pipeline
uv run python main.py run data/raw/1.pdf
uv run python main.py run data/raw/ -r

# Utilities
uv run python main.py check   # verify imports and config
uv run python main.py info    # show loaded configuration
```

`run` skips steps whose output already exists. Run steps individually to inspect intermediate results.

## Output

```
data/processed/1/
├── text.json        # extracted pages + figure metadata
├── chunks.json      # text chunks
├── embeddings.npz   # vectors
└── figures/         # page_3_fig_0.jpeg, …
```

## Configuration

Edit `config/config.yaml`.

| Setting | Default | Notes |
|---|---|---|
| `extraction.method` | `pymupdf` | `pdfplumber` for tables |
| `extraction.extract_figures` | `true` | save embedded images to `figures/` |
| `extraction.min_figure_px` | `100` | skip figures smaller than N px |
| `extraction.extract_tables` | `false` | extract tables separately (pdfplumber only) |
| `chunking.chunk_size` | `512` | chars per chunk |
| `chunking.chunk_overlap` | `128` | ~10–20 % of chunk_size |
| `vectorization.model_type` | `sentence_transformers` | `openai` requires API key |
| `vectorization.model_name` | `all-MiniLM-L6-v2` | override via `EMBEDDING_MODEL` env var |

### OpenAI embeddings

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

```yaml
vectorization:
  model_type: openai
  model_name: text-embedding-3-small
```
