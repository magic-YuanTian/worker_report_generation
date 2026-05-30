import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

LLM_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-large"

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "rag_storage")

# Some PDFs encode text using the Adobe "Symbol" font, whose glyphs are
# extracted as Private Use Area codepoints (U+F020-U+F0FF = 0xF000 + the
# Symbol char code). These render as garbled "tofu" boxes in the browser.
# Map them back to their proper Unicode equivalents.
_SYMBOL_FONT_MAP = {
    0x20: " ", 0x22: "\u2200", 0x24: "\u2203", 0x27: "\u220B",
    0x2A: "\u2217", 0x2D: "\u2212",
    0x40: "\u2245",
    0x41: "\u0391", 0x42: "\u0392", 0x43: "\u03A7", 0x44: "\u0394",
    0x45: "\u0395", 0x46: "\u03A6", 0x47: "\u0393", 0x48: "\u0397",
    0x49: "\u0399", 0x4A: "\u03D1", 0x4B: "\u039A", 0x4C: "\u039B",
    0x4D: "\u039C", 0x4E: "\u039D", 0x4F: "\u039F", 0x50: "\u03A0",
    0x51: "\u0398", 0x52: "\u03A1", 0x53: "\u03A3", 0x54: "\u03A4",
    0x55: "\u03A5", 0x56: "\u03C2", 0x57: "\u03A9", 0x58: "\u039E",
    0x59: "\u03A8", 0x5A: "\u0396",
    0x5C: "\u2234", 0x5E: "\u22A5",
    0x61: "\u03B1", 0x62: "\u03B2", 0x63: "\u03C7", 0x64: "\u03B4",
    0x65: "\u03B5", 0x66: "\u03C6", 0x67: "\u03B3", 0x68: "\u03B7",
    0x69: "\u03B9", 0x6A: "\u03D5", 0x6B: "\u03BA", 0x6C: "\u03BB",
    0x6D: "\u03BC", 0x6E: "\u03BD", 0x6F: "\u03BF", 0x70: "\u03C0",
    0x71: "\u03B8", 0x72: "\u03C1", 0x73: "\u03C3", 0x74: "\u03C4",
    0x75: "\u03C5", 0x76: "\u03D6", 0x77: "\u03C9", 0x78: "\u03BE",
    0x79: "\u03C8", 0x7A: "\u03B6",
    0x7E: "\u223C",
    0xA1: "\u03D2", 0xA2: "\u2032", 0xA3: "\u2264", 0xA4: "\u2044",
    0xA5: "\u221E", 0xA6: "\u0192",
    0xAB: "\u2194", 0xAC: "\u2190", 0xAD: "\u2191", 0xAE: "\u2192",
    0xAF: "\u2193", 0xB0: "\u00B0", 0xB1: "\u00B1", 0xB2: "\u2033",
    0xB3: "\u2265", 0xB4: "\u00D7", 0xB5: "\u221D", 0xB6: "\u2202",
    0xB7: "\u2022", 0xB8: "\u00F7", 0xB9: "\u2260", 0xBA: "\u2261",
    0xBB: "\u2248", 0xBC: "\u2026",
    0xC4: "\u2297", 0xC5: "\u2295", 0xC6: "\u2205", 0xC7: "\u2229",
    0xC8: "\u222A", 0xC9: "\u2283", 0xCA: "\u2287", 0xCB: "\u2284",
    0xCC: "\u2282", 0xCD: "\u2286", 0xCE: "\u2208", 0xCF: "\u2209",
    0xD0: "\u2220", 0xD1: "\u2207", 0xD2: "\u00AE", 0xD3: "\u00A9",
    0xD4: "\u2122", 0xD5: "\u220F", 0xD6: "\u221A", 0xD7: "\u22C5",
    0xD8: "\u00AC", 0xD9: "\u2227", 0xDA: "\u2228", 0xDB: "\u21D4",
    0xDC: "\u21D0", 0xDD: "\u21D1", 0xDE: "\u21D2", 0xDF: "\u21D3",
    0xE5: "\u2211",
}
_SYMBOL_TRANSLATION = {0xF000 + code: repl for code, repl in _SYMBOL_FONT_MAP.items()}


def _sanitize_text(text: str) -> str:
    """Fix garbled glyphs from PDF symbol fonts.

    Remaps Adobe Symbol PUA codepoints to real Unicode, then strips any
    remaining Private Use Area characters so they never render as tofu boxes.
    """
    if not text:
        return text
    text = text.translate(_SYMBOL_TRANSLATION)
    return "".join(ch for ch in text if not (0xE000 <= ord(ch) <= 0xF8FF))


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase word/number tokenizer for BM25 keyword matching."""
    return _TOKEN_RE.findall(text.lower())


class HybridRetriever:
    """Combines dense (vector) retrieval with BM25 keyword retrieval.

    Building codes are full of exact terms and dimensions (e.g. "16d box nail",
    "braced wall panel") that dense embeddings can miss. BM25 catches those
    exact/keyword matches; the vector retriever catches semantic matches. The
    two candidate lists are merged with Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, vector_retriever, nodes: List[Any]):
        self.vector_retriever = vector_retriever
        self.nodes = nodes or []
        self._bm25 = None
        if _BM25_AVAILABLE and self.nodes:
            corpus = [_tokenize(n.get_content()) for n in self.nodes]
            # Guard against an all-empty corpus, which BM25Okapi rejects.
            if any(corpus):
                self._bm25 = BM25Okapi(corpus)

    def _bm25_nodes(self, query: str, k: int) -> List[Any]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.nodes[i] for i in ranked[:k] if scores[i] > 0]


def _node_id(node: Any) -> str:
    inner = getattr(node, "node", node)
    return getattr(inner, "node_id", None) or getattr(inner, "id_", str(id(inner)))


def _reciprocal_rank_fusion(ranked_lists: List[List[Any]], k: int = 60) -> List[Any]:
    """Merge several ranked node lists into one using RRF.

    Returns nodes ordered by fused score (best first), de-duplicated by id.
    """
    scores: dict = {}
    seen: dict = {}
    for ranked in ranked_lists:
        for rank, node in enumerate(ranked):
            nid = _node_id(node)
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank + 1)
            # Prefer the variant carrying a similarity score (dense hit).
            if nid not in seen or getattr(node, "score", None) is not None:
                seen[nid] = node
    ordered_ids = sorted(scores, key=lambda nid: scores[nid], reverse=True)
    return [seen[nid] for nid in ordered_ids]

def init_rag():
    """Initialize the RAG retriever from PDF documents."""
    print("Initializing RAG engine...")

    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
    Settings.llm = None

    index_exists = os.path.exists(os.path.join(STORAGE_DIR, "docstore.json"))

    if index_exists:
        print("Loading existing index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
    else:
        print(f"Building new index from documents in {DOCUMENTS_DIR}...")
        documents = SimpleDirectoryReader(DOCUMENTS_DIR).load_data()
        print(f"Loaded {len(documents)} document chunks. Creating embeddings...")
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=STORAGE_DIR)
        print("Index saved to storage.")

    # Fetch more candidates so the fusion + LLM judge have a richer pool.
    vector_retriever = index.as_retriever(similarity_top_k=DENSE_TOP_K)

    nodes = list(index.docstore.docs.values())
    hybrid = HybridRetriever(vector_retriever, nodes)
    mode = "hybrid (vector + BM25)" if hybrid._bm25 is not None else "vector-only"
    print(f"RAG engine ready ({mode}, {len(nodes)} chunks).")
    return hybrid


DENSE_TOP_K = 10      # Candidates pulled from the vector retriever
BM25_TOP_K = 10       # Candidates pulled from the BM25 keyword retriever
CANDIDATE_POOL = 8    # Fused candidates handed to the LLM relevance judge
MAX_RESULTS = 4       # Max sources to return after judging

RELEVANCE_PROMPT = """\
You are a relevance judge. Given a user query and a document chunk, decide \
whether the chunk contains information that is relevant and useful for \
answering or addressing the query.

Reply with ONLY "yes" or "no". Nothing else.

User query: {query}

Document chunk:
{chunk}"""

# Shared LLM client for relevance judging (created lazily)
_judge_client: OpenAI | None = None


def _get_judge_client() -> OpenAI:
    global _judge_client
    if _judge_client is None:
        _judge_client = OpenAI()
    return _judge_client


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(3))
def _judge_relevance(query: str, chunk: str) -> bool:
    """Ask the LLM whether a chunk is relevant to the query."""
    client = _get_judge_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": RELEVANCE_PROMPT.format(query=query, chunk=chunk),
            }
        ],
        temperature=0,
        max_tokens=3,
    )
    answer = response.choices[0].message.content.strip().lower()
    return answer.startswith("yes")


def query_rag(retriever, query: str) -> dict:
    """Retrieve relevant document passages with LLM-based relevance judging.

    1. Retrieves top-k candidates via vector similarity.
    2. Judges each candidate's relevance in parallel using the LLM.
    3. Returns only the chunks the LLM deems relevant.

    Returns a dict with:
      - context: concatenated text for the LLM system prompt
      - sources: list of dicts with file_name, page_label, text, score
    """
    if retriever is None:
        return {"context": "", "sources": []}

    # Hybrid retrieval: dense (semantic) + BM25 (keyword), fused with RRF.
    # Falls back gracefully to plain vector retrieval for older retrievers.
    if isinstance(retriever, HybridRetriever):
        dense_nodes = retriever.vector_retriever.retrieve(query)
        bm25_nodes = retriever._bm25_nodes(query, BM25_TOP_K)
        fused = _reciprocal_rank_fusion([dense_nodes, bm25_nodes])
        nodes = fused[:CANDIDATE_POOL]
    else:
        nodes = retriever.retrieve(query)

    if not nodes:
        return {"context": "", "sources": []}

    # Prepare candidate data
    candidates = []
    for node in nodes:
        full_text = _sanitize_text(node.get_content())
        text = full_text[:600] + "..." if len(full_text) > 600 else full_text

        metadata = node.metadata if hasattr(node, "metadata") else {}
        if not metadata and hasattr(node, "node") and hasattr(node.node, "metadata"):
            metadata = node.node.metadata

        node_score = getattr(node, "score", None)
        score = round(node_score, 4) if node_score is not None else None
        candidates.append({
            "file_name": metadata.get("file_name", "Unknown Document"),
            "page_label": metadata.get("page_label", None),
            "text": text,
            "score": score,
        })

    # Judge all candidates in parallel
    relevance = [False] * len(candidates)

    def judge_one(idx: int) -> tuple[int, bool]:
        return idx, _judge_relevance(query, candidates[idx]["text"])

    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        futures = {pool.submit(judge_one, i): i for i in range(len(candidates))}
        for future in as_completed(futures):
            try:
                idx, is_relevant = future.result()
                relevance[idx] = is_relevant
            except Exception:
                pass  # On error, treat as not relevant

    # Collect relevant sources in original ranking order
    sources = []
    chunks = []
    for i, cand in enumerate(candidates):
        if not relevance[i]:
            continue
        if len(sources) >= MAX_RESULTS:
            break
        sources.append(cand)
        chunks.append(cand["text"])

    context = "\n\n---\n\n".join(chunks) if chunks else ""
    return {"context": context, "sources": sources}
