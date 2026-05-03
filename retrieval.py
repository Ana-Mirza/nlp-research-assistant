"""Hybrid retrieval pipeline: dense (ChromaDB) + sparse (BM25) + RRF + cross-encoder re-ranking."""

import os
import pickle
import numpy as np
from config import (
    EMBEDDING_MODEL, CHROMA_DB_PATH, COLLECTION_NAME, DATA_DIR,
    TOP_K_DENSE, TOP_K_SPARSE, TOP_K_RERANK, RRF_K, RERANKER_MODEL,
)

# Lazy-loaded resources
_collection = None
_bm25 = None
_bm25_mapping = None
_reranker = None


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    return _collection


def _get_bm25():
    global _bm25, _bm25_mapping
    if _bm25 is None:
        with open(os.path.join(DATA_DIR, "bm25_index.pkl"), "rb") as f:
            _bm25 = pickle.load(f)
        with open(os.path.join(DATA_DIR, "bm25_mapping.pkl"), "rb") as f:
            _bm25_mapping = pickle.load(f)
    return _bm25, _bm25_mapping


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _parse_document(doc):
    """Extract abstract from document text formatted as 'Title: ...\\nAuthors: ...\\nCategories: ...\\n\\n{abstract}'."""
    parts = doc.split("\n\n", 1)
    return parts[1] if len(parts) > 1 else ""


def _dense_retrieval(query):
    """Return list of (arxiv_id, rank) from ChromaDB."""
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=TOP_K_DENSE, include=["documents", "metadatas"])
    return [(aid, rank + 1) for rank, aid in enumerate(results["ids"][0])]


def _sparse_retrieval(query):
    """Return list of (arxiv_id, rank) from BM25."""
    bm25, mapping = _get_bm25()
    scores = bm25.get_scores(query.lower().split())
    top_indices = scores.argsort()[-TOP_K_SPARSE:][::-1]
    results = []
    for rank, idx in enumerate(top_indices):
        if scores[idx] > 0:
            results.append((mapping[int(idx)], rank + 1))
    return results


def _reciprocal_rank_fusion(dense_results, sparse_results):
    """Merge results using RRF with k=RRF_K. Returns list of (arxiv_id, rrf_score) sorted descending."""
    scores = {}
    for aid, rank in dense_results:
        scores[aid] = scores.get(aid, 0) + 1 / (RRF_K + rank)
    for aid, rank in sparse_results:
        scores[aid] = scores.get(aid, 0) + 1 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve top-k papers using hybrid retrieval + cross-encoder re-ranking."""
    dense_results = _dense_retrieval(query)
    sparse_results = _sparse_retrieval(query)
    fused = _reciprocal_rank_fusion(dense_results, sparse_results)

    # Fetch full paper info from ChromaDB for fused candidates
    fused_ids = [aid for aid, _ in fused]
    collection = _get_collection()
    fetched = collection.get(ids=fused_ids, include=["documents", "metadatas"])

    # Build lookup by id
    docs_by_id = {}
    for i, aid in enumerate(fetched["ids"]):
        docs_by_id[aid] = {
            "document": fetched["documents"][i],
            "metadata": fetched["metadatas"][i],
        }

    # Prepare pairs for cross-encoder
    candidates = []
    for aid, rrf_score in fused:
        if aid in docs_by_id:
            abstract = _parse_document(docs_by_id[aid]["document"])
            candidates.append((aid, abstract, docs_by_id[aid]["metadata"], docs_by_id[aid]["document"]))

    if not candidates:
        return []

    # Cross-encoder re-ranking
    reranker = _get_reranker()
    pairs = [(query, c[1]) for c in candidates]
    ce_scores = reranker.predict(pairs)

    # Sort by cross-encoder score and take top_k
    ranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for (aid, abstract, meta, doc), score in ranked:
        results.append({
            "id": aid,
            "title": meta.get("title", ""),
            "authors": meta.get("authors", ""),
            "categories": meta.get("categories", ""),
            "year": meta.get("year", ""),
            "abstract": abstract,
            "score": float(score),
            "arxiv_url": f"https://arxiv.org/abs/{aid}",
        })
    return results


# --- Feature: Cosine similarity ---
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def compute_cosine_similarities(query: str, paper_ids: list[str]) -> dict:
    """Compute cosine similarity between query and each paper using embeddings."""
    if not paper_ids:
        return {}
    embedder = _get_embedder()
    collection = _get_collection()
    fetched = collection.get(ids=paper_ids, include=["embeddings"])
    if fetched["embeddings"] is None:
        return {}
    q_emb = embedder.encode(query)
    sims = {}
    for pid, emb in zip(fetched["ids"], fetched["embeddings"]):
        emb = np.array(emb)
        cos = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-10)
        sims[pid] = float(cos)
    return sims