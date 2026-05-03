"""Ingest filtered arXiv papers into ChromaDB (parallel) and build BM25 index."""

import pickle
import os
import time
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from config import EMBEDDING_MODEL, CHROMA_DB_PATH, COLLECTION_NAME, DATA_DIR

BATCH_SIZE = 500
EMBED_BATCH_SIZE = 256  # Batch size for sentence-transformers encoding


def find_parquet_file():
    for f in os.listdir(DATA_DIR):
        if f.endswith(".parquet"):
            return os.path.join(DATA_DIR, f)
    raise FileNotFoundError(f"No parquet file found in {DATA_DIR}")


def make_document(row):
    return f"Title: {row['title']}\nAuthors: {row['authors']}\nCategories: {row['categories']}\n\n{row['abstract']}"


def extract_year(date_val):
    try:
        if hasattr(date_val, "year"):
            return date_val.year
        return int(str(date_val)[:4])
    except (ValueError, TypeError):
        return 0


def ingest_chromadb(parquet_path):
    """Embed all documents in parallel using sentence-transformers, then batch insert into ChromaDB."""
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete existing collection and recreate
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Deleted existing collection.")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print("Loading parquet...")
    df = pd.read_parquet(parquet_path, columns=["id", "title", "abstract", "authors", "categories", "update_date"])
    total = len(df)
    print(f"Total papers: {total:,}")

    # Prepare all documents
    print("Preparing documents...")
    documents = []
    ids = []
    metadatas = []
    for _, row in df.iterrows():
        doc = make_document(row)
        documents.append(doc)
        ids.append(str(row["id"]))
        metadatas.append({
            "id": str(row["id"]),
            "title": str(row["title"]),
            "authors": str(row["authors"]),
            "categories": str(row["categories"]),
            "year": extract_year(row.get("update_date", "")),
        })

    # Batch embed using sentence-transformers directly (much faster than ChromaDB's per-batch embedding)
    print(f"Embedding {total:,} documents with {EMBEDDING_MODEL} (batch_size={EMBED_BATCH_SIZE})...")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    t0 = time.time()
    # encode() handles batching internally and uses GPU/MPS if available
    all_embeddings = model.encode(
        documents,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    t1 = time.time()
    print(f"Embedding done in {t1-t0:.0f}s ({total/(t1-t0):.0f} docs/s)")

    # Batch insert into ChromaDB (no re-embedding needed)
    print("Inserting into ChromaDB...")
    t0 = time.time()
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=all_embeddings[start:end].tolist(),
        )
        if (start // BATCH_SIZE) % 20 == 0:
            print(f"  Inserted {end:,}/{total:,} ({100*end/total:.0f}%)")

    t1 = time.time()
    print(f"ChromaDB insert done in {t1-t0:.0f}s. Total: {collection.count():,} documents.")


def build_bm25_index(parquet_path):
    bm25_path = os.path.join(DATA_DIR, "bm25_index.pkl")
    mapping_path = os.path.join(DATA_DIR, "bm25_mapping.pkl")

    if os.path.exists(bm25_path) and os.path.exists(mapping_path):
        print("BM25 index already exists. Skipping.")
        return

    print("Building BM25 index...")
    df = pd.read_parquet(parquet_path, columns=["id", "abstract"])

    tokenized_corpus = [str(abstract).lower().split() for abstract in df["abstract"]]
    id_mapping = {i: str(aid) for i, aid in enumerate(df["id"])}

    print(f"  Tokenized {len(tokenized_corpus):,} documents.")
    bm25 = BM25Okapi(tokenized_corpus)

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    with open(mapping_path, "wb") as f:
        pickle.dump(id_mapping, f)
    print(f"BM25 index saved.")


def main():
    parquet_path = find_parquet_file()
    print(f"Using: {parquet_path}")
    ingest_chromadb(parquet_path)
    # Delete old BM25 to rebuild with new data
    for f in ["bm25_index.pkl", "bm25_mapping.pkl"]:
        p = os.path.join(DATA_DIR, f)
        if os.path.exists(p):
            os.remove(p)
    build_bm25_index(parquet_path)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
