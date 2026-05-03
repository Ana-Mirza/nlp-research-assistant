"""Download PubMed papers and add to existing ChromaDB + BM25 indexes."""

import os
import time
import pickle
import numpy as np
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import EMBEDDING_MODEL, CHROMA_DB_PATH, COLLECTION_NAME, DATA_DIR

BATCH_SIZE = 500
EMBED_BATCH_SIZE = 256


def download_pubmed():
    """Download PubMed dataset from HuggingFace."""
    output_path = os.path.join(DATA_DIR, "pubmed_filtered.parquet")
    if os.path.exists(output_path):
        print(f"[1/4] PubMed data already exists at {output_path}")
        return output_path

    print("[1/4] Downloading PubMed dataset from HuggingFace...")
    from datasets import load_dataset

    t0 = time.time()
    ds = load_dataset("brainchalov/pubmed_arxiv_abstracts_data", split="train")
    print(f"  Downloaded {len(ds):,} papers in {time.time()-t0:.0f}s")

    # Filter: keep only papers with substantial abstracts
    print("  Filtering empty/short abstracts...")
    filtered = ds.filter(lambda x: len(x["abstr"].strip()) > 100, num_proc=4)
    print(f"  After filtering: {len(filtered):,} papers")

    filtered.to_parquet(output_path)
    print(f"  Saved to {output_path}")
    return output_path


def prepare_documents(parquet_path):
    """Load parquet and prepare documents + metadata."""
    print("[2/4] Preparing documents...")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df):,} papers")

    documents = []
    ids = []
    metadatas = []
    abstracts = []

    for i, row in df.iterrows():
        pid = f"pubmed_{i}"
        title = str(row.get("title", ""))
        abstract = str(row.get("abstr", ""))
        authors = "N/A"  # Dataset doesn't have authors
        categories = str(row.get("field", ""))
        journal = str(row.get("journal", ""))

        doc = f"Title: {title}\nAuthors: {authors}\nCategories: {categories}\nJournal: {journal}\n\n{abstract}"
        documents.append(doc)
        ids.append(pid)
        abstracts.append(abstract)
        metadatas.append({
            "id": pid,
            "title": title,
            "authors": authors,
            "categories": categories,
            "year": 0,  # Not available in this dataset
            "source": "pubmed",
        })

        if (i + 1) % 50000 == 0:
            print(f"  Prepared {i+1:,}/{len(df):,} documents...")

    print(f"  Prepared {len(documents):,} documents total")
    return documents, ids, metadatas, abstracts


def embed_and_insert(documents, ids, metadatas):
    """Embed documents and insert into ChromaDB."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    print(f"[3/4] Embedding {len(documents):,} documents with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    t0 = time.time()
    all_embeddings = model.encode(
        documents,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - t0
    print(f"  Embedding done in {elapsed:.0f}s ({len(documents)/elapsed:.0f} docs/s)")

    # Insert into existing ChromaDB
    print(f"  Inserting into ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    existing = collection.count()
    print(f"  Existing documents: {existing:,}")

    t0 = time.time()
    total = len(documents)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=all_embeddings[start:end].tolist(),
        )
        done = end
        pct = 100 * done / total
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  Inserted {done:,}/{total:,} ({pct:.0f}%) | {rate:.0f} docs/s | ETA: {eta:.0f}s", end="\r")

    print(f"\n  ChromaDB insert done. New total: {collection.count():,} documents")


def rebuild_bm25(pubmed_abstracts, pubmed_ids):
    """Rebuild BM25 index combining arXiv + PubMed."""
    import bm25s

    print("[4/4] Rebuilding BM25 index with arXiv + PubMed...")

    # Load existing arXiv data
    arxiv_parquet = os.path.join(DATA_DIR, "arxiv_filtered.parquet")
    arxiv_df = pd.read_parquet(arxiv_parquet, columns=["id", "abstract"])
    print(f"  arXiv papers: {len(arxiv_df):,}")
    print(f"  PubMed papers: {len(pubmed_abstracts):,}")

    # Combine
    all_abstracts = [str(a).lower() for a in arxiv_df["abstract"]] + [str(a).lower() for a in pubmed_abstracts]
    all_ids = [str(aid) for aid in arxiv_df["id"]] + pubmed_ids
    print(f"  Total: {len(all_abstracts):,} papers")

    # Tokenize
    print("  Tokenizing...")
    t0 = time.time()
    corpus_tokens = bm25s.tokenize(all_abstracts, show_progress=True)
    print(f"  Tokenized in {time.time()-t0:.0f}s")

    # Build index
    print("  Building BM25 index...")
    t0 = time.time()
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens, show_progress=True)
    print(f"  Indexed in {time.time()-t0:.0f}s")

    # Save
    save_path = os.path.join(DATA_DIR, "bm25s_index")
    bm25.save(save_path)

    mapping = {i: aid for i, aid in enumerate(all_ids)}
    with open(os.path.join(DATA_DIR, "bm25_mapping.pkl"), "wb") as f:
        pickle.dump(mapping, f)

    print(f"  BM25 index saved ({len(all_ids):,} documents)")


def main():
    print("=" * 60)
    print("Adding PubMed papers to the Research Assistant")
    print("=" * 60)
    t_start = time.time()

    # Step 1: Download
    parquet_path = download_pubmed()

    # Step 2: Prepare
    documents, ids, metadatas, abstracts = prepare_documents(parquet_path)

    # Step 3: Embed + insert into ChromaDB
    embed_and_insert(documents, ids, metadatas)

    # Step 4: Rebuild BM25 with combined data
    rebuild_bm25(abstracts, ids)

    total_time = time.time() - t_start
    print("=" * 60)
    print(f"DONE in {total_time/60:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
