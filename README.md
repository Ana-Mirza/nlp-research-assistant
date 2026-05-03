# Academic Research Assistance System

A RAG-based system that retrieves relevant academic papers and generates grounded analyses comparing existing work with your research direction. Built with a hybrid retrieval pipeline over **888K papers** from arXiv (CS/AI/ML) and PubMed (biomedical sciences).

**GitHub:** [REPO_URL]  
**Demo Video:** [VIDEO_URL]

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data and build indexes (~1 hour, one-time)
KMP_DUPLICATE_LIB_OK=TRUE python setup.py

# 3. Run the app
KMP_DUPLICATE_LIB_OK=TRUE streamlit run app.py
```

> **Note:** If you have the pre-built `data/` and `chroma_db/` folders (e.g., from a shared drive), place them in the project root and skip step 2.

## Architecture

```
User Query → Language Detection (langid) → NLLB Translation → English Query
  ├── Dense Retrieval (ChromaDB, all-MiniLM-L6-v2) → Top 20
  └── Sparse Retrieval (BM25s) → Top 20                        [parallel]
      └── Reciprocal Rank Fusion (k=60) → Merged candidates
          └── Cross-Encoder Re-ranking (ms-marco-MiniLM-L6-v2) → Top 5
              └── LLM Generation (llama3.1:8b via UC3M API) → Response
```

## Dataset

| Source | Papers | Categories | Years | Description |
|--------|--------|-----------|-------|-------------|
| arXiv | 388K | cs.CL, cs.AI, cs.LG | 2018+ | NLP, AI, and ML papers |
| PubMed | 500K | All fields | All | Biomedical and scientific papers |
| **Total** | **888K** | | | |

- arXiv data from [librarian-bots/arxiv-metadata-snapshot](https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot) on HuggingFace
- PubMed data from [brainchalov/pubmed_arxiv_abstracts_data](https://huggingface.co/datasets/brainchalov/pubmed_arxiv_abstracts_data) on HuggingFace

## Features

### Core (Mandatory)
- **Hybrid retrieval**: Dense (ChromaDB) + Sparse (BM25s) + RRF fusion + cross-encoder re-ranking
- **Grounded analysis**: LLM generates structured comparison with retrieved papers only
- **Hallucination mitigation**: Relevance threshold (score < 3.0 filtered), constrained prompting, source attribution
- **Multi-language**: Queries in 55+ languages via NLLB-200 translation for retrieval; LLM responds in the user's language
- **Streamlit frontend**: Interactive UI with settings sidebar, paper expanders, relevance indicators

### Additional (Grade > 8)
- **Paper comparison table**: Side-by-side comparison across methodology, datasets, contributions
- **State-of-the-art review**: Groups papers by theme and traces field evolution
- **Research gap detection**: Identifies unexplored areas based on retrieved literature
- **Methodology classification**: Categorizes papers by approach (supervised, unsupervised, etc.)
- **Per-paper summaries**: On-demand LLM summaries, progressively loaded without blocking UI
- **Similarity metrics**: Cosine similarity + cross-encoder scores with color-coded relevance (🟢 High / 🟠 Medium / 🔵 Low)
- **BibTeX export**: One-click export of retrieved papers for LaTeX workflows

## Project Structure

| File | Purpose |
|------|---------|
| `config.py` | All configuration constants with documented justifications |
| `download_data.py` | Downloads arXiv dataset, filters by category (cs.CL/AI/LG) and year (2018+) |
| `add_pubmed.py` | Downloads PubMed dataset, embeds, and adds to existing indexes |
| `ingest.py` | Embeds papers with sentence-transformers, builds ChromaDB + BM25s indexes |
| `retrieval.py` | Hybrid retrieval: parallel dense + BM25s, RRF fusion, cross-encoder re-ranking |
| `llm.py` | UC3M Ollama API integration (llama3.1:8b, qwen3:8b, gemma3:4b) |
| `translate.py` | Language detection (langid + langdetect fallback) and NLLB-200 translation |
| `rag.py` | Full RAG pipeline, prompt engineering, and additional feature functions |
| `app.py` | Streamlit frontend with progressive summary loading |
| `evaluate.py` | Evaluation: Precision@K, MRR, NDCG@K, citation faithfulness, cross-family LLM-as-Judge |
| `setup.py` | One-command setup: download, embed, index |
| `paper.tex` | 4-page academic report |

## Evaluation

Evaluated on 15 diverse research queries. Generator: `llama3.1:8b`, Judge: `qwen3:8b` (cross-family to avoid self-evaluation bias).

**Retrieval metrics:** Precision@5, MRR, NDCG@5, Document Coverage  
**Generation metrics:** Citation faithfulness (programmatic), LLM-as-Judge (1-5 scale)  
**Efficiency:** Response time (mean, median, min, max)

Run evaluation:
```bash
KMP_DUPLICATE_LIB_OK=TRUE python evaluate.py
```

Results saved to `data/evaluation_results.json`.

## Configuration

All parameters are in `config.py` with documented justifications. Key parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Embedding model | all-MiniLM-L6-v2 | Best speed/quality tradeoff for 888K corpus |
| Re-ranker | ms-marco-MiniLM-L6-v2 | MS MARCO-trained, +200ms for significant precision gain |
| Top-K dense/sparse | 20 | Sufficient candidate pool; diminishing returns beyond 20 |
| Top-K rerank | 5 | 5 abstracts ≈ 2K tokens; balances coverage and readability |
| RRF k | 60 | Standard from Cormack et al. (2009) |
| Temperature | 0.1 | Low for factual responses; best LLM-as-Judge scores |
| Relevance threshold | 3.0 | Papers below this are filtered; system refuses to answer if none pass |

## Requirements

- Python 3.10+
- ~14GB disk space (data + indexes)
- macOS with Apple Silicon recommended (MPS acceleration for embedding)
- Access to UC3M LLM API (yiyuan.tsc.uc3m.es)

## Dependencies

```
streamlit, chromadb, sentence-transformers, bm25s, requests,
datasets, transformers, langid, langdetect
```
