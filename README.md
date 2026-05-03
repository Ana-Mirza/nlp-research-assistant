# Academic Research Assistance System

## Overview

RAG-based system that helps researchers find relevant papers, compare them with their research direction, identify gaps, and generate structured state-of-the-art reviews.

**NLP Final Project** — Master in Machine Learning for Health, UC3M (2025/2026).

## Architecture

```
User Query → Language Detection → Translation (NLLB-200) → Hybrid Retrieval → RRF Fusion → Cross-Encoder Re-ranking → LLM Generation → Translation Back → Response
```

| Component | Description |
|-----------|-------------|
| **Hybrid Retrieval** | Dense (ChromaDB + all-MiniLM-L6-v2) + Sparse (BM25S) |
| **Fusion** | Reciprocal Rank Fusion (k=60) |
| **Re-ranking** | cross-encoder/ms-marco-MiniLM-L6-v2 |
| **Generation** | UC3M Ollama API (llama3.1:8b, qwen3:8b, gemma3:4b) |
| **Translation** | facebook/nllb-200-distilled-600M (200+ languages) |
| **Frontend** | Streamlit |

## Features

### Core (Mandatory)

- Research direction analysis with retrieved papers
- Paper comparison (key differences table)
- No hallucination — refuses to answer when no relevant papers are found
- Multi-language support (query and response in user's language)

### Extensions

- State-of-the-art classification (thematic grouping + field evolution)
- Research gap detector (identifies unexplored areas)
- Paper methodology classification
- Per-paper auto-summaries
- BibTeX export for retrieved papers
- Similarity metrics with visual indicators (cosine similarity + cross-encoder scores)
- Multi-language translation (200+ languages via NLLB-200)

## Dataset

- **arXiv**: 388K papers from cs.CL, cs.AI, cs.LG (2018–present)
- **PubMed**: Extended with biomedical abstracts via `add_pubmed.py`

## Evaluation Results

| Metric | Score |
|--------|-------|
| Document Coverage | 100% (15/15 queries) |
| Avg Retrieval Score | 7.676 |
| Avg LLM Judge Score | 3.87/5 |

## Performance & Cold Start

On first launch, the app preloads all retrieval models into memory using `@st.cache_resource`:

| Component | Loaded At | Approx. Time |
|-----------|-----------|---------------|
| ChromaDB collection | Startup | ~2s |
| BM25 index | Startup | ~1s |
| Cross-encoder reranker | Startup | ~3s |
| SentenceTransformer embedder | Startup | ~2s |
| NLLB translation model | First non-English query | ~5s |

After the cold start, retrieval is fast (<1s). The main latency bottleneck is the **LLM API** — each call to UC3M Ollama takes 3–10s depending on response length. A typical search makes 6 LLM calls (1 analysis + 5 paper summaries).

## Setup

```bash
pip install -r requirements.txt
```

Set the API key (optional, has default):

```bash
export UC3M_API_KEY="your-key-here"
```

### Data Preparation

```bash
python download_data.py   # Download arXiv metadata
python ingest.py           # Build ChromaDB + BM25 indices
python add_pubmed.py       # (Optional) Add PubMed papers
```

### Run

```bash
streamlit run app.py
```

On **macOS**, if you get an `OMP: Error #15` (duplicate libiomp5), prefix with:

```bash
KMP_DUPLICATE_LIB_OK=TRUE streamlit run app.py
```

This is a known macOS issue when PyTorch and NumPy both link against OpenMP. It is not needed on Linux.

### Evaluate

```bash
python evaluate.py
```

## Project Structure

```
research_assistant/
├── app.py              # Streamlit frontend
├── rag.py              # RAG pipeline (research_assistant, compare, classify, gaps, SOTA, BibTeX)
├── retrieval.py        # Hybrid retrieval: dense + sparse + RRF + cross-encoder
├── llm.py              # UC3M Ollama API integration
├── translate.py        # Multi-language translation (NLLB-200)
├── config.py           # Configuration with parameter justifications
├── evaluate.py         # Evaluation suite (coverage, retrieval, LLM-as-Judge)
├── download_data.py    # arXiv dataset download
├── ingest.py           # Embedding + indexing pipeline
├── add_pubmed.py       # PubMed dataset extension
├── requirements.txt    # Python dependencies
└── data/               # Datasets and indices
```

## Technologies

| Component | Technology |
|-----------|------------|
| Vector DB | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Sparse Retrieval | BM25S |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L6-v2 |
| LLM | llama3.1:8b / qwen3:8b / gemma3:4b (UC3M Ollama) |
| Translation | facebook/nllb-200-distilled-600M |
| Frontend | Streamlit |

## Parameter Justification

All retrieval and generation parameters are documented in `config.py`. Key choices:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `TOP_K_DENSE` / `TOP_K_SPARSE` | 20 | Sufficiently large candidate pool for RRF fusion; increasing beyond 20 showed diminishing returns |
| `TOP_K_RERANK` | 5 | Balances comprehensiveness with readability; 5 abstracts ≈ 2000 tokens for the LLM context |
| `RRF_K` | 60 | Standard smoothing constant from Cormack et al. (2009); prevents top-ranked docs from dominating |
| `DEFAULT_TEMPERATURE` | 0.1 | Low temperature for factual, grounded responses; best LLM-as-Judge scores (3.87/5) among tested values 0.0–0.5 |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | STSB Spearman 0.8492; best speed/quality tradeoff for 388K corpus |
| `RERANKER_MODEL` | ms-marco-MiniLM-L6-v2 | Trained on MS MARCO passage ranking; adds ~200ms latency but significantly improves precision |
| `MIN_YEAR` | 2018 | Post-transformer era; papers before 2018 predate the architecture and are less relevant |
