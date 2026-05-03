# Pipeline Documentation

## System Overview

The Academic Research Assistance System is a RAG (Retrieval-Augmented Generation) pipeline that helps researchers explore how their work relates to existing literature. Given a research direction in any of 55+ languages, it retrieves relevant papers from a corpus of 888K academic papers and generates a grounded analysis.

---

## Data Sources

### arXiv (388,100 papers)
- **Source**: `librarian-bots/arxiv-metadata-snapshot` on HuggingFace
- **Filtering**: Categories `cs.CL` (NLP), `cs.AI` (AI), `cs.LG` (Machine Learning), year ≥ 2018
- **Fields**: id, title, authors, abstract, categories, update_date
- **Script**: `download_data.py`

### PubMed (500,318 papers)
- **Source**: `brainchalov/pubmed_arxiv_abstracts_data` on HuggingFace
- **Filtering**: Abstracts > 100 characters (no category filtering — intentionally broad for interdisciplinary coverage)
- **Fields**: title, abstract, journal, field
- **Script**: `add_pubmed.py`
- **Limitation**: No year metadata, no category filtering, possible overlap with arXiv subset

---

## Indexing Pipeline

### Embedding (ingest.py)
- **Model**: `all-MiniLM-L6-v2` (384 dimensions, STSB Spearman: 0.8492)
- **Method**: Batch encoding with `sentence-transformers` (batch_size=256), using Apple MPS GPU acceleration
- **Storage**: ChromaDB PersistentClient with HNSW cosine similarity index
- **Pre-computed**: Embeddings are generated once during ingestion and passed directly to ChromaDB — no re-embedding at query time
- **Performance**: ~230 docs/s on Apple M3 Pro

### BM25 Index (ingest.py / add_pubmed.py)
- **Library**: `bm25s` (numpy-vectorized, 10x faster than `rank_bm25`)
- **Tokenization**: `bm25s.tokenize()` — consistent between build-time and query-time
- **Storage**: Saved to disk via `bm25.save()`, loaded lazily on first query
- **Mapping**: Pickle file mapping index positions to paper IDs

---

## Retrieval Pipeline (retrieval.py)

### Stage 1: Parallel Candidate Retrieval
Two retrieval methods run **in parallel** using `ThreadPoolExecutor`:

**Dense Retrieval (ChromaDB)**
- Query is embedded with `all-MiniLM-L6-v2`
- HNSW approximate nearest neighbor search returns top 20 candidates
- Captures semantic similarity (synonyms, paraphrases)

**Sparse Retrieval (BM25s)**
- Query is tokenized with `bm25s.tokenize()`
- BM25 scoring returns top 20 candidates
- Captures exact keyword matches (specific terms, acronyms, method names)

**Why hybrid?** Dense retrieval finds semantically similar papers but can miss exact terminology. BM25 catches specific terms but lacks semantic understanding. Together they cover both cases.

### Stage 2: Reciprocal Rank Fusion (RRF)
- Merges dense and sparse results using RRF with k=60 (Cormack et al., 2009)
- Score: `RRF(d) = Σ 1/(k + rank_i)` across both retrievers
- No score calibration needed — works purely on rank positions
- Papers appearing in both result sets get boosted

### Stage 3: Cross-Encoder Re-ranking
- **Model**: `cross-encoder/ms-marco-MiniLM-L6-v2` (trained on MS MARCO passage ranking)
- Scores each (query, paper) pair jointly — more accurate than independent embeddings
- Top 5 papers selected from the fused candidates
- Adds ~500ms latency but significantly improves precision

### Stage 4: Relevance Filtering
- Papers with cross-encoder score < 3.0 are removed individually
- If no papers pass the threshold, the system returns a "no relevant articles" message
- Prevents showing irrelevant results to the user

---

## Language Pipeline (translate.py)

### Query Translation (Input)
- **Detection**: `langid` as primary detector, `langdetect` as fallback for short Romance language texts (ro/fr/it/es confusion)
- **Translation**: NLLB-200-distilled-600M (Meta) translates non-English queries to English for retrieval
- **Why needed**: The corpus and embedding model are English-only. Without translation, non-English queries get poor retrieval results
- **Overhead**: ~0.5-1s per query translation

### Response Language (Output)
- **Method**: LLM generates directly in the target language via explicit system prompt instruction ("You MUST respond in Romanian")
- **Why not NLLB?**: NLLB mangles markdown formatting (tables, bullet points, headers). The LLM produces better structured multilingual output
- **Reinforcement**: Language instruction appears in both system prompt (start) and user prompt (end) for maximum compliance
- **Supported**: 55+ languages for detection/translation, UI labels for en/es/fr/ro/de

### UI Localization
- Section headers, button labels, and placeholders are translated via a `UI_LABELS` dictionary
- Language is set based on the detected query language
- Falls back to English for unsupported UI languages

---

## Generation Pipeline (rag.py)

### Main Analysis
- **LLM**: UC3M Ollama API (`llama3.1:8b` default, `qwen3:8b` and `gemma3:4b` available)
- **Temperature**: 0.1 (low for factual, grounded responses)
- **System prompt** enforces:
  - Only reference provided papers
  - Never fabricate citations
  - Structured output: (1) Overview, (2) Key differences, (3) Gaps/opportunities
  - Concise (max 300 words)
  - Respond in detected language
- **Hallucination mitigation**: Relevance threshold + constrained prompting + source attribution

### Additional Features

#### Paper Comparison
- Produces a markdown table comparing each paper against the user's research direction
- Columns: Paper | Methodology | Datasets | Key Contribution | Relation to Direction
- Uses **full abstract** for each paper as context
- Explicit paper count in prompt to prevent hallucinated extra papers

#### Methodology Classification
- Categorizes retrieved papers by approach (supervised, unsupervised, RL, theoretical, etc.)
- Uses **full abstract** for accurate classification
- Output as bullet points

#### State-of-the-Art Review
- Groups papers into thematic clusters
- Traces field evolution using publication years
- Identifies dominant methodologies and paradigms
- Uses **full abstract** for each paper

#### Research Gap Detection
- Identifies aspects of the user's direction not covered by existing papers
- Suggests unexplored combinations of methods/datasets
- Highlights limitations that create opportunities
- Uses **full abstract** for each paper

#### Per-Paper Summaries
- 2-3 sentence summaries highlighting key contributions and methodology
- Generated on-demand using the LLM
- **Progressive loading**: Results display immediately, summaries fill in one-by-one without blocking the UI
- Uses title + authors + **full abstract**
- Cached in session state — won't regenerate on page refresh

#### Similarity Metrics
- **Cross-encoder score**: Re-ranker relevance score (0-10 scale)
- **Cosine similarity**: Embedding-space similarity between query and paper (0-1 scale)
- **Color-coded relevance indicator**: 🟢 High (≥7.0) | 🟠 Medium (4.0-6.99) | 🔵 Low (<4.0)

#### BibTeX Export
- One-click download of retrieved papers as a `.bib` file
- Includes title, authors, year, and URL (arXiv or Google Scholar)
- Directly usable in LaTeX workflows

#### Multi-Source Paper Links
- arXiv papers link to `https://arxiv.org/abs/{id}`
- PubMed papers link to Google Scholar search for the title
- Source badge shows "📎 arXiv Link" or "📎 PubMed Link"

---

## Evaluation Pipeline (evaluate.py)

### Retrieval Metrics
| Metric | What It Measures | How |
|--------|-----------------|-----|
| **Precision@5** | Fraction of top-5 papers that are relevant | Cross-encoder score > 5.0 as relevance proxy |
| **MRR** | Rank of first highly-relevant paper | First paper with score > 7.0 |
| **NDCG@5** | Ranking quality with graded relevance | Cross-encoder scores normalized to [0,1] |
| **Document Coverage** | Can the system find something for any topic? | Keyword match in title/abstract |

### Generation Metrics
| Metric | What It Measures | How |
|--------|-----------------|-----|
| **Citation Faithfulness** | Are cited papers real? | Programmatic check: quoted titles exist in retrieved set |
| **LLM-as-Judge** | Overall answer quality | Cross-family: `qwen3:8b` judges `llama3.1:8b` outputs (avoids self-evaluation bias) |

### Design Choices
- **Cross-encoder as relevance proxy**: Trained on MS MARCO for relevance prediction, avoids need for manual annotations
- **Cross-family judge**: Different model family for evaluation vs generation to mitigate self-preference bias
- **Score-before-justify**: Judge outputs only a number first, avoiding the score-then-justify antipattern (Turpin et al., 2023)
- **15 diverse test queries**: Spanning NLP, CV, RL, biomedical AI, and cross-disciplinary topics

---

## Configuration (config.py)

All parameters are documented with justifications in `config.py`. Key decisions:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Embedding model | all-MiniLM-L6-v2 | Best speed/quality for 888K corpus |
| Re-ranker | ms-marco-MiniLM-L6-v2 | MS MARCO-trained, +500ms for significant precision gain |
| Top-K dense/sparse | 20 | Sufficient candidate pool; diminishing returns beyond 20 |
| Top-K rerank | 5 | 5 abstracts ≈ 2K tokens; balances coverage and readability |
| RRF k | 60 | Standard from Cormack et al. (2009) |
| Temperature | 0.1 | Low for factual responses; best LLM-as-Judge scores |
| Relevance threshold | 3.0 | Individual paper filter; below this = not shown |
| BM25 library | bm25s | 10x faster than rank_bm25 for 888K corpus |

---

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Retrieval (parallel dense + BM25) | ~1s | After model warmup |
| Cross-encoder re-ranking (20 pairs) | ~0.5s | |
| LLM generation | 2-15s | Varies with UC3M server load |
| Total query (warm) | ~4-20s | Dominated by LLM latency |
| Cold start (first query) | ~15-30s | Loading embedding models + BM25 index |
| Embedding 888K papers | ~65 min | One-time, Apple M3 Pro MPS |
| BM25 index build | ~2 min | One-time |
