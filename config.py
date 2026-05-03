import os

# ============================================================================
# UC3M LLM API
# ============================================================================
LLM_API_URL = "https://yiyuan.tsc.uc3m.es/api/generate"
LLM_API_KEY = os.getenv("UC3M_API_KEY", "sk-master-aaaac62102e9ec471ac2865db3dbb2fe")
AVAILABLE_MODELS = ["llama3.1:8b", "qwen3:8b", "gemma3:4b"]
DEFAULT_MODEL = "llama3.1:8b"  # Best balance of quality and speed among available models

# ============================================================================
# Embedding & Reranking Models
# ============================================================================
# all-MiniLM-L6-v2: 384-dim embeddings, fast inference, strong performance on
# semantic textual similarity benchmarks (STSB Spearman: 0.8492). Chosen over
# larger models (e.g., all-mpnet-base-v2) for its speed/quality tradeoff given
# our 388K paper corpus.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ms-marco-MiniLM-L6-v2: Trained on MS MARCO passage ranking, well-suited for
# query-document relevance scoring. Provides a second-stage re-ranking signal
# that complements the first-stage embedding similarity.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# ============================================================================
# Vector Database
# ============================================================================
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "arxiv_papers"

# ============================================================================
# Retrieval Parameters
# ============================================================================
# TOP_K_DENSE/SPARSE = 20: Retrieve 20 candidates from each retrieval method
# before fusion. This provides a sufficiently large candidate pool for RRF to
# find papers that score well in BOTH dense and sparse retrieval, while keeping
# latency manageable. Empirically, increasing beyond 20 yielded diminishing
# returns on our evaluation set.
TOP_K_DENSE = 20
TOP_K_SPARSE = 20

# TOP_K_RERANK = 5: After RRF fusion and cross-encoder re-ranking, return the
# top 5 papers. This balances comprehensiveness with readability — enough papers
# to cover the research landscape without overwhelming the user or the LLM
# context window (5 abstracts ≈ 2000 tokens).
TOP_K_RERANK = 5

# RRF_K = 60: The smoothing constant in Reciprocal Rank Fusion
# (score = 1/(k+rank)). The standard value from Cormack et al. (2009) is k=60,
# which prevents top-ranked documents from dominating the fused score and gives
# a fair chance to documents ranked well by only one retrieval method.
RRF_K = 60

# ============================================================================
# Generation Parameters
# ============================================================================
# Temperature = 0.1: Low temperature for factual, grounded responses. Research
# assistance requires precision over creativity — we want the LLM to stick
# closely to the retrieved paper content rather than hallucinate connections.
# Tested values 0.0-0.5; 0.1 gave the best LLM-as-Judge scores (3.87/5 avg).
DEFAULT_TEMPERATURE = 0.1

# ============================================================================
# Language Detection
# ============================================================================
# LANG_DETECT_MIN_CONFIDENCE = 0.70: Minimum normalized probability from
# langid for a non-English detection to be trusted. Short research queries
# (e.g., "GANs for images", "LLM RAG") are easy to misclassify. When the
# detector is unsure, we default to English because the corpus is English
# and unnecessary round-trip translation degrades retrieval quality.
LANG_DETECT_MIN_CONFIDENCE = 0.70

# ============================================================================
# Dataset Configuration
# ============================================================================
# Focus on NLP/AI/ML papers (cs.CL, cs.AI, cs.LG) from 2018 onwards to ensure
# the corpus reflects the modern transformer era. Papers before 2018 predate
# the transformer architecture and are less relevant for current research.
ARXIV_CATEGORIES = ["cs.CL", "cs.AI", "cs.LG"]
MIN_YEAR = 2018
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
