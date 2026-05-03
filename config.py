import os

# UC3M LLM API
LLM_API_URL = "https://yiyuan.tsc.uc3m.es/api/generate"
LLM_API_KEY = "sk-master-aaaac62102e9ec471ac2865db3dbb2fe"
AVAILABLE_MODELS = ["llama3.1:8b", "qwen3:8b", "gemma3:4b"]
DEFAULT_MODEL = "llama3.1:8b"

# Retrieval
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "arxiv_papers"

# Retrieval params
TOP_K_DENSE = 20
TOP_K_SPARSE = 20
TOP_K_RERANK = 5
RRF_K = 60

# Generation params
DEFAULT_TEMPERATURE = 0.1

# Dataset
ARXIV_CATEGORIES = ["cs.CL", "cs.AI", "cs.LG"]
MIN_YEAR = 2018
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
