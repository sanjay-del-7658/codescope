"""Central configuration. Everything can be overridden with environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("CODESCOPE_DATA", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Ingestion -------------------------------------------------------------
MAX_FILE_KB = int(os.getenv("MAX_FILE_KB", "500"))

EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "bower_components", "vendor",
    "venv", ".venv", "env", ".env", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".tox", ".nox", "dist", "build", "out", "target", "bin", "obj", ".next",
    ".nuxt", ".cache", "coverage", "htmlcov", ".idea", ".vscode", ".gradle",
    "site-packages", ".eggs", "generated", "__generated__",
}
EXCLUDED_FILE_PATTERNS = [
    "*.min.js", "*.min.css", "*.map", "*.lock", "package-lock.json", "yarn.lock",
    "pnpm-lock.yaml", "poetry.lock", "*.pb.go", "*_pb2.py", "*.generated.*",
    "*.d.ts", "*.bundle.js",
]

LANG_BY_EXT = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
}

# --- Chunking --------------------------------------------------------------
MAX_CHUNK_LINES = 120       # long functions are split into windows of this size
CHUNK_OVERLAP = 20
MODULE_WINDOW = 60          # top-level code between symbols is chunked in windows

# --- Embeddings ------------------------------------------------------------
# auto | st (sentence-transformers) | api (OpenAI-compatible endpoint) | lsa (offline fallback)
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "auto")
ST_MODEL = os.getenv("ST_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_API_MODEL = os.getenv("EMBED_API_MODEL", "")

# --- LLM (Featherless, OpenAI-compatible) ----------------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.featherless.ai/v1")
LLM_API_KEY = os.getenv("FEATHERLESS_API_KEY", os.getenv("LLM_API_KEY", ""))
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

# --- Ranking ---------------------------------------------------------------
FUSION = os.getenv("FUSION", "linear")      # linear | rrf
RRF_K = 60
# Weights per embedding backend: a weak offline LSA embedder gets less say than a real model.
WEIGHTS = {
    "lsa": {"bm25": 1.0, "dense": 0.3, "symbol": 0.25},
    "st":  {"bm25": 0.7, "dense": 1.0, "symbol": 0.1},
    "api": {"bm25": 1.0, "dense": 0.8, "symbol": 0.3},
}


def weights_for(backend: str) -> dict:
    return WEIGHTS.get(backend, WEIGHTS["lsa"])
CANDIDATES_PER_RETRIEVER = 100
