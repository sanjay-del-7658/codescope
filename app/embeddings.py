"""Pluggable embedding backends.

- st  : local sentence-transformers model (best quality; needs `pip install sentence-transformers`
        and internet once to download the model)
- api : any OpenAI-compatible /embeddings endpoint (e.g. Featherless, if it offers an embedding model)
- lsa : offline fallback — TF-IDF + truncated SVD (latent semantic analysis). Always available.
"""
from __future__ import annotations

import logging

import numpy as np

from . import config

log = logging.getLogger(__name__)


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1
    return (m / n).astype(np.float32)


class LSAEmbedder:
    name = "lsa"

    def __init__(self, dims: int = 256):
        self.dims = dims
        self.vec = None
        self.svd = None

    def fit(self, texts: list[str]):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from .text import tokenize
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                   sublinear_tf=True, min_df=1, max_features=50000)
        X = self.vec.fit_transform(texts)
        k = max(2, min(self.dims, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0)
        return _normalize(self.svd.fit_transform(X))

    def encode(self, texts: list[str]) -> np.ndarray:
        return _normalize(self.svd.transform(self.vec.transform(texts)))


class STEmbedder:
    name = "st"

    def __init__(self, model_name: str = config.ST_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]):
        return self.encode(texts, is_query=False)

    def encode(self, texts: list[str], is_query: bool = True) -> np.ndarray:
        if is_query and "bge" in self.model_name:
            texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
        return _normalize(np.asarray(self.model.encode(texts, batch_size=32, show_progress_bar=False)))

    def __getstate__(self):          # don't pickle the model weights
        return {"model_name": self.model_name}

    def __setstate__(self, s):
        self.__init__(s["model_name"])


class APIEmbedder:
    name = "api"

    def __init__(self, model: str = config.EMBED_API_MODEL):
        if not (model and config.LLM_API_KEY):
            raise RuntimeError("EMBED_API_MODEL and FEATHERLESS_API_KEY must be set for api embeddings")
        self.model = model

    def _call(self, texts: list[str]) -> np.ndarray:
        import httpx
        out = []
        for i in range(0, len(texts), 64):
            r = httpx.post(f"{config.LLM_BASE_URL}/embeddings",
                           headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
                           json={"model": self.model, "input": texts[i:i + 64]}, timeout=120)
            r.raise_for_status()
            out.extend(d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"]))
        return _normalize(np.asarray(out))

    def fit(self, texts):
        return self._call(texts)

    def encode(self, texts):
        return self._call(texts)


def make_embedder(backend: str | None = None):
    backend = backend or config.EMBED_BACKEND
    if backend in ("st", "auto"):
        try:
            return STEmbedder()
        except Exception as e:  # not installed / no network
            if backend == "st":
                raise
            log.info("sentence-transformers unavailable (%s); falling back", e)
    if backend in ("api", "auto") and config.EMBED_API_MODEL:
        try:
            return APIEmbedder()
        except Exception as e:
            if backend == "api":
                raise
            log.info("API embeddings unavailable (%s); falling back", e)
    return LSAEmbedder()
