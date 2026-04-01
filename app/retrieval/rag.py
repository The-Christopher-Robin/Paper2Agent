"""FAISS-backed retrieval-augmented generation module.

Embeds document chunks via the OpenAI embeddings API (with a deterministic
hash-based fallback for offline / test use) and stores them in a FAISS
inner-product index for fast approximate nearest-neighbour search.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from config import Config

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore[import-untyped]
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    logger.warning("faiss-cpu not installed; falling back to numpy search")

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


class RAGRetriever:
    """Semantic search over chunked documents."""

    def __init__(self, index_path: str | None = None, dim: int = 1536):
        self.dim = dim
        self.index_path = Path(index_path or Config.FAISS_INDEX_PATH)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.documents: list[dict[str, Any]] = []
        self.embeddings: list[np.ndarray] = []

        if _HAS_FAISS:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None

        self._openai = (
            OpenAI(api_key=Config.OPENAI_API_KEY)
            if _HAS_OPENAI and Config.OPENAI_API_KEY
            else None
        )

        self._load_state()

    # ── Public API ───────────────────────────────────────────────────

    def add_document(
        self,
        text: str,
        metadata: dict | None = None,
        chunk_size: int = 800,
        overlap: int = 200,
    ) -> str:
        """Chunk *text*, embed each chunk, and append to the index."""
        doc_id = hashlib.sha256(text[:500].encode()).hexdigest()[:12]
        chunks = self._chunk_text(text, chunk_size, overlap)

        for i, chunk in enumerate(chunks):
            embedding = self._embed(chunk)
            if embedding is None:
                continue

            self.documents.append({
                "doc_id": doc_id,
                "chunk_index": i,
                "text": chunk,
                "metadata": metadata or {},
            })
            self.embeddings.append(embedding)

            if self.index is not None:
                vec = np.array([embedding], dtype=np.float32)
                faiss.normalize_L2(vec)
                self.index.add(vec)

        self._save_state()
        logger.info("Indexed document %s (%d chunks)", doc_id, len(chunks))
        return doc_id

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the *top_k* most relevant chunks for *query*."""
        if not self.documents:
            return []

        q_vec = self._embed(query)
        if q_vec is None:
            return []

        if self.index is not None and self.index.ntotal > 0:
            vec = np.array([q_vec], dtype=np.float32)
            faiss.normalize_L2(vec)
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(vec, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.documents):
                    continue
                doc = self.documents[idx].copy()
                doc["score"] = float(score)
                results.append(doc)
            return results

        return self._numpy_search(q_vec, top_k)

    # ── Embedding helpers ────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray | None:
        if self._openai:
            try:
                resp = self._openai.embeddings.create(
                    input=text[:8000],
                    model=Config.EMBEDDING_MODEL,
                )
                return np.array(resp.data[0].embedding, dtype=np.float32)
            except Exception as exc:
                logger.error("Embedding API call failed: %s", exc)
                return None

        return self._hash_embed(text)

    def _hash_embed(self, text: str) -> np.ndarray:
        """Deterministic hash-based embedding for offline / test use."""
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    # ── Fallback search ──────────────────────────────────────────────

    def _numpy_search(self, q_vec: np.ndarray, top_k: int) -> list[dict]:
        if not self.embeddings:
            return []

        matrix = np.array(self.embeddings, dtype=np.float32)
        q = np.array(q_vec, dtype=np.float32)

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        q_norm = q / (np.linalg.norm(q) or 1.0)

        scores = (matrix / norms) @ q_norm
        top_idx = np.argsort(scores)[-top_k:][::-1]

        return [
            {**self.documents[i], "score": float(scores[i])}
            for i in top_idx
        ]

    # ── Chunking ─────────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
        words = text.split()
        if len(words) <= size:
            return [text]

        chunks, start = [], 0
        while start < len(words):
            end = min(start + size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += size - overlap
        return chunks

    # ── Persistence ──────────────────────────────────────────────────

    def _save_state(self):
        try:
            (self.index_path / "documents.json").write_text(
                json.dumps(self.documents), encoding="utf-8"
            )
            if self.embeddings:
                np.save(
                    str(self.index_path / "embeddings.npy"),
                    np.array(self.embeddings),
                )
        except Exception as exc:
            logger.error("Failed to persist index: %s", exc)

    def _load_state(self):
        meta = self.index_path / "documents.json"
        emb = self.index_path / "embeddings.npy"
        if not meta.exists():
            return
        try:
            self.documents = json.loads(meta.read_text(encoding="utf-8"))
            if emb.exists():
                self.embeddings = list(np.load(str(emb)))
                if self.index is not None:
                    for e in self.embeddings:
                        vec = np.array([e], dtype=np.float32)
                        faiss.normalize_L2(vec)
                        self.index.add(vec)
            logger.info("Loaded %d documents from disk", len(self.documents))
        except Exception as exc:
            logger.error("Failed to load index: %s", exc)
