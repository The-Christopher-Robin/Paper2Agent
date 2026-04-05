"""pgvector-based retriever for PostgreSQL document storage.

Stores document chunks with embeddings in PostgreSQL using the pgvector
extension for similarity search.  Falls back gracefully when pgvector
is not available by delegating to FAISS-backed RAGRetriever.
"""

import hashlib
import json
import logging
from typing import Any

import numpy as np

from config import Config

logger = logging.getLogger(__name__)

try:
    from sqlalchemy import create_engine, text
    _HAS_SQLALCHEMY = True
except ImportError:
    _HAS_SQLALCHEMY = False

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


class PgVectorStore:
    """Vector store backed by PostgreSQL + pgvector extension.

    Provides the same public interface as RAGRetriever (search / add_document)
    so modules can swap between them transparently.
    """

    def __init__(self, database_url: str | None = None, dim: int = 1536):
        self.dim = dim
        self.database_url = database_url or Config.DATABASE_URL
        self._engine = None
        self._available = False

        self._openai = (
            OpenAI(api_key=Config.OPENAI_API_KEY)
            if _HAS_OPENAI and Config.OPENAI_API_KEY
            else None
        )

        if _HAS_SQLALCHEMY and "postgresql" in self.database_url:
            try:
                self._engine = create_engine(self.database_url, pool_pre_ping=True)
                self._ensure_pgvector()
                self._available = True
                logger.info("PgVectorStore initialised (dim=%d)", dim)
            except Exception as exc:
                logger.warning("pgvector unavailable, falling back: %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    def _ensure_pgvector(self):
        with self._engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS pgvector_docs (
                    id SERIAL PRIMARY KEY,
                    doc_id VARCHAR(64),
                    chunk_index INTEGER DEFAULT 0,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    embedding vector({self.dim})
                )
            """))
            conn.commit()

    def add_document(
        self,
        text_content: str,
        metadata: dict | None = None,
        chunk_size: int = 800,
        overlap: int = 200,
    ) -> str:
        doc_id = hashlib.sha256(text_content[:500].encode()).hexdigest()[:12]
        chunks = self._chunk_text(text_content, chunk_size, overlap)

        with self._engine.connect() as conn:
            for i, chunk in enumerate(chunks):
                embedding = self._embed(chunk)
                if embedding is None:
                    continue
                vec_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
                conn.execute(
                    text("""
                        INSERT INTO pgvector_docs (doc_id, chunk_index, content, metadata, embedding)
                        VALUES (:doc_id, :idx, :content, :meta, :embedding)
                    """),
                    {
                        "doc_id": doc_id,
                        "idx": i,
                        "content": chunk,
                        "meta": json.dumps(metadata or {}),
                        "embedding": vec_str,
                    },
                )
            conn.commit()

        logger.info("PgVector indexed doc %s (%d chunks)", doc_id, len(chunks))
        return doc_id

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        q_vec = self._embed(query)
        if q_vec is None:
            return []

        vec_str = "[" + ",".join(str(float(x)) for x in q_vec) + "]"

        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"""
                    SELECT doc_id, chunk_index, content, metadata,
                           1 - (embedding <=> :vec::vector) AS score
                    FROM pgvector_docs
                    ORDER BY embedding <=> :vec::vector
                    LIMIT :k
                """),
                {"vec": vec_str, "k": top_k},
            ).fetchall()

        results = []
        for row in rows:
            meta = row[3] if isinstance(row[3], dict) else json.loads(row[3] or "{}")
            results.append({
                "doc_id": row[0],
                "chunk_index": row[1],
                "text": row[2],
                "metadata": meta,
                "score": float(row[4]) if row[4] is not None else 0.0,
            })
        return results

    def _embed(self, text_content: str) -> np.ndarray | None:
        if self._openai:
            try:
                resp = self._openai.embeddings.create(
                    input=text_content[:8000],
                    model=Config.EMBEDDING_MODEL,
                )
                return np.array(resp.data[0].embedding, dtype=np.float32)
            except Exception as exc:
                logger.error("Embedding API call failed: %s", exc)
                return None
        return self._hash_embed(text_content)

    def _hash_embed(self, text_content: str) -> np.ndarray:
        seed = int(hashlib.md5(text_content.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    @staticmethod
    def _chunk_text(text_content: str, size: int, overlap: int) -> list[str]:
        words = text_content.split()
        if len(words) <= size:
            return [text_content]
        chunks, start = [], 0
        while start < len(words):
            end = min(start + size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += size - overlap
        return chunks
