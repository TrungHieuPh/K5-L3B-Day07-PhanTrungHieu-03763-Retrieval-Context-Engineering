"""Connection layer to the Day 6 index: same embedder + the live pgvector table.

Dense retrieval in Day 7 goes through `Day6VectorStore.search`, which embeds the
query with the SAME model Day 6 used for documents and runs cosine search in
PostgreSQL + pgvector. Nothing is cached or faked: errors propagate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "corpus" / "index_manifest.json"
ALLOWED_FILTERS = {"article": "int", "chapter_roman": "text"}


def vec_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in vector) + "]"


class Day6Embedder:
    """Query encoder identical to Day 6 (sentence-transformers, L2-normalized, no prefix)."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, load on demand

        if model_name is None:
            model_name = json.loads(MANIFEST.read_text(encoding="utf-8"))["embedding"]["model"]
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dimensions = get_dim()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class Day6VectorStore:
    def __init__(self, database_url: str, embedder: Day6Embedder, strategy: str = "B") -> None:
        self.database_url = database_url
        self.embedder = embedder
        self.strategy = strategy
        self.dimensions = self._index_dims()
        if self.dimensions != embedder.dimensions:
            raise ValueError(f"query dim {embedder.dimensions} != index dim {self.dimensions}")

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url)

    def _index_dims(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT DISTINCT vector_dims(embedding) FROM chunks WHERE strategy = %s",
                        (self.strategy,))
            dims = [r[0] for r in cur.fetchall()]
        if len(dims) != 1:
            raise RuntimeError(f"strategy {self.strategy}: expected one embedding dim, got {dims}")
        return dims[0]

    def stored_vectors(self, chunk_ids: list[str]) -> dict[str, list[float]]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT chunk_id, embedding::text FROM chunks WHERE strategy = %s AND chunk_id = ANY(%s)",
                        (self.strategy, chunk_ids))
            return {cid: json.loads(vec) for cid, vec in cur.fetchall()}

    def search_vector(self, vector: list[float], *, top_k: int,
                      filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        filters = filters or {}
        unknown = set(filters) - ALLOWED_FILTERS.keys()
        if unknown:
            raise ValueError(f"unsupported filter keys: {sorted(unknown)}")
        clauses, params = ["strategy = %s"], [self.strategy]
        for key, cast in ALLOWED_FILTERS.items():
            if filters.get(key) is not None:
                clauses.append(f"{key} = %s::{cast}")
                params.append(filters[key])
        q = vec_literal(vector)
        sql = f"""SELECT chunk_id, document_id, source, page, text, metadata,
                         1 - (embedding <=> %s::vector) AS score
                  FROM chunks WHERE {' AND '.join(clauses)}
                  ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s"""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, [q, *params, q, top_k])
            rows = cur.fetchall()
        return [
            {"id": cid, "text": text, "source_id": doc, "source": src, "page": page,
             "metadata": meta, "score": float(score), "rank": rank, "method": "dense"}
            for rank, (cid, doc, src, page, text, meta, score) in enumerate(rows, start=1)
        ]

    def search(self, query: str, *, top_k: int,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self.search_vector(self.embedder.embed_query(query), top_k=top_k, filters=filters)
