"""Step 3 — retrieval methods beyond dense: BM25 (lexical) and hybrid dense+BM25 via RRF.

All retrievers share one interface and one result format (same as Day6VectorStore.search):
  search(query, *, top_k, filters) -> [{id, text, source_id, source, page, metadata, score, rank, method, ...}]

  DenseRetriever   Day 6 pgvector cosine search (live PostgreSQL), unchanged.
  BM25Retriever    Okapi BM25 over the SAME exported chunks (data/corpus/chunks_{strategy}.json), in memory.
  HybridRRF        Reciprocal Rank Fusion of dense and BM25 candidate lists (rank-based, no score mixing).

Filters are the same user-provided metadata filters as Day 6 (article, chapter_roman). No gold is used here.
"""
from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections import Counter
from typing import Any

from .day6_store import ALLOWED_FILTERS, ROOT, Day6VectorStore

CORPUS = ROOT / "data" / "corpus"


def tokenize(text: str) -> list[str]:
    """Vietnamese syllable tokens: NFC + casefold, split on non-word chars (e.g. 'hợp đồng' -> ['hợp', 'đồng'])."""
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def check_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    filters = {k: v for k, v in (filters or {}).items() if v is not None}
    unknown = set(filters) - ALLOWED_FILTERS.keys()
    if unknown:
        raise ValueError(f"unsupported filter keys: {sorted(unknown)}")
    return filters


def load_chunks(strategy: str) -> list[dict[str, Any]]:
    return json.loads((CORPUS / f"chunks_{strategy}.json").read_text(encoding="utf-8"))


class DenseRetriever:
    """Thin wrapper so dense has the same name/shape as the others."""

    name = "dense"

    def __init__(self, store: Day6VectorStore) -> None:
        self.store = store
        self.strategy = store.strategy

    def search(self, query: str, *, top_k: int, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self.store.search(query, top_k=top_k, filters=check_filters(filters) or None)


class BM25Retriever:
    """Okapi BM25 (Lucene-style idf, always >= 0). k1 = term-frequency saturation, b = length normalization."""

    name = "bm25"

    def __init__(self, chunks: list[dict[str, Any]], *, k1: float = 1.2, b: float = 0.75, strategy: str = "B") -> None:
        t = time.perf_counter()
        self.chunks, self.k1, self.b, self.strategy = chunks, k1, b, strategy
        self.tfs = [Counter(tokenize(c["text"])) for c in chunks]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lens) / len(self.lens)
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}
        self.index_seconds = time.perf_counter() - t

    def _allowed(self, chunk: dict[str, Any], filters: dict[str, Any]) -> bool:
        meta = chunk["metadata"]
        return all((str(meta.get(k)) == str(v)) if k == "chapter_roman" else
                   (int(v) in (meta.get("articles") or [meta.get("article")])) for k, v in filters.items())

    def score(self, query_terms: list[str], i: int) -> float:
        tf, norm = self.tfs[i], self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg_len)
        return sum(self.idf[q] * tf[q] * (self.k1 + 1) / (tf[q] + norm) for q in query_terms if q in tf)

    def search(self, query: str, *, top_k: int, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        filters = check_filters(filters)
        terms = tokenize(query)  # repeated query terms count once per occurrence, as in Okapi BM25
        scored = [(self.score(terms, i), c["id"], i) for i, c in enumerate(self.chunks) if self._allowed(c, filters)]
        scored = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))[:top_k]
        out = []
        for rank, (s, _, i) in enumerate(scored, start=1):
            c = self.chunks[i]
            out.append({"id": c["id"], "text": c["text"], "source_id": c["source_id"], "source": c["source"],
                        "page": c["page"], "metadata": c["metadata"], "score": round(s, 4), "rank": rank,
                        "method": "bm25"})
        return out


class HybridRRF:
    """RRF: score(d) = sum over retrievers of 1 / (k + rank_r(d)); only ranks are fused, so the very
    different score scales of cosine and BM25 never get added together."""

    name = "hybrid_rrf"

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, *, k: int = 60, candidates: int = 20) -> None:
        if dense.strategy != bm25.strategy:
            raise ValueError("dense and BM25 must index the same chunk strategy")
        self.dense, self.bm25, self.k, self.candidates = dense, bm25, k, candidates
        self.strategy = dense.strategy

    def search(self, query: str, *, top_k: int, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        n = max(self.candidates, top_k)
        lists = {"dense": self.dense.search(query, top_k=n, filters=filters),
                 "bm25": self.bm25.search(query, top_k=n, filters=filters)}
        fused: dict[str, dict[str, Any]] = {}
        for name, results in lists.items():
            for r in results:
                item = fused.setdefault(r["id"], {**{k: r[k] for k in ("id", "text", "source_id", "source",
                                                                          "page", "metadata")},
                                                  "score": 0.0, "components": {"dense": None, "bm25": None}})
                item["score"] += 1 / (self.k + r["rank"])
                item["components"][name] = {"rank": r["rank"], "score": round(r["score"], 4)}
        ranked = sorted(fused.values(), key=lambda d: (-d["score"], d["id"]))[:top_k]
        for rank, d in enumerate(ranked, start=1):
            d.update(rank=rank, score=round(d["score"], 6), method="hybrid_rrf")
        return ranked
