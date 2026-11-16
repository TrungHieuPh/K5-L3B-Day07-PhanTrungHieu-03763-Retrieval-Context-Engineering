"""Step 3 — retrieval evaluator: compare dense / BM25 / hybrid RRF on the SAME dev questions and index.

Run: python -m src.eval_retrieval                       (dev: all methods, used to choose)
     python -m src.eval_retrieval --split test          (report only: refused unless results/step5/frozen_config.json
                                                         exists; runs only the two frozen retrievers dense + bm25)

Conditions (identical for every method):
  corpus / index   strategy B chunks (by_article, 220 chunks) — same text in pgvector and chunks_B.json
  query            the ORIGINAL question text (no rewriting); filter = the user-provided `filter` field
  ranking          top-10 is kept; Hit@1, Evidence Hit@3, Hit@5, Hit@10, MRR@10 use the same hit definition
  context          starter pack_context(top-3, budget=4000 chars) = what one search would put in front of the model

Hit definition (same as Day 6 / step 1): a chunk covers a gold item if the chunk belongs to the gold article
AND contains the whole evidence after NFC + lowercase + whitespace collapse. A question is covered by a set of
chunks when EVERY required group (multi-hop: 2 groups) is covered by at least one chunk; a single-hop question
is one group of alternatives. Unanswerable questions are not in the denominator (BENCHMARK.md).
Gold is read only here, after retrieval, to score; it never reaches a retriever.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .day6_store import ROOT, Day6Embedder, Day6VectorStore
from .retrievers import BM25Retriever, DenseRetriever, HybridRRF, load_chunks

QUESTIONS = ROOT / "data" / "benchmark" / "questions.json"
CUTOFFS = (1, 3, 5, 10)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).lower()).strip()


def load_pack_context():
    starter = str(ROOT / "starter")
    if starter not in sys.path:
        sys.path.insert(0, starter)
    from baseline import pack_context  # type: ignore[import-not-found]
    return pack_context


def required_groups(q: dict[str, Any]) -> list[list[dict[str, Any]]]:
    return q["gold_groups"] if q.get("gold_groups") else [q["gold"]]


def chunk_covers(chunk: dict[str, Any], group: list[dict[str, Any]]) -> bool:
    articles = chunk["metadata"].get("articles") or [chunk["metadata"].get("article")]
    text = norm(chunk["text"])
    return any(g["article"] in articles and norm(g["evidence"]) in text for g in group)


def groups_covered(chunks: list[dict[str, Any]], groups) -> list[bool]:
    return [any(chunk_covers(c, g) for c in chunks) for g in groups]


def full_coverage_rank(results, groups, max_k: int) -> int | None:
    """Smallest k such that top-k covers every group (for single-hop = rank of the first hit)."""
    for k in range(1, min(max_k, len(results)) + 1):
        if all(groups_covered(results[:k], groups)):
            return k
    return None


def evaluate(method, questions, pack_context, *, budget: int, ctx_k: int = 3, max_k: int = 10):
    rows = []
    for q in questions:
        t = time.perf_counter()
        results = method.search(q["question"], top_k=max_k, filters=q["filter"])
        ms = 1000 * (time.perf_counter() - t)
        groups = required_groups(q)
        context, ctx_ids = pack_context(results[:ctx_k], budget=budget)
        ctx_chunks = [r for r in results[:ctx_k] if r["id"] in ctx_ids]
        rank = full_coverage_rank(results, groups, max_k)
        gold_articles = sorted({g["article"] for grp in groups for g in grp})
        rows.append({
            "query_id": q["id"], "type": q["type"], "question": q["question"], "filter": q["filter"],
            "n_groups": len(groups), "gold_articles": gold_articles,
            "results": [{"rank": r["rank"], "id": r["id"], "article": r["metadata"].get("article"),
                         "score": r["score"], "components": r.get("components"),
                         "hit": any(chunk_covers(r, g) for g in groups)} for r in results],
            **{f"hit@{k}": rank is not None and rank <= k for k in CUTOFFS},
            "full_coverage_rank": rank,
            "groups_covered@3": sum(groups_covered(results[:ctx_k], groups)),
            "context_ids": ctx_ids, "context_chars": len(context),
            "dropped_by_budget": [r["id"] for r in results[:ctx_k] if r["id"] not in ctx_ids],
            "context_evidence": all(groups_covered(ctx_chunks, groups)),
            "ms": round(ms, 1),
        })
    return rows


def summarize(name: str, rows: list[dict[str, Any]], extra: dict[str, Any]) -> dict[str, Any]:
    n = len(rows)
    frac = lambda key: {"n": sum(r[key] for r in rows), "N": n, "rate": round(sum(r[key] for r in rows) / n, 3)}
    multi = [r for r in rows if r["n_groups"] > 1]
    ms = sorted(r["ms"] for r in rows)
    return {
        "method": name, **extra, "N_answerable": n,
        "hit@1": frac("hit@1"), "evidence_hit@3": frac("hit@3"), "hit@5": frac("hit@5"), "hit@10": frac("hit@10"),
        "mrr@10": round(sum(1 / r["full_coverage_rank"] for r in rows if r["full_coverage_rank"]) / n, 3),
        "context_evidence_rate": frac("context_evidence"),
        "multi_hop_groups_covered@3": {"n": sum(r["groups_covered@3"] for r in multi),
                                       "N": sum(r["n_groups"] for r in multi)},
        "avg_context_chars": round(statistics.mean(r["context_chars"] for r in rows), 1),
        "questions_with_chunk_dropped_by_budget": sum(bool(r["dropped_by_budget"]) for r in rows),
        "latency_ms": {"median": round(statistics.median(ms), 1), "p90": round(ms[int(0.9 * (n - 1))], 1)},
        "cost_usd": 0.0,
        "misses@3": [r["query_id"] for r in rows if not r["hit@3"]],
    }


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description="Compare retrieval methods on DEV (original queries)")
    ap.add_argument("--strategy", default="B", choices=("A", "B", "C"))
    ap.add_argument("--budget", type=int, default=4000, help="context budget in characters (agent baseline)")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--rrf-candidates", type=int, default=20)
    ap.add_argument("--split", choices=("dev", "test"), default="dev",
                    help="test = reporting after the freeze only (frozen retrievers, written to results/step5)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    frozen = ROOT / "results" / "step5" / "frozen_config.json"
    if args.split == "test" and not (frozen.exists()
                                     and json.loads(frozen.read_text(encoding="utf-8")).get("frozen_before_test")):
        raise SystemExit("test retrieval is reported only after the configuration is frozen (results/step5)")
    if args.out is None:
        args.out = ROOT / "results" / ("step3" if args.split == "dev" else "step5")

    questions = [q for q in json.loads(QUESTIONS.read_text(encoding="utf-8")) if q["split"] == args.split]
    assert questions and all(q["split"] == args.split for q in questions)
    answerable = [q for q in questions if q["answerable"]]
    pack_context = load_pack_context()

    dense = DenseRetriever(Day6VectorStore(os.environ["DATABASE_URL"], Day6Embedder(), strategy=args.strategy))
    bm25 = BM25Retriever(load_chunks(args.strategy), strategy=args.strategy)
    hybrid = HybridRRF(dense, bm25, k=args.rrf_k, candidates=args.rrf_candidates)
    methods = {
        "dense": (dense, {"config": f"pgvector cosine, {dense.store.embedder.model_name}, strategy {args.strategy}"}),
        "bm25": (bm25, {"config": f"Okapi BM25 k1={bm25.k1} b={bm25.b}, syllable tokens, strategy {args.strategy}",
                        "index_seconds": round(bm25.index_seconds, 3)}),
        "hybrid_rrf": (hybrid, {"config": f"RRF k={hybrid.k} over dense top-{hybrid.candidates} + "
                                          f"BM25 top-{hybrid.candidates}, strategy {args.strategy}"}),
    }
    if args.split == "test":  # only what was frozen: baseline (dense) and selected config (bm25)
        methods = {k: v for k, v in methods.items() if k in ("dense", "bm25")}
    common = {"split": args.split, "query": "original question", "context": f"pack_context(top-3, {args.budget} chars)",
              "strategy": args.strategy}

    all_rows, summaries = {}, []
    for name, (m, extra) in methods.items():
        rows = evaluate(m, answerable, pack_context, budget=args.budget)
        all_rows[name] = rows
        summaries.append(summarize(name, rows, extra))

    # ---- print comparison ----
    pct = lambda d: f"{d['n']}/{d['N']} ({100 * d['rate']:.0f}%)"
    print(f"\n{args.split.upper()} retrieval, original queries, strategy {args.strategy}, N={len(answerable)} answerable "
          f"({len(questions) - len(answerable)} unanswerable excluded)\n")
    print(f"{'method':<11} {'Hit@1':>12} {'EvHit@3':>12} {'Hit@5':>12} {'Hit@10':>12} {'MRR@10':>7} "
          f"{'CtxEvid':>12} {'MH grp@3':>9} {'ctx chars':>9} {'med ms':>7}")
    for s in summaries:
        mh = s["multi_hop_groups_covered@3"]
        print(f"{s['method']:<11} {pct(s['hit@1']):>12} {pct(s['evidence_hit@3']):>12} {pct(s['hit@5']):>12} "
              f"{pct(s['hit@10']):>12} {s['mrr@10']:>7} {pct(s['context_evidence_rate']):>12} "
              f"{mh['n']}/{mh['N']:<7} {s['avg_context_chars']:>9} {s['latency_ms']['median']:>7}")
    print("\nPer question (rank at which all evidence is covered; '-' = not in top-10):")
    print(f"{'id':<6} {'type':<16} " + " ".join(f"{n:>10}" for n in methods))
    for i, q in enumerate(answerable):
        ranks = [all_rows[n][i]["full_coverage_rank"] for n in methods]
        mark = "" if len({r is not None and r <= 3 for r in ranks}) == 1 else "   <- differs @3"
        print(f"{q['id']:<6} {q['type']:<16} " + " ".join(f"{(r or '-'):>10}" for r in ranks) + mark)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"retrieval_{args.split}_summary.json").write_text(json.dumps(
        {"conditions": common, "methods": summaries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / f"retrieval_{args.split}_traces.json").write_text(json.dumps(
        {"conditions": common, "traces": all_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved -> {(args.out / f'retrieval_{args.split}_summary.json').relative_to(ROOT)}, "
          f"retrieval_{args.split}_traces.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
