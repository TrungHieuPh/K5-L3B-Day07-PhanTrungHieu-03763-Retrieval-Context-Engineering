"""Step 1c — prove the Day 7 query encoder is compatible with the Day 6 index.

Run from the repo root:  python -m src.check_embedding

Two checks, both against the live pgvector table:
  1. Re-embed a fixed sample of stored chunks; cosine(new, stored) must be ~1.0.
     -> same model, same normalization, same text as Day 6.
  2. Replay every Day 6 strategy-B query (same question, same filter) and compare
     the top-3 chunk IDs and scores with the Day 6 trace file.
     -> dense search in Day 7 reproduces Day 6 results.
Output: results/step1/embedding_check.json
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from .day6_store import ROOT, Day6Embedder, Day6VectorStore

SAMPLE_SIZE = 20
SEED = 7
COSINE_MIN = 0.9999
SCORE_TOL = 1e-3


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


def main() -> None:
    load_dotenv(ROOT / ".env")
    day6_root = Path(os.environ.get("DAY6_ROOT", "../K5-L3-Day06-PhanTrungHieu-03763-DataIngestionEngineering/starter"))
    day6_root = (day6_root if day6_root.is_absolute() else ROOT / day6_root).resolve()

    embedder = Day6Embedder()
    store = Day6VectorStore(os.environ["DATABASE_URL"], embedder, strategy="B")
    print(f"Embedder {embedder.model_name}: dim={embedder.dimensions}; index B dim={store.dimensions}")

    # 1. Re-embed a seeded sample of stored chunks.
    chunks = json.loads((ROOT / "data" / "corpus" / "chunks_B.json").read_text(encoding="utf-8"))
    sample = random.Random(SEED).sample(chunks, SAMPLE_SIZE)
    stored = store.stored_vectors([c["id"] for c in sample])
    fresh = embedder.embed([c["text"] for c in sample])
    cosines = {c["id"]: round(cosine(v, stored[c["id"]]), 6) for c, v in zip(sample, fresh)}
    min_cos = min(cosines.values())
    print(f"Check 1: re-embedded {len(sample)} chunks, min cosine vs stored = {min_cos:.6f}")

    # 2. Replay Day 6 strategy-B retrieval.
    trace_rows = [json.loads(line) for line in
                  (day6_root / "results" / "traces_B.jsonl").read_text(encoding="utf-8").splitlines() if line]
    replay, times = [], []
    for row in trace_rows:
        t = time.perf_counter()
        hits = store.search(row["question"], top_k=3, filters=row.get("filters_applied"))
        times.append(time.perf_counter() - t)
        old_ids = [h["chunk_id"] for h in row["hits"]]
        new_ids = [h["id"] for h in hits]
        max_diff = max((abs(h["score"] - o["score"]) for h, o in zip(hits, row["hits"])), default=0.0)
        replay.append({"question_id": row["question_id"], "filter": row.get("filters_applied"),
                       "day6_top3": old_ids, "day7_top3": new_ids, "same_ids": old_ids == new_ids,
                       "max_score_diff": round(max_diff, 6)})
    same = sum(r["same_ids"] for r in replay)
    score_ok = sum(r["max_score_diff"] <= SCORE_TOL for r in replay)
    print(f"Check 2: replayed {len(replay)} Day 6 queries: identical top-3 IDs {same}/{len(replay)}, "
          f"scores within {SCORE_TOL}: {score_ok}/{len(replay)}, "
          f"avg dense query {1000 * sum(times) / len(times):.1f} ms")

    passed = min_cos >= COSINE_MIN and same == len(replay)
    out = ROOT / "results" / "step1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "embedding_check.json").write_text(json.dumps({
        "embedding_model": embedder.model_name, "query_dim": embedder.dimensions,
        "index_dim": store.dimensions, "strategy": "B",
        "check1_reembed": {"sample_size": SAMPLE_SIZE, "seed": SEED, "threshold": COSINE_MIN,
                           "min_cosine": min_cos, "per_chunk": cosines},
        "check2_replay_day6": {"n": len(replay), "identical_top3": same, "score_tolerance": SCORE_TOL,
                               "scores_within_tolerance": score_ok,
                               "avg_query_ms": round(1000 * sum(times) / len(times), 1), "rows": replay},
        "passed": passed,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PASSED" if passed else "FAILED — see results/step1/embedding_check.json")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
