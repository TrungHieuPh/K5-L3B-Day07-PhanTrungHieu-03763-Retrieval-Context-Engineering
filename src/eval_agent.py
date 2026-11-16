"""Steps 4-5 — automatic metrics for agent runs (no model calls; gold is read only here, never by the agent).

Run: python -m src.eval_agent results/step4/baseline_dense_B_dev_all.json results/step4/bm25_B_dev_all.json
     python -m src.eval_agent --split test results/step5/baseline_dense_B_test_all.json \
                                           results/step5/bm25_excerpt_B_test_all.json   (only after freezing)

Per config (same 30 dev questions, same model / prompt / top-k / budget / max searches):
  status            answered / abstain / error (provider errors are reported, not scored)
  tool use          questions with >= 1 search / successful runs; distribution of searches per question
  answerable (27)   Context Evidence Rate  = every required evidence group is in the FINAL context TEXT sent to the
                                             model (checked on the text as sent, so an excerpt that cut the evidence
                                             does not count)
                    Ever retrieved         = every group appeared in some search result (before budget packing)
                    Cited evidence (proxy) = answered AND the cited blocks (as sent) contain every required group
                    False abstention       = abstained although answerable
  unanswerable (3)  Abstention             = abstained / 3
  cost / time       tokens, model calls, time without rate-limit waiting, $ (free tier = 0)

"Cited evidence" is an automatic PROXY: a cited chunk containing the evidence does not prove the answer text is
right. Answer grades (0/1/2) and citation support are read by a human (step 5, test); for dev the answers are
printed side by side with the expected answer for error analysis.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from .day6_store import ROOT
from .eval_retrieval import QUESTIONS, chunk_covers, required_groups
from .retrievers import load_chunks


HEADER = re.compile(r"^\[([^\s|\]]+) \| [^\n]*\]$", re.M)


def final_context(trace) -> tuple[list[str], str]:
    searches = [e for e in trace["events"] if e["event"] == "search"]
    return (searches[-1]["context_ids"], searches[-1]["context"]) if searches else ([], "")


def context_blocks(context: str) -> dict[str, str]:
    """chunk id -> the text of that chunk AS SENT to the model (may be an excerpt in config 2)."""
    heads = list(HEADER.finditer(context))
    return {m.group(1): context[m.end(): heads[k + 1].start() if k + 1 < len(heads) else len(context)]
            for k, m in enumerate(heads)}


def sent_covers(blocks: dict[str, str], chunks_by_id, group) -> bool:
    """Same hit rule as retrieval, but on the text the model actually saw (whole chunk or excerpt)."""
    for cid, text in blocks.items():
        meta = chunks_by_id[cid]["metadata"]
        if chunk_covers({"metadata": meta, "text": text}, group):
            return True
    return False


def evaluate_run(run, gold_by_id, chunks_by_id):
    rows = []
    for t in run["traces"]:
        q = gold_by_id[t["query_id"]]
        row = {"id": q["id"], "type": q["type"], "answerable": q["answerable"], "status": t["status"],
               "decision": t.get("final_decision"), "n_searches": t["n_searches"],
               "queries": [c["query"] for c in t["retriever_calls"]],
               "answer": t["answer"], "abstain_reason": t.get("abstain_reason"), "citations": t["citations"],
               "expected": q.get("expected_answer"), "tokens": t["tokens"], "n_model_calls": t["n_model_calls"],
               "active_s": round((t["elapsed_ms"] / 1000) - t.get("wait_s", 0), 1), "wait_s": t.get("wait_s", 0),
               "error": t["error"]}
        if q["answerable"] and t["status"] != "error":
            groups = required_groups(q)
            ctx_ids, ctx_text = final_context(t)
            blocks = context_blocks(ctx_text)
            assert list(blocks) == ctx_ids, (t["query_id"], list(blocks), ctx_ids)
            seen = {r["id"] for e in t["events"] if e["event"] == "search" for r in e["results"]}
            cited = {i: blocks[i] for i in t["citations"] if i in blocks}  # the cited text as the model saw it
            cover = lambda chs: all(any(chunk_covers(c, g) for c in chs) for g in groups)
            row.update(context_evidence=all(sent_covers(blocks, chunks_by_id, g) for g in groups),
                       ever_retrieved=cover([chunks_by_id[i] for i in seen]),
                       cited_evidence=t["status"] == "answered" and all(sent_covers(cited, chunks_by_id, g)
                                                                         for g in groups),
                       gold_articles=sorted({g["article"] for grp in groups for g in grp}),
                       final_context_articles=[chunks_by_id[i]["metadata"]["article"] for i in ctx_ids],
                       context_chars=len(ctx_text))
        rows.append(row)
    return rows


def summarize(name, cfg, rows):
    ok = [r for r in rows if r["status"] != "error"]
    ans = [r for r in ok if r["answerable"]]
    una = [r for r in ok if not r["answerable"]]
    nN = lambda n, N: {"n": n, "N": N, "rate": round(n / N, 3) if N else None}
    return {
        "config": name, "settings": cfg, "questions": len(rows),
        "status": dict(Counter(r["status"] for r in rows)),
        "provider_errors": [r["id"] for r in rows if r["status"] == "error"],
        "tool_use": nN(sum(r["n_searches"] > 0 for r in ok), len(ok)),
        "searches_per_question": dict(sorted(Counter(r["n_searches"] for r in ok).items())),
        "context_evidence_rate": nN(sum(r["context_evidence"] for r in ans), len(ans)),
        "ever_retrieved": nN(sum(r["ever_retrieved"] for r in ans), len(ans)),
        "cited_evidence_proxy": nN(sum(r["cited_evidence"] for r in ans), len(ans)),
        "false_abstention": nN(sum(r["status"] == "abstain" for r in ans), len(ans)),
        "abstention_unanswerable": nN(sum(r["status"] == "abstain" for r in una), len(una)),
        "tokens_total": {k: sum(r["tokens"][k] for r in ok) for k in ("prompt_tokens", "output_tokens",
                                                                       "thinking_tokens")},
        "tokens_per_question": round(statistics.mean(r["tokens"]["prompt_tokens"] + r["tokens"]["output_tokens"]
                                                     + r["tokens"]["thinking_tokens"] for r in ok), 1),
        "model_calls_total": sum(r["n_model_calls"] for r in ok),
        "active_s_median": round(statistics.median(r["active_s"] for r in ok), 2),
        "wait_s_total": round(sum(r["wait_s"] for r in ok), 1),
        "cost_usd": 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Automatic metrics for agent traces (dev, or test after freezing)")
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--split", choices=("dev", "test"), default="dev",
                    help="test only for runs made after the configuration was frozen (results/step5)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = ROOT / "results" / ("step4/agent_dev_summary.json" if args.split == "dev"
                                       else "step5/agent_test_auto_summary.json")

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    gold_by_id = {q["id"]: q for q in questions if q["split"] == args.split}
    summaries, per_config = [], {}
    for path in args.runs:
        run = json.loads(path.read_text(encoding="utf-8"))
        if any(t["split"] != args.split for t in run["traces"]):
            raise SystemExit(f"{path}: contains traces that are not split={args.split}")
        cfg = run["config"]
        chunks_by_id = {c["id"]: c for c in load_chunks(cfg["strategy"])}
        rows = evaluate_run(run, gold_by_id, chunks_by_id)
        per_config[cfg["name"]] = rows
        summaries.append(summarize(cfg["name"], cfg, rows))

    pct = lambda d: f"{d['n']}/{d['N']} ({100 * d['rate']:.0f}%)" if d["N"] else "0/0"
    print(f"{'config':<20} {'CtxEvid':>12} {'EverRetr':>12} {'CitedEv*':>12} {'FalseAbst':>11} "
          f"{'Abst(unans)':>11} {'tok/q':>7} {'calls':>5} {'act s':>5} errors")
    for s in summaries:
        print(f"{s['config']:<20} {pct(s['context_evidence_rate']):>12} {pct(s['ever_retrieved']):>12} "
              f"{pct(s['cited_evidence_proxy']):>12} {pct(s['false_abstention']):>11} "
              f"{pct(s['abstention_unanswerable']):>11} {s['tokens_per_question']:>7} {s['model_calls_total']:>5} "
              f"{s['active_s_median']:>5} {len(s['provider_errors'])}")
    print("* proxy: cited blocks (as sent) contain all required evidence - not a human answer grade\n")

    names = list(per_config)
    print(f"{'id':<6} {'type':<16} " + " ".join(f"{n[:22]:>24}" for n in names))
    for i, row in enumerate(per_config[names[0]]):
        cells = []
        for n in names:
            r = per_config[n][i]
            mark = r["status"][:3]
            if r["answerable"] and r["status"] != "error":
                mark += " ctx" + ("+" if r["context_evidence"] else "-") + " cite" + ("+" if r["cited_evidence"] else "-")
            cells.append(f"{mark:>24}")
        print(f"{row['id']:<6} {row['type']:<16} " + " ".join(cells))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summaries": summaries, "per_question": per_config}, ensure_ascii=False,
                                   indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
