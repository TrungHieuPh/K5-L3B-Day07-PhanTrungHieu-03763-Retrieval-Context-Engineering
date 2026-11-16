"""Local demo: ask the agent one question in the browser and see every step it took.

Run from the repo root:   python -m src.demo_server          then open http://127.0.0.1:8000
                          python -m src.demo_server --port 8001

Same code path as the benchmark runs: src.agent.run_case -> starter run_agent + GeminiController + ToolRetriever,
with the chosen retrieval (dense / bm25 / hybrid_rrf) and context selection (starter / excerpt_rr).
  * listens on 127.0.0.1 only; GEMINI_KEY stays in this process and never reaches the browser
  * one question at a time (a lock); the free tier allows 5 model requests per minute, shared with any
    benchmark run of src.agent -> do not demo while a benchmark run is in progress
  * the example list contains DEV questions only (question text, filter, type) — no gold answers and no test
    questions, so the demo cannot leak test items before the configuration is frozen
Standard library only (http.server); no extra package.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the Day 6 embedder is already in the local HF cache

from dotenv import load_dotenv  # noqa: E402

from .agent import (QUESTIONS, AgentConfig, GeminiController, ToolRetriever, build_retriever,  # noqa: E402
                    load_run_agent, run_case)
from .day6_store import ROOT  # noqa: E402
from .retrievers import load_chunks  # noqa: E402

RETRIEVALS = ("dense", "bm25", "hybrid_rrf")
CONTEXTS = ("starter", "excerpt_rr")
HEADER = re.compile(r"^\[([^\s|\]]+) \| [^\n]*\]$", re.M)


def context_blocks(context: str) -> dict[str, str]:
    heads = list(HEADER.finditer(context))
    return {m.group(1): context[m.end(): heads[k + 1].start() if k + 1 < len(heads) else len(context)].strip("\n")
            for k, m in enumerate(heads)}


class Demo:
    def __init__(self, model: str, rpm: int) -> None:
        key = os.environ.get("GEMINI_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("GEMINI_KEY is not set in .env")
        self.base = AgentConfig(model=model)
        self.controller = GeminiController(key, model, max_searches=self.base.max_searches,
                                           temperature=self.base.temperature, rpm=rpm)
        self.retrievers: dict[str, ToolRetriever] = {}
        self.chunks = {c["id"]: c for c in load_chunks(self.base.strategy)}
        self.lock = threading.Lock()
        dev = [q for q in json.loads(QUESTIONS.read_text(encoding="utf-8")) if q["split"] == "dev"]
        self.examples = [{"id": q["id"], "type": q["type"], "question": q["question"], "filter": q.get("filter")}
                         for q in dev]

    def retriever(self, kind: str) -> ToolRetriever:
        if kind not in self.retrievers:
            print(f"  loading retriever {kind} ...", flush=True)
            self.retrievers[kind] = ToolRetriever(build_retriever(kind, self.base.strategy))
        return self.retrievers[kind]

    def ask(self, question: str, retrieval: str, context: str, filters: dict[str, Any] | None) -> dict[str, Any]:
        cfg = AgentConfig(name=f"demo_{retrieval}_{context}", model=self.base.model, retrieval=retrieval,
                          context=context)
        with self.lock:
            run_agent, packer = load_run_agent(cfg)  # sets agentic.pack_context for this run
            retriever = self.retriever(retrieval)
            case = {"id": "demo", "split": "demo", "question": question, "filter": filters}
            tr = run_case(case, self.controller, retriever, cfg, run_agent, packer)
        return self.view(tr, cfg)

    def article(self, cid: str) -> int | None:
        c = self.chunks.get(cid)
        return c["metadata"].get("article") if c else None

    def view(self, tr: dict[str, Any], cfg: AgentConfig) -> dict[str, Any]:
        """What the page shows: every model decision, every search with its results and the context after it."""
        searches = [e for e in tr["events"] if e["event"] == "search"]
        steps, s = [], 0
        for c in tr["controller_calls"]:
            step = {"decision": c.get("decision") or (c["function_calls"][0]["name"] if c["function_calls"] else "?"),
                    "ms": c["ms"], "wait_s": round((c["total_ms"] - c["ms"]) / 1000, 1),
                    "tokens": c["prompt_tokens"] + c["output_tokens"] + c["thinking_tokens"],
                    "retries": [f"{r['code']} {r['status']}" for r in c["retries"]],
                    "normalized_from": c.get("normalized_from")}
            if step["decision"] == "search" and s < len(searches):
                e, pack = searches[s], (tr["pack_log"][s] if tr.get("pack_log") else None)
                how = {p["id"]: p for p in pack} if pack else {}
                step.update(query=e["query"], context_ids=e["context_ids"], context_chars=len(e["context"]),
                            context_articles=[self.article(i) for i in e["context_ids"]],
                            results=[{"id": r["id"], "article": self.article(r["id"]), "rank": r["rank"],
                                      "score": r["score"], "chars": len(r["text"]),
                                      "in_context": r["id"] in e["context_ids"],
                                      "pack": how.get(r["id"])} for r in e["results"]])
                s += 1
            elif step["decision"] == "abstain":
                step["reason"] = c.get("abstain_reason")
            steps.append(step)
        final_ctx = searches[-1]["context"] if searches else ""
        blocks = context_blocks(final_ctx)
        last_event = tr["events"][-1]["event"] if tr["events"] else None
        return {
            "status": tr["status"], "answer": tr["answer"], "abstain_reason": tr["abstain_reason"],
            "final_decision": tr["final_decision"], "run_agent_event": last_event, "error": tr["error"],
            "citations": [{"id": i, "article": self.article(i), "text_as_sent": blocks.get(i, "")}
                          for i in tr["citations"]],
            "steps": steps, "final_context": final_ctx,
            "totals": {"model_calls": tr["n_model_calls"], "searches": tr["n_searches"], "tokens": tr["tokens"],
                       "elapsed_s": round(tr["elapsed_ms"] / 1000, 1), "wait_s": tr["wait_s"],
                       "retries": tr["n_retries"], "cost_usd": tr["cost_usd"]},
            "config": asdict(cfg),
        }


def parse_request(body: dict[str, Any]) -> tuple[str, str, str, dict[str, Any] | None]:
    question = str(body.get("question", "")).strip()
    if not question or len(question) > 1000:
        raise ValueError("Câu hỏi phải có từ 1 đến 1000 ký tự.")
    retrieval, context = body.get("retrieval", "dense"), body.get("context", "starter")
    if retrieval not in RETRIEVALS or context not in CONTEXTS:
        raise ValueError("retrieval/context không hợp lệ.")
    filters: dict[str, Any] = {}
    chapter = str(body.get("chapter_roman") or "").strip().upper()
    if chapter:
        if not re.fullmatch(r"[IVXLC]{1,6}", chapter):
            raise ValueError("Chương phải là số La Mã, ví dụ III hoặc XIV.")
        filters["chapter_roman"] = chapter
    article = str(body.get("article") or "").strip()
    if article:
        if not article.isdigit() or not 1 <= int(article) <= 220:
            raise ValueError("Điều phải là số từ 1 đến 220.")
        filters["article"] = int(article)
    return question, retrieval, context, filters or None


def make_handler(demo: Demo):
    page = (ROOT / "src" / "demo_page.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/examples":
                self._json(200, demo.examples)
            elif self.path == "/api/config":
                self._json(200, {"model": demo.base.model, "strategy": demo.base.strategy,
                                 "top_k": demo.base.top_k, "budget_chars": demo.base.budget_chars,
                                 "max_searches": demo.base.max_searches, "rpm": demo.controller.rpm,
                                 "retrievals": RETRIEVALS, "contexts": CONTEXTS})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/ask":
                return self._json(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                question, retrieval, context, filters = parse_request(json.loads(self.rfile.read(n) or b"{}"))
            except (ValueError, json.JSONDecodeError) as e:
                return self._json(400, {"error": str(e)})
            t = time.perf_counter()
            print(f"[ask] {retrieval}/{context} filter={filters} q={question[:70]!r}", flush=True)
            try:
                out = demo.ask(question, retrieval, context, filters)
            except Exception as e:  # e.g. database not running for dense/hybrid
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})
            print(f"      -> {out['status']} in {time.perf_counter() - t:.1f}s", flush=True)
            self._json(200, out)

        def log_message(self, fmt: str, *args: Any) -> None:  # keep the console readable
            pass

    return Handler


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description="Local web demo of the agent (127.0.0.1 only)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", AgentConfig.model))
    ap.add_argument("--rpm", type=int, default=int(os.environ.get("GEMINI_RPM", 5)))
    args = ap.parse_args()
    demo = Demo(args.model, args.rpm)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(demo))
    except OSError as e:
        raise SystemExit(f"Port {args.port} is busy ({e}). Try: python -m src.demo_server --port 8001")
    print(f"Demo ready: http://127.0.0.1:{args.port}   (model {args.model}, Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
