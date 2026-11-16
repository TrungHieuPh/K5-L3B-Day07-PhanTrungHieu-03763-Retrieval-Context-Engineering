"""Step 1d — build and validate the Day 7 benchmark (dev + new independent test).

Run from the repo root:
  python -m src.build_benchmark            build a DRAFT and validate it
  python -m src.build_benchmark --freeze   freeze it (only after a human reviewed the draft)

Inputs
  <DAY6_ROOT>/benchmark/questions.json   Day 6 bench v2: all 25 questions become Day 7 DEV
                                         (its 5 test questions were exposed and relabelled in Day 6)
  data/benchmark/day7_authored.json      new dev multi-hop / unanswerable + the new TEST set
  data/corpus/chunks_{A,B,C}.json, pages.json

Outputs
  data/benchmark/questions.json          full benchmark WITH gold — evaluator only, never sent to the agent
  data/benchmark/cases.json              id/split/type/answerable for starter/agent_metrics.py
  data/benchmark/manifest.json           counts, rules, metric definitions, file hashes, status
  results/step1/benchmark_validation.json  per-question checks + every absence-check match for review

Hit definition (kept from Day 6): a chunk is a hit when it belongs to a gold article AND contains the
whole evidence after NFC + lowercase + whitespace collapse. Single-hop = one group of alternatives;
multi-hop = several groups, all of which must be covered.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
BENCH = ROOT / "data" / "benchmark"
SOURCE_ID = "BLLD_45_2019_QH14"
ALLOWED_FILTER_KEYS = {"chapter_roman"}
MINIMUMS = {("dev", True): 15, ("dev", False): 2, ("test", True): 5, ("test", False): 2}  # BENCHMARK.md §1


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).lower()).strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Corpus:
    def __init__(self) -> None:
        self.pages = json.loads((CORPUS / "pages.json").read_text(encoding="utf-8"))
        self.starts = [p["char_offset"] for p in self.pages]
        parts, pos = [], 0
        for p in self.pages:  # rebuild the Day 6 joined text so chunk char offsets are exact
            parts.append("\n" * (p["char_offset"] - pos))
            parts.append(p["text"])
            pos = p["char_offset"] + len(p["text"])
        self.joined = "".join(parts)
        if unicodedata.normalize("NFC", self.joined) != self.joined:
            raise RuntimeError("pages.json is not NFC; offsets would drift")
        self.chunks = {s: json.loads((CORPUS / f"chunks_{s}.json").read_text(encoding="utf-8")) for s in "ABC"}
        for s, cs in self.chunks.items():
            bad = [c["id"] for c in cs
                   if self.joined[c["metadata"]["char_start"]:c["metadata"]["char_end"]].strip() != c["text"].strip()]
            if bad:
                raise RuntimeError(f"chunks_{s}: {len(bad)} chunks do not match pages.json, e.g. {bad[:3]}")
        self.article = {c["metadata"]["article"]: c for c in self.chunks["B"]}
        self.b_spans = sorted((c["metadata"]["char_start"], c["metadata"]["char_end"], c["metadata"]["article"])
                              for c in self.chunks["B"])

    def page_of(self, offset: int) -> int:
        return self.pages[bisect.bisect_right(self.starts, offset) - 1]["page"]

    def article_at(self, offset: int) -> int | None:
        i = bisect.bisect_right([s for s, _, _ in self.b_spans], offset) - 1
        if i >= 0 and self.b_spans[i][0] <= offset < self.b_spans[i][1]:
            return self.b_spans[i][2]
        return None

    def locate(self, evidence: str, article: int) -> tuple[int, int] | None:
        """Absolute [start, end) of the evidence inside the article's B chunk (whitespace/case tolerant)."""
        m = self.article[article]["metadata"]
        pattern = r"\s+".join(re.escape(tok) for tok in norm(evidence).split(" "))
        hit = re.search(pattern, self.joined[m["char_start"]:m["char_end"]], re.IGNORECASE)
        return (m["char_start"] + hit.start(), m["char_start"] + hit.end()) if hit else None

    def single_chunk_fit(self, start: int, end: int) -> dict[str, bool]:
        """Diagnostic: does any one chunk of each strategy cover the whole evidence span?"""
        return {s: any(c["metadata"]["char_start"] <= start and c["metadata"]["char_end"] >= end for c in cs)
                for s, cs in self.chunks.items()}


def convert_day6(q: dict[str, Any]) -> dict[str, Any]:
    """Day 6 bench v2 item -> Day 7 schema. gold_page there is the ARTICLE number, not a PDF page."""
    articles = [q["gold_page"], *q.get("alt_gold_pages", [])]
    item = {
        "id": q["id"], "split": "dev", "type": q["type"],
        "origin": f"day6:{q['id']}" + (" (Day 6 test, đã lộ điểm và sửa nhãn ở Day 6)" if q["split"] == "test" else ""),
        "question": q["question"], "expected_answer": q["expected_answer"],
        "answerable": True, "expected_behavior": "search_and_answer",
        "gold": [{"source_id": q["gold_source_id"], "article": a, "evidence": q["evidence_quote"]} for a in articles],
        "filter": {"chapter_roman": q["filter"]["chapter"]} if q.get("filter") else None,
    }
    if q.get("filter_context"):
        item["filter_context"] = q["filter_context"]
    return item


def complete_authored(item: dict[str, Any]) -> dict[str, Any]:
    item = dict(item)
    item.setdefault("filter", None)
    if item.get("answerable", True):
        item["answerable"] = True
        item["expected_behavior"] = "search_and_answer"
    for group in item.get("gold_groups") or [item.get("gold", [])]:
        for alt in group:
            alt.setdefault("source_id", SOURCE_ID)
    return item


def groups_of(item: dict[str, Any]) -> list[list[dict[str, Any]]]:
    return item["gold_groups"] if item.get("gold_groups") else [item["gold"]]


def validate_answerable(item: dict[str, Any], corpus: Corpus) -> tuple[list[str], dict[str, Any]]:
    errors, detail = [], {"groups": []}
    if not item.get("expected_answer"):
        errors.append("missing expected_answer")
    if bool(item.get("gold")) == bool(item.get("gold_groups")):
        errors.append("need exactly one of gold / gold_groups")
        return errors, detail
    gold_articles = set()
    for gi, group in enumerate(groups_of(item)):
        g = []
        for alt in group:
            a = alt["article"]
            gold_articles.add(a)
            if a not in corpus.article:
                errors.append(f"group {gi}: article {a} not in corpus")
                continue
            chunk = corpus.article[a]
            if norm(alt["evidence"]) not in norm(chunk["text"]):
                errors.append(f"group {gi}: evidence not verbatim in B chunk of Điều {a}")
                continue
            span = corpus.locate(alt["evidence"], a)
            if span is None:
                errors.append(f"group {gi}: evidence matched normalized text but could not be located in Điều {a}")
                continue
            p0, p1 = corpus.page_of(span[0]), corpus.page_of(span[1] - 1)
            alt["page"] = p0  # PDF page where the evidence starts (computed, not typed)
            if p1 != p0:
                alt["page_end"] = p1
            g.append({"article": a, "chunk_b": chunk["id"], "pdf_pages": [p0, p1],
                      "chapter_roman": chunk["metadata"]["chapter_roman"],
                      "fits_single_chunk": corpus.single_chunk_fit(*span)})
        detail["groups"].append(g)
    q = norm(item["question"])
    for a in gold_articles:
        if re.search(rf"\bđiều {a}\b", q):
            errors.append(f"question names gold article Điều {a} (leaks the label)")
    flt = item.get("filter") or {}
    if set(flt) - ALLOWED_FILTER_KEYS:
        errors.append(f"filter keys {sorted(set(flt) - ALLOWED_FILTER_KEYS)} not allowed (article filter leaks gold)")
    if flt.get("chapter_roman"):
        if not item.get("filter_context"):
            errors.append("filter without filter_context (must say it is user-provided)")
        for a in gold_articles:
            if a in corpus.article and corpus.article[a]["metadata"]["chapter_roman"] != flt["chapter_roman"]:
                errors.append(f"filter chapter {flt['chapter_roman']} excludes gold Điều {a}")
    return errors, detail


def validate_unanswerable(item: dict[str, Any], corpus: Corpus) -> tuple[list[str], dict[str, Any]]:
    errors = []
    for key in ("absence_reason", "absence_check"):
        if not item.get(key):
            errors.append(f"missing {key}")
    if item.get("expected_behavior") != "search_and_abstain":
        errors.append("expected_behavior must be search_and_abstain")
    if item.get("gold") or item.get("gold_groups") or item.get("expected_answer"):
        errors.append("unanswerable item must not carry gold / expected_answer")
    matches = []
    for pat in (item.get("absence_check") or {}).get("patterns", []):
        rx = re.compile(pat.replace(" ", r"\s+"), re.IGNORECASE)
        found = list(rx.finditer(corpus.joined))  # whole document, not top-k
        matches.append({
            "pattern": pat, "n": len(found),
            "matches": [{"page": corpus.page_of(m.start()), "article": corpus.article_at(m.start()),
                         "snippet": norm(corpus.joined[max(0, m.start() - 80):m.end() + 80])}
                        for m in found[:12]],
        })
    return errors, {"absence_matches": matches, "human_review_required": True}


def build(day6_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    corpus = Corpus()
    day6 = json.loads((day6_root / "benchmark" / "questions.json").read_text(encoding="utf-8"))
    authored = json.loads((BENCH / "day7_authored.json").read_text(encoding="utf-8"))
    items = [convert_day6(q) for q in day6["questions"]] + [complete_authored(x) for x in authored]

    report: dict[str, Any] = {"items": [], "errors": [], "warnings": []}
    ids = Counter(i["id"] for i in items)
    questions = Counter(norm(i["question"]) for i in items)
    report["errors"] += [f"duplicate id {k}" for k, v in ids.items() if v > 1]
    report["errors"] += [f"duplicate question: {k[:60]}" for k, v in questions.items() if v > 1]
    for item in items:
        if item["split"] not in ("dev", "test"):
            report["errors"].append(f"{item['id']}: bad split")
        fn = validate_answerable if item["answerable"] else validate_unanswerable
        errs, detail = fn(item, corpus)
        report["errors"] += [f"{item['id']}: {e}" for e in errs]
        report["items"].append({"id": item["id"], "split": item["split"], "type": item["type"],
                                "ok": not errs, "errors": errs, **detail})

    counts = Counter((i["split"], i["answerable"]) for i in items)
    for key, need in MINIMUMS.items():
        if counts[key] < need:
            report["errors"].append(f"{key[0]} {'answerable' if key[1] else 'unanswerable'}: {counts[key]} < {need}")
    gold_by_split = {s: {alt["article"] for i in items if i["split"] == s and i["answerable"]
                         for g in groups_of(i) for alt in g} for s in ("dev", "test")}
    overlap = sorted(gold_by_split["dev"] & gold_by_split["test"])
    if overlap:
        report["warnings"].append(f"test gold articles also used in dev: {overlap}")
    report["summary"] = {
        "n": len(items),
        "by_split": {s: {"answerable": counts[(s, True)], "unanswerable": counts[(s, False)],
                         "multi_hop": sum(1 for i in items if i["split"] == s and i.get("gold_groups")),
                         "with_filter": sum(1 for i in items if i["split"] == s and i.get("filter"))}
                     for s in ("dev", "test")},
        "types": dict(Counter(f"{i['split']}:{i['type']}" for i in items)),
        "gold_articles": {s: sorted(v) for s, v in gold_by_split.items()},
        "dev_test_article_overlap": overlap,
    }
    return items, report


def manifest(status: str, items: list[dict[str, Any]], report: dict[str, Any], **extra: Any) -> dict[str, Any]:
    corpus_manifest = json.loads((CORPUS / "index_manifest.json").read_text(encoding="utf-8"))
    return {
        "benchmark_name": "BLLD_45_2019_QH14_day7_v1",
        "status": status,
        **extra,
        "corpus": {"source_id": SOURCE_ID, "pdf_sha256": corpus_manifest["source"]["sha256"],
                   "chunks_sha256": corpus_manifest["files_sha256"]},
        "summary": report["summary"],
        "rules": [
            "Chọn cấu hình chỉ trên dev; chốt cấu hình và tiêu chí chọn trước khi chạy test.",
            "Agent và retrieval chỉ nhận `question` và `filter` (filter do người dùng cung cấp); không nhận gold, expected_answer, type, answerable.",
            "Test là bộ mới soạn ở Day 7 từ các Điều Day 6 chưa hỏi; 5 câu test cũ của Day 6 đã lộ nên chuyển sang dev.",
            "Câu thiếu evidence không vào mẫu số của retrieval metrics.",
        ],
        "schema": {
            "gold": "Câu một bước: danh sách evidence THAY THẾ {source_id, article, page, evidence}; khớp một phần tử là đủ.",
            "gold_groups": "Câu multi-hop: danh sách NHÓM BẮT BUỘC, mỗi nhóm là danh sách evidence thay thế; phải phủ mọi nhóm.",
            "article": "Số Điều (đơn vị gold, giữ từ Day 6).",
            "page": "Trang PDF (1-based) nơi evidence bắt đầu, script tự tính từ pages.json; page_end nếu evidence qua trang.",
            "filter": "{chapter_roman} do người dùng cung cấp trước khi truy xuất (xem filter_context); không bao giờ là article.",
        },
        "metric_definitions": {
            "hit": "Chunk thuộc một Điều gold VÀ chứa trọn evidence sau NFC + lowercase + gộp khoảng trắng (giữ từ Day 6).",
            "hit@1": "Câu một bước: chunk top-1 là hit.",
            "evidence_hit@k": "Câu một bước: có ít nhất một hit trong top-k.",
            "multi_hop_all_groups@k": "Câu multi-hop: mọi nhóm có ít nhất một hit trong tập chunk truy xuất (gộp các lượt search của agent); báo riêng, kèm tỷ lệ nhóm được phủ.",
            "context_evidence_rate": "Context đã đóng gói (sau budget) chứa evidence của mọi nhóm.",
            "answer_grade": "0 = không có evidence hoặc sai; 1 = đúng một phần / thiếu điều kiện hoặc nguồn; 2 = đủ ý và citation hỗ trợ. Từ chối câu có đáp án = 0. Xem grading_note từng câu.",
            "abstention": "Câu thiếu evidence: đúng khi agent đã search rồi từ chối, không bịa con số; nói rõ Bộ luật dẫn chiếu luật khác cũng được tính là từ chối đúng.",
        },
        "files_sha256": {name: sha256_file(BENCH / name) for name in ("questions.json", "cases.json")},
    }


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Build and validate the Day 7 benchmark")
    parser.add_argument("--freeze", action="store_true", help="freeze after human review")
    parser.add_argument("--approved-by", default="học viên 03763", help="who reviewed the draft")
    args = parser.parse_args()
    day6_root = Path(os.environ.get("DAY6_ROOT", "../K5-L3-Day06-PhanTrungHieu-03763-DataIngestionEngineering/starter"))
    day6_root = (day6_root if day6_root.is_absolute() else ROOT / day6_root).resolve()

    man_path = BENCH / "manifest.json"
    if man_path.exists() and json.loads(man_path.read_text(encoding="utf-8")).get("status") == "frozen":
        raise SystemExit("Benchmark is FROZEN. Changing it after test runs breaks test independence; "
                         "make a new version instead of overwriting.")

    items, report = build(day6_root)
    write_json(ROOT / "results" / "step1" / "benchmark_validation.json", report)
    if report["errors"]:
        print("VALIDATION FAILED:")
        for e in report["errors"]:
            print("  -", e)
        sys.exit(1)

    write_json(BENCH / "questions.json", items)
    write_json(BENCH / "cases.json", [{"id": i["id"], "split": i["split"], "type": i["type"],
                                       "answerable": i["answerable"],
                                       "expected_behavior": i["expected_behavior"]} for i in items])
    if args.freeze:
        man = manifest("frozen", items, report, frozen_at=date.today().isoformat(),
                       approved_by=args.approved_by,
                       absence_checks_reviewed=True)
    else:
        man = manifest("draft", items, report, frozen_at=None,
                       note="DRAFT — cần người duyệt câu test và kết quả absence check trước khi chạy --freeze.")
    write_json(man_path, man)

    s = report["summary"]
    for split in ("dev", "test"):
        b = s["by_split"][split]
        print(f"{split}: {b['answerable']} answerable ({b['multi_hop']} multi-hop, {b['with_filter']} filter), "
              f"{b['unanswerable']} unanswerable")
    print("dev/test gold-article overlap:", s["dev_test_article_overlap"] or "none")
    for w in report["warnings"]:
        print("WARNING:", w)
    print(f"status: {man['status']}  ->  data/benchmark/questions.json, cases.json, manifest.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
