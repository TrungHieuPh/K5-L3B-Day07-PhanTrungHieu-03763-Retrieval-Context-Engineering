"""Step 1b — export the Day 6 index (pgvector) to Day 7 JSON files.

Run from the repo root:  python -m src.export_day6

Reads chunks straight from the Day 6 PostgreSQL + pgvector table (it does NOT
re-chunk), so every Day 7 method searches the exact corpus Day 6 indexed.

Outputs (data/corpus/):
  chunks_{A,B,C}.json   Day 7 chunk contract: id, text, source_id, source, page, metadata
  pages.json            page text exactly as Day 6 indexed it (rebuilt with Day 6 code),
                        used to locate evidence pages and to check "absent" questions
  index_manifest.json   DB/pgvector versions, counts, dims, embedding model, file hashes

Embeddings are not exported: dense retrieval queries the live vector store.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_ID = "BLLD_45_2019_QH14"
PDF_NAME = "bo_luat_lao_dong_45_2019_QH14.pdf"
STRATEGY_LABELS = {"A": "fixed_size(500,50)", "B": "by_article", "C": "by_clause(200,800)"}


def load_day6_package(day6_root: Path) -> dict[str, ModuleType]:
    """Import Day 6 `src` as package `day6src` (Day 7 also has a package named `src`)."""
    pkg_dir = day6_root / "src"
    spec = importlib.util.spec_from_file_location(
        "day6src", pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Day 6 code from {pkg_dir}")
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["day6src"] = pkg
    spec.loader.exec_module(pkg)
    return {name: importlib.import_module(f"day6src.{name}") for name in ("models", "ingestion", "chunking")}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj: Any) -> str:
    data = (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.write_bytes(data)
    return sha256_bytes(data)


def rebuild_pages(day6_root: Path, mods: dict[str, ModuleType]) -> tuple[list[dict[str, Any]], str, str]:
    """Rebuild normalized page text with Day 6 code; verify the PDF hash first."""
    pdf = day6_root / "data" / PDF_NAME
    manifest = json.loads((day6_root / "data" / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest["sources"][0]["file"]["sha256"]
    actual = sha256_bytes(pdf.read_bytes())
    if actual != expected:
        raise RuntimeError(f"PDF sha256 {actual} != Day 6 manifest {expected}")
    raw = mods["ingestion"].load_pdf(pdf, DOCUMENT_ID, PDF_NAME)
    # Same steps as Day 6 run.py::_load_pages, so char offsets match the DB metadata.
    pages = [
        mods["models"].DocumentPage(p.document_id, p.source, p.page,
                                    mods["ingestion"].normalize_text(p.text), p.metadata)
        for p in raw
    ]
    joined, offsets = mods["chunking"].concat_with_offsets(pages)
    rows = [
        {"source_id": p.document_id, "source": p.source, "page": p.page,
         "char_offset": off, "text": p.text}
        for p, off in zip(pages, offsets)
    ]
    return rows, joined, actual


def fetch_strategy(conn: psycopg.Connection, strategy: str) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT chunk_id, document_id, source, page, text, metadata, vector_dims(embedding)
               FROM chunks WHERE strategy = %s
               ORDER BY (metadata->>'char_start')::int, chunk_id""",
            (strategy,),
        )
        rows = cur.fetchall()
    return [
        {"id": cid, "text": text, "source_id": doc, "source": src, "page": page,
         "metadata": meta, "_dims": dims}
        for cid, doc, src, page, text, meta, dims in rows
    ]


def check_alignment(chunks: list[dict[str, Any]], joined: str) -> list[str]:
    """Every chunk text must equal the Day 6 document slice it claims to come from."""
    bad = []
    for c in chunks:
        m = c["metadata"]
        if joined[m["char_start"]:m["char_end"]].strip() != c["text"].strip():
            bad.append(c["id"])
    return bad


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Export the Day 6 pgvector index to Day 7 JSON")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--day6-root", type=Path, default=Path(os.environ.get(
        "DAY6_ROOT", "../K5-L3-Day06-PhanTrungHieu-03763-DataIngestionEngineering/starter")))
    parser.add_argument("--strategies", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "corpus")
    args = parser.parse_args()
    if not args.database_url:
        raise SystemExit("DATABASE_URL is not set (copy .env.example to .env)")
    day6_root = (args.day6_root if args.day6_root.is_absolute() else ROOT / args.day6_root).resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    mods = load_day6_package(day6_root)
    pages, joined, pdf_sha = rebuild_pages(day6_root, mods)
    print(f"Pages rebuilt with Day 6 code: {len(pages)} ({sum(1 for p in pages if p['text'])} non-empty)")

    files: dict[str, str] = {"pages.json": write_json(args.out / "pages.json", pages)}
    strategies: dict[str, Any] = {}
    with psycopg.connect(args.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SHOW server_version")
            pg_version = cur.fetchone()[0]
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            pgvector_version = cur.fetchone()[0]
        for s in args.strategies:
            chunks = fetch_strategy(conn, s)
            if not chunks:
                raise RuntimeError(f"Strategy {s}: no rows in Day 6 table `chunks`")
            dims = {c.pop("_dims") for c in chunks}
            if len(dims) != 1:
                raise RuntimeError(f"Strategy {s}: mixed embedding dims {dims}")
            if len({c["id"] for c in chunks}) != len(chunks):
                raise RuntimeError(f"Strategy {s}: duplicate chunk ids")
            bad = check_alignment(chunks, joined)
            if bad:
                raise RuntimeError(f"Strategy {s}: {len(bad)} chunks differ from Day 6 page text, e.g. {bad[:3]}")
            name = f"chunks_{s}.json"
            files[name] = write_json(args.out / name, chunks)
            strategies[s] = {"label": STRATEGY_LABELS.get(s, s), "n_chunks": len(chunks),
                             "embedding_dims": dims.pop(), "file": name,
                             "text_aligned_with_pages": True}
            print(f"Strategy {s} ({STRATEGY_LABELS.get(s, s)}): {len(chunks)} chunks -> {name}")

    day6_summary = json.loads((day6_root / "results" / "summary.json").read_text(encoding="utf-8"))
    manifest = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"source_id": DOCUMENT_ID, "file": PDF_NAME, "sha256": pdf_sha,
                   "physical_pages": len(pages),
                   "content_pages": sum(1 for p in pages if p["text"]),
                   "manifest": "Day 6 starter/data/manifest.json"},
        "vector_store": {"engine": "PostgreSQL + pgvector (Day 6 Docker, 127.0.0.1:55433)",
                         "postgres": pg_version, "pgvector": pgvector_version,
                         "table": "chunks", "key": "(strategy, chunk_id)",
                         "distance": "cosine (<=>), score = 1 - distance"},
        "embedding": day6_summary["run"]["embedding"],
        "day6_selected_strategy": "B",
        "strategies": strategies,
        "files_sha256": files,
    }
    write_json(args.out / "index_manifest.json", manifest)
    print(f"Wrote {args.out.relative_to(ROOT)}/index_manifest.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
