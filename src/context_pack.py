"""Step 4, config 2 — context selection that does not throw evidence away.

Two failure modes of starter pack_context (first-come, whole chunk or nothing):
  P1  a chunk that does not fit WHOLE is dropped entirely: Điều 219 (7248 chars) can never enter a 4000-char
      context; hybrid q24 lost art020 (1486 chars) because art021 + art014 had already used 2515 chars.
  P2  candidates accumulate in arrival order across searches: the first search fills the budget, so a better
      chunk found by a later, reformulated search cannot get in.

ExcerptPacker (same signature as pack_context(results, budget) -> (context, ids)):
  1  order   rank-major round robin over searches: rank 1 of every search, then rank 2, ... (dedupe by id)
  2  cap     one chunk may use at most `cap` chars; a longer chunk becomes a query-aware excerpt
  3  fill    a chunk that does not fit the remaining space whole is excerpted to that space (if >= min_chars)
  excerpt    the chunk is split into legal units (title "Điều N.", clause "1.", point "a)") and grouped into
             clause blocks (clause + its points); the title is kept, then the blocks with the highest idf-weighted
             overlap with THAT search's query (a block that does not fit keeps its lead-in + first points);
             units are restored to document order and gaps are marked "[…]".
Only the model's own search query and corpus idf are used — never gold. Block format = starter format, so
citations and run_agent's citation check work unchanged.
"""
from __future__ import annotations

import re
from typing import Any

from .retrievers import BM25Retriever, load_chunks, tokenize

# "Điều 54." starts a (quoted) article; a wrapped PDF line "Điều 169 của ..." does not
UNIT_START = re.compile(r"^\s*[“\"]?(Điều\s+\d+\.|\d+\.|[a-zđ]\))")
POINT = re.compile(r"^\s*[“\"]?[a-zđ]\)")
GAP = "[…]"


def split_units(text: str) -> list[str]:
    units: list[str] = []
    for line in text.split("\n"):
        if not units or UNIT_START.match(line):
            units.append(line)
        else:
            units[-1] += "\n" + line
    return units


def render(units: list[str], selected: set[int]) -> str:
    out, prev = [], -1
    for i in sorted(selected):
        if i - prev > 1:
            out.append(GAP)
        out.append(units[i])
        prev = i
    if prev < len(units) - 1:
        out.append(GAP)
    return "\n".join(out)


class ExcerptPacker:
    name = "excerpt_rr"

    def __init__(self, strategy: str, *, cap: int = 1500, min_chars: int = 300) -> None:
        self.idf = BM25Retriever(load_chunks(strategy), strategy=strategy).idf
        self.cap, self.min_chars = cap, min_chars
        self.stats: list[dict[str, Any]] = []  # per call: which chunks were excerpted / skipped (for the trace)

    def settings(self) -> dict[str, Any]:
        return {"name": self.name, "cap_chars": self.cap, "min_chars": self.min_chars,
                "order": "rank-major round robin over searches", "excerpt": "idf overlap with the search query"}

    # ---- excerpt -------------------------------------------------------------------------------------------
    def excerpt(self, text: str, query: str, limit: int) -> str:
        units = split_units(text)
        # a clause block = one clause "N. ..." + its points "a) ...": the lead-in ("... trường hợp sau đây:")
        # matches the query but the answer is the list of points, so they are scored and kept together
        blocks: list[list[int]] = []
        for i in range(len(units)):
            if i == 0 or not POINT.match(units[i]):
                blocks.append([i])
            else:
                blocks[-1].append(i)
        q = set(tokenize(query))
        score = [sum(self.idf.get(t, 0.0) for t in q & {t for i in b for t in tokenize(units[i])}) for b in blocks]
        selected = {0}
        if len(render(units, selected)) > limit:  # title alone too long: hard cut
            return text[: max(0, limit - len(GAP) - 1)] + "\n" + GAP
        for b in sorted(range(len(blocks)), key=lambda b: (-score[b], b)):
            if score[b] <= 0:
                break
            if len(render(units, selected | set(blocks[b]))) <= limit:
                selected |= set(blocks[b])
                continue
            for i in blocks[b]:  # block too long: lead-in + first points, in order, while they fit
                if len(render(units, selected | {i})) > limit:
                    break
                selected.add(i)
        if len(selected) == 1:  # no query term inside: keep the beginning of the article
            for i in range(1, len(units)):
                if len(render(units, selected | {i})) > limit:
                    break
                selected.add(i)
        return render(units, selected)

    # ---- pack ----------------------------------------------------------------------------------------------
    @staticmethod
    def _search_order(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        searches: list[list[dict[str, Any]]] = []
        for r in results:  # run_agent concatenates each search's ranked list; ToolRetriever tags search_no
            n = r.get("search_no")
            new = (n != searches[-1][0].get("search_no")) if (n is not None and searches) else r.get("rank") == 1
            if not searches or new:
                searches.append([])
            searches[-1].append(r)
        ordered, seen = [], set()
        for k in range(max((len(s) for s in searches), default=0)):
            for s in searches:
                if k < len(s) and s[k]["id"] not in seen:
                    seen.add(s[k]["id"])
                    ordered.append(s[k])
        return ordered

    def __call__(self, results: list[dict[str, Any]], budget: int = 800) -> tuple[str, list[str]]:
        blocks, ids, used, log = [], [], 0, []
        for c in self._search_order(results):
            header = f"[{c['id']} | {c['source']} | page {c['page']}]\n"
            room = budget - used - (2 if blocks else 0) - len(header)
            allow = min(self.cap, room)
            text = c["text"]
            if len(text) <= allow:
                body, how = text, "whole"
            elif allow >= self.min_chars:
                body, how = self.excerpt(text, c.get("query", ""), allow), "excerpt"
            else:
                log.append({"id": c["id"], "how": "skipped", "chars": len(text), "room": max(room, 0)})
                continue
            blocks.append(header + body)
            ids.append(c["id"])
            used += (2 if len(blocks) > 1 else 0) + len(header) + len(body)
            log.append({"id": c["id"], "how": how, "chars": len(text), "kept": len(body)})
        self.stats.append(log)
        context = "\n\n".join(blocks)
        assert len(context) <= budget, (len(context), budget)
        return context, ids
