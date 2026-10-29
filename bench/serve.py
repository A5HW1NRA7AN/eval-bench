"""`bench serve`: a small read-only JSON API over every stage's output, for Postman or a browser.

Only local files are read (data/), except /search, which embeds the query with Ollama and searches the
local Qdrant test bed. It listens on 127.0.0.1 only.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from bench.agents import FACTS, QUESTIONS
from bench.checks import PAGES, RESULTS, all_checks
from bench.config import DATA_DIR
from bench.evaluate import REPORT
from bench.pdf import read_jsonl
from bench.segments import SEGMENTS, slug
from bench.testbed import COLLECTION, embed, qdrant

ENDPOINTS = {
    "/pages?doc=&page=": "one page: clean text, raw text, printed page number, junk-OCR flag",
    "/toc": "each PDF's table of contents as the TOC agent parsed it (data/toc/)",
    "/segments?doc=": "segments without their text: title, pages, used for questions, size",
    "/segments/<segment_id>": "one segment with its text, page map and TOC headings",
    "/facts?doc=&segment=": "the fact agent's output per segment: verified facts and rejected ones with reasons",
    "/questions?doc=&type=": "the question agent's output per segment: verified questions (with evidence) and rejected ones",
    "/results?type=&misses=1": "each question's search result: rank of the first evidence chunk, hit@k, top 5",
    "/report": "the scored report (Markdown)",
    "/checks": "every stage check (same as `bench check`)",
    "/search?q=&k=5": "search the local test bed now: the top k chunks with their text",
}


def _load(path: Path, stage: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise LookupError(f"{path.name} is missing: run `bench {stage}` first")
    return read_jsonl(path)


def _match(segment_id: str, doc: str | None) -> bool:
    return not doc or slug(doc) in segment_id


def route(path: str, query: dict[str, str]) -> tuple[int, Any]:
    """(status, body) for a GET request; a str body is sent as Markdown, anything else as JSON."""
    parts = [p for p in path.split("/") if p]
    name, rest, doc = (parts[0] if parts else ""), parts[1:], query.get("doc")
    if not name:
        return 200, {"endpoints": ENDPOINTS, "data": str(DATA_DIR)}
    if name == "pages":
        pages = [p for p in _load(PAGES, "segment") if _match(slug(p["file_name"]), doc)]
        if "page" in query:
            pages = [p for p in pages if p["page"] == int(query["page"])]
        return (200, pages[:1]) if pages and "page" in query else (200, [
            {"file_name": p["file_name"], "page": p["page"], "printed": p["printed"], "garbled": p["garbled"],
             "chars": len(p["text"])} for p in pages])
    if name == "toc":
        return 200, {f.stem: json.loads(f.read_text()) for f in sorted((DATA_DIR / "toc").glob("*.json"))}
    if name == "segments":
        segments = _load(SEGMENTS, "segment")
        if rest:
            found = [s for s in segments if s["segment_id"] == rest[0]]
            return (200, found[0]) if found else (404, {"error": f"no segment {rest[0]}"})
        return 200, [{"segment_id": s["segment_id"], "title": s["title"], "pages": s["pages"],
                      "use_for_questions": s["use_for_questions"], "chars": len(s["text"]), "headings": len(s["headings"])}
                     for s in segments if _match(s["segment_id"], doc)]
    if name in ("facts", "questions"):
        rows = _load(FACTS if name == "facts" else QUESTIONS, name)
        rows = [r for r in rows if _match(r["segment_id"], doc) and query.get("segment", r["segment_id"]) == r["segment_id"]]
        if name == "questions" and "type" in query:
            rows = [{**r, "questions": [q for q in r["questions"] if q["type"] == query["type"]]} for r in rows]
        return 200, rows
    if name == "results":
        results = [r for r in _load(RESULTS, "evaluate") if _match(r["question_id"], doc)
                   and query.get("type", r["type"]) == r["type"]]
        if query.get("misses"):
            results = [r for r in results if not r["hit@10"]]
        return 200, results
    if name == "report":
        return (200, REPORT.read_text()) if REPORT.exists() else (404, {"error": "run `bench evaluate` first"})
    if name == "checks":
        return 200, {stage: [{"check": n, "ok": ok, "detail": d} for n, ok, d in checks] for stage, checks in all_checks().items()}
    if name == "search":
        if not query.get("q"):
            return 400, {"error": "add ?q=<question>"}
        hits = qdrant().query_points(COLLECTION, query=embed([query["q"]])[0], limit=int(query.get("k", 5)), with_payload=True).points
        return 200, [{"rank": n, "score": round(h.score, 4), **h.payload} for n, h in enumerate(hits, start=1)]
    return 404, {"error": f"unknown path {path}", "endpoints": ENDPOINTS}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        url = urlparse(self.path)
        try:
            status, body = route(url.path, {k: v[0] for k, v in parse_qs(url.query).items()})
        except LookupError as exc:
            status, body = 404, {"error": str(exc)}
        except Exception as exc:  # e.g. Qdrant or Ollama not running
            status, body = 500, {"error": f"{type(exc).__name__}: {exc}"}
        markdown = isinstance(body, str)
        data = (body if markdown else json.dumps(body, ensure_ascii=False, indent=1)).encode()
        self.send_response(status)
        self.send_header("Content-Type", f"{'text/markdown' if markdown else 'application/json'}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run_server(port: int) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"serving data/ on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
