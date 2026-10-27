"""Clean page text from the PDFs: running headers, page numbers and stray glyphs removed, paragraphs kept."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

from bench.config import DATA_DIR, PDF_DIR

EDGE_LINES = 3  # lines at the top and bottom of a page that may be headers or footers
REPEATS = 4  # a header/footer must repeat on this many pages
GLYPHS = {"yy", "□"}  # bullet/end marks that PyMuPDF extracts as lines of their own
NUMBER = re.compile(r"^[\d.,%()\-–—:/]+$|^NIL$")
PAGE_NO = re.compile(r"^\d{1,4}\b\s*|\s*\b\d{1,4}$")
BARE_INT = re.compile(r"^\d{1,4}$")
YEAR = re.compile(r"^(19|20)\d{2}$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _edges(lines: list[str]) -> list[str]:
    return lines if len(lines) <= 2 * EDGE_LINES else lines[:EDGE_LINES] + lines[-EDGE_LINES:]


def _key(line: str) -> str | None:
    """'Dairy Farming 12' and 'Dairy Farming 14' share a key; number-only lines get none (table values)."""
    norm = " ".join(line.lower().split())
    if any(c.isalpha() for c in norm):
        return PAGE_NO.sub("#", norm)
    return norm if YEAR.match(norm) else None  # a repeated header year still counts


def running_lines(pages: list[list[str]]) -> tuple[set[str], set[int]]:
    """Header/footer keys repeated across pages, and page-number offsets (printed minus PDF page)."""
    keys: Counter[str] = Counter()
    offsets: Counter[int] = Counter()
    for index, lines in enumerate(pages, start=1):
        edges = [l.strip() for l in _edges(lines)]
        keys.update({k for l in edges if (k := _key(l))})
        offsets.update({int(l) - index for l in edges if BARE_INT.match(l)})
    return {k for k, n in keys.items() if n >= REPEATS}, {o for o, n in offsets.items() if n >= REPEATS}


def clean_page(blocks: list[str], index: int, keys: set[str], offsets: set[int]) -> str:
    """Drop header/footer lines and glyphs; blocks stay separated by a blank line."""
    lines = [(b, l) for b, block in enumerate(blocks) for l in block.splitlines() if l.strip()]
    edge = set(range(min(EDGE_LINES, len(lines)))) | set(range(max(0, len(lines) - EDGE_LINES), len(lines)))
    kept: dict[int, list[str]] = {}
    for i, (b, line) in enumerate(lines):
        s = line.strip()
        page_number = BARE_INT.match(s) and int(s) - index in offsets  # only numbers that advance with the page
        if i in edge and (page_number or _key(s) in keys):
            continue
        if s not in GLYPHS:
            kept.setdefault(b, []).append(line)
    return "\n\n".join("\n".join(kept[b]) for b in sorted(kept))


def printed_number(lines: list[str], index: int, offsets: set[int]) -> int | None:
    """The page number printed on the page: an edge line holding a number that advances with the page."""
    edges = [l.strip() for l in _edges(lines)]
    return next((int(s) for s in edges if BARE_INT.match(s) and int(s) - index in offsets), None)


def garbled(text: str) -> bool:
    """Junk from a bad OCR layer ('0 CL o c —'): mostly 1-2 character tokens once numbers are set aside."""
    words = [t for t in text.split() if not NUMBER.match(t)]
    return len(words) >= 50 and sum(len(w) <= 2 for w in words) / len(words) > 0.6


def extract(path: Path) -> list[dict[str, Any]]:
    doc_id = hashlib.sha256(path.read_bytes()).hexdigest()
    with pymupdf.open(path) as doc:
        raw = [page.get_text() for page in doc]
        blocks = [[b[4] for b in page.get_text("blocks") if b[6] == 0] for page in doc]  # text blocks only
    lines = [[l for l in text.splitlines() if l.strip()] for text in raw]
    keys, offsets = running_lines(lines)
    pages = []
    for index, (raw_text, page_blocks) in enumerate(zip(raw, blocks), start=1):
        text = clean_page(page_blocks, index, keys, offsets)
        pages.append({"doc_id": doc_id, "file_name": path.name, "page": index, "printed": printed_number(lines[index - 1], index, offsets),
                      "text": text, "raw_text": raw_text, "garbled": garbled(text)})
    return pages


def load_pages(refresh: bool = False) -> list[dict[str, Any]]:
    """All pages of every PDF in pdfs/, kept in data/pages.jsonl and extracted again when the set of
    PDFs changes (or with refresh, e.g. after replacing a PDF under the same name)."""
    out = DATA_DIR / "pages.jsonl"
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs in {PDF_DIR}")
    if out.exists() and not refresh:
        pages = read_jsonl(out)
        if {p["file_name"] for p in pages} == {path.name for path in pdfs}:
            return pages
    pages = [p for path in pdfs for p in extract(path)]
    write_jsonl(out, pages)
    return pages
