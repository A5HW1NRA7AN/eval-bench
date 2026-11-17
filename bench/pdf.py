"""Clean page text from the PDFs, through PyMuPDF's layout engine: reading order, paragraphs and tables with
their headers kept; running headers, page numbers and stray glyphs removed; a status for every page."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf
import pymupdf4llm

from bench.config import DATA_DIR, PDF_DIR, BenchError

EDGE_LINES = 3  # lines at the top and bottom of a page's raw text where a printed page number may be
REPEATS = 4  # a running header/footer must repeat on this many pages
GLYPHS = {"yy", "□", "•"}  # bullet/end marks that come out as blocks of their own
SOURCE_NOTE = re.compile(r"^\s*(source|note|notes)\b", re.I)  # table sources and notes are content, never running lines
LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"})
NUMBER = re.compile(r"^[\d.,%()\-–—:/]+$|^NIL$")
PAGE_NO = re.compile(r"^\d{1,4}\b\s*|\s*\b\d{1,4}$")
BARE_INT = re.compile(r"^\d{1,4}$")
YEAR = re.compile(r"^(19|20)\d{2}$")
PAGES = DATA_DIR / "pages.jsonl"


def read_jsonl(path: Path, missing_ok: bool = False) -> list[dict[str, Any]]:
    """The rows of a JSONL file; with missing_ok, [] for a file no stage has written yet. A bad line is an error,
    except a torn last line of the agent trace, which is appended to rather than rewritten."""
    if missing_ok and not path.exists():
        return []
    rows = []
    lines = path.read_text().splitlines()
    for n, line in enumerate(lines, start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                if n == len(lines) and path.name == "trace.jsonl":
                    break
                raise BenchError(f"{path.name} line {n} isn't valid JSON ({exc.msg}); re-run the stage that writes it") from exc
    return rows


def write_text(path: Path, text: str) -> None:
    """Write to a temporary file and rename it over the target, so a stop mid-write never loses the last version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:  # including Ctrl+C: the target keeps its last complete version
        tmp.unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _edges(lines: list[str]) -> list[str]:
    return lines if len(lines) <= 2 * EDGE_LINES else lines[:EDGE_LINES] + lines[-EDGE_LINES:]


def _key(line: str) -> str | None:
    """'Dairy Farming 12' and 'Page 3 of 69' share a key with their other pages (a page number at either end,
    or 'page N of M', doesn't count); number-only lines get none (table values), but a repeated header year does."""
    norm = " ".join(line.lower().split())
    if any(c.isalpha() for c in norm):
        return PAGE_NO.sub("#", re.sub(r"\bpage \d+ of \d+\b", "page # of #", norm))
    return norm if YEAR.match(norm) else None


def page_offsets(pages: list[list[str]]) -> set[int]:
    """Printed-minus-PDF page offsets: bare numbers at the edges of the raw text that advance with the page."""
    offsets: Counter[int] = Counter()
    for index, lines in enumerate(pages, start=1):
        offsets.update({int(l) - index for l in (l.strip() for l in _edges(lines)) if BARE_INT.match(l)})
    return {o for o, n in offsets.items() if n >= REPEATS}


def printed_number(lines: list[str], index: int, offsets: set[int]) -> int | None:
    """The page number printed on the page: an edge line holding a number that advances with the page."""
    edges = [l.strip() for l in _edges(lines)]
    return next((int(s) for s in edges if BARE_INT.match(s) and int(s) - index in offsets), None)


def garbled(text: str) -> bool:
    """Junk from a bad OCR layer ('0 CL o c —'): mostly 1-2 character tokens once numbers are set aside."""
    words = [t for t in text.split() if not NUMBER.match(t)]
    return len(words) >= 50 and sum(len(w) <= 2 for w in words) / len(words) > 0.6


def normalize(text: str) -> str:
    """One clean form of the text: soft hyphens and control characters dropped, OCR's line-end '¬' as a hyphen,
    ligatures spelled out, private-use bullet glyphs as '•', non-breaking spaces as spaces."""
    text = text.translate(LIGATURES).replace("\u00ad", "").replace("\u00a0", " ")
    text = re.sub(r"¬\s*", "-", text)
    text = re.sub(r"[\ue000-\uf8ff]", "•", text)
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)


def _line_text(line: dict[str, Any]) -> str:
    return " ".join(" ".join(s["text"] for s in line["spans"]).split())


def _box_text(box: dict[str, Any]) -> str:
    """A layout block's lines; a word broken at a line end ('milk-' + 'production') is joined, keeping the hyphen."""
    out = ""
    for line in (t for t in (_line_text(l) for l in box.get("textlines") or []) if t):
        out += line if not out else ("" if out.endswith("-") and line[:1].islower() else "\n") + line
    return out


def _table_text(table: dict[str, Any]) -> str:
    """A table as one line per row, its cells joined by ' | ' and its header rows kept as printed, so a row stays
    whole instead of breaking into one cell per line."""
    rows = [[" ".join(str(c or "").split()) for c in row] for row in table.get("extract") or []]
    return "\n".join(" | ".join(row) for row in rows if any(row))


def _blocks(page: dict[str, Any]) -> list[tuple[str, str]]:
    """(class, text) of a layout page's blocks in reading order, without page headers, footers and formulas."""
    blocks = []
    for box in page["boxes"]:
        kind = box["boxclass"]
        if kind in ("page-header", "page-footer", "formula"):
            continue
        text = _table_text(box["table"]) if kind == "table" and box.get("table") else _box_text(box)
        if text.strip() and text.strip() not in GLYPHS:
            blocks.append((kind, normalize(text).strip()))
    return blocks


def _join(blocks: list[str]) -> str:
    """Blocks separated by a blank line, except a block that continues the previous one's sentence."""
    out = ""
    for block in blocks:
        continues = out and not re.search(r"[.!?:;)\]]\s*$", out) and block[:1].islower()
        out += ("\n" if continues else "\n\n" if out else "") + block
    return out


def page_status(page: pymupdf.Page, text: str) -> str:
    """What a page holds, by code: text, blank, image_only (a scan or picture with no text layer), garbled
    (a bad OCR layer) or broken_font (text that extracts as replacement characters)."""
    if text.count("\ufffd") > max(5, len(text) // 100):
        return "broken_font"
    if garbled(text.replace("|", " ")):  # table cell separators aren't words
        return "garbled"
    if len(text.strip()) >= 100:
        return "text"
    area = page.rect.width * page.rect.height or 1
    covered = sum(abs(pymupdf.Rect(i["bbox"])) for i in page.get_image_info())
    return "image_only" if covered / area > 0.2 or len(page.get_drawings()) > 50 else "blank" if not text.strip() else "text"


def extract(path: Path) -> list[dict[str, Any]]:
    """Every page of one PDF: clean text from PyMuPDF's layout engine (reading order, page headers and footers
    left out, tables written row by row under their headers), the raw text (for the table of contents), the
    printed page number and a status. Running lines the layout engine misses are removed when they repeat at
    the top or bottom of REPEATS pages."""
    doc_id = hashlib.sha256(path.read_bytes()).hexdigest()
    layout = json.loads(pymupdf4llm.to_json(str(path), use_ocr=False, show_progress=False))["pages"]
    with pymupdf.open(path) as doc:
        raw = [page.get_text() for page in doc]
        lines = [[l for l in text.splitlines() if l.strip()] for text in raw]
        offsets = page_offsets(lines)
        blocks = [_blocks(page) for page in layout]
        edges = Counter(k for page in blocks for _, text in page[:1] + page[-1:] if (k := _key(text)) and len(text) <= 80)
        running = {k for k, n in edges.items() if n >= REPEATS}
        pages = []
        for index, (page, page_blocks) in enumerate(zip(doc, blocks), start=1):
            last = len(page_blocks) - 1
            kept = [(kind, text) for i, (kind, text) in enumerate(page_blocks)
                    if not (i in (0, last) and len(text) <= 80 and not SOURCE_NOTE.match(text)
                            and (_key(text) in running or (BARE_INT.match(text) and int(text) - index in offsets)))]
            text = _join([t for _, t in kept])
            status = page_status(page, text)
            pages.append({"doc_id": doc_id, "file_name": path.name, "page": index,
                          "printed": printed_number(lines[index - 1], index, offsets), "text": text, "raw_text": raw[index - 1],
                          "status": status, "garbled": status in ("garbled", "broken_font"),
                          "tables": sum(kind == "table" for kind, _ in page_blocks)})
    return pages


def load_pages(refresh: bool = False) -> list[dict[str, Any]]:
    """All pages of every PDF in pdfs/, kept in data/pages.jsonl and extracted again when the set of
    PDFs changes (or with refresh, e.g. after replacing a PDF under the same name)."""
    out = PAGES
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise BenchError(f"no PDFs in {PDF_DIR}")
    if out.exists() and not refresh:
        pages = read_jsonl(out)
        if {p["file_name"] for p in pages} == {path.name for path in pdfs}:
            return pages
    pages = [p for path in pdfs for p in extract(path)]
    write_jsonl(out, pages)
    return pages
