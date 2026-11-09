"""Cut each PDF into segments along its printed table of contents, one per top-level entry.

The fact agent reads a segment in parts of a few pages, each carrying its heading path and the
tail of the previous part, so it keeps the context of the whole segment.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai.exceptions import UnexpectedModelBehavior

from bench import llm
from bench.config import DATA_DIR, REQUEST_TIMEOUT, BenchError
from bench.pdf import load_pages, read_jsonl, write_jsonl

SEGMENTS = DATA_DIR / "segments.jsonl"
INGEST = DATA_DIR / "ingest.jsonl"  # per PDF: page statuses, tables, how the TOC was used, segment sizes
TOC_SCAN_PAGES = 25  # the printed TOC sits near the front
ENTRY_LINE = re.compile(r"(\.\s*){4,}|…{2,}|\s\d{1,3}\s*$|^\s*\d{1,3}\s*$")  # dot leaders or a page number
LEAD = r"^\W*(?:[\divxlc.]+\s+)?"  # titles are matched at their start, after an optional number ("VI GLOSSARY")
BOILERPLATE = re.compile(LEAD + r"(preface|foreword|messages?|contents|index|references|bibliography|annexures?|"
                         r"appendix|appendices|list of|abbreviations?|glossary|acknowledg\w*|authors?)\b", re.I)
BACK_MATTER = re.compile(LEAD + r"(annexures?|appendix|appendices|references|bibliography|glossary|index)\b", re.I)
OVERLAP_CHARS = 800  # tail of the previous part handed to the next one
COVER_PAGES = 4  # at most this many unnumbered pages at the end are taken as covers
MIN_TEXT = 500  # a segment needs at least a paragraph of text to be used for questions
SPLIT_CHARS = 30_000  # a longer top-level segment is cut at its level-2 TOC entries...
MIN_PIECE = 6_000  # ...into pieces of at least this many characters

TOC_PROMPT = """You turn the text of a PDF's table of contents into JSON. The text comes from a PDF \
library, so dot leaders, numbers and page numbers may sit on separate lines and long titles may wrap.

Return the entries of the top two levels only, in order: parts or chapters (level 1) and their
direct sections (level 2). Front and back matter such as Preface, Abbreviations, Acknowledgement,
References, Glossary or Annexures are level-1 entries too. Skip deeper sub-sections, lists of tables
or figures, the "Contents" heading and column headers like "Page No.". For each entry:
- number: the label as printed ("2", "1.1", "Section IV", "PART-II"), or "" if none.
- title: the title exactly as printed, wrapped lines joined by one space, without the number, dot leaders or page.
- printed_page: the printed page number as an integer, or null if missing or not an Arabic numeral.
- level: 1 or 2, from the numbering ("2" or "PART-II" is 1; "2.1" or "Section IV" under a part is 2).
Never invent entries."""


class TocEntry(BaseModel):
    number: str
    title: str
    printed_page: int | None
    level: int


class TocEntries(BaseModel):
    entries: list[TocEntry]


def normalize(text: str) -> str:
    """Lowercase, unify quotes and dashes, drop dot leaders, collapse whitespace (for title matching)."""
    text = text.lower().translate(str.maketrans("‘’“”–—", "''\"\"--")).replace("&", " and ")
    return " ".join(re.sub(r"(\.\s*){3,}|…+", " ", text).split())


def slug(file_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", Path(file_name).stem.lower()).strip("-")


def find_toc_pages(pages: list[dict[str, Any]]) -> list[int]:
    """The first front page that mentions contents/index and looks like a TOC, plus the TOC pages after it."""
    def toc_like(text: str) -> bool:
        lines = [l for l in text.splitlines() if l.strip()]
        hits = sum(bool(ENTRY_LINE.search(l)) for l in lines)
        return hits >= 8 and hits >= 0.25 * len(lines)

    front = pages[:TOC_SCAN_PAGES]
    for i, page in enumerate(front):
        if re.search(r"\b(contents|index)\b", page["raw_text"], re.I) and toc_like(page["raw_text"]):
            found = [page["page"]]
            for nxt in front[i + 1 :]:
                if not toc_like(nxt["raw_text"]):
                    break
                found.append(nxt["page"])
            return found
    return []


def read_toc(file_name: str, toc_text: str, dropped: list[str] | None = None) -> list[TocEntry]:
    """Ask the LLM to parse the TOC (answers are cached by bench/llm.py, so a re-run needs no LLM),
    then keep only the entries the checks below accept."""
    try:  # a long TOC needs more output, and more time, than the defaults
        raw = llm.call("toc", llm.agent(TocEntries, TOC_PROMPT), toc_text, file_name,
                       model_settings={"max_tokens": 8_000, "request_timeout": 3 * REQUEST_TIMEOUT}).entries
    except UnexpectedModelBehavior:
        raise  # an answer that can't be parsed: segment_pdf falls back to one segment
    except Exception as exc:  # an outage: saving the whole-PDF fallback would look like a real result
        raise BenchError(f"{file_name}: the TOC agent failed ({type(exc).__name__}: {exc}). Nothing was saved; "
                         "re-run `bench segment` when the LLM endpoint is up") from exc
    return keep_valid(raw, toc_text, dropped)


def _words(text: str) -> str:
    """Normalized words without hyphens or bare numbers, so 'I-MAJOR' or a title wrapped around its page
    number still matches."""
    return " " + " ".join(w for w in normalize(text).replace("-", " ").split() if not re.fullmatch(r"[\d.]+", w)) + " "


def _longest_run(pages: list[int]) -> set[int]:
    """Indexes of the longest non-decreasing run (not necessarily adjacent) of page numbers."""
    best: list[list[int]] = []
    for i, page in enumerate(pages):
        prior = [b for j, b in enumerate(best) if pages[j] <= page]
        best.append(max(prior, key=len, default=[]) + [i])
    return set(max(best, key=len, default=[]))


def keep_valid(entries: list[TocEntry], toc_text: str, dropped: list[str] | None = None) -> list[TocEntry]:
    """Entries whose title really is in the TOC text, put back in the text's order (the model can
    move a line), keeping the longest run of page numbers that never goes backwards: one misread number
    drops only itself. Dropped titles are added to `dropped`."""
    haystack, cursor, taken, found = _words(toc_text), 0, set(), []
    for e in entries:
        needle = _words(e.title)
        if not needle.strip():
            continue
        at = haystack.find(needle, cursor)
        if at < 0:  # earlier in the text: the model moved this entry (or repeated one)
            at = haystack.find(needle)
            while at in taken:
                at = haystack.find(needle, at + 1)
        if at < 0:
            continue  # not in the TOC text, or every match is used: invented
        taken.add(at)
        cursor = max(cursor, at + len(needle) - 1)
        found.append((at, e))
    ordered = [e for _, e in sorted(found, key=lambda pair: pair[0])]
    paged = [i for i, e in enumerate(ordered) if e.printed_page is not None]
    run = {paged[k] for k in _longest_run([ordered[i].printed_page for i in paged])}
    if dropped is not None:
        dropped += [e.title for e in entries if e not in ordered]  # not in the TOC text: invented
        dropped += [ordered[i].title for i in paged if i not in run]  # a page number out of order
    return [e.model_copy(update={"title": e.title.strip(), "level": max(1, e.level)})
            for i, e in enumerate(ordered) if e.printed_page is None or i in run]


def label(e: TocEntry) -> str:
    """'Number Title', without repeating a title the model also put in the number."""
    return e.number.strip() if _words(e.title) in _words(e.number) else f"{e.number} {e.title}".strip()


def _has_line(title: str, page_text: str) -> bool:
    """True if a line of the page, or two adjacent lines joined, is the title itself (a bracketed
    note may follow), as a heading is."""
    heading = re.compile(re.escape(normalize(title)) + r"(\s*\(.*\))?[\s:.]*")
    rows = [normalize(t) for t in page_text.splitlines() if t.strip()]
    return any(heading.fullmatch(r) for r in rows + [f"{a} {b}" for a, b in zip(rows, rows[1:])])


def locate(entries: list[TocEntry], pages: list[dict[str, Any]], after: int) -> tuple[float, list[tuple[TocEntry, int]]]:
    """PDF start page of each entry: the most common (PDF page - printed page) offset among titles
    found in the body, then each entry's own match within 2 pages. An entry without a printed page
    starts at the next page that has its title as a line, or is dropped. Returns (agreement, placed)."""
    body = [p for p in pages if p["page"] > after]
    texts = {p["page"]: normalize(p["raw_text"]) for p in body}
    matches = [[n for n, t in texts.items() if len(normalize(e.title)) >= 4 and normalize(e.title) in t] for e in entries]
    votes = Counter(n - e.printed_page for e, m in zip(entries, matches) if e.printed_page for n in set(m))
    if not votes:
        return 0.0, []
    offset = max(votes.items(), key=lambda kv: (kv[1], -abs(kv[0])))[0]
    titled = [(e, m) for e, m in zip(entries, matches) if e.printed_page and len(normalize(e.title)) >= 4]
    agreement = sum(any(abs(n - (e.printed_page + offset)) <= 1 for n in m) for e, m in titled) / max(len(titled), 1)
    guesses = [e.printed_page + offset if e.printed_page else None for e in entries]
    placed, prev = [], after + 1
    for i, (e, m) in enumerate(zip(entries, matches)):
        if e.printed_page:
            near = [n for n in m if abs(n - guesses[i]) <= 2]
            start = min(near, key=lambda n: (abs(n - guesses[i]), n)) if near else guesses[i]
        else:  # searched no further than the next entry that has a page
            limit = next((g + 2 for g in guesses[i + 1 :] if g), len(pages))
            start = next((p["page"] for p in body if prev <= p["page"] <= limit and _has_line(e.title, p["raw_text"])), None)
            if start is None and placed and placed[-1][0].level > 1 and BOILERPLATE.search(label(e)):
                start = prev + 1  # e.g. a table annex with no heading, right after a (short) sub-section
            if start is None:
                continue  # no page number and no heading in the body: cannot be placed
        start = min(max(start, prev), len(pages))  # never before the previous entry
        placed.append((e, start))
        prev = start
    return agreement, placed


def _segment(seg_id: str, file_name: str, doc_id: str, title: str, pages: list[dict[str, Any]],
             headings: list[dict[str, Any]], for_questions: bool) -> dict[str, Any]:
    """Join the clean, non-garbled pages and record where each page sits in the segment text. A page that
    ends mid-sentence joins the next one as the same paragraph."""
    text, spans = "", []
    for p in pages:
        if p["garbled"] or not p["text"].strip():
            continue
        if text:
            text += "\n" if not re.search(r"[.!?:;)\]]\s*$", text) and p["text"].lstrip()[:1].islower() else "\n\n"
        spans.append({"page": p["page"], "start": len(text), "end": len(text) + len(p["text"])})
        text += p["text"]
    return {
        "segment_id": seg_id, "doc_id": doc_id, "file_name": file_name, "title": title,
        "pages": [pages[0]["page"], pages[-1]["page"]] if pages else [],
        "use_for_questions": for_questions and len(text) >= MIN_TEXT,
        "headings": headings, "page_spans": spans, "text": text,
    }


def _split(seg_id: str, segment_pages: list[dict[str, Any]], headings: list[dict[str, Any]], make: Any) -> list[dict[str, Any]]:
    """A top-level segment longer than SPLIT_CHARS, cut at the pages where its level-2 TOC entries start.
    Pieces are grouped in order until a group holds MIN_PIECE characters (a short last group joins the one
    before), so many short sections, such as one table per page, don't collapse back into one. Each piece keeps
    the parent heading, so its heading path still names the chapter."""
    top, first = headings[0], segment_pages[0]["page"]
    cuts = sorted({h["page"] for h in headings[1:] if h["level"] == top["level"] + 1 and h["page"] > first})
    bounds = [first, *cuts, segment_pages[-1]["page"] + 1]
    pieces = [[p for p in segment_pages if a <= p["page"] < b] for a, b in zip(bounds, bounds[1:])]
    size = lambda group: sum(len(p["text"]) for p in group)
    merged: list[list[dict[str, Any]]] = []
    for piece in pieces:
        if merged and size(merged[-1]) < MIN_PIECE:
            merged[-1] = merged[-1] + piece
        else:
            merged.append(piece)
    if len(merged) > 1 and size(merged[-1]) < MIN_PIECE:
        merged[-2:] = [merged[-2] + merged[-1]]
    if len(merged) < 2:
        return [make(seg_id, top["title"], segment_pages, headings)]
    out = []
    for j, piece in enumerate(merged, start=1):
        inside = [h for h in headings[1:] if piece[0]["page"] <= h["page"] <= piece[-1]["page"]]
        title = top["title"] if j == 1 else f"{top['title']} > {inside[0]['title'] if inside else f'part {j}'}"
        out.append(make(f"{seg_id}-{j}", title, piece, [top, *inside]))
    return out


def segment_pdf(pages: list[dict[str, Any]], report: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Segments of one PDF: front matter, one per top-level TOC entry (or the whole PDF if no TOC),
    then the covers: unnumbered pages after the last printed page number. A top-level entry longer than
    SPLIT_CHARS is cut at its level-2 entries. `report` receives how the TOC was used (for data/ingest.jsonl)."""
    report = {} if report is None else report
    file_name, doc_id, base = pages[0]["file_name"], pages[0]["doc_id"], slug(pages[0]["file_name"])
    numbered = [p["page"] for p in pages if p["printed"] is not None]
    last = numbered[-1] if numbered and len(pages) - numbered[-1] <= COVER_PAGES else len(pages)
    toc_pages, dropped = find_toc_pages(pages), []
    try:
        entries = read_toc(file_name, "\n".join(pages[n - 1]["raw_text"] for n in toc_pages), dropped) if toc_pages else []
    except UnexpectedModelBehavior as exc:  # an answer that can't be parsed must not stop the other PDFs
        print(f"  {file_name}: TOC could not be parsed ({type(exc).__name__})")
        entries = []
    agreement, placed = locate(entries, pages, toc_pages[-1]) if entries else (0.0, [])
    unplaced = [e.title for e in entries if e not in [p[0] for p in placed]]
    report.update({"toc_pages": toc_pages, "toc_entries": len(entries), "placed": len(placed), "agreement": round(agreement, 2),
                   "dropped_titles": dropped, "unplaced_titles": unplaced})
    if len(placed) < 3 or agreement < 0.5:
        report["segmented_by"] = "whole PDF (no usable TOC)"
        print(f"  {file_name}: no usable TOC (agreement {agreement:.0%}); one segment for the whole PDF")
        segments = [_segment(f"{base}-01", file_name, doc_id, Path(file_name).stem, pages[:last], [], True)]
        segments.append(_segment(f"{base}-99", file_name, doc_id, "Covers", pages[last:], [], False))
        return [s for s in segments if s["text"]]

    headings = [{"title": label(e), "level": e.level, "page": start} for e, start in placed]
    top_level = min(h["level"] for h in headings)
    tops = [i for i, h in enumerate(headings) if h["level"] == top_level]
    first = headings[tops[0]]["page"]
    last = max(last, headings[tops[-1]]["page"])  # the covers never swallow the last entry
    report["segmented_by"] = "TOC"
    segments = [_segment(f"{base}-00", file_name, doc_id, "Front matter", pages[: first - 1], [], False)] if first > 1 else []
    body_started = body_ended = False
    for k, i in enumerate(tops, start=1):
        nxt = tops[k] if k < len(tops) else None
        end = headings[nxt]["page"] - 1 if nxt is not None else last
        title = headings[i]["title"]
        # the back matter starts at the first annexure, references, index... after the body; none of it is used
        body_ended = body_ended or (body_started and bool(BACK_MATTER.match(title)))
        usable = not body_ended and not BOILERPLATE.search(title)
        body_started = body_started or usable
        seg_pages = pages[headings[i]["page"] - 1 : end]

        def make(seg_id: str, seg_title: str, piece: list[dict[str, Any]], heads: list[dict[str, Any]], usable: bool = usable) -> dict[str, Any]:
            return _segment(seg_id, file_name, doc_id, seg_title, piece, heads, usable)

        whole = make(f"{base}-{k:02d}", title, seg_pages, headings[i:nxt])
        segments += _split(f"{base}-{k:02d}", seg_pages, headings[i:nxt], make) if len(whole["text"]) > SPLIT_CHARS else [whole]
    segments.append(_segment(f"{base}-99", file_name, doc_id, "Covers", pages[last:], [], False))
    return [s for s in segments if s["text"]]  # an entry that shares its page with the next has no text of its own


def heading_path(segment: dict[str, Any], page: int) -> str:
    """'Chapter > Section > Sub-section' in force at a page, from the segment's TOC headings."""
    stack: list[dict[str, Any]] = []
    for h in segment["headings"]:
        if h["page"] > page:
            break
        while stack and stack[-1]["level"] >= h["level"]:
            stack.pop()
        stack.append(h)
    return " > ".join(h["title"] for h in stack) or segment["title"]


def page_at(segment: dict[str, Any], pos: int) -> int:
    """The page a position falls on (a separator between pages counts with the page before it)."""
    return next((s["page"] for s in reversed(segment["page_spans"]) if s["start"] <= pos), segment["pages"][0])


def parts(segment: dict[str, Any], max_chars: int) -> list[dict[str, Any]]:
    """Consecutive parts of at most max_chars, cut at paragraph breaks where possible.
    Each part carries its heading path and the tail of the previous part as context."""
    text, cuts, start = segment["text"], [], 0
    while len(text) - start > max_chars:
        window = text[start : start + max_chars]
        cut = window.rfind("\n\n")  # a paragraph break in the second half...
        if cut < max_chars // 2:
            cut = window.rfind(". ") + 1  # ...else a sentence end...
        if cut < max_chars // 2:
            cut = max_chars  # ...else a hard cut
        cuts.append((start, start + cut))
        start += cut
    cuts.append((start, len(text)))
    return [
        {"start": s, "end": e, "text": text[s:e].strip(), "heading": heading_path(segment, page_at(segment, s)),
         "context": text[max(0, s - OVERLAP_CHARS) : s].strip() if s else ""}
        for s, e in cuts
    ]


PARAGRAPH = re.compile(r"(?:[^\n]|\n(?![ \t]*\n))+")  # lines up to a blank line
SENTENCE_END = re.compile(r"[.!?][\"”’)]?\s+(?=[\"“‘(]?[A-Z0-9])|[।॥]\s+")  # also the Hindi danda
# Not a sentence end: common abbreviations, Indian-English ones (Rs. 1,500; Govt. hospital; Sl. No.) and list numbers.
NOT_AN_END = re.compile(r"(?:\b(?:e\.g|i\.e|viz|etc|approx|nos?|fig|vs|dr|mr|mrs|ms|st|rs|re|govt|deptt?|sl|sh|smt|pvt|"
                        r"ltd|distt?|prof|asst|addl|dy|jt|para|sec|vol|al)|(?:^|\s)\d{1,2})\.$", re.I)
INITIAL = re.compile(r"\b[A-Z]\.$")  # "J. Smith"


def _units(para: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence of a paragraph; a table row (cells joined by " | ", bench/pdf.py) is one unit."""
    pieces: list[tuple[int, int, bool]] = []  # each table row alone, the lines between rows together
    for line in re.finditer(r"[^\n]*(?:\n|$)", para):
        if not line.group():
            break
        row = " | " in line.group()
        if pieces and not row and not pieces[-1][2]:
            pieces[-1] = (pieces[-1][0], line.end(), False)
        else:
            pieces.append((line.start(), line.end(), row))
    units = []
    for a, b, row in pieces:
        chunk = para[a:b]
        cuts = [] if row else [m.start() + 1 for m in SENTENCE_END.finditer(chunk)
                               if not NOT_AN_END.search(chunk[: m.start() + 1]) and not INITIAL.search(chunk[: m.start() + 1])]
        bounds = [0, *cuts, len(chunk)]
        units += [(a + x, a + y) for x, y in zip(bounds, bounds[1:])]
    return units


def sentences(text: str) -> list[dict[str, Any]]:
    """The text cut into paragraphs (at blank lines) and sentences, each with an id such as 'p3.s2' and
    its (start, end) in the text; each table row is a sentence of its own. Agents point at these ids and the
    quote is cut here, never retyped."""
    found = []
    for p, para in enumerate((m for m in PARAGRAPH.finditer(text) if m.group().strip()), start=1):
        s = 0
        for a, b in _units(para.group()):
            piece = para.group()[a:b]
            if not piece.strip():
                continue
            s += 1
            start = para.start() + a + len(piece) - len(piece.lstrip())
            found.append({"id": f"p{p}.s{s}", "start": start, "end": para.start() + a + len(piece.rstrip())})
    return found


def run_segments(refresh: bool = False) -> list[dict[str, Any]]:
    """Segments for every PDF -> data/segments.jsonl."""
    pages = load_pages(refresh=refresh)
    by_pdf: dict[str, list[dict[str, Any]]] = {}
    for p in pages:
        by_pdf.setdefault(p["file_name"], []).append(p)
    segments, ingest = [], []
    for file_name, pdf_pages in sorted(by_pdf.items()):
        report: dict[str, Any] = {}
        ordered = sorted(pdf_pages, key=lambda p: p["page"])
        found = segment_pdf(ordered, report)
        usable = sum(s["use_for_questions"] for s in found)
        print(f"{file_name}: {len(found)} segments ({usable} used for questions)")
        segments += found
        ingest.append({"file_name": file_name, "doc_id": ordered[0]["doc_id"], "pages": len(ordered),
                       "status": dict(Counter(p.get("status", "garbled" if p["garbled"] else "text") for p in ordered)),
                       "tables": sum(p.get("tables", 0) for p in ordered), **report, "segments": len(found),
                       "used_for_questions": usable, "largest_segment_chars": max((len(s["text"]) for s in found), default=0)})
    write_jsonl(SEGMENTS, segments)
    write_jsonl(INGEST, ingest)
    return segments


def load_segments() -> list[dict[str, Any]]:
    if not SEGMENTS.exists():
        raise BenchError("data/segments.jsonl missing: run `bench segment` first")
    return read_jsonl(SEGMENTS)
