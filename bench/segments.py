"""Cut each PDF into segments along its printed table of contents, one per top-level entry.

The fact agent reads a segment in parts of a few pages, each carrying its heading path and the
tail of the previous part, so it keeps the context of the whole segment.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput

from bench import llm
from bench.config import DATA_DIR
from bench.pdf import load_pages, read_jsonl, write_jsonl

SEGMENTS = DATA_DIR / "segments.jsonl"
TOC_SCAN_PAGES = 25  # the printed TOC sits near the front
ENTRY_LINE = re.compile(r"(\.\s*){4,}|…{2,}|\s\d{1,3}\s*$|^\s*\d{1,3}\s*$")  # dot leaders or a page number
LEAD = r"^\W*(?:[\divxlc.]+\s+)?"  # titles are matched at their start, after an optional number ("VI GLOSSARY")
BOILERPLATE = re.compile(LEAD + r"(preface|foreword|messages?|contents|index|references|bibliography|annexures?|"
                         r"appendix|appendices|list of|abbreviations?|glossary|acknowledg\w*|authors?)\b", re.I)
BACK_MATTER = re.compile(LEAD + r"(annexures?|appendix|appendices|references|bibliography|glossary|index)\b", re.I)
OVERLAP_CHARS = 800  # tail of the previous part handed to the next one
COVER_PAGES = 4  # at most this many unnumbered pages at the end are taken as covers
MIN_TEXT = 500  # a segment needs at least a paragraph of text to be used for questions
TOC_VERSION = "toc-v3"  # bump when TOC_PROMPT changes; older caches are re-parsed

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
    text = text.lower().translate(str.maketrans("‘’“”–—", "''\"\"--"))
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


def read_toc(file_name: str, toc_text: str) -> list[TocEntry]:
    """Ask the LLM to parse the TOC once. Its raw answer is cached in data/toc/, so re-runs are
    offline and changes to the checks below never need the LLM again."""
    cache = DATA_DIR / "toc" / f"{slug(file_name)}.json"
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    if cached.get("version") == TOC_VERSION:
        raw = [TocEntry(**e) for e in cached["entries"]]
    else:
        agent = Agent(llm.model(), output_type=NativeOutput(TocEntries, strict=True), instructions=TOC_PROMPT)
        # a long TOC needs more than the default output cap
        raw = llm.call("toc", agent, toc_text, file_name, model_settings={"max_tokens": 8_000}).entries
        cache.parent.mkdir(parents=True, exist_ok=True)
        record = {"version": TOC_VERSION, "model": llm.settings().llm_model, "entries": [e.model_dump() for e in raw]}
        cache.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    return keep_valid(raw, toc_text)


def _words(text: str) -> str:
    """Normalized words without hyphens or bare numbers, so 'I-MAJOR' or a title wrapped around its page
    number still matches."""
    return " " + " ".join(w for w in normalize(text).replace("-", " ").split() if not re.fullmatch(r"[\d.]+", w)) + " "


def keep_valid(entries: list[TocEntry], toc_text: str) -> list[TocEntry]:
    """Entries whose title really is in the TOC text, put back in the text's order (the model can
    move a line), whose page numbers never go backwards."""
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
    kept, last = [], 0
    for _, e in sorted(found, key=lambda pair: pair[0]):
        if e.printed_page is not None:
            if e.printed_page < last:
                continue
            last = e.printed_page
        kept.append(e.model_copy(update={"title": e.title.strip(), "level": max(1, e.level)}))
    return kept


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
    """Join the clean, non-garbled pages and record where each page sits in the segment text."""
    texts, spans, pos = [], [], 0
    for p in pages:
        if p["garbled"] or not p["text"].strip():
            continue
        texts.append(p["text"])
        spans.append({"page": p["page"], "start": pos, "end": pos + len(p["text"])})
        pos += len(p["text"]) + 2
    text = "\n\n".join(texts)
    return {
        "segment_id": seg_id, "doc_id": doc_id, "file_name": file_name, "title": title,
        "pages": [pages[0]["page"], pages[-1]["page"]] if pages else [],
        "use_for_questions": for_questions and len(text) >= MIN_TEXT,
        "headings": headings, "page_spans": spans, "text": text,
    }


def segment_pdf(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Segments of one PDF: front matter, one per top-level TOC entry (or the whole PDF if no TOC),
    then the covers: unnumbered pages after the last printed page number."""
    file_name, doc_id, base = pages[0]["file_name"], pages[0]["doc_id"], slug(pages[0]["file_name"])
    numbered = [p["page"] for p in pages if p["printed"] is not None]
    last = numbered[-1] if numbered and len(pages) - numbered[-1] <= COVER_PAGES else len(pages)
    toc_pages = find_toc_pages(pages)
    try:
        entries = read_toc(file_name, "\n".join(pages[n - 1]["raw_text"] for n in toc_pages)) if toc_pages else []
    except Exception as exc:  # an unparseable TOC must not stop the other PDFs
        print(f"  {file_name}: TOC could not be parsed ({type(exc).__name__})")
        entries = []
    agreement, placed = locate(entries, pages, toc_pages[-1]) if entries else (0.0, [])
    if len(placed) < 3 or agreement < 0.5:
        print(f"  {file_name}: no usable TOC (agreement {agreement:.0%}); one segment for the whole PDF")
        segments = [_segment(f"{base}-01", file_name, doc_id, Path(file_name).stem, pages[:last], [], True)]
        segments.append(_segment(f"{base}-99", file_name, doc_id, "Covers", pages[last:], [], False))
        return [s for s in segments if s["text"]]

    headings = [{"title": label(e), "level": e.level, "page": start} for e, start in placed]
    top_level = min(h["level"] for h in headings)
    tops = [i for i, h in enumerate(headings) if h["level"] == top_level]
    first = headings[tops[0]]["page"]
    last = max(last, headings[tops[-1]]["page"])  # the covers never swallow the last entry
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
        segments.append(_segment(f"{base}-{k:02d}", file_name, doc_id, title, pages[headings[i]["page"] - 1 : end],
                                 headings[i:nxt], usable))
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
    return next((s["page"] for s in segment["page_spans"] if s["start"] <= pos < s["end"] + 2), segment["pages"][0])


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
SENTENCE_END = re.compile(r"[.!?][\"”’)]?\s+(?=[\"“‘(]?[A-Z0-9])")
NOT_AN_END = re.compile(r"(?:\b(?:e\.g|i\.e|viz|etc|approx|nos?|fig|vs|dr|mr|mrs|ms|st)|(?:^|\s)\d{1,2})\.$", re.I)
INITIAL = re.compile(r"\b[A-Z]\.$")  # "J. Smith"


def sentences(text: str) -> list[dict[str, Any]]:
    """The text cut into paragraphs (at blank lines) and sentences, each with an id such as 'p3.s2' and
    its (start, end) in the text. Agents point at these ids and the quote is cut here, never retyped."""
    found = []
    for p, para in enumerate((m for m in PARAGRAPH.finditer(text) if m.group().strip()), start=1):
        cuts = [m.start() + 1 for m in SENTENCE_END.finditer(para.group())
                if not NOT_AN_END.search(para.group()[: m.start() + 1]) and not INITIAL.search(para.group()[: m.start() + 1])]
        bounds = [0, *cuts, len(para.group())]
        for s, (a, b) in enumerate(zip(bounds, bounds[1:]), start=1):
            piece = para.group()[a:b]
            start = para.start() + a + len(piece) - len(piece.lstrip())
            found.append({"id": f"p{p}.s{s}", "start": start, "end": para.start() + a + len(piece.rstrip())})
    return found


def run_segments(refresh: bool = False) -> list[dict[str, Any]]:
    """Segments for every PDF -> data/segments.jsonl."""
    pages = load_pages(refresh=refresh)
    by_pdf: dict[str, list[dict[str, Any]]] = {}
    for p in pages:
        by_pdf.setdefault(p["file_name"], []).append(p)
    segments = []
    for file_name, pdf_pages in sorted(by_pdf.items()):
        found = segment_pdf(sorted(pdf_pages, key=lambda p: p["page"]))
        usable = sum(s["use_for_questions"] for s in found)
        print(f"{file_name}: {len(found)} segments ({usable} used for questions)")
        segments += found
    write_jsonl(SEGMENTS, segments)
    return segments


def load_segments() -> list[dict[str, Any]]:
    if not SEGMENTS.exists():
        raise SystemExit("data/segments.jsonl missing: run `bench segment` first")
    return read_jsonl(SEGMENTS)
