"""Checks on every stage's saved output, without the LLM: `bench check`, and after each stage of `bench run`.

Each check is (name, ok, detail). A failed check means a stage's output is wrong, missing, or out of step
with the stage before it, and the fix is to re-run that stage.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bench.config import DATA_DIR, PDF_DIR, question_types
from bench.evaluate import REPORT
from bench.facts import FACTS, FACTS_VERSION
from bench.pdf import PAGES, read_jsonl
from bench.questions import LIBRARY, QUESTIONS, QUESTIONS_VERSION
from bench.segments import SEGMENTS, TOC_VERSION, find_toc_pages, slug
from bench.testbed import CHUNKS, COLLECTION, qdrant
from bench.verify import MAX_QUOTE_CHARS, claim_problem, find, question_problem, table_row

RESULTS = DATA_DIR / "results.jsonl"
Check = tuple[str, bool, str]


def reason_group(reason: str) -> str:
    """One name per audit rule: '"goats" is in the library…' and 'needs 2-5 words, has 9' lose their specifics."""
    rule = re.sub(r"^overlaps .*", "overlaps a kept fact", reason.split(":")[0])
    return re.sub(r"\d+", "N", re.sub(r'"[^"]*"', "a term", rule))


def _rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _missing(path: Path, stage: str) -> list[Check]:
    return [(f"{path.name} exists", False, f"run `bench {stage}` first")]


def check_segment() -> list[Check]:
    """Pages, TOC parses and segments: every page with text sits in exactly one segment."""
    if not PAGES.exists() or not SEGMENTS.exists():
        return _missing(SEGMENTS, "segment")
    pages, segments = read_jsonl(PAGES), read_jsonl(SEGMENTS)
    by_pdf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in pages:
        by_pdf[p["file_name"]].append(p)
    pdfs = {path.name for path in PDF_DIR.glob("*.pdf")}
    stale_toc = []
    for name, pdf_pages in by_pdf.items():
        cache = DATA_DIR / "toc" / f"{slug(name)}.json"
        has_toc = bool(find_toc_pages(sorted(pdf_pages, key=lambda p: p["page"])))
        if has_toc and (not cache.exists() or json.loads(cache.read_text()).get("version") != TOC_VERSION):
            stale_toc.append(name)
    text = {(p["file_name"], p["page"]): p["text"] for p in pages}
    expected = {(p["file_name"], p["page"]) for p in pages if p["text"].strip() and not p["garbled"]}
    seen = Counter((s["file_name"], span["page"]) for s in segments for span in s["page_spans"])
    twice, lost = [k for k, n in seen.items() if n > 1], sorted(expected - set(seen))
    misplaced = [s["segment_id"] for s in segments for span in s["page_spans"]
                 if s["text"][span["start"] : span["end"]] != text[(s["file_name"], span["page"])]]
    used = sum(s["use_for_questions"] for s in segments)
    return [
        ("every PDF in pdfs/ is extracted", set(by_pdf) == pdfs,
         f"{len(pdfs)} PDFs, {len(pages)} pages, {sum(p['garbled'] for p in pages)} junk-OCR pages left out"),
        ("every TOC parse is cached at the current version", not stale_toc, ", ".join(stale_toc) or TOC_VERSION),
        ("every page with text is in exactly one segment", not twice and not lost,
         f"{len(segments)} segments, {used} used for questions" + (f"; twice: {twice[:3]}" if twice else "")
         + (f"; in none: {lost[:3]}" if lost else "")),
        ("page positions point at each page's text", not misplaced, ", ".join(sorted(set(misplaced))[:3]) or "ok"),
    ]


def check_facts() -> list[Check]:
    """Facts: current prompt version, quotes still verbatim at their positions, fact rules hold."""
    if not FACTS.exists():
        return _missing(FACTS, "facts")
    segments = {s["segment_id"]: s for s in _rows(SEGMENTS)}
    rows = read_jsonl(FACTS)
    done = {r["segment_id"] for r in rows if r["version"] == FACTS_VERSION}
    todo = [sid for sid, s in segments.items() if s["use_for_questions"] and sid not in done]
    moved, broken = [], []
    for r in rows:
        segment = segments.get(r["segment_id"], {"text": "", "title": "", "file_name": "", "headings": []})
        for f in r["facts"]:
            if find(f["quote"], segment["text"][f["start"] : f["end"]]) is None:
                moved.append(f["fact_id"])
            if len(f["quote"]) > MAX_QUOTE_CHARS or table_row(f["quote"]) or claim_problem(f["claim"], f["quote"], segment):
                broken.append(f["fact_id"])
    kept, rejected = sum(len(r["facts"]) for r in rows), Counter(x["reason"] for r in rows for x in r["rejected"])
    reasons = Counter(reason_group(reason) for reason in rejected.elements())
    return [
        (f"every question segment has facts ({FACTS_VERSION})", not todo, f"{len(done)} segments done" + (f"; to do: {todo[:3]}" if todo else "")),
        ("every quote is verbatim at its recorded position", not moved, f"{kept} facts" + (f"; moved: {moved[:3]}" if moved else "")),
        ("every fact follows the fact rules", not broken, f"the audit rejected {sum(rejected.values())}: {dict(reasons)}"
         + (f"; breaking: {broken[:3]}" if broken else "")),
    ]


def check_questions() -> list[Check]:
    """Questions: valid types, current version, still pass the audit against their facts, evidence still in place."""
    try:
        types = question_types()
    except SystemExit as exc:
        return [("question_types.toml is valid", False, str(exc))]
    if not QUESTIONS.exists():
        return _missing(QUESTIONS, "questions")
    segments = {s["segment_id"]: s for s in _rows(SEGMENTS)}
    facts = {f["fact_id"]: f for r in _rows(FACTS) for f in r["facts"]}
    corpus = "\n".join(p["text"] for p in _rows(PAGES))
    rows = read_jsonl(QUESTIONS)
    facts_version = {r["segment_id"]: r["version"] for r in _rows(FACTS)} | {LIBRARY: FACTS_VERSION}
    done = {r["segment_id"] for r in rows if r["version"] == QUESTIONS_VERSION and r.get("facts_version") == facts_version.get(r["segment_id"])}
    todo = [sid for sid in {r["segment_id"] for r in _rows(FACTS) if r["facts"]} if sid not in done]
    failing, moved = [], []
    questions = [q for r in rows for q in r["questions"]]
    for q in questions:
        if q["type"] not in types:
            failing.append(f"{q['question_id']} (type {q['type']} is not in question_types.toml)")
        elif problem := question_problem(q, facts, types[q["type"]].get("rules", {}), corpus):
            failing.append(f"{q['question_id']} ({problem})")
        if any(find(ev["quote"], segments.get(ev.get("segment_id", ""), {}).get("text", "")[ev["start"] : ev["end"]]) is None
               for ev in q["evidence"]):  # rows before questions-v6 have no segment_id: they fail here, and are redone
            moved.append(q["question_id"])
    counts = Counter(q["type"] for q in questions)
    rejected = Counter(reason_group(x["reason"]) for r in rows for x in r["rejected"])
    return [
        ("question_types.toml is valid", True, f"{len(types)} types: {', '.join(types)}"),
        (f"every segment with facts has questions ({QUESTIONS_VERSION})", not todo, f"{len(done)} rows done" + (f"; to do: {todo[:3]}" if todo else "")),
        ("every question still passes the audit", not failing, f"{len(questions)} questions {dict(counts)}" + (f"; failing: {failing[:3]}" if failing else "")),
        ("every evidence quote is at its recorded position", not moved, ", ".join(moved[:3]) or f"the audit rejected {sum(rejected.values())}: {dict(rejected)}"),
    ]


def check_index() -> list[Check]:
    """Test bed: chunks cover each segment's text exactly, and Qdrant holds every chunk."""
    if not CHUNKS.exists():
        return _missing(CHUNKS, "index")
    chunks, segments = read_jsonl(CHUNKS), _rows(SEGMENTS)
    by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        by_segment[c["segment_id"]].append(c)
    gaps = [s["segment_id"] for s in segments
            if "".join(c["text"] for c in sorted(by_segment[s["segment_id"]], key=lambda c: c["start"])) != s["text"]]
    try:
        points = qdrant().count(COLLECTION, exact=True).count
    except Exception:  # Qdrant not running
        points = None
    return [
        ("chunks cover every segment's text exactly", not gaps, f"{len(chunks)} chunks" + (f"; out of step: {gaps[:3]}" if gaps else "")),
        ("Qdrant holds every chunk", points == len(chunks),
         f"{points} points in {COLLECTION}" if points is not None else "Qdrant is not reachable: docker compose up -d"),
    ]


def check_evaluate() -> list[Check]:
    """Results: one per current question, and a report."""
    if not RESULTS.exists():
        return _missing(RESULTS, "evaluate")
    scored = {r["question_id"] for r in read_jsonl(RESULTS)}
    current = {q["question_id"] for r in _rows(QUESTIONS) for q in r["questions"]}
    return [("every current question is scored", scored == current and REPORT.exists(),
             f"{len(scored)} scored of {len(current)} questions; report {'written' if REPORT.exists() else 'missing'}")]


STAGES = {"segment": check_segment, "facts": check_facts, "questions": check_questions, "index": check_index,
          "evaluate": check_evaluate}


def all_checks(stages: list[str] | None = None) -> dict[str, list[Check]]:
    return {name: STAGES[name]() for name in stages or STAGES}


def print_checks(stages: list[str] | None = None) -> bool:
    """Print each stage's checks; True if all pass."""
    ok = True
    for stage, checks in all_checks(stages).items():
        print(f"{stage}")
        for name, passed, detail in checks:
            print(f"  {'✓' if passed else '✗'} {name}: {detail}")
            ok &= passed
    return ok
