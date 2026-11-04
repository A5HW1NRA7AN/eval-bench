"""Checks on every stage's saved output, without the LLM: `bench check`, and after each stage of `bench run`.

Each check is (name, ok, detail). A failed check means a stage's output is wrong, missing, or out of step
with the stage before it, and the fix is to re-run that stage.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bench.config import PDF_DIR, BenchError, question_types
from bench.evaluate import REPORT, RESULTS
from bench.facts import FACTS, fact_budget
from bench.facts import fingerprint as facts_fingerprint
from bench.pdf import PAGES, read_jsonl
from bench.questions import LIBRARY, QUESTIONS, library_fingerprint, split_types
from bench.questions import fingerprint as questions_fingerprint
from bench.segments import INGEST, SEGMENTS
from bench.templates import load_topics, templates
from bench.testbed import CHUNKS, COLLECTION, qdrant
from bench.verify import find, kept_fact_problem, question_problem, reason_group

Check = tuple[str, bool, str]


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
    text = {(p["file_name"], p["page"]): p["text"] for p in pages}
    expected = {(p["file_name"], p["page"]) for p in pages if p["text"].strip() and not p["garbled"]}
    seen = Counter((s["file_name"], span["page"]) for s in segments for span in s["page_spans"])
    twice, lost = [k for k, n in seen.items() if n > 1], sorted(expected - set(seen))
    misplaced = [s["segment_id"] for s in segments for span in s["page_spans"]
                 if s["text"][span["start"] : span["end"]] != text[(s["file_name"], span["page"])]]
    used = sum(s["use_for_questions"] for s in segments)
    ingest = {r["file_name"]: r for r in _rows(INGEST)}
    no_toc = [name for name, r in ingest.items() if r.get("segmented_by") != "TOC"]
    textless = sum(n for r in ingest.values() for status, n in r["status"].items() if status != "text")
    return [
        ("every PDF in pdfs/ is extracted", set(by_pdf) == pdfs,
         f"{len(pdfs)} PDFs, {len(pages)} pages, {sum(p['garbled'] for p in pages)} junk-OCR pages left out"),
        ("every page with text is in exactly one segment", not twice and not lost,
         f"{len(segments)} segments, {used} used for questions" + (f"; twice: {twice[:3]}" if twice else "")
         + (f"; in none: {lost[:3]}" if lost else "")),
        ("page positions point at each page's text", not misplaced, ", ".join(sorted(set(misplaced))[:3]) or "ok"),
        ("the ingest report covers every PDF", set(ingest) == set(by_pdf),
         f"{textless} pages without usable text (see data/ingest.jsonl)"
         + (f"; cut without a TOC: {', '.join(no_toc)}" if no_toc else "; every PDF cut along its TOC")),
    ]


def check_facts() -> list[Check]:
    """Facts: every row made from today's inputs, and kept facts still pass the fact rules (models are listed)."""
    if not FACTS.exists():
        return _missing(FACTS, "facts")
    segments = {s["segment_id"]: s for s in _rows(SEGMENTS)}
    used = [sid for sid, s in segments.items() if s["use_for_questions"]]
    rows = {r["segment_id"]: r for r in read_jsonl(FACTS)}
    stale = [sid for sid in used if sid not in rows
             or rows[sid].get("fingerprint") != facts_fingerprint(segments[sid], fact_budget(segments[sid], rows[sid]))]
    stale += [sid for sid in rows if sid not in used]  # rows of sections that are gone or no longer used
    broken = [f"{f['fact_id']} ({problem})" for sid, r in rows.items() if sid in segments for f in r["facts"]
              if (problem := kept_fact_problem(f, segments[sid]))]
    kept = sum(len(r["facts"]) for r in rows.values())
    flagged = sum(f.get("review", {}).get("verdict") == "flag" for r in rows.values() for f in r["facts"])
    rejected = Counter(reason_group(x["reason"]) for r in rows.values() for x in r["rejected"])
    models = ", ".join(f"{m}: {n} rows" for m, n in Counter(r.get("model") for r in rows.values()).items()) or "no rows"
    return [
        ("every question segment has facts made from today's inputs", not stale,
         f"{len(used) - len([s for s in stale if s in used])} of {len(used)} segments current"
         + (f"; stale or missing: {stale[:3]} (run `bench facts`)" if stale else "")),
        ("every kept fact still passes the fact rules", not broken,
         f"{kept} facts ({flagged} flagged by the AI reviewer; models {models}); the audit rejected "
         f"{sum(rejected.values())}: {dict(rejected)}" + (f"; failing: {broken[:3]}" if broken else "")),
    ]


def check_questions() -> list[Check]:
    """Questions: valid types, every row made from today's inputs, still pass the audit, evidence in place."""
    try:
        types = question_types()
    except BenchError as exc:
        return [("question_types.toml is valid", False, str(exc))]
    if not QUESTIONS.exists():
        return _missing(QUESTIONS, "questions")
    segments = {s["segment_id"]: s for s in _rows(SEGMENTS)}
    facts_rows = {r["segment_id"]: r for r in _rows(FACTS)}
    facts = {f["fact_id"]: f for r in facts_rows.values() for f in r["facts"]}
    corpus = "\n".join(p["text"] for p in _rows(PAGES))
    rows = {r["segment_id"]: r for r in read_jsonl(QUESTIONS)}
    section_types, library = split_types(types)
    expected = {sid: questions_fingerprint(section_types, facts_rows.get(sid, {}).get("facts", []), s["title"])
                for sid, s in segments.items() if s["use_for_questions"]}
    if library:
        expected[LIBRARY] = library_fingerprint(library, list(facts_rows.values()), {sid: s["title"] for sid, s in segments.items()})
    stale = [sid for sid, fp in expected.items() if rows.get(sid, {}).get("fingerprint") != fp]
    stale += [sid for sid in rows if sid not in expected]  # rows of sections that are gone, or a library row no type needs
    failing, moved = [], []
    questions = [q for r in rows.values() for q in r["questions"]]
    for q in questions:
        if q["type"] not in types:
            failing.append(f"{q['question_id']} (type {q['type']} is switched off or not in question_types.toml)")
        elif problem := question_problem(q, facts, types[q["type"]]["rules"], corpus):
            failing.append(f"{q['question_id']} ({problem})")
        if any(find(ev["quote"], segments.get(ev.get("segment_id", ""), {}).get("text", "")[ev["start"] : ev["end"]]) is None
               for ev in q["evidence"]):
            moved.append(q["question_id"])
    counts = Counter(q["type"] for q in questions)
    flagged = sum(q.get("review", {}).get("verdict") == "flag" for q in questions)
    rejected = Counter(reason_group(x["reason"]) for r in rows.values() for x in r["rejected"])
    models = ", ".join(f"{m}: {n} rows" for m, n in Counter(r.get("model") for r in rows.values()).items()) or "no rows"
    return [
        ("question_types.toml is valid", True, f"{len(types)} active types: {', '.join(types)}"),
        ("every segment with facts has questions made from today's inputs", not stale,
         f"{len(expected) - len([s for s in stale if s in expected])} of {len(expected)} rows current"
         + (f"; stale or missing: {stale[:3]} (run `bench questions`)" if stale else "")),
        ("every question still passes the audit", not failing,
         f"{len(questions)} questions {dict(counts)}, {flagged} flagged by the AI reviewer; models {models}"
         + (f"; failing: {failing[:3]}" if failing else "")),
        ("every evidence quote is at its recorded position", not moved,
         ", ".join(moved[:3]) or f"the audit rejected {sum(rejected.values())}: {dict(rejected)}"),
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
    """Results: exactly the current questions (matched by content id, so stale results can't pass), and a report."""
    if not RESULTS.exists():
        return _missing(RESULTS, "evaluate")
    scored = [r.get("uid") for r in read_jsonl(RESULTS)]
    current = [q.get("uid") for r in _rows(QUESTIONS) for q in r["questions"]]
    if None in scored or None in current:  # made before content ids: they can't be matched
        return [("every current question is scored", False, "questions or results have no content id: run `bench questions` "
                 "and `bench evaluate`")]
    gone = set(scored) - set(current)
    return [("every current question is scored", set(scored) == set(current) and REPORT.exists(),
             f"{len(set(scored) & set(current))} of {len(set(current))} current questions scored"
             + (f"; {len(gone)} results are for questions that no longer exist" if gone else "")
             + f"; report {'written' if REPORT.exists() else 'missing'}")]


def check_templates() -> list[Check]:
    """Templates: every file in templates/ is valid, and the overrides in data/topics.json name known sections and templates."""
    try:
        tmpls = templates()
    except BenchError as exc:
        return [("templates/ are valid", False, str(exc))]
    topics = load_topics()
    sections = {s["segment_id"] for s in _rows(SEGMENTS)}
    bad = [f"{sid}: {names}" for sid, names in topics["sections"].items() if sid not in sections or set(names) - set(tmpls)]
    bad += [key for key in topics["switches"] if key.split(".")[0] not in tmpls
            or ("." in key and key.split(".", 1)[1] not in tmpls[key.split(".")[0]].intents)]
    return [("templates/ are valid", True, ", ".join(f"{n} ({len(t.intents)} intents)" for n, t in tmpls.items()) or "no templates"),
            ("data/topics.json names known sections, templates and intents", not bad, ", ".join(bad[:3]) or
             f"{len(topics['sections'])} overrides, {len(topics['switches'])} switches")]


STAGES = {"segment": check_segment, "facts": check_facts, "questions": check_questions, "templates": check_templates, "index": check_index,
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
