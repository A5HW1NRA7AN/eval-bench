"""`bench export`: every stage's output as one Excel workbook (data/bench.xlsx), a sheet per table.

Sheets that need a stage not run yet are left out. Each sheet has a frozen header, filters and wrapped text.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill

from bench.checks import all_checks
from bench.config import DATA_DIR, BenchError
from bench.evaluate import KS, RESULTS, by_method, summary, test_set_id
from bench.facts import FACTS
from bench.llm import TRACE
from bench.pdf import PAGES, read_jsonl, write_text
from bench.questions import question_rows
from bench.segments import SEGMENTS
from bench.templates import current_questions, pdf_rows
from bench.testbed import CHUNKS
from bench.verify import reason_group

WORKBOOK = DATA_DIR / "bench.xlsx"
CELL_LIMIT = 32_000  # Excel holds at most 32,767 characters in a cell
WIDE = 60  # width of text columns; others fit their content
HEADER = PatternFill("solid", fgColor="DDE4EE")
BAD = PatternFill("solid", fgColor="F8D7DA")


def _rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    return ILLEGAL_CHARACTERS_RE.sub("", value[:CELL_LIMIT]) if isinstance(value, str) else value  # PDFs hold control characters


def _sheet(book: Workbook, title: str, rows: list[dict[str, Any]], bad: Callable[[dict], bool] | None = None) -> None:
    """One sheet from a list of dicts; rows for which `bad` is true are shaded red."""
    if not rows:
        return
    ws = book.create_sheet(title)
    head = list(rows[0])
    ws.append(head)
    for row in rows:
        ws.append([_cell(row.get(k)) for k in head])
        for cell in ws[ws.max_row]:
            if isinstance(cell.value, float):
                cell.number_format = "0.00"
            if bad and bad(row):
                cell.fill = BAD
    for cell in ws[1]:
        cell.font, cell.fill = Font(bold=True), HEADER
    for col, name in zip(ws.columns, head):
        longest = max(len(str(c.value or "")) for c in col)
        ws.column_dimensions[col[0].column_letter].width = min(max(len(name), longest) + 2, WIDE)
        if longest > WIDE:
            for c in col[1:]:
                c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes, ws.auto_filter.ref = "B2", ws.dimensions


def flags(items: Any) -> int:
    """How many items carry an AI reviewer flag (an AI verdict; the regex audit decided they're kept)."""
    return sum(item.get("review", {}).get("verdict") == "flag" for item in items)


def home(q: dict[str, Any]) -> str | None:
    """A general question's section: its own, or its first fact's for a multi-hop question."""
    return q.get("segment_id") or (q["fact_ids"][0].rsplit(":", 1)[0] if q["fact_ids"] else None)


def run_export() -> Path:
    segments, facts, questions, topical = _rows(SEGMENTS), _rows(FACTS), question_rows(), pdf_rows()
    results, trace = _rows(RESULTS), _rows(TRACE)
    chunks = {c["chunk_id"]: c for c in _rows(CHUNKS)}
    facts_by = {r["segment_id"]: r for r in facts}
    kept_in, rejected_in = (Counter(home(q) for r in questions for q in r[field]) for field in ("questions", "rejected"))
    scored = {r.get("uid") or r["question_id"]: r for r in results}  # by content id, so stale results don't match
    templated = [r for r in results if "template" in r and r.get("answerable", True)]
    answerable = [r for r in results if r.get("answerable", True) and "template" not in r]

    def missed(r: dict[str, Any]) -> bool:  # an answerable question whose evidence wasn't in the top 10: shaded red
        return bool(r.get("answerable", True) and r.get("hit@10") == 0)

    def passage(cid: str) -> str:
        c = chunks.get(cid)
        return f"{c['file_name']} p{c['pages'][0]}: {' '.join(c['text'].split())[:300]}" if c else cid

    book = Workbook()
    book.remove(book.active)
    _sheet(book, "Overview", [
        {"stage": "pages", "rows": len(_rows(PAGES)), "detail": "cleaned PDF pages"},
        {"stage": "segments", "rows": len(segments), "detail": f"{sum(s['use_for_questions'] for s in segments)} used for questions"},
        {"stage": "facts", "rows": sum(len(r["facts"]) for r in facts),
         "detail": f"{sum(len(r['rejected']) for r in facts)} rejected by the regex audit; "
                   f"{flags(f for r in facts for f in r['facts'])} kept with an AI reviewer flag"},
        {"stage": "questions", "rows": sum(len(r["questions"]) for r in questions),
         "detail": f"{sum(len(r['rejected']) for r in questions)} rejected by the regex audit; "
                   f"{flags(q for r in questions for q in r['questions'])} kept with an AI reviewer flag; "
                   f"test set {test_set_id([q for r in questions for q in r['questions']])}"},
        {"stage": "test bed", "rows": len(chunks), "detail": "chunks in Qdrant"},
        {"stage": "evaluate", "rows": len(results),
         "detail": ", ".join(f"{m} {v:.2f}" for m, v in summary(answerable, "type")[0].items() if m.startswith(("hit", "rr")))
         + f"; {len(results) - len(answerable)} unanswerable" if answerable else "not run"},
        {"stage": "agent calls", "rows": len(trace), "detail": f"{sum(t['seconds'] for t in trace) / 60:.1f} min in data/trace.jsonl"},
    ])
    _sheet(book, "Segments", [
        {"segment_id": s["segment_id"], "title": s["title"], "pdf": s["file_name"], "pages": f"{s['pages'][0]}-{s['pages'][-1]}" if s["pages"] else "",
         "used_for_questions": s["use_for_questions"], "chars": len(s["text"]),
         "facts": len(facts_by.get(s["segment_id"], {}).get("facts", [])),
         "facts_rejected": len(facts_by.get(s["segment_id"], {}).get("rejected", [])),
         "questions": kept_in[s["segment_id"]], "questions_rejected": rejected_in[s["segment_id"]]} for s in segments])
    _sheet(book, "Facts", [{"fact_id": f["fact_id"], "size": f.get("size"), "claim": f["claim"], "quote": f["quote"],
                            "pages": f"{f['pages'][0]}-{f['pages'][-1]}", "chars": len(f["quote"]), "topic": f.get("topic"),
                            "ai_review": f.get("review", {}).get("verdict"), "claim_before_review": f.get("review", {}).get("original"),
                            "ai_review_reason": f.get("review", {}).get("reason"), "model": r.get("model"), "version": r["version"]}
                           for r in facts for f in r["facts"]])
    _sheet(book, "Rejected facts", [{"segment_id": r["segment_id"], "reason": f["reason"], "claim": f["claim"], "quote": f.get("quote"),
                                     "sentences": f"{f['first']}–{f['last']}" if "first" in f else None, "topic": f.get("topic")}
                                    for r in facts for f in r["rejected"]])
    _sheet(book, "Questions", [
        {"question_id": q["question_id"], "uid": q.get("uid"), "type": q["type"], "question": q["question"],
         "answer": q.get("answer"), "absent_term": q.get("absent_term") or None, "item": q.get("item"),
         "repaired_from": q.get("original"), "repair_of": q.get("repair_of"),
         "ai_review": q.get("review", {}).get("verdict"), "ai_review_reason": q.get("review", {}).get("reason"),
         "question_before_review": q.get("review", {}).get("original"),
         "fact_ids": ", ".join(q["fact_ids"]), "evidence": "\n---\n".join(ev["quote"] for ev in q["evidence"]),
         "first_rank": scored.get(q.get("uid") or q["question_id"], {}).get("first_rank"),
         **{f"hit@{k}": scored.get(q.get("uid") or q["question_id"], {}).get(f"hit@{k}") for k in KS}}
        for r in questions for q in r["questions"]], bad=lambda row: missed(scored.get(row["uid"] or row["question_id"], {})))
    _sheet(book, "Rejected questions", [{"group": r["group"], "segment_id": home(q), "item": q.get("item"), "reason": q["reason"],
                                         "type": q["type"], "question": q["question"], "answer": q.get("answer"),
                                         "fact_ids": ", ".join(q["fact_ids"]), "repair_of": q.get("repair_of")}
                                        for r in questions for q in r["rejected"]])
    _sheet(book, "Question plan", [  # every item the plan gave, and what became of it; invalid ones with why
        *({"group": r["group"], "item": it["id"], "type": it["type"], "fact_ids": ", ".join(it["fact_ids"]),
           "absent_term": it["absent_term"] or None, "replaces": it["replaces"] or None, "brief": it["brief"] or None,
           "planned_by": it["by"], "outcome": it.get("outcome")} for r in questions for it in r["plan"]["items"]),
        *({"group": r["group"], "item": None, "type": x["type"], "fact_ids": x["ref"], "absent_term": None, "replaces": None,
           "brief": None, "planned_by": "orchestrator", "outcome": f"not planned: {x['why']}"}
          for r in questions for x in r["plan"]["invalid"])])
    if topical:
        _sheet(book, "Plan", [  # every entry a reader gave, and what became of it
            {"pdf": r["file_name"], "entry": e["id"], "section": e["segment_id"], "template": e["template"], "intent": e["intent"],
             "reader_subject": e["subject"], "detail": e["detail"], "paragraphs": ", ".join(e["paragraphs"]),
             "dropped_by_code": e.get("dropped"), "picked_by": (e.get("pick") or {}).get("by"),
             "subject": (e.get("pick") or {}).get("subject"), "brief": (e.get("pick") or {}).get("brief"),
             "fact": next((f["fact_id"] for f in r["facts"] if f["entry"] == e["id"]), None)
             or next((f["reason"] for f in r["rejected_facts"] if f.get("entry") == e["id"]), None),
             "questions": ", ".join(q["question_id"] for q in r["questions"]
                                    for f in r["facts"] if f["entry"] == e["id"] and f["fact_id"] in q["fact_ids"])}
            for r in topical for e in r["entries"]])
        _sheet(book, "Template facts", [{"fact_id": f["fact_id"], "template": f["template"], "intent": f["intent"],
                                         "subject": f["subject"], "entry": f.get("entry"), "brief": f.get("brief"), "claim": f["claim"],
                                         "header": f.get("header"), "quote": f["quote"], "pages": f"{f['pages'][0]}-{f['pages'][-1]}",
                                         "ai_review": f.get("review", {}).get("verdict"), "ai_review_reason": f.get("review", {}).get("reason")}
                                        for r in topical for f in r["facts"]])
        _sheet(book, "Template questions", [
            {"question_id": q["question_id"], "uid": q["uid"], "template": q["template"], "intent": q["intent"], "subject": q["subject"],
             "type": q["type"], "question": q["question"], "answer": q.get("answer"), "repaired_from": q.get("original"),
             "repair_of": q.get("repair_of"), "ai_review": q.get("review", {}).get("verdict"),
             "ai_review_reason": q.get("review", {}).get("reason"), "evidence": "\n---\n".join(ev["quote"] for ev in q["evidence"]),
             "first_rank": scored.get(q["uid"], {}).get("first_rank"), **{f"hit@{k}": scored.get(q["uid"], {}).get(f"hit@{k}") for k in KS}}
            for r in topical for q in r["questions"]], bad=lambda row: missed(scored.get(row["uid"], {})))
        _sheet(book, "Rejected template items", [{"segment_id": x["segment_id"], "stage": stage, "reason": x["reason"],
                                                  "intent": x.get("intent"), "text": x.get("question") or x.get("claim"),
                                                  "repair_of": x.get("repair_of"), "quote": x.get("quote")}
                                                 for r in topical for stage, items in (("fact", r["rejected_facts"]), ("question", r["rejected"]))
                                                 for x in items])
    _sheet(book, "Results", [
        {"question_id": r["question_id"], "type": r["type"], "intent": r.get("intent"), "question": r["question"],
         "answerable": r.get("answerable", True),
         "first_rank": r.get("first_rank"), **{f"hit@{k}": r.get(f"hit@{k}") for k in KS}, "rr": r.get("rr"),
         "all_facts@10": r.get("all_facts@10"), "top_score": r["top_score"] if r.get("ranked") else None,
         "expected": "\n---\n".join(r["expected"]),
         **{f"top{n}": f"{score:.3f} {passage(cid)}" for n, (cid, score) in enumerate(r["ranked"][:3], start=1)}}
        for r in results], bad=missed)
    if answerable:
        pdf_of = {s["segment_id"]: s["file_name"] for s in segments}
        _sheet(book, "Metrics by type", summary(answerable, "type"))
        if all("baselines" in r for r in answerable):
            _sheet(book, "Metrics by method", by_method(answerable))
        _sheet(book, "Metrics by PDF", summary([{**r, "pdf": pdf_of.get(r["question_id"].rsplit(":", 1)[0], "several PDFs")}
                                                for r in answerable], "pdf"))
    if templated:
        _sheet(book, "Metrics by intent", summary(templated, "intent"))
    rejects = Counter((stage, reason_group(item["reason"])) for stage, rows, field in
                      (("facts", facts, "rejected"), ("questions", questions, "rejected"),
                       ("template facts", topical, "rejected_facts"), ("template questions", topical, "rejected"))
                      for r in rows for item in r[field])
    repairs = Counter((stage, reason_group(q["repair_of"]), passed) for stage, rows in (("questions", questions), ("template questions", topical))
                      for r in rows for items, passed in ((r["questions"], True), (r["rejected"], False)) for q in items if q.get("repair_of"))
    keys = dict.fromkeys([key for key, _ in rejects.most_common()] + [(stage, rule) for stage, rule, _ in repairs])
    _sheet(book, "Audit reasons", [  # rejections by rule; for questions, also the repairs each rule set off
        {"stage": s, "reason": reason, "count": rejects.get((s, reason), 0),
         "repairs_tried": repairs[(s, reason, True)] + repairs[(s, reason, False)] if "questions" in s else None,
         "repairs_passed": repairs[(s, reason, True)] if "questions" in s else None} for s, reason in keys])
    _sheet(book, "Checks", [{"stage": stage, "check": name, "ok": ok, "detail": detail}
                            for stage, checks in all_checks().items() for name, ok, detail in checks], bad=lambda row: not row["ok"])
    _sheet(book, "Agent calls", [{"time": t["time"], "run": t.get("run"), "role": t["role"], "label": t["label"],
                                  "model": t["model"], "seconds": t["seconds"], "input_tokens": t.get("input_tokens"),
                                  "output_tokens": t.get("output_tokens"), "error": t.get("error"), "output": t.get("output"),
                                  "prompt": t["prompt"]} for t in trace], bad=lambda row: bool(row["error"]))
    book.save(WORKBOOK)
    print(f"wrote {WORKBOOK}: {', '.join(book.sheetnames)}")
    return WORKBOOK


TESTSET_JSON, TESTSET_CSV = DATA_DIR / "testset.json", DATA_DIR / "testset.csv"
FORMULA_START = ("=", "+", "-", "@", "\t", "\r")  # a spreadsheet would run such a cell as a formula


def testset_rows() -> list[dict[str, Any]]:
    """Every current question, general then template, with its answer and evidence (PDF, pages, quote, position)."""
    pdf_of = {s["segment_id"]: s["file_name"] for s in _rows(SEGMENTS)}
    out = []
    for kind, questions in (("general", [q for r in question_rows() for q in r["questions"]]), ("template", current_questions())):
        for q in questions:
            out.append({"uid": q["uid"], "question_id": q["question_id"], "set": kind, "type": q["type"],
                        "template": q.get("template"), "intent": q.get("intent"), "subject": q.get("subject"),
                        "question": q["question"], "answer": q.get("answer") or None, "absent_term": q.get("absent_term") or None,
                        "answerable": bool(q["evidence"]),
                        "evidence": [{"file_name": pdf_of.get(ev["segment_id"]), "segment_id": ev["segment_id"], "pages": ev["pages"],
                                      "quote": ev["quote"], "start": ev["start"], "end": ev["end"]} for ev in q["evidence"]]})
    return out


def _safe(value: Any) -> str:
    text = "" if value is None else str(value)
    return "'" + text if text.startswith(FORMULA_START) else text


def run_testset() -> tuple[Path, Path]:
    """The test set to hand to another team: data/testset.json (with the test-set ids) and data/testset.csv."""
    rows = testset_rows()
    if not rows:
        raise BenchError("no verified questions: run `bench questions` first")
    general, topical = [r for r in rows if r["set"] == "general"], [r for r in rows if r["set"] == "template"]
    header = {"test_set": test_set_id(general), "template_test_set": test_set_id(topical) if topical else None,
              "questions": len(general), "template_questions": len(topical),
              "note": "Each answer is copied from its evidence quotes, which are verbatim PDF text; start and end are "
                      "character positions in the section (segment) text. uid stays the same across runs."}
    write_text(TESTSET_JSON, json.dumps({**header, "items": rows}, indent=1, ensure_ascii=False))
    columns = ["uid", "question_id", "set", "type", "template", "intent", "subject", "question", "answer", "absent_term",
               "answerable", "pdf", "pages", "evidence"]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    for r in rows:
        flat = {**r, "pdf": "; ".join(sorted({e["file_name"] or "" for e in r["evidence"]})),
                "pages": "; ".join(f"{e['pages'][0]}-{e['pages'][-1]}" for e in r["evidence"] if e["pages"]),
                "evidence": "\n---\n".join(e["quote"] for e in r["evidence"])}
        writer.writerow([_safe(flat[c]) for c in columns])
    write_text(TESTSET_CSV, buffer.getvalue())
    print(f"wrote {TESTSET_JSON.name} and {TESTSET_CSV.name}: {len(general)} general and {len(topical)} template questions "
          f"(test set {header['test_set']})")
    return TESTSET_JSON, TESTSET_CSV
