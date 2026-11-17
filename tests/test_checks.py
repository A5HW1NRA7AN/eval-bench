"""bench check on small data written here, the agent trace and response cache (bench/llm.py), and the workbook."""

import json

from bench import checks, facts
from bench.pdf import write_jsonl

TEXT = "Give the calf colostrum within two hours of birth."


def _data(tmp_path, monkeypatch, fact_start=0):
    (tmp_path / "pdfs").mkdir(exist_ok=True)
    (tmp_path / "pdfs" / "book.pdf").write_bytes(b"")
    pages = [{"file_name": "book.pdf", "page": n, "text": TEXT, "raw_text": TEXT, "garbled": False, "printed": n} for n in (1, 2, 3)]
    segment = {"segment_id": "book-01", "file_name": "book.pdf", "title": "1 Calves", "pages": [1, 2], "use_for_questions": True,
               "headings": [], "text": f"{TEXT}\n\n{TEXT}",
               "page_spans": [{"page": 1, "start": 0, "end": len(TEXT)}, {"page": 2, "start": len(TEXT) + 2, "end": 2 * len(TEXT) + 2}]}
    fact = {"fact_id": "book-01:f01", "claim": "Calves need colostrum early.", "quote": "colostrum within two hours",
            "start": fact_start, "end": fact_start + 26, "pages": [1, 1]}
    paths = {"PAGES": tmp_path / "pages.jsonl", "SEGMENTS": tmp_path / "segments.jsonl", "FACTS": tmp_path / "facts.jsonl"}
    write_jsonl(paths["PAGES"], pages)
    write_jsonl(paths["SEGMENTS"], [segment])
    write_jsonl(paths["FACTS"], [{"segment_id": "book-01", "version": facts.FACTS_VERSION, "fingerprint": facts.fingerprint(segment, facts.fact_budget(segment)),
                             "max_facts": facts.fact_budget(segment), "model": "m", "facts": [fact], "rejected": []}])
    for name, path in paths.items():
        monkeypatch.setattr(checks, name, path)
    monkeypatch.setattr(checks, "PDF_DIR", tmp_path / "pdfs")
    return paths


def test_checks_catch_a_lost_page_and_a_moved_quote(tmp_path, monkeypatch):
    _data(tmp_path, monkeypatch, fact_start=3)  # the quote really starts at TEXT.index("colostrum")
    results = {name: ok for name, ok, _ in checks.check_segment() + checks.check_facts()}
    assert results["every PDF in pdfs/ is extracted"] and results["page positions point at each page's text"]
    assert not results["every page with text is in exactly one segment"]  # page 3 is in no segment
    assert results["every question segment has facts made from today's inputs"]
    assert not results["every kept fact still passes the fact rules"]  # the quote isn't at its position
    fixed = tmp_path / "fixed"
    fixed.mkdir()
    _data(fixed, monkeypatch, fact_start=TEXT.index("colostrum"))
    assert {n: ok for n, ok, _ in checks.check_facts()}["every kept fact still passes the fact rules"]


def test_a_changed_prompt_or_setting_makes_a_row_stale(tmp_path, monkeypatch):
    _data(tmp_path, monkeypatch, fact_start=TEXT.index("colostrum"))
    monkeypatch.setattr(facts, "PLANNER_PROMPT", facts.PLANNER_PROMPT + " Changed.")
    results = {name: ok for name, ok, _ in checks.check_facts()}
    assert not results["every question segment has facts made from today's inputs"]


def test_agent_calls_are_traced_and_the_workbook_has_every_sheet(tmp_path, monkeypatch):
    from openpyxl import load_workbook
    from pydantic import BaseModel
    from pydantic_ai import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    from bench import export, llm

    class Out(BaseModel):
        items: list[str]

    asked = []
    monkeypatch.setattr(llm, "model", lambda: FunctionModel(
        lambda m, i: asked.append(1) or ModelResponse(parts=[TextPart('{"items": ["a", "b"]}')])))
    agent = llm.agent(Out, "List things.")
    assert llm.call("tester", agent, "say two things", "book-01").items == ["a", "b"]
    call = json.loads(llm.TRACE.read_text().splitlines()[0])
    assert call["role"] == "tester" and call["prompt"] == "say two things" and call["output"] == {"items": ["a", "b"]}
    assert llm.call("tester", agent, "say two things", "book-01").items == ["a", "b"] and len(asked) == 1  # cached
    llm.call("tester", llm.agent(Out, "List other things."), "say two things", "book-01")  # new instructions: asked
    monkeypatch.setattr(llm, "USE_CACHE", False)
    llm.call("tester", agent, "say two things", "book-01")  # --no-cache: asked again
    assert len(asked) == 3

    paths = _data(tmp_path, monkeypatch, fact_start=TEXT.index("colostrum"))
    for name in ("SEGMENTS", "FACTS", "PAGES"):
        monkeypatch.setattr(export, name, paths[name])
    for name, path in {"RESULTS": "r.jsonl", "CHUNKS": "c.jsonl", "WORKBOOK": "bench.xlsx"}.items():
        monkeypatch.setattr(export, name, tmp_path / path)
    monkeypatch.setattr(export, "TRACE", llm.TRACE)
    book = load_workbook(export.run_export())
    assert book.sheetnames == ["Overview", "Segments", "Facts", "Checks", "Agent calls"]  # no questions or results yet
    row = {head.value: cell.value for head, cell in zip(book["Facts"][1], book["Facts"][2])}
    assert row["fact_id"] == "book-01:f01" and row["claim"] == "Calves need colostrum early."


def test_a_csv_cell_never_runs_as_a_formula():
    from bench.export import _safe
    assert _safe("=HYPERLINK(1)") == "'=HYPERLINK(1)" and _safe("-0.50") == "'-0.50" and _safe("Milk") == "Milk" and _safe(None) == ""
