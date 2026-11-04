"""End to end: `bench run` on a small generated PDF, with fake agents, an in-memory Qdrant and fake embeddings
(see conftest.py), plus the safety net around it: the run lock, failing sections, outages and atomic writes."""

import json

import pymupdf
import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.function import FunctionModel
from typer.testing import CliRunner

from bench import cli, config, facts, llm, pdf, segments
from bench.checks import all_checks
from bench.config import BenchError

PAGES = ["Calves should get colostrum within two hours of birth. Colostrum gives the newborn calf antibodies against "
         "common infections and diseases, so the first feed matters a great deal for its health and growth.",
         "A dairy cow giving ten litres of milk a day needs about four kilograms of concentrate mixture. Clean water "
         "should always be available, because a milking cow drinks a lot of it every day in summer and winter alike.",
         "Sheds should be cleaned every day and the floor kept dry. A dry floor prevents hoof problems and mastitis, "
         "and good drainage keeps flies away from the animals and from the milk they give to the dairy farmer."]


def _pdf(path):
    doc = pymupdf.open()
    for text in PAGES:
        doc.new_page().insert_textbox(pymupdf.Rect(72, 72, 520, 760), text, fontsize=11)
    doc.save(path)


def _agents(messages, info):
    """Every agent's reply, told apart by its instructions."""
    system, prompt = messages[-1].instructions, messages[-1].parts[-1].content
    if system.startswith("You plan fact extraction"):
        reply = {"topics": [{"topic": "colostrum timing", "kind": "figure", "paragraphs": ["p1"]}] if "colostrum" in prompt else []}
    elif system.startswith("You write one fact"):
        reply = {"facts": [{"claim": "Calves should get colostrum within two hours of birth.", "first": "p1.s1", "last": "p1.s1"}]}
    elif system.startswith("You review facts"):
        reply = {"reviews": [{"id": "f1", "verdict": "keep", "claim": "", "reason": ""}]}
    elif system.startswith("You review benchmark questions"):
        reply = {"reviews": []}
    elif "TYPE: direct" in system and "whole library" not in prompt:
        reply = {"questions": [{"question": "How soon after birth should a calf be fed colostrum?", "fact_ids": ["f1"],
                                "answer": "within two hours of birth", "absent_term": ""}]}
    else:
        reply = {"questions": []}
    return ModelResponse(parts=[TextPart(json.dumps(reply))])


@pytest.fixture
def book(isolated, monkeypatch):
    _pdf(pdf.PDF_DIR / "book.pdf")
    monkeypatch.setattr(llm, "model", lambda: FunctionModel(_agents))
    return isolated


def test_bench_run_end_to_end(book):
    result = CliRunner().invoke(cli.app, ["run"])
    assert result.exit_code == 0, result.output
    assert all(ok for checks in all_checks().values() for _, ok, _ in checks)
    (scored,) = [json.loads(line) for line in (config.DATA_DIR / "results.jsonl").read_text().splitlines()]
    assert scored["uid"] and scored["first_rank"] == 1  # the fake embedding finds the colostrum chunk first
    report = (config.DATA_DIR / "report.md").read_text()
    assert "| direct | 1 |" in report and "test set `" in report and (config.DATA_DIR / "bench.xlsx").exists()

    calls = len((config.DATA_DIR / "trace.jsonl").read_text().splitlines())
    assert CliRunner().invoke(cli.app, ["run"]).exit_code == 0  # a second run resumes: no agent call
    assert len((config.DATA_DIR / "trace.jsonl").read_text().splitlines()) == calls


def test_errors_are_one_line_and_the_lock_stops_a_second_writer(book):
    assert CliRunner().invoke(cli.app, ["segment"]).exit_code == 0
    result = CliRunner().invoke(cli.app, ["facts", "--doc", "nothing-like-this"])
    assert result.exit_code == 1 and "matches no PDF" in result.output and "Traceback" not in result.output
    with cli._writing("facts"):
        result = CliRunner().invoke(cli.app, ["facts"])
    assert result.exit_code == 1 and "another bench command is writing data/" in result.output


def test_a_failing_section_is_left_to_do_and_the_rest_is_saved(book, monkeypatch):
    seen = []

    def work(segment):
        seen.append(segment["segment_id"])
        if segment["segment_id"] == "b":
            raise ConnectionError("endpoint down at [/v1] [type=json_invalid]")  # brackets must not break the printout

    with pytest.raises(BenchError, match="1 section"):
        facts.run_each([{"segment_id": s} for s in "abc"], work, "facts")
    assert seen == ["a", "b", "c"]
    with pytest.raises(BenchError, match="in a row"):  # stops instead of trying every section of a long outage
        facts.run_each([{"segment_id": s} for s in "bbbbb"], work, "facts")
    assert seen.count("b") == 1 + config.MAX_FAILURES


def test_a_toc_outage_saves_nothing_but_an_unreadable_toc_falls_back(book, monkeypatch):
    pages = pdf.load_pages()
    monkeypatch.setattr(segments, "find_toc_pages", lambda pages: [1])

    def outage(*args, **kwargs):
        raise ConnectionError("endpoint down")

    monkeypatch.setattr(segments.llm, "call", outage)
    with pytest.raises(BenchError, match="Nothing was saved"):
        segments.segment_pdf(pages)
    assert not (config.DATA_DIR / "toc").exists()  # no cache written

    def garbage(*args, **kwargs):
        raise UnexpectedModelBehavior("not JSON")

    monkeypatch.setattr(segments.llm, "call", garbage)
    assert [s["segment_id"] for s in segments.segment_pdf(pages)] == ["book-01"]  # one segment for the whole PDF


def test_rows_of_removed_sections_are_saved_away_and_max_facts_sticks(book, monkeypatch):
    assert CliRunner().invoke(cli.app, ["run"]).exit_code == 0
    rows = [json.loads(line) for line in facts.FACTS.read_text().splitlines()]
    pdf.write_jsonl(facts.FACTS, rows + [{**rows[0], "segment_id": "gone-01"}])  # a section that no longer exists
    assert CliRunner().invoke(cli.app, ["facts"]).exit_code == 0
    assert [json.loads(line)["segment_id"] for line in facts.FACTS.read_text().splitlines()] == ["book-01"]

    assert CliRunner().invoke(cli.app, ["facts", "--max-facts", "3", "--force"]).exit_code == 0
    calls = len(llm.TRACE.read_text().splitlines())
    result = CliRunner().invoke(cli.app, ["run"])  # keeps each row's own max_facts: nothing is redone
    assert result.exit_code == 0, result.output
    assert len(llm.TRACE.read_text().splitlines()) == calls


def test_a_failed_call_in_the_trace_does_not_break_the_export(book, monkeypatch):
    assert CliRunner().invoke(cli.app, ["run"]).exit_code == 0
    with llm.TRACE.open("a") as f:
        f.write(json.dumps({"time": "t", "run": "r", "role": "planner", "label": "x [/v1] [type=json]", "model": "m",
                            "seconds": 1.0, "prompt": "p", "error": "ConnectionError: [/]"}) + "\n")
    result = CliRunner().invoke(cli.app, ["export"])
    assert result.exit_code == 0, result.output


def test_writes_are_atomic(tmp_path, monkeypatch):
    target = tmp_path / "rows.jsonl"
    pdf.write_jsonl(target, [{"n": 1}])
    real_dumps = json.dumps

    def failing(row, **kwargs):
        if row == {"n": 3}:
            raise KeyboardInterrupt
        return real_dumps(row, **kwargs)

    monkeypatch.setattr(pdf.json, "dumps", failing)
    with pytest.raises(KeyboardInterrupt):
        pdf.write_jsonl(target, [{"n": 2}, {"n": 3}])
    assert target.read_text() == '{"n": 1}\n' and not (tmp_path / ".rows.jsonl.tmp").exists()
