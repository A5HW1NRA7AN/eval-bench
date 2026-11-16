import dataclasses
import json
import re
import threading
import time

import pytest
from pydantic import BaseModel
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from bench import llm, orchestrator
from bench import questions as question_stage
from bench.config import BenchError
from bench.orchestrator import (
    Brief,
    Plan,
    Subject,
    categorise,
    cut,
    fingerprint,
    groups,
    plan,
    read,
    validate,
)
from bench.templates import styles
from bench.templates import templates as load_templates
from bench.verify import plain

REPO = load_templates()  # the committed templates/ files
NO_TOPICS = {"switches": {}, "sections": {}}
# Written for these tests (not copied from the PDFs).
TEXT = ("Mastitis\n\nMastitis is an infection of the udder. Symptoms include swelling and a watery discharge from the teat.\n\n"
        "To prevent mastitis, keep the shed floor dry and clean the udder before milking.\n\n"
        "Cows were fed hay and grain on most farms in the district.")
P2, P3 = TEXT.index("Mastitis is"), TEXT.index("To prevent")
P2_END, P3_END = TEXT.index("teat.") + len("teat."), TEXT.index("milking.") + len("milking.")


def _segment(sid="book-01", text=TEXT, title="Udder health"):
    return {"segment_id": sid, "file_name": "book.pdf", "title": title, "headings": [], "use_for_questions": True,
            "text": text, "pages": [1, 1], "page_spans": [{"page": 1, "start": 0, "end": len(text)}]}


def _model(monkeypatch, reply):
    """Every agent answers reply(instructions, prompt), as JSON; answers are never taken from the cache."""
    monkeypatch.setattr(llm, "USE_CACHE", False)
    monkeypatch.setattr(llm, "model", lambda: FunctionModel(
        lambda messages, info: ModelResponse(parts=[TextPart(json.dumps(reply(messages[-1].instructions, messages[-1].parts[-1].content)))])))


def _unreadable(monkeypatch):
    monkeypatch.setattr(llm, "model", lambda: FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("not JSON")])))


def test_groups_keep_order_and_fit_the_limit_and_briefs_are_cut_at_a_word():
    assert groups([3, 3, 3, 9, 1], lambda n: n, 6) == [[3, 3], [3], [9], [1]]
    assert cut("one two three", 9) == "one two" and cut(" a \n b ") == "a b"


def test_the_map_lists_every_section_and_the_full_text_only_when_it_fits():
    items = [("s1", _segment()), ("s2", _segment("book-02", title="Feeding"))]
    prompt, fits = orchestrator.categorise_prompt("book", items, REPO, budget=10_000)
    assert fits and "s1 | Udder health" in prompt and "s2 | Feeding" in prompt and "TEXT:\n== s1 Udder health ==" in prompt
    prompt, fits = orchestrator.categorise_prompt("book", items, REPO, budget=100)
    assert not fits and "TEXT:" not in prompt and 'opens: "Mastitis Mastitis is an infection' in prompt


def test_your_override_wins_and_a_section_the_answer_leaves_out_falls_back_to_the_rules(monkeypatch):
    segs = [_segment(), _segment("book-02", "A shed should face east and stay dry. " * 20, "Housing"), _segment("book-03")]
    _model(monkeypatch, lambda system, prompt: {"sections": [{"section": "s2", "templates": ["animal_care"]}]})
    sections, views = categorise(segs, REPO, {"switches": {}, "sections": {"book-03": []}}, failed := [])
    assert sections == {"book-01": {"templates": ["animal_care"], "by": "rules"},  # left out of the answer: the density rules
                        "book-02": {"templates": ["animal_care"], "by": "orchestrator"},
                        "book-03": {"templates": [], "by": "you"}}
    assert views == ["map"] and not failed  # the tests' settings give the orchestrator no text budget
    _unreadable(monkeypatch)
    sections, _ = categorise(segs, REPO, NO_TOPICS, failed)
    assert {c["by"] for c in sections.values()} == {"rules"} and failed == ["book.pdf"]  # recorded, so the PDF is redone

    def down(messages, info):
        raise ConnectionError("endpoint down")

    monkeypatch.setattr(llm, "model", lambda: FunctionModel(down))
    with pytest.raises(ConnectionError):  # an outage isn't an answer: the PDF is left to do
        categorise(segs, REPO, NO_TOPICS, [])


def test_reader_entries_that_the_audit_could_never_pass_are_dropped_by_code(monkeypatch):
    def reply(system, prompt):
        assert system.startswith("You read one part") and "- prevention:" in prompt and "[p3] To prevent mastitis" in prompt
        entry = {"intent": "prevention", "subject": "mastitis", "paragraphs": ["p2"], "detail": "x"}
        return {"entries": [{**entry, "paragraphs": ["[p3]"], "detail": "keep the floor dry"},
                            {**entry, "intent": "symptoms", "detail": "swelling and discharge"},
                            {**entry, "intent": "treatment_first_aid", "paragraphs": ["p4"]},  # no cue word near it
                            {**entry, "subject": "lumpy skin disease", "paragraphs": ["p3"]},  # a subject the text never names
                            {**entry, "intent": "symptoms", "paragraphs": ["p9"]},  # not in the part
                            entry, entry, {**entry, "detail": "over the cap"}]}  # prevention entries 4 and 5 are over the cap

    _model(monkeypatch, reply)
    entries = read([_segment()], {"book-01": {"animal_care": list(REPO["animal_care"].intents)}}, REPO, [])
    assert [e["id"] for e in entries] == ["e1", "e2", "e3", "e4", "e5", "e6"]
    assert [(e["intent"], e["paragraphs"]) for e in entries if not e.get("dropped")] == [("symptoms", ["p2"]), ("prevention", ["p3"])]
    assert sorted(e["dropped"] for e in entries if e.get("dropped")) == [
        "its paragraphs aren't in the part", "its subject isn't named near it", "no cue word of its intent near it",
        "no cue word of its intent near it"]


def test_the_plan_is_checked_by_code_an_empty_plan_is_kept_and_an_unreadable_one_falls_back(monkeypatch):
    seg = _segment()
    e1 = {"id": "e1", "segment_id": "book-01", "template": "animal_care", "intent": "symptoms", "subject": "mastitis",
          "paragraphs": ["p2"], "detail": "swelling and discharge", "start": P2, "end": P2_END}
    e2 = {**e1, "id": "e2", "intent": "prevention", "paragraphs": ["p3"], "detail": "a dry floor", "start": P3, "end": P3_END}
    # Models sometimes copy an entry's line or detail instead of its id: code maps the copy back.
    answer = Plan(subjects=[Subject(name="Mastitis (udder infection)", entries=["[e1]", "e9"]),
                            Subject(name="Bovine mastitis", entries=[orchestrator.entry_line(e2), "e1", "A dry floor"])],
                  briefs=[Brief(entry="e1", brief="word " * 100)])
    picks, invalid = validate(answer, [e1, e2], {"book-01": seg})
    assert picks["e1"]["subject"] == "Mastitis (udder infection)" and len(picks["e1"]["brief"]) <= orchestrator.BRIEF_CHARS
    assert picks["e2"] == {"subject": "mastitis", "proposed": "Bovine mastitis"}  # the text never says "bovine mastitis"
    assert invalid == [{"entry": "e9", "why": "no such entry"}, {"entry": "e1", "why": "listed twice"},
                       {"entry": "A dry floor", "why": "listed twice"}]

    _model(monkeypatch, lambda system, prompt: {"subjects": [], "briefs": []})  # nothing worth a question
    entries = [dict(e1), dict(e2)]
    plans = plan("book.pdf", [seg], entries, REPO, llm.agent(Plan, orchestrator.PLAN_PROMPT), [])
    assert plans["animal_care"][0]["by"] == "orchestrator" and [e["pick"] for e in entries] == [None, None]
    _unreadable(monkeypatch)
    entries, failed = [dict(e1), dict(e2)], []
    plans = plan("book.pdf", [seg], entries, REPO, llm.agent(Plan, orchestrator.PLAN_PROMPT), failed)
    assert plans["animal_care"][0]["by"] == "code" and failed == ["book.pdf animal_care"]
    assert [e["pick"] for e in entries] == [{"by": "code", "brief": "swelling and discharge", "subject": "mastitis"},
                                            {"by": "code", "brief": "a dry floor", "subject": "mastitis"}]


def test_a_long_plan_is_split_into_groups_of_sections(monkeypatch):
    monkeypatch.setattr(orchestrator, "PLAN_ENTRIES", 1)
    _model(monkeypatch, lambda system, prompt: {"subjects": [{"name": "mastitis", "entries": re.findall(r"^e\d+", prompt, re.MULTILINE)}],
                                                "briefs": []})
    segs = [_segment(), _segment("book-02")]
    entries = [{"id": f"e{n}", "segment_id": s["segment_id"], "template": "animal_care", "intent": "prevention", "subject": "mastitis",
                "paragraphs": ["p3"], "detail": "a dry floor", "start": P3, "end": P3_END} for n, s in enumerate(segs, start=1)]
    plans = plan("book.pdf", segs, entries, REPO, llm.agent(Plan, orchestrator.PLAN_PROMPT), [])
    assert [g["entries"] for g in plans["animal_care"]] == [1, 1] and [e["pick"]["by"] for e in entries] == ["orchestrator"] * 2
    labels = [json.loads(line)["label"] for line in llm.TRACE.read_text().splitlines()]
    assert labels == ["book.pdf animal_care group 1/2", "book.pdf animal_care group 2/2"]


def test_a_rejected_question_is_repaired_once_with_the_reason_but_never_to_differ_from_a_general_one(monkeypatch):
    quote = "To prevent mastitis, keep the shed floor dry and clean the udder before milking."
    fact = {"fact_id": "book-01:tf01", "segment_id": "book-01", "template": "animal_care", "intent": "prevention", "subject": "mastitis",
            "claim": "Keeping the shed floor dry prevents mastitis.", "quote": quote, "start": P3, "end": P3 + len(quote), "pages": [1, 1],
            "example": "How can a farmer prevent mastitis?", "brief": "a dry floor"}
    prompts = []

    def reply(system, prompt):
        prompts.append(prompt)
        if system.startswith("You review benchmark questions"):
            return {"reviews": []}
        draft = {"fact_ids": ["f1"], "absent_term": "", "answer": "keep the shed floor dry"}
        if "A CHECK REJECTED" in prompt:
            return {"questions": [{**draft, "question": "How can farmers prevent mastitis in a dairy shed?"}]}
        if "TYPE: direct" in system:  # its answer isn't in the quote
            return {"questions": [{**draft, "question": "How can a farmer prevent mastitis?", "answer": "keep the floor very dry"}]}
        return {"questions": [{**draft, "question": "How do I stop mastitis in cows?"}]}  # a general question already asks this

    _model(monkeypatch, reply)
    kinds = styles(REPO)
    writers = {n: llm.agent(question_stage.Drafts, question_stage.WRITER_PROMPT + f"{n}: {t['prompt']}") for n, t in kinds.items()}
    reviewer = llm.agent(question_stage.Reviews, question_stage.REVIEWER_PROMPT)
    questions, rejected = orchestrator.write_questions([fact], {"book-01": _segment()}, REPO, kinds, writers, reviewer, "",
                                                       {plain("How do I stop mastitis in cows?")}, [])
    assert [(q["question_id"], q["repair_of"], q["original"], q["subject"]) for q in questions] == [
        ("book-01:t01", "answer not found in its facts' quotes", "How can a farmer prevent mastitis?", "mastitis")]
    assert [(x["reason"], "repair_of" in x) for x in rejected] == [("same as a general question", False)]
    (repair,) = [p for p in prompts if "A CHECK REJECTED" in p]  # one repair call, for the one repairable draft
    assert "reason: answer not found in its facts' quotes" in repair and "Copy the answer word for word" in repair


def test_a_pdf_with_failed_calls_is_redone_and_rows_of_pdfs_that_are_gone_are_dropped(monkeypatch):
    monkeypatch.setattr(orchestrator, "load_segments", lambda: [_segment()])
    done = {"book.pdf": {"fingerprint": "fp", "failed_calls": ["book-01 animal_care part 1/1"]}, "gone.pdf": {}}
    assert [name for name, _ in orchestrator.todo(done, None, False, lambda name, segs: "fp", lambda row, segs: True)] == ["book.pdf"]
    assert list(done) == ["book.pdf"]
    done["book.pdf"]["failed_calls"] = []
    assert orchestrator.todo(done, None, False, lambda name, segs: "fp", lambda row, segs: True) == []
    with pytest.raises(BenchError, match="matches no PDF"):
        orchestrator.todo(done, "nothing-like-this", False, lambda name, segs: "fp", lambda row, segs: True)


def test_the_fingerprint_changes_with_the_inputs_and_only_then(monkeypatch):
    segs, kinds = [_segment()], styles(REPO)
    base = fingerprint(segs, REPO, NO_TOPICS, kinds)
    assert fingerprint([_segment()], REPO, NO_TOPICS, kinds) == base
    assert fingerprint([_segment(text=TEXT + " More.")], REPO, NO_TOPICS, kinds) != base
    assert fingerprint(segs, REPO, {"switches": {"animal_care.symptoms": False}, "sections": {}}, kinds) != base
    assert fingerprint(segs, REPO, {"switches": {}, "sections": {"book-01": ["animal_care"]}}, kinds) != base
    monkeypatch.setattr(orchestrator, "PLAN_PROMPT", orchestrator.PLAN_PROMPT + " Changed.")
    assert fingerprint(segs, REPO, NO_TOPICS, kinds) != base


def test_an_outage_stops_new_calls_and_identical_calls_are_asked_once(monkeypatch):
    class Out(BaseModel):
        n: int

    asked = []

    def answer(messages, info):
        n = int(messages[-1].parts[-1].content)
        asked.append(n)
        if n == 0:
            raise ConnectionError("endpoint down")
        time.sleep(0.05)  # still running when the first call fails
        return ModelResponse(parts=[TextPart(json.dumps({"n": n}))])

    monkeypatch.setattr(llm, "model", lambda: FunctionModel(answer))
    monkeypatch.setattr(llm, "USE_CACHE", False)
    base = llm.settings()
    monkeypatch.setattr(llm, "settings", lambda: dataclasses.replace(base, concurrency=2))
    agent = llm.agent(Out, "Echo the number.")
    with pytest.raises(ConnectionError):
        llm.call_all([("echo", agent, str(n), f"job {n}") for n in range(6)])
    assert sorted(asked) == [0, 1]  # the first wave only: the other four never started
    asked.clear()
    monkeypatch.setattr(llm, "USE_CACHE", True)
    answers = llm.call_all([("echo", agent, "7", "job a"), ("echo", agent, "7", "job b")])
    assert [a.n for a in answers] == [7, 7] and asked == [7]  # one call for both
    assert [("cached" in json.loads(line)) for line in llm.TRACE.read_text().splitlines()[-2:]] == [False, True]


def test_answers_come_back_in_job_order_at_any_concurrency(monkeypatch):
    class Out(BaseModel):
        n: int

    running, peak, lock = [0], [0], threading.Lock()

    def answer(messages, info):  # later jobs finish first
        n = int(messages[-1].parts[-1].content)
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.04 * (5 - n))
        with lock:
            running[0] -= 1
        return ModelResponse(parts=[TextPart(json.dumps({"n": n}))])

    monkeypatch.setattr(llm, "model", lambda: FunctionModel(answer))
    monkeypatch.setattr(llm, "USE_CACHE", False)
    base, agent = llm.settings(), llm.agent(Out, "Echo the number.")
    for limit in (1, 4):
        monkeypatch.setattr(llm, "settings", lambda limit=limit: dataclasses.replace(base, concurrency=limit))
        peak[0] = 0
        assert [o.n for o in llm.call_all([("echo", agent, str(n), f"job {n}") for n in range(5)])] == [0, 1, 2, 3, 4]
        assert peak[0] == 1 if limit == 1 else 1 < peak[0] <= limit
