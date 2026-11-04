import json

import pytest

from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from bench import facts, questions
from bench.config import question_types
from bench.verify import verify_facts

TEXT = ("Milking animals should be fed 2 kg of compound cattle feed.\n\nIn addition to this quantity, 1 kg good quality "
        "oil cake should also be given to pregnant animals during the last two months of pregnancy.")
FEEDING = {"segment_id": "book-01", "title": "Feeding", "file_name": "book.pdf", "use_for_questions": True, "headings": [],
           "pages": [5, 5], "text": TEXT, "page_spans": [{"page": 5, "start": 0, "end": len(TEXT)}]}
EMPTY = {**FEEDING, "segment_id": "book-02", "title": "Empty"}
CALVES = {**FEEDING, "segment_id": "book-04", "title": "Calves", "text": "Calves need 4 litres of milk a day.",
          "page_spans": [{"page": 9, "start": 0, "end": 35}]}
TYPES = {name: t for name, t in question_types(active_only=False).items() if name in ("direct", "unanswerable", "noisy", "multi_hop")}
TYPES["noisy"] = {**TYPES["noisy"], "enabled": True, "from": ["direct"]}  # off in the config; switched on here to test rewrites


def _replies(messages, info):
    """Each agent's reply, told apart by its instructions."""
    system = messages[-1].instructions
    if system.startswith("You review"):
        reviews = [] if "whole library" in messages[-1].parts[-1].content else [
            {"id": "d1", "verdict": "rewrite", "reason": "name the feed",
             "question": "How much oil cake should pregnant animals get in the last two months of pregnancy?"}]
        reply = {"reviews": reviews}
    elif "TYPE: direct" in system:
        reply = {"questions": [{"question": "How much oil cake do pregnant animals get?", "fact_ids": ["f2"], "answer": "1 kg", "absent_term": ""},
                               {"question": "What do milking animals eat?", "fact_ids": ["f9"], "answer": "feed", "absent_term": ""}]}
    elif "TYPE: unanswerable" in system:
        reply = {"questions": [{"question": "How much oil cake should a pregnant camel get?", "fact_ids": ["f2"], "answer": "",
                                "absent_term": "camel"}]}
    elif "STYLE: noisy" in system:
        reply = {"questions": [{"source": "book-01:q01", "question": "how much oil cake for pregnent animals last 2 months"}]}
    else:  # multi_hop, written from the whole library: x1, x2 are book-01's facts and x3 is book-04's
        reply = {"questions": [{"question": "How much oil cake for pregnant animals and how much milk for calves?",
                                "fact_ids": ["x2", "x3"], "answer": "1 kg; 4 litres", "absent_term": ""}]}
    return ModelResponse(parts=[TextPart(json.dumps(reply))])


def test_question_stage_writes_reviews_audits_rewrites_and_links_sections(tmp_path, monkeypatch):
    feeding_facts, _ = verify_facts(FEEDING, [{"claim": "Milking animals get 2 kg of feed.", "first": "p1.s1", "last": "p1.s1"},
                                        {"claim": "Pregnant animals get 1 kg oil cake.", "first": "p2.s1", "last": "p2.s1"}])
    calves, _ = verify_facts(CALVES, [{"claim": "Calves need 4 litres of milk a day.", "first": "p1.s1", "last": "p1.s1"}])
    (tmp_path / "facts.jsonl").write_text("\n".join(json.dumps({"segment_id": s, "version": "facts-v4", "facts": f, "rejected": []})
                                                    for s, f in (("book-01", feeding_facts), ("book-02", []), ("book-04", calves))))
    (tmp_path / "questions.jsonl").write_text(json.dumps({"segment_id": "book-03", "questions": []}))  # no longer a segment
    (tmp_path / "pages.jsonl").write_text(json.dumps({"text": TEXT}))
    calls = []
    monkeypatch.setattr(questions.llm, "model", lambda: FunctionModel(lambda m, i: calls.append(1) or _replies(m, i)))
    monkeypatch.setattr(questions, "question_types", lambda: TYPES)
    monkeypatch.setattr(questions, "load_segments", lambda: [FEEDING, EMPTY, CALVES])
    monkeypatch.setattr(facts, "load_segments", lambda: [FEEDING, EMPTY, CALVES])
    for name in ("FACTS", "QUESTIONS", "PAGES"):
        monkeypatch.setattr(questions, name, tmp_path / f"{name.lower()}.jsonl")
    questions.run_questions()

    rows = {r["segment_id"]: r for r in map(json.loads, (tmp_path / "questions.jsonl").read_text().splitlines())}
    assert list(rows) == ["book-01", "book-02", "book-04", "library"]  # book-03's row is gone
    direct, unanswerable, noisy = rows["book-01"]["questions"]
    assert direct["question_id"] == "book-01:q01" and direct["review"]["original"] == "How much oil cake do pregnant animals get?"
    assert direct["evidence"] == [{"segment_id": "book-01", **{k: feeding_facts[1][k] for k in ("quote", "start", "end", "pages")}}]
    assert unanswerable["absent_term"] == "camel" and unanswerable["evidence"] == []
    assert noisy["type"] == "noisy" and noisy["source_question"] == direct["question"] and noisy["evidence"] == direct["evidence"]
    assert noisy["question_id"] == "book-01:q03"
    assert [r["reason"] for r in rows["book-01"]["rejected"]] == ["refers to a fact that isn't verified"]
    assert rows["book-02"]["questions"] == [] and rows["book-01"]["fingerprint"] and direct["uid"] != noisy["uid"]
    (hop,) = rows["library"]["questions"]
    assert [ev["segment_id"] for ev in hop["evidence"]] == ["book-01", "book-04"] and hop["question_id"] == "library:q01"

    before = len(calls)
    questions.run_questions()  # resumes: every row is current, so no agent is called
    assert len(calls) == before


def test_reviewer_flags_never_remove_and_switched_off_types_make_no_call(tmp_path, monkeypatch):
    feeding_facts, _ = verify_facts(FEEDING, [{"claim": "Pregnant animals get 1 kg oil cake.", "first": "p2.s1", "last": "p2.s1"}])
    (tmp_path / "facts.jsonl").write_text(json.dumps({"segment_id": "book-01", "facts": feeding_facts, "rejected": []}))
    (tmp_path / "pages.jsonl").write_text(json.dumps({"text": TEXT}))
    roles = []

    def replies(messages, info):
        system = messages[-1].instructions
        roles.append(system.split("TYPE: ")[-1].split(":")[0] if "TYPE: " in system else "reviewer")
        if system.startswith("You review"):
            reply = {"reviews": [{"id": "d1", "verdict": "flag", "question": "", "reason": "too vague"}]}
        else:
            reply = {"questions": [{"question": "How much oil cake should pregnant animals get?", "fact_ids": ["f1"],
                                    "answer": "1 kg", "absent_term": ""}]}
        return ModelResponse(parts=[TextPart(json.dumps(reply))])

    types = {**{n: t for n, t in TYPES.items() if n == "direct"},
             "unanswerable": {**TYPES["unanswerable"], "enabled": False}}  # what question_types() would leave out
    monkeypatch.setattr(questions.llm, "model", lambda: FunctionModel(replies))
    monkeypatch.setattr(questions, "question_types", lambda: {n: t for n, t in types.items() if t["enabled"]})
    monkeypatch.setattr(questions, "load_segments", lambda: [FEEDING])
    monkeypatch.setattr(facts, "load_segments", lambda: [FEEDING])
    for name in ("FACTS", "QUESTIONS", "PAGES"):
        monkeypatch.setattr(questions, name, tmp_path / f"{name.lower()}.jsonl")
    questions.run_questions()
    (q,) = json.loads((tmp_path / "questions.jsonl").read_text())["questions"]
    assert q["review"] == {"verdict": "flag", "reason": "too vague"}  # flagged by the AI, kept by the audit
    assert roles == ["direct", "reviewer"]  # no unanswerable writer, no library call


def test_config_switches_types_off_and_rejects_unknown_keys(tmp_path):
    from bench.config import BenchError

    path = tmp_path / "types.toml"
    path.write_text('[a]\ncount = 1\nsource = "facts"\nprompt = "p"\n[b]\ncount = 0\nsource = "facts"\nprompt = "p"\n'
                    '[c]\nenabled = false\ncount = 2\nsource = "facts"\nprompt = "p"\n')
    assert list(question_types(path)) == ["a"] and list(question_types(path, active_only=False)) == ["a", "b", "c"]
    path.write_text('[a]\ncount = 1\nsource = "facts"\nprompt = "p"\nrule = { words = [2, 5] }\n')  # typo: rule
    with pytest.raises(BenchError, match="rule"):
        question_types(path)
    path.write_text('[a]\ncount = 1\nsource = "facts"\nprompt = "p"\nrules = { word = [2, 5] }\n')  # unknown rule
    with pytest.raises(BenchError, match="word"):
        question_types(path)
