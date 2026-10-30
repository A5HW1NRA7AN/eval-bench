import json

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
TYPES = {name: t for name, t in question_types().items() if name in ("direct", "unanswerable", "noisy", "multi_hop")}
TYPES["noisy"] = {**TYPES["noisy"], "from": ["direct"]}


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
        reply = {"questions": [{"source": "book-01:q01", "question": "pregnent bhains ko kitna oil cake dena last 2 month"}]}
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
    assert rows["book-02"]["questions"] == [] and rows["book-01"]["facts_version"] == "facts-v4"
    (hop,) = rows["library"]["questions"]
    assert [ev["segment_id"] for ev in hop["evidence"]] == ["book-01", "book-04"] and hop["question_id"] == "library:q01"

    before = len(calls)
    questions.run_questions()  # resumes: every row is current, so no agent is called
    assert len(calls) == before
