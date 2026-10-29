import json
import re

from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from bench import agents
from bench.verify import find, question_problem, verify_facts

TEXT = ("Milking animals should be fed 2 kg of compound cattle feed.\n\nIn addition to this quantity, 1 kg "
        "compound cattle feed and 1 kg good quality oil cake should\nalso be given to pregnant animals during the "
        "last two months of pregnancy. The farmer’s “best” cows are dewormed every 3-4 months.")
SEGMENT = {"segment_id": "book-01", "title": "Feeding", "file_name": "book.pdf", "use_for_questions": True,
           "headings": [], "pages": [5, 6], "text": TEXT,
           "page_spans": [{"page": 5, "start": 0, "end": 59}, {"page": 6, "start": 61, "end": len(TEXT)}]}


def test_quotes_match_across_line_breaks_quote_styles_and_dashes():
    assert find("1 kg good quality oil cake should also be given to pregnant animals", TEXT)
    assert find("The farmer's \"best\" cows are dewormed every 3–4 months.", TEXT)
    assert find("1 kg good quality oil cake must also be given", TEXT) is None  # a changed word fails


def test_fact_audit_keeps_verbatim_quotes_and_records_where_they_are():
    facts = [{"claim": "Pregnant animals get 1 kg extra feed.", "quote": "1 kg good quality oil cake should also be given"},
             {"claim": "Invented.", "quote": "Calves need 5 litres of milk a day."},
             {"claim": "Dup.", "quote": "1 kg good quality oil cake should also be given"},
             {"claim": "", "quote": "Milking animals should be fed 2 kg"},
             {"claim": "Punjab grew 464.", "quote": "Punjab 444 434 432 436 480 464"}]
    kept, rejected = verify_facts(SEGMENT, facts)
    assert [f["fact_id"] for f in kept] == ["book-01:f01"] and kept[0]["pages"] == [6, 6]
    assert TEXT[kept[0]["start"] : kept[0]["end"]].startswith("1 kg good")
    assert [r["reason"] for r in rejected] == ["quote not found in the PDF text", "duplicate quote", "empty claim",
                                               "quote is a table row, not a sentence"]


def test_question_audit_rules():
    facts = {"f1": {"quote": "1 kg good quality oil cake should also be given to pregnant animals during the last two months"},
             "f2": {"quote": "Milking animals should be fed 2 kg of compound cattle feed."}}
    ok = {"type": "precise", "question": "How much oil cake in late pregnancy?", "fact_ids": ["f1"], "answer": "1 kg"}
    assert question_problem(ok, facts) is None
    assert question_problem({**ok, "question": "For how long before calving?", "answer": "two months"}, facts) is None
    assert question_problem({**ok, "answer": "good quality"}, facts) == "precise answer has no number"
    assert question_problem({**ok, "answer": "2 kg"}, facts) == "answer not found in its facts' quotes"
    assert question_problem({**ok, "fact_ids": ["f9"]}, facts) == "refers to a fact that isn't verified"
    assert question_problem({**ok, "type": "keyword", "question": "oil cake pregnant animals late pregnancy dose"}, facts)
    assert question_problem({**ok, "type": "multi_fact"}, facts) == "multi_fact needs two facts"
    yearly = {"f3": {"quote": "Milk production reached 247.87 million tonnes in 2024-25."}}
    asked = {"type": "precise", "fact_ids": ["f3"], "answer": "247.87 million tonnes"}
    assert question_problem({**asked, "question": "What was India's milk production?"}, yearly) == "question drops its quote's year"
    assert question_problem({**asked, "question": "What was India's milk production in 2024-25?"}, yearly) is None
    both = {"type": "multi_fact", "question": "How much feed for milking and pregnant animals?", "fact_ids": ["f1", "f2"]}
    assert question_problem({**both, "answer": "1 kg; 2 kg"}, facts) is None
    assert question_problem({**both, "answer": "1 kg"}, facts) == "multi_fact answer needs a span from each fact"
    assert question_problem({**both, "answer": "1 kg; 3 kg"}, facts) == "answer not found in its facts' quotes"
    assert question_problem({**ok, "type": "paraphrased", "question": "quality oil cake given pregnant animals?",
                             "answer": "oil cake"}, facts) == "most of the paraphrase's words come from the quote"


def test_fact_stage_keeps_only_verified_facts(tmp_path, monkeypatch):
    reply = {"facts": [{"claim": "Pregnant animals get 1 kg oil cake in the last two months.",
                        "quote": "1 kg good quality oil cake should also be given to pregnant animals"},
                       {"claim": "Made up.", "quote": "This sentence is not in the text."}]}
    model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(json.dumps(reply))]))
    monkeypatch.setattr(agents.llm, "model", lambda: model)
    monkeypatch.setattr(agents, "load_segments", lambda: [SEGMENT])
    monkeypatch.setattr(agents, "FACTS", tmp_path / "facts.jsonl")
    agents.run_facts()
    (row,) = [json.loads(l) for l in (tmp_path / "facts.jsonl").read_text().splitlines()]
    assert [f["fact_id"] for f in row["facts"]] == ["book-01:f01"] and len(row["rejected"]) == 1
    agents.run_facts()  # resumes: the segment is already done, nothing changes
    assert len((tmp_path / "facts.jsonl").read_text().splitlines()) == 1
    (tmp_path / "facts.jsonl").write_text(json.dumps({**row, "version": "facts-v0", "facts": []}))
    agents.run_facts()  # a row from an older version is redone
    assert json.loads((tmp_path / "facts.jsonl").read_text())["version"] == agents.FACTS_VERSION


def test_question_stage_attaches_evidence_and_clears_segments_without_facts(tmp_path, monkeypatch):
    (fact,), _ = verify_facts(SEGMENT, [{"claim": "Pregnant animals get 1 kg oil cake.",
                                         "quote": "1 kg good quality oil cake should also be given to pregnant animals"}])
    other = {**SEGMENT, "segment_id": "book-02"}
    (tmp_path / "facts.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"segment_id": "book-01", "facts": [fact], "rejected": []}, {"segment_id": "book-02", "facts": [], "rejected": []}]))
    (tmp_path / "questions.jsonl").write_text("\n".join(json.dumps({"segment_id": s, "questions": [{"question_id": f"{s}:q01"}]})
                                                        for s in ("book-02", "book-03")))  # book-03 is no longer a segment
    reply = {"questions": [{"type": "precise", "question": "How much oil cake for pregnant animals?", "fact_ids": ["f1"], "answer": "1 kg"}]}
    monkeypatch.setattr(agents.llm, "model", lambda: FunctionModel(lambda m, i: ModelResponse(parts=[TextPart(json.dumps(reply))])))
    monkeypatch.setattr(agents, "load_segments", lambda: [SEGMENT, other])
    monkeypatch.setattr(agents, "FACTS", tmp_path / "facts.jsonl")
    monkeypatch.setattr(agents, "QUESTIONS", tmp_path / "questions.jsonl")
    agents.run_questions(force=True)
    rows = {r["segment_id"]: r for r in map(json.loads, (tmp_path / "questions.jsonl").read_text().splitlines())}
    (q,) = rows["book-01"]["questions"]
    assert q["question_id"] == "book-01:q01" and q["fact_ids"] == ["book-01:f01"]
    assert q["evidence"] == [{k: fact[k] for k in ("quote", "start", "end", "pages")}]
    assert rows["book-02"]["questions"] == []  # no facts now: the old questions are gone
    assert "book-03" not in rows  # and so are the rows of a segment that no longer exists


def test_facts_are_asked_for_evenly_across_parts(tmp_path, monkeypatch):
    asked = []

    def nothing_found(messages, info):  # records how many facts each part asks for
        asked.append(int(re.search(r"Return at most (\d+)", messages[-1].parts[-1].content).group(1)))
        return ModelResponse(parts=[TextPart(json.dumps({"facts": []}))])

    monkeypatch.setattr(agents.llm, "model", lambda: FunctionModel(nothing_found))
    monkeypatch.setattr(agents, "load_segments", lambda: [SEGMENT])
    monkeypatch.setattr(agents, "FACTS", tmp_path / "facts.jsonl")
    monkeypatch.setattr(agents, "PART_CHARS", 120)  # three parts
    agents.run_facts(max_facts=10)
    assert asked == [3, 6, 10]  # an even share each, plus what earlier parts didn't find
