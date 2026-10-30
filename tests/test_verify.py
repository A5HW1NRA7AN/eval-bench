import json
import re

from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from bench import facts
from bench.config import question_types
from bench.segments import sentences
from bench.verify import claim_problem, find, numbers, question_problem, size_of, verify_facts

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


def test_fact_audit_cuts_quotes_from_sentence_ranges_and_checks_claims():
    pregnant = "Pregnant animals get 1 kg extra compound cattle feed and 1 kg oil cake in the last two months of pregnancy."
    found = [{"claim": pregnant, "first": "p2.s1", "last": "p2.s1"},
             {"claim": "Cows are dewormed every 3-4 months.", "first": "p2.s1", "last": "p2.s2"},
             {"claim": "Calves need 5 litres of milk.", "first": "p9.s1", "last": "p9.s1"},
             {"claim": "Backwards.", "first": "p2.s2", "last": "p2.s1"},
             {"claim": "Milking animals get 3 kg of feed.", "first": "p1.s1", "last": "p1.s1"},
             {"claim": "It is fed 2 kg.", "first": "p1.s1", "last": "p1.s1"},
             {"claim": "NDDB says milking animals get 2 kg of feed.", "first": "p1.s1", "last": "p1.s1"},
             {"claim": "", "first": "p1.s1", "last": "p1.s1"},
             {"claim": "Milking animals should get two kg of compound cattle feed.", "first": "p1.s1", "last": "p1.s1"}]
    kept, rejected = verify_facts(SEGMENT, found)
    assert [f["fact_id"] for f in kept] == ["book-01:f01", "book-01:f02"]
    assert kept[0]["quote"].startswith("In addition to this quantity") and kept[0]["pages"] == [6, 6]
    assert kept[0]["size"] == "sentence" and TEXT[kept[0]["start"] : kept[0]["end"]] == kept[0]["quote"]
    assert [r["reason"] for r in rejected] == [
        "overlaps book-01:f01", "points at a sentence that isn't in the section", "its last sentence comes before its first",
        "claim has a number its quote lacks: 3", "claim starts with a pronoun",
        "claim names NDDB, which the section never mentions", "empty claim"]


def test_claim_years_come_from_the_section_and_values_from_the_quote():
    quote = "Milking animals should be fed 2 kg of compound cattle feed."
    dated = {**SEGMENT, "text": "Feeding norms for 2024-25. " + TEXT}
    assert claim_problem("In 2024-25, milking animals get 2 kg of feed.", quote, dated) is None
    assert claim_problem("As of March 31, 2024, milking animals get 2 kg.", quote, dated) is None
    assert claim_problem("In 2024-25, milking animals get 2 kg of feed.", quote, SEGMENT) == \
        "claim names a year the section never mentions: 2024"
    assert claim_problem("In 2024-25, milking animals get 3 kg of feed.", quote, dated) == "claim has a number its quote lacks: 3"


def test_fact_sizes_and_numbers():
    units = {u["id"]: u for u in sentences(TEXT)}
    assert size_of({"first": "p2.s1", "last": "p2.s2"}, units) == "paragraph"
    assert size_of({"first": "p2.s2", "last": "p2.s2"}, units) == "sentence"
    three = {u["id"]: u for u in sentences("One. Two. Three.")}
    assert size_of({"first": "p1.s1", "last": "p1.s2"}, three) == "passage"
    assert numbers("clears in four days, 1,000 ppb or 0.5") == {"4", "1000", "0.5"}
    assert numbers("Aflatoxin B1 at 20 ppb") == {"20"}  # B1 is a name, not the number 1


RULES = {name: t.get("rules", {}) for name, t in question_types().items()}


def test_question_audit_rules():
    facts_ = {"f1": {"fact_id": "book-01:f01", "quote": "1 kg good quality oil cake should also be given to pregnant animals during the last two months"},
              "f2": {"fact_id": "book-01:f02", "quote": "Milking animals should be fed 2 kg of compound cattle feed."},
              "f3": {"fact_id": "book-02:f01", "quote": "Calves need 4 litres of milk a day."}}
    ok = {"type": "precise", "question": "How much oil cake in late pregnancy?", "fact_ids": ["f1"], "answer": "1 kg"}

    def check(q, corpus=None):
        return question_problem(q, facts_, RULES[q["type"]], corpus)

    assert check(ok) is None
    assert check({**ok, "question": "For how long before calving?", "answer": "two months"}) is None
    assert check({**ok, "answer": "good quality"}) == "the answer has no number"
    assert check({**ok, "answer": "2 kg"}) == "answer not found in its facts' quotes"
    assert check({**ok, "fact_ids": ["f9"]}) == "refers to a fact that isn't verified"
    assert check({**ok, "type": "keyword", "question": "oil cake pregnant animals late pregnancy dose"}) == "needs 2-5 words, has 7"
    assert check({**ok, "type": "multi_fact"}) == "needs 2-3 facts, uses 1"
    yearly = {"f4": {"fact_id": "stats-01:f01", "quote": "Milk production reached 247.87 million tonnes in 2024-25."}}
    asked = {"type": "precise", "fact_ids": ["f4"], "answer": "247.87 million tonnes"}
    assert question_problem({**asked, "question": "What was India's milk production?"}, yearly, RULES["precise"]) == "question drops its quote's year"
    assert question_problem({**asked, "question": "What was India's milk production in 2024-25?"}, yearly, RULES["precise"]) is None
    both = {"type": "multi_fact", "question": "How much feed for milking and pregnant animals?", "fact_ids": ["f1", "f2"]}
    assert check({**both, "answer": "1 kg; 2 kg"}) is None
    assert check({**both, "answer": "1 kg"}) == "needs 2-3 answer parts, has 1"
    assert check({**both, "answer": "1 kg; 1 kg"}) == "the answer needs a span from each fact"
    assert check({**both, "answer": "1 kg; 3 kg"}) == "answer not found in its facts' quotes"
    assert check({**ok, "type": "paraphrased", "question": "quality oil cake given pregnant animals?",
                  "answer": "oil cake"}) == "too many of the question's words come from the quote"
    listing = {"type": "comparison", "question": "What feed do pregnant animals get extra?", "fact_ids": ["f1"]}
    assert check({**listing, "answer": "1 kg good quality oil cake; last two months"}) is None
    hop = {"type": "multi_hop", "question": "How much oil cake for pregnant cows and milk for calves?", "answer": "1 kg; 4 litres"}
    assert check({**hop, "fact_ids": ["f1", "f3"]}) is None
    assert check({**hop, "fact_ids": ["f1", "f2"], "answer": "1 kg; 2 kg"}) == "needs facts from 2 sections"
    absent = {"type": "unanswerable", "question": "How much oil cake should a pregnant camel get?", "fact_ids": ["f1"],
              "answer": "", "absent_term": "camel"}
    assert check(absent, corpus="cows, buffaloes and calves") is None
    assert check(absent, corpus="Camel milk is rich in iron.") == '"camel" is in the library, so the question may be answerable'
    assert check({**absent, "absent_term": "yak"}) == "the question doesn't name its absent term"
    noisy = {"type": "noisy", "question": "how much oil cake in late pregnancy?", "fact_ids": ["f1"], "answer": "1 kg",
             "source_question": "How much oil cake in late pregnancy?"}
    assert check(noisy) == "the rewrite is the same as its source"
    assert check({**noisy, "question": "pregnent bhains ko kitna oil cake"}) is None


def _fake_agents(messages, info):
    """Planner, extractor and reviewer replies, told apart by their prompts."""
    prompt = messages[-1].parts[-1].content
    if "FACTS TO REVIEW" in prompt:
        reply = {"reviews": [{"id": "f1", "verdict": "rewrite", "reason": "missed the compound feed",
                              "claim": "Pregnant animals get 1 kg compound cattle feed and 1 kg oil cake in the last two months."},
                             {"id": "f2", "verdict": "drop", "claim": "", "reason": "repeats f1"}]}
    elif "TOPIC:" in prompt:
        reply = {"facts": [{"claim": "Pregnant animals get 1 kg oil cake in the last two months.", "first": "p2.s1", "last": "p2.s1"}]}
    else:
        reply = {"topics": [{"topic": "feed in late pregnancy", "kind": "figure", "paragraphs": ["p2"]}, {"topic": "not here", "kind": "figure", "paragraphs": ["p7"]},
                            {"topic": "feed again", "kind": "explanation", "paragraphs": ["p1"]}]}
    return ModelResponse(parts=[TextPart(json.dumps(reply))])


def test_fact_stage_plans_extracts_reviews_and_audits(tmp_path, monkeypatch):
    monkeypatch.setattr(facts.llm, "model", lambda: FunctionModel(_fake_agents))
    monkeypatch.setattr(facts, "load_segments", lambda: [SEGMENT])
    monkeypatch.setattr(facts, "FACTS", tmp_path / "facts.jsonl")
    facts.run_facts()
    (row,) = [json.loads(l) for l in (tmp_path / "facts.jsonl").read_text().splitlines()]
    (fact,) = row["facts"]
    assert fact["claim"].startswith("Pregnant animals get 1 kg compound cattle feed") and fact["topic"] == "feed in late pregnancy"
    assert fact["review"]["original"] == "Pregnant animals get 1 kg oil cake in the last two months."
    assert fact["quote"] == TEXT[fact["start"] : fact["end"]] and fact["size"] == "sentence"
    assert [r["reason"] for r in row["rejected"]] == ["reviewer: repeats f1"]  # "not here" named no paragraph of the part
    facts.run_facts()  # resumes: the segment is already done, nothing changes
    assert len((tmp_path / "facts.jsonl").read_text().splitlines()) == 1
    (tmp_path / "facts.jsonl").write_text(json.dumps({**row, "version": "facts-v0", "facts": []}))
    facts.run_facts()  # a row from an older version is redone
    assert json.loads((tmp_path / "facts.jsonl").read_text())["version"] == facts.FACTS_VERSION


def test_facts_are_asked_for_evenly_across_parts(tmp_path, monkeypatch):
    asked = []

    def nothing_found(messages, info):  # records how many topics each part asks for
        asked.append(int(re.search(r"Return at most (\d+) topics", messages[-1].parts[-1].content).group(1)))
        return ModelResponse(parts=[TextPart(json.dumps({"topics": []}))])

    monkeypatch.setattr(facts.llm, "model", lambda: FunctionModel(nothing_found))
    monkeypatch.setattr(facts, "load_segments", lambda: [SEGMENT])
    monkeypatch.setattr(facts, "FACTS", tmp_path / "facts.jsonl")
    monkeypatch.setattr(facts, "PART_CHARS", 120)  # three parts
    facts.run_facts(max_facts=10)
    # an even share each, plus what earlier parts didn't find; part 2 only holds the end of a sentence
    # that part 1 already showed, so it is skipped and its share moves on to part 3
    assert asked == [3, 10]
