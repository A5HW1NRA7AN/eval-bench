"""General questions: the candidates code builds (bench/questions.py), the orchestrator's plan checked by code, item
writing with one repair, groups of PDFs and the rows (bench/orchestrator.py)."""

import json
import os
import subprocess
import sys

import pytest

from bench import checks, llm, orchestrator, pdf, questions, segments, templates
from bench import facts as fact_stage
from bench.config import BenchError, question_types
from bench.verify import verify_facts

TEXT = ("Milking animals should be fed 2 kg of compound cattle feed.\n\nIn addition to this quantity, 1 kg good quality "
        "oil cake should also be given to pregnant animals during the last two months of pregnancy.")
CALF_TEXT = "Calves of milking animals need 4 litres of milk a day."
FEEDING = {"segment_id": "book-01", "title": "Feeding", "file_name": "book.pdf", "use_for_questions": True, "headings": [],
           "pages": [5, 5], "text": TEXT, "page_spans": [{"page": 5, "start": 0, "end": len(TEXT)}]}
EMPTY = {**FEEDING, "segment_id": "book-02", "title": "Empty"}  # a section with no facts
CALVES = {**FEEDING, "segment_id": "book-04", "title": "Calves", "text": CALF_TEXT, "page_spans": [{"page": 9, "start": 0, "end": len(CALF_TEXT)}]}
HERD = {**CALVES, "segment_id": "herd-01", "file_name": "herd.pdf", "title": "Herd"}  # another PDF
CLAIMS = {"book-01": [("Milking animals get 2 kg of compound cattle feed.", "p1.s1"),
                      ("Pregnant animals get 1 kg oil cake in the last two months.", "p2.s1")],
          "book-04": [(CALF_TEXT, "p1.s1")], "herd-01": [(CALF_TEXT, "p1.s1")]}
ALL = question_types(active_only=False)
TYPES = {n: ALL[n] for n in ("direct", "unanswerable", "multi_hop")}
LISTS = {"species": ["cattle", "calf", "camel", "yak"]}  # the library says cattle and calves: camel and yak are absent


def _rows(*segs):
    """data/facts.jsonl's rows by section: the claims above, verified."""
    return {s["segment_id"]: {"segment_id": s["segment_id"], "rejected": [],
                              "facts": verify_facts(s, [{"claim": c, "first": p, "last": p} for c, p in CLAIMS.get(s["segment_id"], [])])[0]}
            for s in segs}


def _material(segs=(FEEDING, EMPTY, CALVES)):
    """g1, g2 (book-01) and g3 (book-04), with their pairs, word lists and absent terms."""
    return orchestrator.material(list(segs), _rows(*segs), LISTS, questions.absent_terms(LISTS, TEXT + CALF_TEXT))


def test_code_builds_the_candidates_eligibility_pairs_and_absent_terms_word_by_word(monkeypatch):
    assert [questions.needs(ALL[n]) for n in ("multi_hop", "multi_fact", "unanswerable", "comparison", "precise")] == [
        "cross", "pair", "absent", "fact", "fact"]
    mat = _material()
    g1 = mat["facts"]["g1"]
    assert questions.fits(ALL["precise"], g1) and not questions.fits(ALL["comparison"], g1)  # a number; but one sentence
    assert [(p["id"], p["a"], p["b"], p["word"], p["cross"]) for p in mat["pairs"]] == [
        ("p1", "g1", "g3", "milking", True), ("p2", "g1", "g2", "animals", False), ("p3", "g2", "g3", "animals", True)]
    assert mat["named"] == {"g1": {"species": "cattle"}, "g2": {}, "g3": {}} and mat["absent"] == {"species": ["camel", "yak"]}
    library = "Sheep and goats graze. Calves need milk. Blue Tongue outbreaks."  # any word of a term rules it out, in any form
    assert questions.absent_terms({"species": ["sheep and lambs", "goat", "calf", "yak", "lamb"], "diseases": ["bluetongue", "anthrax"]},
                                  library) == {"species": ["yak", "lamb"], "diseases": ["anthrax"]}  # "Blue Tongue" is bluetongue
    prompt = orchestrator.question_plan_prompt("book.pdf", {n: TYPES[n] for n in ("direct", "unanswerable")}, mat, [FEEDING, CALVES])
    assert "book-01 Feeding | QUOTA: direct 2, unanswerable 1" in prompt and "book-04 Calves | QUOTA: direct 1\n" in prompt
    assert "PAIRS" not in prompt and "- species: camel, yak" in prompt
    prompt = orchestrator.question_plan_prompt("book.pdf", {n: ALL[n] for n in ("multi_fact", "multi_hop")}, mat, pairs=mat["pairs"])
    assert "p2 one section (Feeding) | shared: animals\n  g1: Milking animals get 2 kg of compound cattle feed.\n  g2: Pregnant" in prompt
    assert "p1 two sections (Feeding; Calves) | shared: milking" in prompt and "FACTS" not in prompt  # both kinds, claims side by side
    monkeypatch.setattr(questions, "PAIR_PARTNERS", 1)
    assert [p["id"] for p in questions.pairs(mat["facts"], LISTS)] == ["p1", "p2"]  # g3 has its one partner across sections
    monkeypatch.setattr(questions, "PAIR_TERM_FACTS", 2)  # "animals" is in 3 facts of the group: too common to pair them
    assert [(p["a"], p["b"]) for p in questions.pairs(mat["facts"], LISTS)] == [("g1", "g3")]


def test_the_candidates_are_the_same_under_any_hash_seed():
    script = ("import json\nfrom bench import questions\n"
              "claims = ['Cows need green fodder and clean water daily.', 'Buffaloes need green fodder in summer.',\n"
              "          'Clean water prevents disease in cows.', 'Summer heat lowers the milk yield of buffaloes.',\n"
              "          'Green fodder raises the milk yield.', 'The daily water need rises in summer.']\n"
              "facts = {f'g{n}': {'fact_id': f's{n % 3}:f{n}', 'claim': c} for n, c in enumerate(claims, start=1)}\n"
              "print(json.dumps(questions.pairs(facts, {'species': ['cow', 'buffalo']})))")
    outs = {subprocess.run([sys.executable, "-c", script], env={**os.environ, "PYTHONHASHSEED": seed}, capture_output=True,
                           text=True, check=True).stdout for seed in ("1", "2")}
    assert len(outs) == 1 and len(json.loads(outs.pop())) > 3


def test_the_plan_caps_hold_per_section_per_fact_and_per_absent_term():
    types = {**{n: ALL[n] for n in ("direct", "paraphrased", "keyword", "multi_hop")}, "unanswerable": {**ALL["unanswerable"], "count": 5}}
    take = orchestrator.Tally(types, _material()).take
    assert take("direct", ["g1"]) is None and take("direct", ["g1"]) == "planned twice"
    assert take("unanswerable", ["g1"], "camel") is None and take("paraphrased", ["g1"]) == "the fact has its questions"
    assert take("keyword", ["g2"]) is None and take("keyword", ["g1"]) == "over the type's count"  # one per section
    assert take("unanswerable", ["g2"], "Camel") is None and take("unanswerable", ["g3"], "camel") == "the absent term is used up"
    assert take("multi_hop", ["g2", "g3"]) is None and take("unanswerable", ["g3"], "yak") is None  # pairs don't count per fact


def test_code_checks_every_planned_item():
    types, mat = {n: ALL[n] for n in ("direct", "comparison", "unanswerable")}, _material()
    answer = orchestrator.question_plan_schema([*types, "multi_hop"]).model_validate({"items": [
        {"type": "direct", "ref": "g2 Pregnant animals get 1 kg oil cake", "absent_term": "", "brief": "ask about the last two months"},
        {"type": "unanswerable", "ref": "g1", "absent_term": " Camel ", "brief": ""},
        {"type": "unanswerable", "ref": "g2", "absent_term": "yak", "brief": ""},  # g2 names no word list
        {"type": "unanswerable", "ref": "g1", "absent_term": "cattle", "brief": ""},  # the library says cattle
        {"type": "comparison", "ref": "g1", "absent_term": "", "brief": ""},  # one sentence: nothing to compare
        {"type": "multi_hop", "ref": "p1", "absent_term": "", "brief": ""},  # planned in a call of its own
        {"type": "direct", "ref": "g9", "absent_term": "", "brief": ""},
        {"type": "direct", "ref": "g2", "absent_term": "", "brief": ""}]})
    items, invalid = orchestrator.check_items(answer, types, mat, orchestrator.Tally(types, mat))
    assert [(it["type"], it["facts"], it["absent_term"], it["replaces"], it["brief"]) for it in items] == [
        ("direct", ["g2"], "", "", "ask about the last two months"), ("unanswerable", ["g1"], "camel", "cattle", "")]
    assert [x["why"] for x in invalid] == ["the absent term isn't one for this fact"] * 2 + [
        "the fact doesn't suit the type", "a type this call doesn't plan", "no such fact", "planned twice"]
    cross = {"multi_hop": ALL["multi_hop"]}
    answer = orchestrator.question_plan_schema(["multi_hop"]).model_validate({"items": [
        {"type": "multi_hop", "ref": ref, "absent_term": "", "brief": ""} for ref in ("p3", "p2", "p7")]})
    items, invalid = orchestrator.check_items(answer, cross, mat, orchestrator.Tally(cross, mat))
    assert [it["facts"] for it in items] == [["g2", "g3"]] and [x["why"] for x in invalid] == ["a pair of the wrong kind", "no such pair"]


def test_code_plans_an_unreadable_answer_and_a_section_the_plan_leaves_out(monkeypatch, agents):
    mat = _material()
    spread = {n: {**ALL[n], "count": 1} for n in ("direct", "paraphrased", "keyword")}
    items = orchestrator.code_items(spread, mat, orchestrator.Tally(spread, mat), [FEEDING])
    assert [(it["type"], it["facts"]) for it in items] == [("direct", ["g1"]), ("paraphrased", ["g2"]), ("keyword", ["g1"])]
    agents(lambda system, prompt: {"no": "items"} if "- multi_hop" in prompt else
           {"items": [{"type": "direct", "ref": "g1", "absent_term": "", "brief": ""}]})  # book-04 (g3) left out
    planner = llm.agent(orchestrator.question_plan_schema(list(TYPES)), orchestrator.QUESTION_PLAN_PROMPT)
    items, invalid, record = orchestrator.plan_questions("book.pdf", mat, TYPES, planner, failed := [])
    assert [(it["id"], it["type"], it["facts"], it["by"]) for it in items] == [
        ("i1", "direct", ["g1"], "orchestrator"), ("i2", "direct", ["g3"], "code")]  # code plans no pairs
    assert items[1]["fact_ids"] == ["book-04:f01"] and not invalid
    assert [r["by"] for r in record] == ["orchestrator", "code"] and failed == ["book.pdf plan 2/2"]  # retried next run


def test_a_pdf_is_a_group_of_its_own_unless_data_topics_json_groups_it():
    segs = [FEEDING, EMPTY, CALVES, HERD]
    assert [(n, [s["segment_id"] for s in ss]) for n, ss in orchestrator.question_units({"groups": {}}, segs)] == [
        ("book.pdf", ["book-01", "book-02", "book-04"]), ("herd.pdf", ["herd-01"])]
    ((name, grouped),) = orchestrator.question_units({"groups": {"dairy": ["book.pdf", "herd.pdf"]}}, segs)
    together = orchestrator.material(grouped, _rows(*grouped), LISTS, {})
    assert name == "dairy" and not orchestrator.material([HERD], _rows(HERD), LISTS, {})["pairs"]
    assert ("g3", "g4", True) in [(p["a"], p["b"], p["cross"]) for p in together["pairs"]]  # book-04 and herd-01: two PDFs
    assert together["pdf_of"]["herd-01"] == "herd.pdf"


def test_code_attaches_each_items_facts_and_absent_term_and_counts_an_unfilled_item(monkeypatch, agents):
    batch = [{"id": "i2", "type": "unanswerable", "facts": ["g1"], "absent_term": "camel"}]
    answer = questions.ItemDrafts(questions=[questions.ItemDraft(item="", question="How much feed should camels get?", answer="2 kg"),
                                             questions.ItemDraft(item="i2", question="And again?", answer="")])
    assert orchestrator.drafts_from(answer, batch) == [{"type": "unanswerable", "question": "How much feed should camels get?",
                                                        "answer": "", "fact_ids": ["g1"], "absent_term": "camel", "item": "i2"}]
    agents(lambda system, prompt: {"reviews": []} if system.startswith("You review") else {"questions": []})
    items = [{"id": "i1", "type": "direct", "facts": ["g1"], "absent_term": "", "replaces": "", "brief": "", "by": "code", "fact_ids": ["book-01:f01"]}]
    writers = {n: llm.agent(questions.ItemDrafts, questions.ITEM_WRITER_PROMPT + n) for n in TYPES}
    kept, rejected = orchestrator.write_items("book.pdf", items, _material(), TYPES, writers, None, TEXT, [])
    row = {"plan": {"items": items, "invalid": []}, "questions": kept, "rejected": rejected, "calls": {}, "failed_calls": []}
    assert kept == rejected == [] and items[0]["outcome"] == "unfilled" and orchestrator.question_stats([row])["unfilled"] == {"direct": 1}


def _replies(system, prompt):
    """The plan, the writers (one repair) and the reviewer of the stage test, told apart by their instructions."""
    if system.startswith("You plan the general questions"):
        items = ([{"type": "multi_hop", "ref": "p3"}, {"type": "multi_hop", "ref": "p2"}] if "- multi_hop" in prompt else
                 [{"type": "direct", "ref": "g2 Pregnant animals get 1 kg oil cake"}, {"type": "unanswerable", "ref": "g1", "absent_term": "Camel"},
                  {"type": "unanswerable", "ref": "g2", "absent_term": "yak"}])  # book-04 is left out: code plans it
        return {"items": [{"absent_term": "", "brief": "", **it} for it in items]}
    if system.startswith("You review"):
        return {"reviews": [{"id": "d1", "verdict": "rewrite", "reason": "name the stage",
                             "question": "How much oil cake should pregnant animals get in the last two months of pregnancy?"}]
                if "SECTION: Feeding" in prompt else []}
    if "A CHECK REJECTED THIS QUESTION" in prompt:  # the repair, told why
        return {"questions": [{"item": "i3", "question": "How much milk do calves of milking animals need each day?", "answer": "4 litres of milk"}]}
    if "TYPE: direct" in system:
        return {"questions": [{"item": "i1", "question": "How much oil cake do pregnant animals get?", "answer": "1 kg"}] if "- i1:" in prompt else
                [{"item": "i3", "question": "How much milk do calves of milking animals need each day?", "answer": "5 litres"}]}
    if "TYPE: unanswerable" in system:
        return {"questions": [{"item": "i2", "question": "How much compound feed should milking camels get?", "answer": "2 kg"}]}
    return {"questions": [{"item": "i4", "question": "How much oil cake do pregnant animals get, and how much milk do calves need?",
                           "answer": "1 kg; 4 litres"}]}


def test_the_stage_plans_writes_repairs_once_and_resumes_with_no_call(monkeypatch, agents):
    rows = _rows(FEEDING, EMPTY, CALVES)
    pdf.write_jsonl(segments.SEGMENTS, [FEEDING, EMPTY, CALVES])
    pdf.write_jsonl(fact_stage.FACTS, list(rows.values()))
    pdf.write_jsonl(pdf.PAGES, [{"file_name": "book.pdf", "page": 5, "text": TEXT}, {"file_name": "book.pdf", "page": 9, "text": CALF_TEXT}])
    pdf.write_jsonl(questions.QUESTIONS, [{"segment_id": "book-03", "questions": []}])  # an older format: left out, then dropped
    monkeypatch.setattr(orchestrator, "question_types", lambda: TYPES)
    monkeypatch.setattr(checks, "question_types", lambda: TYPES)
    calls = []
    agents(lambda system, prompt: calls.append(1) or _replies(system, prompt))
    orchestrator.run_questions()

    (row,) = questions.question_rows()
    assert row["group"] == "book.pdf" and row["pdfs"] == ["book.pdf"] and row["failed_calls"] == [] and row["rejected"] == []
    assert [(it["id"], it["type"], it["fact_ids"], it["absent_term"], it["by"], it["outcome"]) for it in row["plan"]["items"]] == [
        ("i1", "direct", ["book-01:f02"], "", "orchestrator", "book-01:q01"),
        ("i2", "unanswerable", ["book-01:f01"], "camel", "orchestrator", "book-01:q02"),
        ("i3", "direct", ["book-04:f01"], "", "code", "book-04:q01"),
        ("i4", "multi_hop", ["book-01:f02", "book-04:f01"], "", "orchestrator", "book-01:h01")]
    assert [x["why"] for x in row["plan"]["invalid"]] == ["the absent term isn't one for this fact", "a pair of the wrong kind"]
    direct, unanswerable, calves, hop = row["questions"]
    feeding = rows["book-01"]["facts"]
    assert direct["review"]["original"] == "How much oil cake do pregnant animals get?" and direct["segment_id"] == "book-01"
    assert direct["evidence"] == [{"segment_id": "book-01", **{k: feeding[1][k] for k in ("quote", "start", "end", "pages")}}]
    assert unanswerable["absent_term"] == "camel" and unanswerable["answer"] == "" and unanswerable["evidence"] == []
    assert calves["repair_of"] == "answer not found in its facts' quotes" and calves["answer"] == "4 litres of milk"
    assert [ev["segment_id"] for ev in hop["evidence"]] == ["book-01", "book-04"] and hop["across_pdfs"] is False
    assert all(ok for _, ok, _ in checks.check_questions())

    before = len(calls)
    orchestrator.run_questions()  # resumes: the row is current, so no agent is called
    assert len(calls) == before and [json.loads(line).get("group") for line in questions.QUESTIONS.read_text().splitlines()] == ["book.pdf"]
    templates.save_topics({"switches": {}, "sections": {}, "groups": {"a": ["book.pdf"], "b": ["book.pdf", "nope.pdf"]}})
    assert not {n: ok for n, ok, _ in checks.check_questions()}["data/topics.json groups name known PDFs, each in one group at most"]


def test_config_switches_types_off_and_rejects_unknown_keys(tmp_path):
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
    path.write_text('[a]\ncount = 1\nsource = "library"\nprompt = "p"\nrules = { facts = [2, 3] }\n')  # library without sections
    with pytest.raises(BenchError, match="library"):
        question_types(path)
    path.write_text('species = ["yak", 3]\n')
    with pytest.raises(BenchError, match="species"):
        questions.vocab(path)
