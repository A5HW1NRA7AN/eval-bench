import pytest

from bench.config import BenchError
from bench.evaluate import check_evidence, gold_chunks, score
from bench.testbed import CHUNK_CHARS, chunk

SENT = "Feed the cow well and keep her shed clean and dry. "  # 51 chars


def test_chunks_cover_the_text_exactly_and_prefer_paragraph_breaks():
    text = ("Para one. " * 40 + "\n\n") * 6 + SENT * 50
    spans = chunk(text)
    assert "".join(text[s:e] for s, e in spans) == text
    assert all(e - s <= CHUNK_CHARS for s, e in spans)
    assert text[: spans[0][1]].endswith("\n\n")


def test_gold_chunks_are_the_ones_holding_each_quote():
    chunks = {"seg-01": [{"chunk_id": "seg-01:c001", "start": 0, "end": 100}, {"chunk_id": "seg-01:c002", "start": 100, "end": 200}],
              "seg-02": [{"chunk_id": "seg-02:c001", "start": 0, "end": 100}]}
    evidence = [{"segment_id": "seg-01", "start": 10, "end": 50}, {"segment_id": "seg-01", "start": 90, "end": 120},
                {"segment_id": "seg-02", "start": 10, "end": 50}]
    gold = gold_chunks({"evidence": evidence}, chunks)
    # a quote across a boundary has two gold chunks; a fact from another section has that section's chunk
    assert gold == [{"seg-01:c001"}, {"seg-01:c001", "seg-01:c002"}, {"seg-02:c001"}]


def test_stale_evidence_stops_the_evaluation():
    q = {"question_id": "seg-01:q01", "evidence": [{"segment_id": "seg-01", "quote": "keep her shed clean", "start": 22, "end": 41}]}
    check_evidence(q, {"seg-01": SENT})  # the quote is still where it was verified
    with pytest.raises(BenchError):
        check_evidence(q, {"seg-01": "Segment text changed. " + SENT})


def test_scores():
    gold = [{"a"}, {"b"}]
    hit = score(["x", "a", "y", "b"], gold)
    assert hit["first_rank"] == 2 and hit["rr"] == 0.5 and hit["hit@1"] == 0.0 and hit["hit@3"] == 1.0
    assert hit["all_facts@10"] == 1.0
    miss = score(["x", "y"], gold)
    assert miss["first_rank"] is None and miss["rr"] == 0.0 and miss["hit@10"] == 0.0 and miss["all_facts@10"] == 0.0


def test_keyword_search_and_rank_fusion():
    from bench.testbed import RRF_K, Keyword, fuse
    chunks = [{"chunk_id": "a", "text": "Give the calf colostrum within two hours of birth."},
              {"chunk_id": "b", "text": "Sheds should be cleaned every day and the floor kept dry."},
              {"chunk_id": "c", "text": "Milk production was 247.87 million tonnes in 2024-25."}]
    keyword = Keyword(chunks)
    assert [cid for cid, _ in keyword.search("When should a calf get colostrum?", 3)] == ["a", "b"]  # b shares "should"
    assert keyword.search("milk 247.87", 3)[0][0] == "c" and keyword.search("tractor", 3) == []
    fused = fuse([("a", 0.9), ("b", 0.8)], [("b", 7.0), ("c", 3.0)], k=3)
    assert [cid for cid, _ in fused] == ["b", "a", "c"] and fused[0][1] == 1 / (RRF_K + 2) + 1 / (RRF_K + 1)


def test_lenient_gold_adds_the_same_fact_stated_elsewhere():
    from bench.evaluate import lenient_gold
    chunks = [{"chunk_id": "a:c1", "text": "Give the calf colostrum within two hours of birth."},
              {"chunk_id": "b:c1", "text": "Feed colostrum within two hours of birth, as every manual says."},
              {"chunk_id": "b:c2", "text": "Calves are born in spring."}]
    q = {"answer": "within two hours of birth", "evidence": [{"quote": "Give the calf colostrum within two hours of birth."}]}
    assert lenient_gold(q, [{"a:c1"}], chunks) == [{"a:c1", "b:c1"}]  # a 5-word answer span stated elsewhere
    assert lenient_gold({**q, "answer": "two hours"}, [{"a:c1"}], chunks) == [{"a:c1"}]  # too short to count elsewhere


def test_the_split_holds_out_whole_sections_and_holds_matches_another_systems_text():
    from bench.evaluate import HELD_OUT, holds, split_of
    from bench.verify import uid
    held = [f"book-{n:02d}" for n in range(200) if int(uid(f"book-{n:02d}")[:8], 16) % HELD_OUT == 0]
    assert 0.1 < len(held) / 200 < 0.3
    dev = next(f"book-{n:02d}" for n in range(200) if f"book-{n:02d}" not in held)
    one = {"question_id": f"{held[0]}:q01", "evidence": [{"segment_id": held[0]}]}
    hop = {"question_id": f"{dev}:h01", "evidence": [{"segment_id": dev}, {"segment_id": held[0]}]}  # touches a held-out section
    assert split_of(one) == split_of(hop) == "held-out" and split_of({"question_id": f"{dev}:q01", "evidence": []}) == "dev"
    quote = "Aflatoxin M1 in milk should not exceed 0.5 micrograms per kilogram according to the FSSAI regulation of 2011"
    assert holds("... AFLATOXIN M1 in milk should not exceed 0.5 micrograms per kilogram ...", quote)  # a run of 10 words, any case
    assert holds("micrograms per kilogram, according to the FSSAI regulation of 2011.", quote[quote.index("0.5"):])  # the last 10 words
    assert holds("Gujarat 12.3 14.5 2023", "Gujarat | 12.3 | 14.5 | 2023") and not holds("Aflatoxin M1 is found in milk.", quote)


def test_bench_score_counts_a_skipped_question_as_a_miss_and_needs_the_answer(isolated):
    from bench import evaluate, questions
    from bench.pdf import write_jsonl
    quote = "Give the calf colostrum within two hours of birth so it gets antibodies early."
    q = {"uid": "u1", "question_id": "book-01:q01", "type": "direct", "answer": "within two hours of birth",
         "evidence": [{"segment_id": "book-01", "quote": quote}]}
    other = {**q, "uid": "u2", "question_id": "book-01:q02", "answer": "antibodies"}
    absent = {**q, "uid": "u3", "question_id": "book-01:q03", "evidence": [], "answer": ""}
    write_jsonl(questions.QUESTIONS, [{"group": "book.pdf", "pdfs": ["book.pdf"], "version": questions.QUESTIONS_VERSION,
                                       "questions": [q, other, absent], "rejected": []}])
    path = isolated / "system.jsonl"
    write_jsonl(path, [{"uid": "u1", "passages": ["Nothing here.", quote.upper()]}, {"uid": "nope", "passages": []}])
    text = evaluate.score_external(path)
    assert "1 of 2 answerable questions have a line" in text and "1 unanswerable" in text and "1 lines with an unknown uid" in text
    assert "| all | 2 | 0.00 | 0.50 | 0.50 | 0.50 | 0.25 |" in text  # u1 found at rank 2; u2 has no line: a miss
    write_jsonl(path, [{"uid": "u1", "passages": [quote.replace("within two hours of birth", "soon")]}])
    assert "| all | 2 | 0.00 | 0.00 |" in evaluate.score_external(path)  # the quote's words, but not the answer


def test_compare_matches_runs_by_content_id_keeps_sets_apart_and_tests_the_change(isolated):
    import json

    from bench import evaluate
    from bench.pdf import write_jsonl
    def row(u, hit, **extra):
        return {"uid": u, "question_id": f"s:{u}", "type": "direct", "question": u, "answerable": True, "hit@1": hit, "hit@10": hit, "rr": hit, **extra}
    for name, hits in (("20261101-old", (1.0, 0.0)), ("20261102-new", (1.0, 1.0))):
        (evaluate.RUNS / name).mkdir(parents=True)
        (evaluate.RUNS / name / "manifest.json").write_text(json.dumps({"test_set": name}))
        write_jsonl(evaluate.RUNS / name / "results.jsonl", [row("u1", hits[0]), row("u2", hits[1]), row(f"only-{name}", 1.0),
                                                             row("t1", 1.0, template="animal_care")])
    (evaluate.RUNS / "hand-named").mkdir()
    write_jsonl(evaluate.RUNS / "hand-named" / "results.jsonl", [])
    text = evaluate.compare("20261101-old", "20261102-new")
    assert "answerable questions in both: 3; only in 20261101-old: 1; only in 20261102-new: 1" in text
    assert "| general | all | 2 | hit@10 | 0.50 | 1.00 | 0.50 | 1.000 |" in text and "| template | all | 1 | hit@1 |" in text
    assert "## Found in the top 10 by 20261102-new only: 1\n\n- s:u2 (general direct)" in text
    assert evaluate.runs() == ["20261101-old", "20261102-new"]  # a folder without a manifest isn't a kept run
    assert evaluate.mcnemar([0.0] * 10, [1.0] * 10) < 0.01 and evaluate.mcnemar([1.0, 0.0], [0.0, 1.0]) == 1.0
