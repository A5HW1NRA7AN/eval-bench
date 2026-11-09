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
