"""Search the test bed with every verified question and score where its evidence lands.

Gold chunks are the chunks that overlap a question's verified quotes (same segment offsets), so no
fuzzy matching or LLM judging is needed. Unanswerable questions have no evidence; they are reported apart,
by how confident the search was. Questions are never dropped because of these results.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from bench.config import DATA_DIR, settings
from bench.pdf import read_jsonl, write_jsonl
from bench.questions import QUESTIONS, QUESTIONS_VERSION
from bench.testbed import COLLECTION, embed, load_chunks, qdrant, search
from bench.verify import find

KS = (1, 3, 5, 10)
REPORT = DATA_DIR / "report.md"


def gold_chunks(question: dict[str, Any], chunks: dict[str, list[dict[str, Any]]]) -> list[set[str]]:
    """For each piece of the question's evidence, the chunks of its segment that overlap the quote."""
    return [{c["chunk_id"] for c in chunks[ev["segment_id"]] if c["start"] < ev["end"] and c["end"] > ev["start"]}
            for ev in question["evidence"]]


def check_evidence(question: dict[str, Any], texts: dict[str, str]) -> None:
    """Stop if an evidence quote is no longer at its position in the indexed text: an earlier stage was re-run."""
    for ev in question["evidence"]:
        if find(ev["quote"], texts.get(ev["segment_id"], "")[ev["start"] : ev["end"]]) is None:
            raise SystemExit(f"{question['question_id']}: its evidence no longer matches the indexed text; "
                             "re-run the stages after the one you changed (segment, facts, questions, index)")


def score(ranked: list[str], gold: list[set[str]]) -> dict[str, Any]:
    """First gold rank, hit@k and reciprocal rank; for multi-fact questions also whether every fact was found."""
    any_gold = set().union(*gold)
    first = next((rank for rank, cid in enumerate(ranked, start=1) if cid in any_gold), None)
    result: dict[str, Any] = {"first_rank": first, "rr": 1 / first if first else 0.0}
    result.update({f"hit@{k}": float(first is not None and first <= k) for k in KS})
    result["all_facts@10"] = float(all(set(ranked[:10]) & g for g in gold))
    return result


METRICS = [f"hit@{n}" for n in KS] + ["rr"]


def summary(results: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Mean of each metric over all results, then per value of `key` (e.g. type): one row each."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[r[key]].append(r)
    return [{key: name, "n": len(rows), **{m: sum(r[m] for r in rows) / len(rows) for m in METRICS}}
            for name, rows in [("all", results), *sorted(groups.items())]]


def _table(rows: list[dict[str, Any]]) -> list[str]:
    head = list(rows[0])
    lines = ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    return lines + ["| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row.values()) + " |" for row in rows]


def unanswerable_lines(answerable: list[dict[str, Any]], unanswerable: list[dict[str, Any]]) -> list[str]:
    """How sure the search was of its top result for questions the library can't answer, against the answerable
    ones: a search that is as sure about both can't tell a farmer when it has no answer."""
    if not unanswerable or not answerable:
        return []
    median = statistics.median(r["top_score"] for r in answerable)
    confident = sum(r["top_score"] >= median for r in unanswerable)
    return ["## Unanswerable questions", "",
            f"{len(unanswerable)} questions the library can't answer. Median top-1 score: "
            f"{statistics.median(r['top_score'] for r in unanswerable):.3f}, against {median:.3f} for answerable questions. "
            f"{confident} of {len(unanswerable)} ({confident / len(unanswerable):.0%}) score at least that median, "
            "where the search looks as sure of having no answer as of a real one.", ""]


def run_evaluate() -> None:
    rows = read_jsonl(QUESTIONS)
    if any(row["version"] != QUESTIONS_VERSION for row in rows):
        raise SystemExit(f"questions.jsonl has rows from an older version: run `bench questions` ({QUESTIONS_VERSION})")
    questions = [q for row in rows for q in row["questions"]]
    chunks = load_chunks()
    by_id = {c["chunk_id"]: c for c in chunks}
    chunks_by_segment: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        chunks_by_segment[c["segment_id"]].append(c)
    if not questions:
        raise SystemExit("no verified questions: run `bench facts` and `bench questions` first")
    # chunks cover their segment exactly, so joined they are the indexed segment text
    texts = {sid: "".join(c["text"] for c in sorted(cs, key=lambda c: c["start"])) for sid, cs in chunks_by_segment.items()}
    for q in questions:
        check_evidence(q, texts)

    client, vectors = qdrant(), embed([q["question"] for q in questions])
    results = []
    for q, vector in zip(questions, vectors):
        hits = search(client, vector, max(KS))
        found = {"top": hits[:5], "top_score": hits[0][1] if hits else 0.0}
        if q["evidence"]:
            found |= score([cid for cid, _ in hits], gold_chunks(q, chunks_by_segment))
        results.append({"question_id": q["question_id"], "type": q["type"], "question": q["question"],
                        "answerable": bool(q["evidence"]), "expected": [ev["quote"] for ev in q["evidence"]], **found})
    write_jsonl(DATA_DIR / "results.jsonl", results)
    answerable = [r for r in results if r["answerable"]]

    models = ", ".join(f"`{m}`" for m in sorted({row["model"] for row in rows}))
    lines = [
        "# Search benchmark report", "",
        f"Test bed: local Qdrant `{COLLECTION}`, embeddings `{settings().embed_model}`; questions written by {models} "
        "and verified by the regex audit. Hit@k: a chunk overlapping the question's evidence quote is in the top k. "
        "rr: 1 / rank of the first such chunk.", "",
        *_table(summary(answerable, "type")), "",
    ]
    multi = [r for r in answerable if len(r["expected"]) > 1]
    if multi:
        lines += [f"Questions with several facts that found every fact in the top 10: {sum(r['all_facts@10'] for r in multi):.0f} of {len(multi)}.", ""]
    lines += unanswerable_lines(answerable, [r for r in results if not r["answerable"]])
    misses = [r for r in answerable if r["first_rank"] is None or r["first_rank"] > 10]
    lines += [f"## Misses: {len(misses)} of {len(answerable)} questions had no answer chunk in the top 10", ""]
    for r in misses:
        lines += [f"### {r['question_id']} ({r['type']})", f"- Question: {r['question']}"]
        lines += [f"- Expected: \"{quote}\"" for quote in r["expected"]]
        lines += [f"  {n}. {score_:.3f} {by_id[cid]['file_name']} p{by_id[cid]['pages'][0]}: {' '.join(by_id[cid]['text'].split())[:160]}…"
                  for n, (cid, score_) in enumerate(r["top"][:3], start=1)]
        lines.append("")
    REPORT.write_text("\n".join(lines))
    print(f"scored {len(results)} questions -> {REPORT.relative_to(DATA_DIR.parent)}")
