"""Search the test bed with every verified question and score where its evidence lands.

Gold chunks are the chunks that overlap a question's verified quotes (same segment offsets), so no
fuzzy matching or LLM judging is needed. Questions are never dropped because of these results.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from bench.agents import QUESTIONS, QUESTIONS_VERSION
from bench.config import DATA_DIR, settings
from bench.pdf import read_jsonl, write_jsonl
from bench.testbed import COLLECTION, embed, load_chunks, qdrant, search
from bench.verify import find

KS = (1, 3, 5, 10)
REPORT = DATA_DIR / "report.md"


def gold_chunks(question: dict[str, Any], chunks: list[dict[str, Any]]) -> list[set[str]]:
    """For each piece of the question's evidence, the chunks of its segment that overlap the quote."""
    return [{c["chunk_id"] for c in chunks if c["start"] < ev["end"] and c["end"] > ev["start"]} for ev in question["evidence"]]


def check_evidence(question: dict[str, Any], text: str) -> None:
    """Stop if an evidence quote is no longer at its position in the indexed text: an earlier stage was re-run."""
    for ev in question["evidence"]:
        if find(ev["quote"], text[ev["start"] : ev["end"]]) is None:
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


def _table(rows: dict[str, list[dict]], metrics: list[str]) -> list[str]:
    lines = ["| type | n | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 2) + "|"]
    for name, results in rows.items():
        means = [sum(r[m] for r in results) / len(results) for m in metrics]
        lines.append(f"| {name} | {len(results)} | " + " | ".join(f"{v:.2f}" for v in means) + " |")
    return lines


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
    for q in questions:  # chunks cover their segment exactly, so joined they are the indexed segment text
        segment_chunks = chunks_by_segment[q["question_id"].rsplit(":", 1)[0]]
        check_evidence(q, "".join(c["text"] for c in sorted(segment_chunks, key=lambda c: c["start"])))

    client, vectors = qdrant(), embed([q["question"] for q in questions])
    results = []
    for q, vector in zip(questions, vectors):
        hits = search(client, vector, max(KS))
        ranked = [cid for cid, _ in hits]
        gold = gold_chunks(q, chunks_by_segment[q["question_id"].rsplit(":", 1)[0]])
        results.append({"question_id": q["question_id"], "type": q["type"], "question": q["question"],
                        "expected": [ev["quote"] for ev in q["evidence"]], "top": hits[:5], **score(ranked, gold)})
    write_jsonl(DATA_DIR / "results.jsonl", results)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)
    metrics = [f"hit@{n}" for n in KS] + ["rr"]
    models = ", ".join(f"`{m}`" for m in sorted({row["model"] for row in rows}))
    lines = [
        "# Search benchmark report", "",
        f"Test bed: local Qdrant `{COLLECTION}`, embeddings `{settings().embed_model}`; questions written by {models} "
        "and verified by the regex audit. Hit@k: a chunk overlapping the question's evidence quote is in the top k. "
        "rr: 1 / rank of the first such chunk.", "",
        *_table({"all": results, **dict(sorted(by_type.items()))}, metrics), "",
    ]
    multi = by_type.get("multi_fact", [])
    if multi:
        lines += [f"Multi-fact questions with every fact in the top 10: {sum(r['all_facts@10'] for r in multi):.0f} of {len(multi)}.", ""]
    misses = [r for r in results if r["first_rank"] is None or r["first_rank"] > 10]
    lines += [f"## Misses: {len(misses)} of {len(results)} questions had no answer chunk in the top 10", ""]
    for r in misses:
        lines += [f"### {r['question_id']} ({r['type']})", f"- Question: {r['question']}"]
        lines += [f"- Expected: \"{quote}\"" for quote in r["expected"]]
        lines += [f"  {n}. {score_:.3f} {by_id[cid]['file_name']} p{by_id[cid]['pages'][0]}: {' '.join(by_id[cid]['text'].split())[:160]}…"
                  for n, (cid, score_) in enumerate(r["top"][:3], start=1)]
        lines.append("")
    REPORT.write_text("\n".join(lines))
    print(f"scored {len(results)} questions -> {REPORT.relative_to(DATA_DIR.parent)}")
