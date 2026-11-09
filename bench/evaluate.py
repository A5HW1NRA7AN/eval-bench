"""Search the test bed with every verified question and score where its evidence lands.

Gold chunks are the chunks that overlap a question's verified quotes (same segment offsets), so no
fuzzy matching or LLM judging is needed. Unanswerable questions have no evidence; they are reported apart,
by how confident the search was. Questions are never dropped because of these results.

Each run is kept in data/runs/<run>/: the facts, questions and results it scored (with every question's top
RANKED chunks), the report, a manifest of what was run and stats.json (generation and retrieval numbers).
"""

from __future__ import annotations

import json
import math
import shutil
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from bench import llm
from bench.config import DATA_DIR, ROOT, BenchError, question_types, settings
from bench.facts import FACTS
from bench.pdf import PAGES, read_jsonl, write_jsonl, write_text
from bench.questions import QUESTIONS, QUESTIONS_VERSION
from bench.templates import ROWS as TEMPLATE_ROWS
from bench.templates import current_questions
from bench.testbed import (
    CHUNK_CHARS,
    COLLECTION,
    Keyword,
    embed,
    fuse,
    load_chunks,
    qdrant,
    search,
)
from bench.verify import find, reason_group, uid

KS = (1, 3, 5, 10)
RANKED = 100  # chunks kept per question, so later metrics need no new search
METHODS = {"dense": "embedding search (bge-m3, the test bed)", "bm25": "keyword search (BM25)",
           "hybrid": "hybrid (BM25 + embeddings, reciprocal rank fusion)"}
REPORT = DATA_DIR / "report.md"
RESULTS = DATA_DIR / "results.jsonl"
RUNS = DATA_DIR / "runs"
SMALL = 30  # groups with fewer questions are marked: their scores move a lot from run to run


def gold_chunks(question: dict[str, Any], chunks: dict[str, list[dict[str, Any]]]) -> list[set[str]]:
    """For each piece of the question's evidence, the chunks of its segment that overlap the quote."""
    return [{c["chunk_id"] for c in chunks[ev["segment_id"]] if c["start"] < ev["end"] and c["end"] > ev["start"]}
            for ev in question["evidence"]]


def check_evidence(question: dict[str, Any], texts: dict[str, str]) -> None:
    """Stop if an evidence quote is no longer at its position in the indexed text: an earlier stage was re-run."""
    for ev in question["evidence"]:
        if find(ev["quote"], texts.get(ev["segment_id"], "")[ev["start"] : ev["end"]]) is None:
            raise BenchError(f"{question['question_id']}: its evidence no longer matches the indexed text; "
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


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval of a proportion (Wilson score interval)."""
    if not n:
        return 0.0, 0.0
    p, d = successes / n, 1 + z * z / n
    centre, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def with_intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summary rows with 95% intervals on hit@1 and hit@10, and a mark on groups smaller than SMALL."""
    out = []
    for row in rows:
        low1, high1 = wilson(row["hit@1"] * row["n"], row["n"])
        low10, high10 = wilson(row["hit@10"] * row["n"], row["n"])
        out.append({**row, "hit@1 95%": f"{low1:.2f}-{high1:.2f}", "hit@10 95%": f"{low10:.2f}-{high10:.2f}",
                    "note": f"n<{SMALL}" if row["n"] < SMALL else ""})
    return out


def test_set_id(questions: list[dict[str, Any]]) -> str:
    """One id for the exact set of questions: it changes when any question is added, removed or changed."""
    return uid(sorted(q.get("uid", "") for q in questions))


def by_method(answerable: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One summary row per search method over the same questions: the test bed, then the baselines."""
    rows = []
    for method, name in METHODS.items():
        scored = answerable if method == "dense" else [r["baselines"][method] for r in answerable]
        rows.append({"method": name, "n": len(scored), **{m: sum(r[m] for r in scored) / len(scored) for m in METRICS}})
    return with_intervals(rows)


def _git() -> str:
    """The commit the code was at, marked when files were changed since (read-only git)."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "bench", "templates", "question_types.toml"],
                               cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head.stdout.strip() + ("+changes" if dirty.stdout.strip() else "")


def stats(fact_rows: list[dict[str, Any]], question_rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    """The run's numbers: what the audit kept and rejected (by rule), the AI flags, and the scores with intervals."""
    def counts(rows: list[dict[str, Any]], field: str, by: str | None) -> dict[str, Any]:
        items = [x for r in rows for x in r[field]]
        if by is None:
            return dict(Counter(reason_group(x["reason"]) for x in items).most_common()) if field == "rejected" else {"n": len(items)}
        groups: dict[str, list[dict]] = defaultdict(list)
        for x in items:
            groups[x.get(by, "?")].append(x)
        return {k: dict(Counter(reason_group(x["reason"]) for x in v).most_common()) if field == "rejected" else len(v)
                for k, v in sorted(groups.items())}

    flagged = lambda rows, field: sum(x.get("review", {}).get("verdict") == "flag" for r in rows for x in r[field])
    answerable = [r for r in results if r["answerable"]]
    return {
        "facts": {"sections": len(fact_rows), "kept": counts(fact_rows, "facts", None)["n"],
                  "rejected": counts(fact_rows, "rejected", None), "ai_flags": flagged(fact_rows, "facts")},
        "questions": {"kept": counts(question_rows, "questions", "type"), "rejected": counts(question_rows, "rejected", "type"),
                      "ai_flags": flagged(question_rows, "questions")},
        "retrieval": {"by_type": with_intervals(summary(answerable, "type")),
                      "by_document": with_intervals(summary(answerable, "document")),
                      "by_method": by_method(answerable)},
    }


def save_run(questions: list[dict[str, Any]], question_rows: list[dict[str, Any]], results: list[dict[str, Any]],
             topical: list[dict[str, Any]], templated: list[dict[str, Any]]) -> Path:
    """Keep this run: its inputs, results and report, a manifest and stats.json, in data/runs/<run>/."""
    folder = RUNS / llm.RUN_ID
    folder.mkdir(parents=True, exist_ok=True)
    for path in (FACTS, QUESTIONS, TEMPLATE_ROWS, RESULTS, REPORT):
        if path.exists():
            shutil.copy2(path, folder / path.name)
    pages = read_jsonl(PAGES)
    manifest = {
        "run": llm.RUN_ID, "time": datetime.now().isoformat(timespec="seconds"), "code": _git(),
        "test_set": test_set_id(questions), "questions": len(questions),
        "template_test_set": test_set_id(topical) if topical else None, "template_questions": len(topical),
        "question_models": sorted({row["model"] for row in question_rows}),
        "test_bed": {"collection": COLLECTION, "embed_model": settings().embed_model, "chunk_chars": CHUNK_CHARS,
                     "ranked": RANKED},
        "pdfs": sorted({(p["file_name"], p["doc_id"]) for p in pages}),
        "question_types": {name: t["count"] for name, t in question_types().items()},
    }
    write_text(folder / "manifest.json", json.dumps(manifest, indent=1, ensure_ascii=False))
    numbers = stats(read_jsonl(FACTS), question_rows, results)
    if topical:
        template_rows = read_jsonl(TEMPLATE_ROWS)
        numbers["templates"] = {"facts": {"kept": sum(len(r["facts"]) for r in template_rows),
                                          "rejected": dict(Counter(reason_group(x["reason"]) for r in template_rows for x in r["rejected_facts"]))},
                                "questions": {"kept": len(topical),
                                              "rejected": dict(Counter(reason_group(x["reason"]) for r in template_rows for x in r["rejected"]))},
                                "retrieval": {"by_intent": with_intervals(summary(templated, "intent")),
                                              "by_method": by_method(templated)}}
    write_text(folder / "stats.json", json.dumps(numbers, indent=1, ensure_ascii=False))
    return folder


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
        raise BenchError(f"questions.jsonl has rows in an older format: run `bench questions` ({QUESTIONS_VERSION})")
    general = [q for row in rows for q in row["questions"]]
    topical = current_questions()  # template questions, scored and reported apart from the general ones
    questions = general + topical
    chunks = load_chunks()
    by_id = {c["chunk_id"]: c for c in chunks}
    chunks_by_segment: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        chunks_by_segment[c["segment_id"]].append(c)
    by_segment_file = {c["segment_id"]: c["file_name"] for c in chunks}
    if not questions:
        raise BenchError("no verified questions: run `bench facts` and `bench questions` first")
    # chunks cover their segment exactly, so joined they are the indexed segment text
    texts = {sid: "".join(c["text"] for c in sorted(cs, key=lambda c: c["start"])) for sid, cs in chunks_by_segment.items()}
    for q in questions:
        check_evidence(q, texts)

    client, vectors, keyword = qdrant(), embed([q["question"] for q in questions]), Keyword(chunks)
    results = []
    for q, vector in zip(questions, vectors):
        hits = search(client, vector, RANKED)
        found = {"ranked": hits, "top_score": hits[0][1] if hits else 0.0}
        if q["evidence"]:
            gold = gold_chunks(q, chunks_by_segment)
            words = keyword.search(q["question"], RANKED)
            found |= score([cid for cid, _ in hits], gold)
            found["baselines"] = {"bm25": score([cid for cid, _ in words], gold),
                                  "hybrid": score([cid for cid, _ in fuse(hits, words, k=RANKED)], gold)}
        document = by_segment_file.get(q["evidence"][0]["segment_id"], "?") if q["evidence"] else "none"
        results.append({"question_id": q["question_id"], "uid": q.get("uid"), "type": q["type"], "document": document,
                        **({"template": q["template"], "intent": f"{q['template']}.{q['intent']}"} if "template" in q else {}),
                        "question": q["question"], "answerable": bool(q["evidence"]),
                        "expected": [ev["quote"] for ev in q["evidence"]], **found})
    write_jsonl(RESULTS, results)
    templated = [r for r in results if "template" in r and r["answerable"]]
    results = [r for r in results if "template" not in r]  # from here on: the general questions
    questions = general
    answerable = [r for r in results if r["answerable"]]

    models = ", ".join(f"`{m}`" for m in sorted({row["model"] for row in rows}))
    lines = [
        "# Search benchmark report", "",
        f"Test bed: local Qdrant `{COLLECTION}`, embeddings `{settings().embed_model}`; questions written by {models} "
        f"and verified by the regex audit; test set `{test_set_id(questions)}`. Hit@k: a chunk overlapping the "
        "question's evidence quote is in the top k. rr: 1 / rank of the first such chunk (MRR@10 when averaged). "
        f"95%: Wilson interval; groups with fewer than {SMALL} questions are marked n<{SMALL}.", "",
        *_table(with_intervals(summary(answerable, "type"))), "",
        "By the document holding the evidence:", "",
        *_table(with_intervals(summary(answerable, "document"))), "",
        "## Baselines", "",
        "The same questions searched three ways over the same chunks: the embedding search above, keyword search "
        "(BM25) and a hybrid of the two (reciprocal rank fusion). Gold chunks and scoring are the same for all three.", "",
        *_table(by_method(answerable)), "",
    ]
    if empty := sorted(set(question_types()) - {q["type"] for q in questions}):
        lines += [f"Switched-on types with no kept question: {', '.join(empty)}.", ""]
    flagged = [r for r, q in zip(results, questions) if q.get("review", {}).get("verdict") == "flag" and r["answerable"]]
    if flagged:
        lines += [f"{len(flagged)} answerable questions carry an AI reviewer flag (AI verdict; the regex audit kept them). "
                  f"hit@10 for them: {sum(r['hit@10'] for r in flagged) / len(flagged):.2f}; for the rest: "
                  f"{sum(r['hit@10'] for r in answerable if r not in flagged) / max(len(answerable) - len(flagged), 1):.2f}.", ""]
    multi = [r for r in answerable if len(r["expected"]) > 1]
    if multi:
        lines += [f"Questions with several facts that found every fact in the top 10: {sum(r['all_facts@10'] for r in multi):.0f} of {len(multi)}.", ""]
    lines += unanswerable_lines(answerable, [r for r in results if not r["answerable"]])
    if templated:
        lines += ["## Topic templates", "",
                  f"{len(templated)} questions aimed at the intents of each section's templates (test set "
                  f"`{test_set_id(topical)}`), written and audited apart from the general questions above. Per intent:", "",
                  *_table(with_intervals(summary(templated, "intent"))), "",
                  "The template questions searched three ways:", "", *_table(by_method(templated)), ""]
    misses = [r for r in answerable if r["first_rank"] is None or r["first_rank"] > 10]
    lines += [f"## Misses: {len(misses)} of {len(answerable)} questions had no answer chunk in the top 10", ""]
    for r in misses:
        lines += [f"### {r['question_id']} ({r['type']})", f"- Question: {r['question']}"]
        lines += [f"- Expected: \"{quote}\"" for quote in r["expected"]]
        lines += [f"  {n}. {score_:.3f} {by_id[cid]['file_name']} p{by_id[cid]['pages'][0]}: {' '.join(by_id[cid]['text'].split())[:160]}…"
                  for n, (cid, score_) in enumerate(r["ranked"][:3], start=1)]
        lines.append("")
    write_text(REPORT, "\n".join(lines))
    folder = save_run(questions, rows, results, topical, templated)
    print(f"scored {len(results)} general and {len(topical)} template questions -> {REPORT}; "
          f"run kept in {folder.relative_to(DATA_DIR.parent)}")
