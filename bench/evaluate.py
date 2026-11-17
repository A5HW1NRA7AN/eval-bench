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
import re
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
from bench.orchestrator import generation_stats, question_stats
from bench.pdf import PAGES, read_jsonl, write_jsonl, write_text
from bench.questions import QUESTIONS, question_rows
from bench.templates import ROWS as TEMPLATE_ROWS
from bench.templates import current_questions, pdf_rows
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
from bench.verify import find, pattern, reason_group, uid

KS = (1, 3, 5, 10)
RANKED = 100  # chunks kept per question, so later metrics need no new search
METHODS = {"dense": "embedding search (bge-m3, the test bed)", "bm25": "keyword search (BM25)",
           "hybrid": "hybrid (BM25 + embeddings, reciprocal rank fusion)"}
REPORT = DATA_DIR / "report.md"
RESULTS = DATA_DIR / "results.jsonl"
RUNS = DATA_DIR / "runs"
SMALL = 30  # groups with fewer questions are marked: their scores move a lot from run to run
LENIENT_WORDS = 5  # an answer span of at least this many words, found verbatim elsewhere, also counts (lenient score)
HELD_OUT = 5  # one section in this many is held out, with all its questions: tune a search system on dev only
WINDOW = 10  # words: a passage another system returns holds a quote if it holds a run of this many of its words


def gold_chunks(question: dict[str, Any], chunks: dict[str, list[dict[str, Any]]]) -> list[set[str]]:
    """For each piece of the question's evidence, the chunks of its segment that overlap the quote."""
    return [{c["chunk_id"] for c in chunks[ev["segment_id"]] if c["start"] < ev["end"] and c["end"] > ev["start"]}
            for ev in question["evidence"]]


def lenient_gold(question: dict[str, Any], strict: list[set[str]], chunks: list[dict[str, Any]]) -> list[set[str]]:
    """Strict gold plus, per evidence piece, every chunk in the library that holds its quote verbatim or one of its
    answer spans of at least LENIENT_WORDS words: the same fact stated in another passage. A regex match, no judging."""
    spans = [p.strip() for p in question.get("answer", "").split(";") if len(p.split()) >= LENIENT_WORDS]
    gold = []
    for ev, own in zip(question["evidence"], strict):
        needles = [pattern(ev["quote"])] + [pattern(a) for a in spans if find(a, ev["quote"])]
        gold.append(own | {c["chunk_id"] for c in chunks if any(n.search(c["text"]) for n in needles)})
    return gold


def split_of(question: dict[str, Any]) -> str:
    """dev or held-out, by section: one section in HELD_OUT is held out with every question that touches it (its own
    section, or any of its evidence's), so no dev question shares a passage with a held-out one, run after run."""
    sections = {question["question_id"].rsplit(":", 1)[0]} | {ev["segment_id"] for ev in question["evidence"]}
    return "held-out" if any(int(uid(s)[:8], 16) % HELD_OUT == 0 for s in sections) else "dev"


def words_of(text: str) -> str:
    """The text's words and numbers in lower case, between single spaces: case, punctuation, line breaks and table
    pipes don't count when matching another system's text."""
    return " " + " ".join(re.findall(r"\w+(?:[.,]\d+)*", text.lower())) + " "


def holds(passage: str, quote: str) -> bool:
    """Whether a passage another system returned holds the quote: all of it, or a run of WINDOW of its words (a
    passage cut elsewhere than our chunks may hold only part of a long quote), the last run included."""
    words, text = words_of(quote).split(), words_of(passage)
    last = max(len(words) - WINDOW, 0)
    return any(f" {' '.join(words[i : i + WINDOW])} " in text for i in sorted({*range(0, last + 1, WINDOW // 2), last}))


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


def strict_lenient(answerable: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per type, hit@1, hit@10 and rr scored strictly and leniently, and how many questions lenient gold widened."""
    out = []
    for name, rows in [("all", answerable), *sorted(groupby(answerable, "type").items())]:
        loose = [r["lenient"] for r in rows if "lenient" in r] or rows
        out.append({"type": name, "n": len(rows), **{f"{m} {k}": sum(r[m] for r in src) / len(src)
                                                   for m in ("hit@1", "hit@10", "rr") for k, src in (("strict", rows), ("lenient", loose))},
                    "found only leniently@10": sum(bool(r.get("lenient", {}).get("hit@10")) and not r["hit@10"] for r in rows)})
    return out


def groupby(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get(key, "?")].append(r)
    return groups


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
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "bench", "templates", "question_types.toml", "vocab.toml"],
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
                      "by_method": by_method(answerable), "strict_lenient": strict_lenient(answerable),
                      "by_split": with_intervals(summary(answerable, "split"))},
    }


def save_run(questions: list[dict[str, Any]], question_rows: list[dict[str, Any]], results: list[dict[str, Any]],
             topical: list[dict[str, Any]], templated: list[dict[str, Any]]) -> Path:
    """Keep this run: its inputs, results and report, a manifest and stats.json, in data/runs/<run>/."""
    folder = RUNS / llm.RUN_ID
    folder.mkdir(parents=True, exist_ok=True)
    for path in (FACTS, QUESTIONS, TEMPLATE_ROWS, RESULTS, REPORT):
        if path.exists():
            shutil.copy2(path, folder / path.name)
    pages, template_rows = read_jsonl(PAGES), pdf_rows()
    manifest = {
        "run": llm.RUN_ID, "time": datetime.now().isoformat(timespec="seconds"), "code": _git(),
        "test_set": test_set_id(questions), "questions": len(questions),
        "template_test_set": test_set_id(topical) if topical else None, "template_questions": len(topical),
        "question_models": sorted({row["model"] for row in question_rows}),
        "template_models": sorted({row["model"] for row in template_rows}),
        "template_settings": sorted({(row["concurrency"], row["orchestrator_text_chars"]) for row in template_rows}),  # (LLM_CONCURRENCY, ORCHESTRATOR_TEXT_CHARS)
        "test_bed": {"collection": COLLECTION, "embed_model": settings().embed_model, "chunk_chars": CHUNK_CHARS,
                     "ranked": RANKED},
        "pdfs": sorted({(p["file_name"], p["doc_id"]) for p in pages}),
        "question_types": {name: t["count"] for name, t in question_types().items()},
    }
    write_text(folder / "manifest.json", json.dumps(manifest, indent=1, ensure_ascii=False))
    numbers = stats(read_jsonl(FACTS), question_rows, results)
    numbers["questions"]["generation"] = question_stats(question_rows)
    if template_rows:
        numbers["templates"] = {"generation": generation_stats(template_rows, questions)}
    if templated:
        numbers["templates"]["retrieval"] = {"by_intent": with_intervals(summary(templated, "intent")), "by_method": by_method(templated)}
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
    rows = question_rows()  # rows in an older format are left out: `bench questions` redoes them
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
            found["lenient"] = score([cid for cid, _ in hits], lenient_gold(q, gold, chunks))
            found["baselines"] = {"bm25": score([cid for cid, _ in words], gold),
                                  "hybrid": score([cid for cid, _ in fuse(hits, words, k=RANKED)], gold)}
        document = by_segment_file.get(q["evidence"][0]["segment_id"], "?") if q["evidence"] else "none"
        results.append({"question_id": q["question_id"], "uid": q.get("uid"), "type": q["type"], "document": document,
                        **({"template": q["template"], "intent": f"{q['template']}.{q['intent']}"} if "template" in q else {}),
                        "question": q["question"], "answerable": bool(q["evidence"]), "split": split_of(q),
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
        "question's evidence quote is in the top k. rr: 1 / rank of the first such chunk in the top 100 (MRR when averaged). "
        f"95%: Wilson interval; groups with fewer than {SMALL} questions are marked n<{SMALL}.", "",
        *_table(with_intervals(summary(answerable, "type"))), "",
        "By the document holding the evidence:", "",
        *_table(with_intervals(summary(answerable, "document"))), "",
        "## Strict and lenient", "",
        ("Strict (the headline): only the passage a question was written from counts. Lenient: a chunk elsewhere in the "
         f"library that holds the same quote, or an answer span of at least {LENIENT_WORDS} words, counts too (regex, no "
         "judging), since a fact stated twice is found either way."), "",
        *_table(strict_lenient(answerable)), "",
        "## Dev and held-out", "",
        f"One question in {HELD_OUT} is held out, fixed by its content id. Tune a search system on dev; report held-out.", "",
        *_table(with_intervals(summary(answerable, "split"))), "",
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


def run_results(name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A kept run's results and manifest by its folder name in data/runs/, or 'current' for data/results.jsonl."""
    folder = RUNS / name
    path = RESULTS if name == "current" else folder / "results.jsonl"
    if not path.exists():
        raise BenchError(f"no results for {name!r}: give 'current' or a folder of data/runs/ ({', '.join(runs()[-5:])})")
    manifest = folder / "manifest.json"
    return read_jsonl(path), (json.loads(manifest.read_text()) if name != "current" and manifest.exists() else {})


def runs() -> list[str]:
    """The runs `bench evaluate` kept (each has a manifest), oldest first: a run's name starts with its time."""
    return sorted(p.parent.name for p in RUNS.glob("*/manifest.json") if (p.parent / "results.jsonl").exists())


def mcnemar(a: list[float], b: list[float]) -> float:
    """Exact two-sided McNemar p-value for paired hits (0 or 1): how likely a change this large is by chance."""
    gained, lost = sum(y > x for x, y in zip(a, b)), sum(x > y for x, y in zip(a, b))
    n = gained + lost
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(gained, lost) + 1)) / 2**n) if n else 1.0


def set_of(result: dict[str, Any]) -> str:
    return "template" if "template" in result else "general"


def compare(a: str, b: str) -> str:
    """Markdown: two runs on the questions both scored (matched by content id), per set (general first, the headline)
    and type, with McNemar's p for each hit rate, and the questions one found in the top 10 and the other didn't."""
    (ra, ma), (rb, mb) = run_results(a), run_results(b)
    ka, kb = ({r["uid"]: r for r in rs if r.get("uid") and r.get("answerable", True)} for rs in (ra, rb))
    both = sorted(ka.keys() & kb.keys())
    lines = [f"# Compare {a} with {b}", "",
             *[f"- {n}: test set `{m.get('test_set', '?')}`, code `{m.get('code', '?')}`, questions by {', '.join(m.get('question_models', [])) or '?'}"
               for n, m in ((a, ma), (b, mb))],
             f"- answerable questions in both: {len(both)}; only in {a}: {len(ka.keys() - kb.keys())}; only in {b}: {len(kb.keys() - ka.keys())}", ""]
    if not both:
        return "\n".join(lines + ["No question in common: the runs scored different test sets."])
    rows = []
    for kind in ("general", "template"):
        mine = [u for u in both if set_of(ka[u]) == kind]
        for name, ids in [] if not mine else [("all", mine), *sorted(groupby([{"uid": u, "type": ka[u]["type"]} for u in mine], "type").items())]:
            ids = ids if name == "all" else [x["uid"] for x in ids]
            for m in ("hit@1", "hit@10", "rr"):
                x, y = ([k[u][m] for u in ids] for k in (ka, kb))
                rows.append({"set": kind, "type": name, "n": len(ids), "metric": m, a: sum(x) / len(ids), b: sum(y) / len(ids),
                             "change": (sum(y) - sum(x)) / len(ids), "p (McNemar)": f"{mcnemar(x, y):.3f}" if m != "rr" else ""})
    lines += _table(rows) + ["", "p (McNemar): the chance of a change this large in paired hits if the two runs were equally good.", ""]
    for label, gained in ((f"Found in the top 10 by {b} only", True), (f"Found in the top 10 by {a} only", False)):
        flips = [u for u in both if bool(kb[u]["hit@10"]) == gained and bool(ka[u]["hit@10"]) != gained]
        lines += [f"## {label}: {len(flips)}", ""] + [f"- {ka[u]['question_id']} ({set_of(ka[u])} {ka[u]['type']}): {ka[u]['question']}"
                                                      for u in flips[:40]] + [""]
    return "\n".join(lines)


def score_external(path: Path) -> str:
    """Markdown: another search system scored by the text it returned. Each line of the JSONL file is
    {"uid": ..., "passages": [text, ...]} best first. A passage counts when it holds an evidence quote (holds) and, if
    the question has an answer, one of its answer spans. Every current answerable question is scored: one with no
    line is a miss. General questions are the headline; template questions are reported apart."""
    current = [(q, "general") for row in question_rows() for q in row["questions"]] + [(q, "template") for q in current_questions()]
    lines = {line.get("uid"): line for line in read_jsonl(path)}
    scored, answered = [], 0
    for q, kind in current:
        if not q["evidence"]:
            continue
        line = lines.get(q["uid"])
        answered += line is not None
        pieces = [words_of(p).strip() for p in (q.get("answer") or "").split(";") if words_of(p).strip()]
        hits = {n for n, text in enumerate((line or {}).get("passages", []), start=1)
                if any(holds(text, ev["quote"]) for ev in q["evidence"]) and (not pieces or any(f" {p} " in words_of(text) for p in pieces))}
        first = min(hits, default=None)
        scored.append({"set": kind, "type": q["type"], "split": split_of(q), "rr": 1 / first if first else 0.0,
                       **{f"hit@{k}": float(first is not None and first <= k) for k in KS}})
    if not answered:
        raise BenchError(f"{path.name}: no line matches a current answerable question by uid (see data/testset.json)")
    unknown = len(lines.keys() - {q["uid"] for q, _ in current})
    general, template = ([r for r in scored if r["set"] == kind] for kind in ("general", "template"))
    out = [f"# {path.name} scored by its returned text", "",
           (f"{answered} of {len(scored)} answerable questions have a line; a question with no line counts as a miss. "
            f"{sum(not q['evidence'] for q, _ in current)} unanswerable questions aren't scored; {unknown} lines with an unknown uid "
            f"were skipped. A passage counts when it holds the evidence quote (or a run of {WINDOW} of its words) and one of the "
            "answer's spans, word for word."), ""]
    for label, rows in (("General questions (the headline)", general), ("Template questions", template)):
        if rows:
            out += [f"## {label}", "", *_table(with_intervals(summary(rows, "type"))), "", *_table(with_intervals(summary(rows, "split"))), ""]
    return "\n".join(out)
