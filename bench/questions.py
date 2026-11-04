"""The questions stage: a writer agent per question type, a reviewer agent, then the regex audit.

The types are defined in question_types.toml (only enabled types with a count above 0 run). For each section:
1. every type whose source is facts, paragraph or absent gets a writer that reads the section's verified facts;
2. the reviewer reads every draft with its quotes and keeps it, rewords it or flags it. A flag is recorded and
   reported as an AI verdict; it never removes a question;
3. the regex audit (bench/verify.py) alone decides what is kept, by each type's rules;
4. every type whose source is question (e.g. noisy) rewrites some of the kept questions, which are audited too.
After the sections, every type whose source is library links facts across sections, in one extra row.

Output: one line per section in data/questions.jsonl {segment_id, model, version, fingerprint, questions: [...
each with its evidence and a content id (uid)], rejected: [...]}, then the row "library". The fingerprint covers
the prompts, the question types in use and the facts the row was written from, so a run redoes exactly the rows
whose inputs changed, plus rows whose kept questions no longer pass the audit (e.g. a new PDF now holds an
unanswerable question's absent term). --force redoes everything.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_ai import Agent

from bench import llm
from bench.config import DATA_DIR, BenchError, question_types
from bench.facts import FACTS, audit_line, load_rows, run_each, todo
from bench.pdf import PAGES, read_jsonl, write_jsonl
from bench.segments import load_segments
from bench.verify import question_problem, uid, verify_questions

QUESTIONS = DATA_DIR / "questions.jsonl"
QUESTIONS_VERSION = "questions-v7"  # the row format; part of every row's fingerprint
LIBRARY = "library"  # the row of questions that link facts across sections

WRITER_PROMPT = """You write questions for a benchmark that checks whether a search engine finds the \
passage that answers them. You get verified facts, each with an id, a claim and the exact quote it comes \
from. The search covers a library of many documents that may give other values for other years, places, \
animals or schemes, so every question must make sense on its own: name its subject and every qualifier its \
answer depends on (e.g. Aflatoxin, the animal or class, the feed, the stage, the year, the country or state, \
the scale, the scheme). Never write "it", "this" or "the system". Never copy a whole quote into the question. \
Write in English.

Write questions of the TYPE below only. Ask about facts that the QUESTIONS ALREADY WRITTEN don't use, or \
about other parts of them; never ask the same thing again. For each give the question, the fact_ids it uses, and answer: the \
answer copied exactly from the quotes; when it has several parts (one per fact, or the items of a list), copy \
each part and join them with "; ". Leave absent_term empty unless the TYPE asks for it.

TYPE: """

REWRITER_PROMPT = """You rewrite benchmark questions as described below. Each rewrite keeps its source \
question's meaning, so the same answer still answers it. For each give source (the id of the question you \
rewrote) and the new question.

STYLE: """

REVIEWER_PROMPT = """You review benchmark questions before a search engine is tested with them. The search \
covers a library of many documents. Each question has an id, a type, the question, its answer, the ids of the \
facts it uses, its absent_term (unanswerable questions only) and the quotes it was written from. Return a \
verdict for every question:
- keep: it reads naturally, makes sense on its own (it names its subject and every qualifier its answer \
depends on), its quotes give it one clear answer, and it fits its type below. Leave question empty.
- rewrite: rewording can fix it. Give the corrected question; its answer and quotes stay the same, so they \
must still answer it.
- flag: it can't be fixed: it is ambiguous, its answer doesn't answer it, or it isn't its type. Leave \
question empty.
Give a short reason for every rewrite and flag. Never add questions.

TYPES:
"""


class Draft(BaseModel):
    question: str
    fact_ids: list[str]
    answer: str
    absent_term: str


class Drafts(BaseModel):
    questions: list[Draft]


class Rewrite(BaseModel):
    source: str
    question: str


class Rewrites(BaseModel):
    questions: list[Rewrite]


class Review(BaseModel):
    id: str
    verdict: Literal["keep", "rewrite", "flag"]
    question: str
    reason: str


class Reviews(BaseModel):
    reviews: list[Review]


def _fact_key(fact: dict[str, Any]) -> str:
    return fact.get("uid") or uid(fact["fact_id"], fact["start"], fact["end"], fact["claim"], fact["quote"])


def split_types(types: dict[str, dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    """(section types, library types): a section row is written by the first, the library row by the second."""
    return ({n: t for n, t in types.items() if t["source"] != "library"},
            {n: t for n, t in types.items() if t["source"] == "library"})


def fingerprint(types: dict[str, dict], facts: list[dict[str, Any]], title: str) -> str:
    """Everything that made a questions row: the prompts, the output schemas, what its writers and reviewer were
    told (each type's prompt, count, source, `from` and review text), the section title and the facts it was
    written from. Audit rules are left out: kept questions are re-audited on every run instead."""
    schemas = [m.model_json_schema() for m in (Drafts, Rewrites, Reviews)]
    told = {n: {k: t[k] for k in ("prompt", "count", "source", "from", "review")} for n, t in types.items()}
    return uid(QUESTIONS_VERSION, WRITER_PROMPT, REWRITER_PROMPT, REVIEWER_PROMPT, schemas, told, title,
               [_fact_key(f) for f in facts])


def library_fingerprint(types: dict[str, dict], facts_rows: list[dict[str, Any]], titles: dict[str, str]) -> str:
    """The fingerprint of the library row: its types, every verified fact of the library and the section titles."""
    return fingerprint(types, [f for row in facts_rows for f in row["facts"]], json.dumps(titles, sort_keys=True))


def _listing(facts: dict[str, dict[str, Any]], titles: dict[str, str] | None = None) -> str:
    rows = [{"id": k, **({"section": titles[f["fact_id"].rsplit(":", 1)[0]]} if titles else {}),
             "claim": f["claim"], "quote": " ".join(f["quote"].split())} for k, f in facts.items()]
    return json.dumps(rows, ensure_ascii=False, indent=1)


def _sources(t: dict[str, Any], short: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The facts a writer of this type reads: all of them, or only those spanning several sentences."""
    return {k: f for k, f in short.items() if t["source"] != "paragraph" or f.get("size", "sentence") != "sentence"}


def write(agents: dict[str, Agent], types: dict[str, dict], name: str, title: str, facts: dict[str, dict],
          asked: list[dict[str, Any]], label: str, titles: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Drafts of one type from these facts (none if the type needs facts the section doesn't have). The
    writer sees the questions already written, so the types cover different facts instead of repeating one."""
    t = types[name]
    if not facts or len(facts) < t["rules"].get("facts", [1])[0]:
        return []
    memory = "\n".join(f"- [{q['type']}, {', '.join(q['fact_ids'])}] {q['question']}" for q in asked) or "(none yet)"
    prompt = (f"SECTION: {title}\n\nQUESTIONS ALREADY WRITTEN:\n{memory}\n\n"
              f"Write at most {t['count']} questions.\n\nFACTS:\n{_listing(facts, titles)}")
    out = llm.call(f"{name} writer", agents[name], prompt, label)
    return [{"type": name, **d.model_dump()} for d in out.questions[: t["count"]]]


def review(reviewer: Agent[Any, Reviews], title: str, drafts: list[dict[str, Any]], facts: dict[str, dict],
           label: str) -> list[dict[str, Any]]:
    """Apply the reviewer's verdicts: a rewrite keeps the original question; a flag is recorded and the draft
    still goes to the audit."""
    if not drafts:
        return []
    listing = [{"id": f"d{i}", "type": d["type"], "question": d["question"], "answer": d["answer"] or "(none)",
                "fact_ids": d["fact_ids"], **({"absent_term": d["absent_term"]} if d.get("absent_term") else {}),
                "quotes": [" ".join(facts[k]["quote"].split()) for k in d["fact_ids"] if k in facts]}
               for i, d in enumerate(drafts, start=1)]
    prompt = f"SECTION: {title}\n\nQUESTIONS TO REVIEW:\n{json.dumps(listing, ensure_ascii=False, indent=1)}"
    verdicts = {r.id: r for r in llm.call("question reviewer", reviewer, prompt, label).reviews}
    reviewed = []
    for i, d in enumerate(drafts, start=1):
        r = verdicts.get(f"d{i}")
        if r is None:
            reviewed.append({**d, "review": {"verdict": "unreviewed"}})
        elif r.verdict == "rewrite" and r.question.strip() and r.question.strip() != d["question"].strip():
            reviewed.append({**d, "question": r.question, "review": {"verdict": "rewrite", "original": d["question"], "reason": r.reason}})
        elif r.verdict == "flag":
            reviewed.append({**d, "review": {"verdict": "flag", "reason": r.reason}})
        else:  # keep, or a rewrite that changes nothing
            reviewed.append({**d, "review": {"verdict": "keep"}})
    return reviewed


def finish(questions: list[dict[str, Any]], short: dict[str, dict[str, Any]], types: dict[str, dict], prefix: str) -> list[dict]:
    """Full fact ids, the evidence (each fact's section, quote, position and pages; none for an unanswerable
    question), a question id and a content id (uid) that stays the same across runs."""
    for n, q in enumerate(questions, start=1):
        linked = [short[i] for i in q["fact_ids"]]
        q["fact_ids"] = [f["fact_id"] for f in linked]
        q["evidence"] = [] if types[q["type"]]["source"] == "absent" else [
            {"segment_id": f["fact_id"].rsplit(":", 1)[0], **{k: f[k] for k in ("quote", "start", "end", "pages")}} for f in linked]
        q["question_id"] = f"{prefix}:q{n:02d}"
        q["uid"] = question_uid(q)
    return questions


def question_uid(q: dict[str, Any]) -> str:
    return uid(q["type"], q["question"], q.get("absent_term", ""), [(e["segment_id"], e["start"], e["end"]) for e in q["evidence"]])


def run_questions(doc: str | None = None, force: bool = False) -> None:
    """Questions for each section from its verified facts, then the questions that link sections.
    Each question carries its evidence, so scoring never depends on a later facts run."""
    types, facts = question_types(), load_rows(FACTS)
    corpus = "\n".join(p["text"] for p in read_jsonl(PAGES))  # for the unanswerable check
    by_id = {f["fact_id"]: f for row in facts.values() for f in row["facts"]}
    done = load_rows(QUESTIONS)
    library_row = done.pop(LIBRARY, None)
    agents = {name: llm.agent(Rewrites, REWRITER_PROMPT + f"{name}: {t['prompt']}") if t["source"] == "question"
              else llm.agent(Drafts, WRITER_PROMPT + f"{name}: {t['prompt']}") for name, t in types.items()}
    section_types, library_types = split_types(types)

    def reviewer_for(some: dict[str, dict]) -> Agent[Any, Reviews]:
        described = "\n".join(f"- {name}: {t['review'] or ' '.join(t['prompt'].split())}" for name, t in some.items())
        return llm.agent(Reviews, REVIEWER_PROMPT + described)

    reviewer = reviewer_for(section_types)

    def current(segment: dict[str, Any]) -> str:
        return fingerprint(section_types, facts.get(segment["segment_id"], {}).get("facts", []), segment["title"])

    def valid(row: dict[str, Any], segment: dict[str, Any] | None = None) -> bool:
        """Every kept question still passes the audit against today's facts, rules and library text."""
        return all(q["type"] in types and question_problem(q, by_id, types[q["type"]]["rules"], corpus) is None
                   for q in row.get("questions", []))

    def save() -> None:
        write_jsonl(QUESTIONS, list(done.values()) + ([library_row] if library_row else []))

    def work(segment: dict[str, Any]) -> None:
        sid, title = segment["segment_id"], segment["title"]
        short = {f"f{i}": f for i, f in enumerate(facts.get(sid, {}).get("facts", []), start=1)}  # short ids keep prompts small
        drafts: list[dict[str, Any]] = []
        for name, t in types.items():
            if t["source"] in ("facts", "paragraph", "absent"):
                drafts += write(agents, types, name, title, _sources(t, short), drafts, sid)
        kept, rejected = verify_questions(review(reviewer, title, drafts, short, sid), short, types, corpus)
        kept = finish(kept, short, types, sid)
        for name, t in types.items():  # rewrites of the kept questions, which keep their facts, answer and evidence
            if t["source"] != "question":
                continue
            pool = {q["question_id"]: q for q in kept if q["type"] in t["from"]}
            if not pool:
                continue
            listing = json.dumps([{"id": k, "question": q["question"]} for k, q in pool.items()], ensure_ascii=False, indent=1)
            out = llm.call(f"{name} writer", agents[name], f"Rewrite at most {t['count']} of these questions.\n\nQUESTIONS:\n{listing}", sid)
            rewrites = [{**{k: v for k, v in pool[r.source].items() if k not in ("review", "question_id", "uid")}, "type": name,
                         "question": r.question, "source_question": pool[r.source]["question"], "source_id": r.source}
                        for r in out.questions[: t["count"]] if r.source in pool]
            good, bad = verify_questions(rewrites, by_id, types, corpus)
            kept += [{**q, "question_id": f"{sid}:q{len(kept) + n:02d}", "uid": question_uid(q)} for n, q in enumerate(good, start=1)]
            rejected += bad
        done[sid] = {"segment_id": sid, "model": llm.settings().llm_model, "version": QUESTIONS_VERSION,
                     "fingerprint": current(segment), "questions": kept, "rejected": rejected}
        save()
        audit_line(sid, "questions", kept, rejected)

    rows_before = set(done)
    redone = todo(done, doc, force, current, valid)
    if set(done) != rows_before:  # rows of sections that are gone: save that even if nothing else is redone
        save()
    run_each(redone, work, "questions")

    if not library_types:
        if library_row:  # no library type is switched on any more
            library_row = None
            save()
        return
    titles = {s["segment_id"]: s["title"] for s in load_segments()}
    library_print = library_fingerprint(library_types, list(facts.values()), titles)
    if force or redone or not library_row or library_row.get("fingerprint") != library_print or not valid(library_row):
        all_facts = [f for row in facts.values() for f in row["facts"]]
        try:
            row = link_sections(agents, reviewer_for(library_types), types, list(library_types), all_facts, titles, corpus)
        except Exception as exc:  # e.g. the endpoint is down: the section rows are saved, the old library row kept
            raise BenchError(f"the library questions failed ({type(exc).__name__}: {exc}); re-run `bench questions`") from exc
        library_row = {**row, "fingerprint": library_print}
        save()
        audit_line(LIBRARY, "questions", library_row["questions"], library_row["rejected"])


def link_sections(agents: dict[str, Agent], reviewer: Agent[Any, Reviews], types: dict[str, dict], names: list[str],
                  facts: list[dict[str, Any]], titles: dict[str, str], corpus: str) -> dict[str, Any]:
    """Questions that need facts from several sections, written from every verified fact of the library."""
    short = {f"x{i}": f for i, f in enumerate((f for f in facts if f["fact_id"].rsplit(":", 1)[0] in titles), start=1)}
    drafts: list[dict[str, Any]] = []
    for name in names:
        drafts += write(agents, types, name, "the whole library", short, drafts, LIBRARY, titles)
    kept, rejected = verify_questions(review(reviewer, "the whole library", drafts, short, LIBRARY), short, types, corpus)
    kept = finish(kept, short, types, LIBRARY)
    return {"segment_id": LIBRARY, "model": llm.settings().llm_model, "version": QUESTIONS_VERSION,
            "questions": kept, "rejected": rejected}
