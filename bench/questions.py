"""The questions stage: a writer agent per question type, a reviewer agent, then the regex audit.

The types are defined in question_types.toml. For each section:
1. every type whose source is facts, paragraph or absent gets a writer that reads the section's verified facts;
2. the reviewer reads every draft with its quotes and keeps it, rewords it or drops it;
3. the regex audit (bench/verify.py) keeps the questions that pass their type's rules;
4. every type whose source is question (e.g. noisy) rewrites some of the kept questions, which are audited too.
After the sections, every type whose source is library links facts across sections, in one extra row.

Output: one line per section in data/questions.jsonl {segment_id, model, version, facts_version,
questions: [... each with its evidence], rejected: [...]}, then the row "library". A row is redone when its
prompt version or the facts it was written from change (use --force to redo everything).
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput

from bench import llm
from bench.config import DATA_DIR, question_types
from bench.facts import FACTS, FACTS_VERSION, audit_line, load_rows, todo
from bench.pdf import PAGES, read_jsonl, write_jsonl
from bench.segments import load_segments
from bench.verify import verify_questions

QUESTIONS = DATA_DIR / "questions.jsonl"
QUESTIONS_VERSION = "questions-v6"  # bump when a prompt or the row format changes
LIBRARY = "library"  # the row of questions that link facts across sections

WRITER_PROMPT = """You write questions for a benchmark that checks whether a search engine finds the \
passage that answers them. You get verified facts, each with an id, a claim and the exact quote it comes \
from. The search covers a library of many documents that may give other values for other years, places, \
animals or schemes, so every question must make sense on its own: name its subject and every qualifier its \
answer depends on (e.g. Aflatoxin, the animal or class, the feed, the stage, the year, the country or state, \
the scale, the scheme). Never write "it", "this" or "the system". Never copy a whole quote into the question.

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
covers a library of many documents. Each question has an id, a type, the question, its answer and the quotes \
it was written from. Return a verdict for every question:
- keep: it reads naturally, makes sense on its own (it names its subject and every qualifier its answer \
depends on), its quotes give it one clear answer, and it fits its type.
- rewrite: rewording can fix it. Give the corrected question; its answer and quotes stay the same, so they \
must still answer it.
- drop: it can't be fixed: it is ambiguous, its answer doesn't answer it, or it isn't its type.
An unanswerable question is kept when a farmer could plausibly ask it and its quotes don't answer it.
Give a short reason for every rewrite and drop. Never add questions.

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
    verdict: Literal["keep", "rewrite", "drop"]
    question: str
    reason: str


class Reviews(BaseModel):
    reviews: list[Review]


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
    if not facts or len(facts) < t.get("rules", {}).get("facts", [1])[0]:
        return []
    memory = "\n".join(f"- [{q['type']}, {', '.join(q['fact_ids'])}] {q['question']}" for q in asked) or "(none yet)"
    prompt = (f"SECTION: {title}\n\nQUESTIONS ALREADY WRITTEN:\n{memory}\n\n"
              f"Write at most {t['count']} questions.\n\nFACTS:\n{_listing(facts, titles)}")
    out = llm.call(f"{name} writer", agents[name], prompt, label)
    return [{"type": name, **d.model_dump()} for d in out.questions[: t["count"]]]


def review(reviewer: Agent[Any, Reviews], title: str, drafts: list[dict[str, Any]], facts: dict[str, dict],
           label: str) -> tuple[list[dict], list[dict]]:
    """Apply the reviewer's verdicts. Returns (reviewed, dropped); a rewrite keeps the original question."""
    if not drafts:
        return [], []
    listing = [{"id": f"d{i}", "type": d["type"], "question": d["question"], "answer": d["answer"] or "(none)",
                "quotes": [" ".join(facts[k]["quote"].split()) for k in d["fact_ids"] if k in facts]}
               for i, d in enumerate(drafts, start=1)]
    prompt = f"SECTION: {title}\n\nQUESTIONS TO REVIEW:\n{json.dumps(listing, ensure_ascii=False, indent=1)}"
    verdicts = {r.id: r for r in llm.call("question reviewer", reviewer, prompt, label).reviews}
    reviewed, dropped = [], []
    for i, d in enumerate(drafts, start=1):
        r = verdicts.get(f"d{i}")
        if r and r.verdict == "drop":
            dropped.append({**d, "review": {"verdict": "drop", "reason": r.reason}, "reason": f"reviewer: {r.reason}"})
        elif r and r.verdict == "rewrite" and r.question.strip():
            reviewed.append({**d, "question": r.question, "review": {"verdict": "rewrite", "original": d["question"], "reason": r.reason}})
        else:
            reviewed.append({**d, "review": {"verdict": "keep"}})
    return reviewed, dropped


def finish(questions: list[dict[str, Any]], short: dict[str, dict[str, Any]], types: dict[str, dict], prefix: str) -> list[dict]:
    """Full fact ids, the evidence (each fact's section, quote, position and pages; none for an unanswerable
    question) and a question id."""
    for n, q in enumerate(questions, start=1):
        linked = [short[i] for i in q["fact_ids"]]
        q["fact_ids"] = [f["fact_id"] for f in linked]
        q["evidence"] = [] if types[q["type"]]["source"] == "absent" else [
            {"segment_id": f["fact_id"].rsplit(":", 1)[0], **{k: f[k] for k in ("quote", "start", "end", "pages")}} for f in linked]
        q["question_id"] = f"{prefix}:q{n:02d}"
    return questions


def run_questions(doc: str | None = None, force: bool = False) -> None:
    """Questions for each section from its verified facts, then the questions that link sections.
    Each question carries its evidence, so scoring never depends on a later facts run."""
    types, facts = question_types(), load_rows(FACTS)
    corpus = "\n".join(p["text"] for p in read_jsonl(PAGES))  # for the unanswerable check
    done = load_rows(QUESTIONS)
    library_row = done.pop(LIBRARY, None)
    model = llm.model()
    agents = {name: Agent(model, output_type=NativeOutput(Rewrites if t["source"] == "question" else Drafts, strict=True),
                          instructions=(REWRITER_PROMPT if t["source"] == "question" else WRITER_PROMPT) + f"{name}: {t['prompt']}")
              for name, t in types.items()}
    described = "\n".join(f"- {name}: {' '.join(t['prompt'].split())}" for name, t in types.items())
    reviewer = Agent(model, output_type=NativeOutput(Reviews, strict=True), instructions=REVIEWER_PROMPT + described)
    redone = todo(done, doc, force, QUESTIONS_VERSION, needs=facts)
    for segment in redone:
        sid, title = segment["segment_id"], segment["title"]
        short = {f"f{i}": f for i, f in enumerate(facts.get(sid, {}).get("facts", []), start=1)}  # short ids keep prompts small
        drafts: list[dict[str, Any]] = []
        for name, t in types.items():
            if t["source"] in ("facts", "paragraph", "absent"):
                drafts += write(agents, types, name, title, _sources(t, short), drafts, sid)
        reviewed, dropped = review(reviewer, title, drafts, short, sid)
        kept, rejected = verify_questions(reviewed, short, types, corpus)
        kept = finish(kept, short, types, sid)
        by_id = {f["fact_id"]: f for f in short.values()}
        for name, t in types.items():  # rewrites of the kept questions, which keep their facts, answer and evidence
            if t["source"] != "question":
                continue
            pool = {q["question_id"]: q for q in kept if q["type"] in t["from"]}
            if not pool:
                continue
            listing = json.dumps([{"id": k, "question": q["question"]} for k, q in pool.items()], ensure_ascii=False, indent=1)
            out = llm.call(f"{name} writer", agents[name], f"Rewrite at most {t['count']} of these questions.\n\nQUESTIONS:\n{listing}", sid)
            rewrites = [{**{k: v for k, v in pool[r.source].items() if k not in ("review", "question_id")}, "type": name,
                         "question": r.question, "source_question": pool[r.source]["question"], "source_id": r.source}
                        for r in out.questions[: t["count"]] if r.source in pool]
            good, bad = verify_questions(rewrites, by_id, types, corpus)
            kept += [{**q, "question_id": f"{sid}:q{len(kept) + n:02d}"} for n, q in enumerate(good, start=1)]
            rejected += bad
        done[sid] = {"segment_id": sid, "model": llm.settings().llm_model, "version": QUESTIONS_VERSION,
                     "facts_version": facts.get(sid, {}).get("version"), "questions": kept, "rejected": dropped + rejected}
        write_jsonl(QUESTIONS, list(done.values()) + ([library_row] if library_row else []))
        audit_line(sid, "questions", kept, dropped + rejected)

    library = [name for name, t in types.items() if t["source"] == "library"]
    if library and (force or redone or not library_row or library_row.get("version") != QUESTIONS_VERSION):
        library_row = link_sections(agents, reviewer, types, library, facts, corpus)
        write_jsonl(QUESTIONS, list(done.values()) + [library_row])
        audit_line(LIBRARY, "questions", library_row["questions"], library_row["rejected"])


def link_sections(agents: dict[str, Agent], reviewer: Agent[Any, Reviews], types: dict[str, dict], names: list[str],
                  facts: dict[str, dict[str, Any]], corpus: str) -> dict[str, Any]:
    """Questions that need facts from several sections, written from every verified fact of the library."""
    titles = {s["segment_id"]: s["title"] for s in load_segments()}
    short = {f"x{i}": f for i, f in enumerate((f for row in facts.values() for f in row["facts"]), start=1)}
    drafts: list[dict[str, Any]] = []
    for name in names:
        drafts += write(agents, types, name, "the whole library", short, drafts, LIBRARY, titles)
    reviewed, dropped = review(reviewer, "the whole library", drafts, short, LIBRARY)
    kept, rejected = verify_questions(reviewed, short, types, corpus)
    kept = finish(kept, short, types, LIBRARY)
    return {"segment_id": LIBRARY, "model": llm.settings().llm_model, "version": QUESTIONS_VERSION,
            "facts_version": FACTS_VERSION, "questions": kept, "rejected": dropped + rejected}
