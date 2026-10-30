"""The facts stage: a planner, an extractor and a reviewer agent per segment, then the regex audit.

1. planner: reads a part of the segment as numbered paragraphs and picks the searchable topics in it.
2. extractor: for each topic, reads those paragraphs as numbered sentences and writes one claim with the
   range of sentences that states it (one sentence up to a whole paragraph). Code cuts the quote there.
3. reviewer: reads every fact with its quote and keeps it, fixes its claim, or drops it.
4. the regex audit (bench/verify.py) decides what is kept.

Output: one line per segment in data/facts.jsonl {segment_id, model, version, facts: [...], rejected: [...]}.
A stage resumes where it stopped and redoes rows of an older version (--force redoes everything).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput

from bench import llm
from bench.config import DATA_DIR, MAX_FACTS, PART_CHARS
from bench.pdf import read_jsonl, write_jsonl
from bench.segments import load_segments, parts, sentences
from bench.verify import span_of, verify_facts

FACTS = DATA_DIR / "facts.jsonl"
FACTS_VERSION = "facts-v4"  # bump when a prompt or the row format changes

PLANNER_PROMPT = """You plan fact extraction for a search benchmark. The TEXT is one part of a document \
section, split into numbered paragraphs ([p3]). Pick the topics in it that a farmer or veterinarian could \
search for and that the TEXT answers with specific, checkable information: numbers, doses, timings, \
conditions, causes, procedures, recommendations, comparisons or lists.

Spread the topics over the whole TEXT, not just its start. Skip what is under FACTS SO FAR, headings, bare \
tables of numbers, and the CONTEXT. For each topic give:
- topic: a few words naming it with its qualifiers (e.g. "Aflatoxin M1 limit in milk, India").
- kind: "figure" if the answer is one value (a number, dose, limit, time or name); "explanation" if it takes \
several sentences (a procedure, a list, a comparison, a cause with its effects, a rule with its conditions or \
exceptions). Pick a mix of both kinds when the TEXT allows.
- paragraphs: the ids of the paragraphs that state it (usually one, at most three consecutive).
Return at most the number of topics asked for, or an empty list if the TEXT has nothing worth searching for."""

EXTRACTOR_PROMPT = """You write one fact for a search benchmark from the numbered sentences of a document \
([p3.s2] is paragraph 3, sentence 2). Give:
- claim: one self-contained statement of the fact about the TOPIC. Name the subject and every qualifier that \
makes it true (animal, breed, age or stage, condition, quantity and unit, and the year, place, scale or scheme \
it applies to). Never write "it", "this" or "they". Say only what the sentences say.
- first, last: the ids of the first and last sentence that state the fact. For a KIND of figure, the one \
sentence that states it (plus a neighbour only if it holds a qualifier the figure needs). For an explanation, \
every sentence that explains it, usually the whole paragraph; the claim then sums up all of them.
Return one fact, or an empty list if the sentences hold no checkable fact about the TOPIC."""

REVIEWER_PROMPT = """You review facts extracted from one document section for a search benchmark. Each fact \
has an id, a claim and the exact quote it was taken from. Return a verdict for every fact:
- keep: the claim is true to its quote, makes sense on its own and is worth searching for.
- rewrite: the fact is worth keeping but its claim misstates the quote, adds something the quote doesn't say, \
or misses a qualifier its answer depends on (animal, stage, unit, year, place, scheme). Give the corrected claim.
- drop: the quote gives no useful fact (a heading, a fragment, a figure without its meaning) or the fact \
repeats another one.
Give a short reason for every rewrite and drop. Never add facts."""


class Topic(BaseModel):
    topic: str
    kind: Literal["figure", "explanation"]
    paragraphs: list[str]


class Topics(BaseModel):
    topics: list[Topic]


class Pointer(BaseModel):
    claim: str
    first: str
    last: str


class Pointers(BaseModel):
    facts: list[Pointer]


class Review(BaseModel):
    id: str
    verdict: Literal["keep", "rewrite", "drop"]
    claim: str
    reason: str


class Reviews(BaseModel):
    reviews: list[Review]


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {row["segment_id"]: row for row in read_jsonl(path)} if path.exists() else {}


def todo(done: dict[str, Any], doc: str | None, force: bool, version: str, needs: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Segments used for questions that a stage hasn't done at its current version (all of them with --force).
    With `needs` (the facts rows), a row made from older facts is redone too. Rows of segments no longer
    used for questions are dropped from `done`."""
    segments = [s for s in load_segments() if s["use_for_questions"]]
    for gone in done.keys() - {s["segment_id"] for s in segments}:
        del done[gone]
    picked = [s for s in segments if not doc or doc.lower() in s["file_name"].lower()]

    def stale(row: dict[str, Any], segment_id: str) -> bool:
        return row.get("version") != version or (needs is not None and row.get("facts_version") != needs.get(segment_id, {}).get("version"))

    return [s for s in picked if force or stale(done.get(s["segment_id"], {}), s["segment_id"])]


def audit_line(segment_id: str, what: str, kept: list[dict], rejected: list[dict]) -> None:
    """The regex audit's verdict for one segment: counts, then each rejection and its reason."""
    llm.console.print(f"  [bold]audit[/] ▸ {segment_id} → [green]{len(kept)} {what} kept[/], "
                      f"[{'red' if rejected else 'dim'}]{len(rejected)} rejected[/]")
    for r in rejected:
        text = r.get("question") or r.get("claim", "")
        llm.console.print(f"    [red]✗[/] {r['reason']}: [dim]{text[:100]}[/]")


def _flat(text: str) -> str:
    return " ".join(text.split())


def _memory(found: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {f['claim']}" for f in found) or "(none yet)"


def planner_prompt(part: dict[str, Any], units: list[dict[str, Any]], text: str, found: list[dict], wanted: int) -> str:
    paragraphs: dict[str, list[str]] = {}
    for u in units:
        paragraphs.setdefault(u["id"].split(".")[0], []).append(_flat(text[u["start"] : u["end"]]))
    listing = "\n\n".join(f"[{p}] {' '.join(s)}" for p, s in paragraphs.items())
    context = f"CONTEXT (end of the previous part, for understanding only):\n{part['context']}\n\n" if part["context"] else ""
    return (f"SECTION: {part['heading']}\n\n{context}FACTS SO FAR:\n{_memory(found)}\n\n"
            f"Return at most {wanted} topics.\n\nTEXT:\n{listing}")


def extractor_prompt(heading: str, topic: Topic, units: list[dict[str, Any]], text: str, found: list[dict]) -> str:
    listing = "\n".join(f"[{u['id']}] {_flat(text[u['start'] : u['end']])}" for u in units)
    return (f"SECTION: {heading}\nTOPIC: {topic.topic}\nKIND: {topic.kind}\n\n"
            f"FACTS SO FAR (don't repeat them):\n{_memory(found)}\n\nSENTENCES:\n{listing}")


def review(agent: Agent[Any, Reviews], segment: dict[str, Any], found: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the reviewer's verdicts: a rewrite replaces the claim (the original is kept), a drop marks the fact.
    Facts whose sentence range can't be cut skip the review; the audit rejects them."""
    units = {u["id"]: u for u in sentences(segment["text"])}
    listing = [{"id": f"f{i}", "claim": f["claim"], "quote": _flat(segment["text"][span[0] : span[1]])}
               for i, f in enumerate(found, start=1) if isinstance(span := span_of(f, units), tuple)]
    verdicts = {}
    if listing:
        prompt = f"SECTION: {segment['title']}\n\nFACTS TO REVIEW:\n{json.dumps(listing, ensure_ascii=False, indent=1)}"
        verdicts = {r.id: r for r in llm.call("reviewer", agent, prompt, segment["segment_id"]).reviews}
    reviewed = []
    for i, f in enumerate(found, start=1):
        r = verdicts.get(f"f{i}")
        if r and r.verdict == "rewrite" and r.claim.strip():
            f = {**f, "claim": r.claim, "review": {"verdict": "rewrite", "original": f["claim"], "reason": r.reason}}
        elif r and r.verdict == "drop":
            f = {**f, "review": {"verdict": "drop", "reason": r.reason}}
        else:
            f = {**f, "review": {"verdict": "keep"}}
        reviewed.append(f)
    return reviewed


def run_facts(doc: str | None = None, max_facts: int = MAX_FACTS, force: bool = False) -> None:
    """Facts for each segment, read in parts that remember the facts found so far.
    Given a whole chapter the model only takes facts from its first pages, so each part gets an
    even share of max_facts (plus any shortfall of the parts before it)."""
    done, model = load_rows(FACTS), llm.model()
    planner = Agent(model, output_type=NativeOutput(Topics, strict=True), instructions=PLANNER_PROMPT)
    extractor = Agent(model, output_type=NativeOutput(Pointers, strict=True), instructions=EXTRACTOR_PROMPT)
    reviewer = Agent(model, output_type=NativeOutput(Reviews, strict=True), instructions=REVIEWER_PROMPT)
    for segment in todo(done, doc, force, FACTS_VERSION):
        sid, text, units = segment["segment_id"], segment["text"], sentences(segment["text"])
        found: list[dict[str, Any]] = []
        cut = parts(segment, PART_CHARS)
        for i, part in enumerate(cut):
            wanted = (i + 1) * max_facts // len(cut) - len(found)
            inside = [u for u in units if part["start"] <= u["start"] < part["end"]]
            if wanted <= 0 or not inside:
                continue
            label = f"{sid} part {i + 1}/{len(cut)}"
            plan = llm.call("planner", planner, planner_prompt(part, inside, text, found, wanted), label)
            for topic in plan.topics[:wanted]:
                chosen = [u for u in inside if u["id"].split(".")[0] in topic.paragraphs]
                if not chosen:
                    continue  # the planner named paragraphs that aren't in this part
                out = llm.call("extractor", extractor, extractor_prompt(part["heading"], topic, chosen, text, found),
                               f"{label} ▸ {topic.kind} ▸ {topic.topic[:50]}")
                found += [{**f.model_dump(), "topic": topic.topic, "kind": topic.kind} for f in out.facts[:1]]
        reviewed = review(reviewer, segment, found)
        dropped = [{**f, "reason": f"reviewer: {f['review']['reason']}"} for f in reviewed if f["review"]["verdict"] == "drop"]
        kept, rejected = verify_facts(segment, [f for f in reviewed if f["review"]["verdict"] != "drop"])
        done[sid] = {"segment_id": sid, "model": llm.settings().llm_model, "version": FACTS_VERSION,
                     "facts": kept, "rejected": dropped + rejected}
        write_jsonl(FACTS, list(done.values()))  # saved after every segment, so a stop loses little
        audit_line(sid, "facts", kept, dropped + rejected)
