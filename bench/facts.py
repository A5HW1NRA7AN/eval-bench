"""The facts stage: a planner, an extractor and a reviewer agent per segment, then the regex audit.

1. planner: reads a part of the segment as numbered paragraphs and picks the searchable topics in it.
2. extractor: for each topic, reads those paragraphs as numbered sentences and writes one claim with the
   range of sentences that states it (one sentence up to a whole paragraph). Code cuts the quote there.
3. reviewer: reads every fact with its quote and keeps it, fixes its claim, or flags it. A flag is recorded
   and reported as an AI verdict; it never removes a fact.
4. the regex audit (bench/verify.py) alone decides what is kept.

Output: one line per segment in data/facts.jsonl {segment_id, model, version, fingerprint, max_facts, facts,
rejected}. The fingerprint covers everything that made the row (prompts, output schemas, settings, the
segment's text), so a run redoes exactly the rows whose inputs changed, plus rows whose kept facts no longer
pass the audit. --force redoes everything.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from rich.markup import escape

from bench import llm
from bench.config import (
    DATA_DIR,
    FACT_CHARS,
    MAX_FACTS,
    MAX_FAILURES,
    MIN_FACTS,
    PART_CHARS,
    BenchError,
)
from bench.pdf import read_jsonl, write_jsonl
from bench.segments import heading_path, load_segments, page_at, parts, sentences
from bench.verify import (
    kept_fact_problem,
    span_of,
    starts_with_values,
    table_header,
    uid,
    verify_facts,
)

FACTS = DATA_DIR / "facts.jsonl"
FACTS_VERSION = "facts-v5"  # the row format; part of every row's fingerprint

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
- keep: the claim is true to its quote, makes sense on its own and is worth searching for. Leave claim empty.
- rewrite: the fact is worth keeping but its claim misstates the quote, adds something the quote doesn't say, \
or misses a qualifier its answer depends on (animal, stage, unit, year, place, scheme). Give the corrected claim.
- flag: the quote gives no useful fact (a heading, a fragment, a figure without its meaning) or the fact \
repeats another one. Leave claim empty.
Give a short reason for every rewrite and flag. Never add facts."""


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
    verdict: Literal["keep", "rewrite", "flag"]
    claim: str
    reason: str


class Reviews(BaseModel):
    reviews: list[Review]


def fact_budget(segment: dict[str, Any], row: dict[str, Any] | None = None, max_facts: int | None = None) -> int:
    """How many facts a segment gets: --max-facts if given, else the number a row was made with by --max-facts,
    else one per FACT_CHARS characters of text, between MIN_FACTS and MAX_FACTS. Shared by the run and the checks."""
    if max_facts:
        return max_facts
    if row and row.get("max_facts_from") == "option":
        return row["max_facts"]
    return min(MAX_FACTS, max(MIN_FACTS, round(len(segment["text"]) / FACT_CHARS)))


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {row["segment_id"]: row for row in read_jsonl(path)} if path.exists() else {}


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fingerprint(segment: dict[str, Any], max_facts: int) -> str:
    """Everything that made a facts row: the prompts, the output schemas, the settings and the segment's text.
    The model is recorded on the row but left out, so resuming on another model doesn't redo finished rows."""
    schemas = [m.model_json_schema() for m in (Topics, Pointers, Reviews)]
    return uid(FACTS_VERSION, PLANNER_PROMPT, EXTRACTOR_PROMPT, REVIEWER_PROMPT, schemas, PART_CHARS, max_facts,
               text_hash(segment["text"]), segment["headings"])


def todo(done: dict[str, Any], doc: str | None, force: bool, current: Callable[[dict], str],
         valid: Callable[[dict, dict], bool] = lambda row, segment: True) -> list[dict[str, Any]]:
    """Segments used for questions whose row is missing, was made from other inputs (its fingerprint differs
    from `current(segment)`), or holds items that no longer pass the audit (`valid`); all of them with --force.
    Rows of segments no longer used for questions are dropped from `done`."""
    segments = [s for s in load_segments() if s["use_for_questions"]]
    for gone in done.keys() - {s["segment_id"] for s in segments}:
        del done[gone]
    picked = [s for s in segments if not doc or doc.lower() in s["file_name"].lower()]
    if doc and not picked:
        raise BenchError(f"--doc {doc!r} matches no PDF with sections used for questions")

    def stale(segment: dict[str, Any]) -> bool:
        row = done.get(segment["segment_id"])
        return force or row is None or row.get("fingerprint") != current(segment) or not valid(row, segment)

    return [s for s in picked if stale(s)]


def run_each(segments: list[Any], work: Callable[[Any], None], stage: str,
             name: Callable[[Any], str] = lambda segment: segment["segment_id"]) -> None:
    """Run `work` for each segment (or PDF). A failed one (e.g. the endpoint is down) is reported and left to do,
    and the stage carries on; it stops after MAX_FAILURES failures in a row. Any failure ends in a BenchError,
    so `bench run` stops there; finished ones are already saved."""
    failed, in_a_row = [], 0
    for segment in segments:
        try:
            work(segment)
            in_a_row = 0
        except BenchError:
            raise
        except Exception as exc:
            failed.append(name(segment))
            in_a_row += 1
            llm.console.print(f"  [red]✗ {escape(name(segment))}: {type(exc).__name__}: {escape(str(exc))}[/] [dim](left to do)[/]")
            if llm.VERBOSE:
                llm.console.print_exception()
            if in_a_row >= MAX_FAILURES:
                raise BenchError(f"{in_a_row} sections failed in a row (is the LLM endpoint up?). Finished sections "
                                 f"are saved: re-run `bench {stage}` to resume") from exc
    if failed:
        raise BenchError(f"{len(failed)} section(s) failed and are left to do ({', '.join(failed[:3])}): "
                         f"re-run `bench {stage}`")


def audit_line(segment_id: str, what: str, kept: list[dict], rejected: list[dict]) -> None:
    """The regex audit's verdict for one segment: counts (and AI flags among the kept), then each rejection."""
    flagged = sum(item.get("review", {}).get("verdict") == "flag" for item in kept)
    llm.console.print(f"  [bold]audit[/] ▸ {escape(segment_id)} → [green]{len(kept)} {what} kept[/]"
                      + (f" [yellow]({flagged} flagged by the AI reviewer)[/]" if flagged else "")
                      + f", [{'red' if rejected else 'dim'}]{len(rejected)} rejected[/]")
    for r in rejected:
        text = r.get("question") or r.get("claim", "")
        llm.console.print(f"    [red]✗[/] {escape(r['reason'])}: [dim]{escape(text[:100])}[/]")


def _flat(text: str) -> str:
    return " ".join(text.split())


def _memory(found: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {f['claim']}" for f in found) or "(none yet)"


def paragraph_listing(units: list[dict[str, Any]], text: str) -> str:
    """The units' text as numbered paragraphs ('[p3] first sentence second sentence'), as planners read it."""
    paragraphs: dict[str, list[str]] = {}
    for u in units:
        paragraphs.setdefault(u["id"].split(".")[0], []).append(_flat(text[u["start"] : u["end"]]))
    return "\n\n".join(f"[{p}] {' '.join(s)}" for p, s in paragraphs.items())


def planner_prompt(part: dict[str, Any], units: list[dict[str, Any]], text: str, found: list[dict], wanted: int) -> str:
    context = f"CONTEXT (end of the previous part, for understanding only):\n{part['context']}\n\n" if part["context"] else ""
    return (f"SECTION: {part['heading']}\n\n{context}FACTS SO FAR:\n{_memory(found)}\n\n"
            f"Return at most {wanted} topics.\n\nTEXT:\n{paragraph_listing(units, text)}")


def extractor_prompt(heading: str, topic: Topic, units: list[dict[str, Any]], text: str, found: list[dict],
                     brief: str = "", others: list[str] | None = None) -> str:
    """The extractor's input. A planned pick (bench/orchestrator.py) has a brief, and instead of the facts found so
    far it lists the other picks in its paragraphs, so picks don't depend on each other's order."""
    listing = "\n".join(f"[{u['id']}] {_flat(text[u['start'] : u['end']])}" for u in units)
    memory = ("OTHER PICKS IN THESE PARAGRAPHS (other workers write these; don't repeat them):\n"
              + ("\n".join(f"- {o}" for o in others) or "(none)") if others is not None
              else f"FACTS SO FAR (don't repeat them):\n{_memory(found)}")
    return (f"SECTION: {heading}\nTOPIC: {topic.topic}\nKIND: {topic.kind}\n" + (f"BRIEF: {brief}\n" if brief else "")
            + f"\n{memory}\n\nSENTENCES:\n{listing}")


def review_prompt(segment: dict[str, Any], found: list[dict[str, Any]]) -> str | None:
    """The reviewer's input: each fact with its claim and quote, or None if no fact's sentence range can be cut
    (the audit rejects those, so the reviewer doesn't see them)."""
    units = {u["id"]: u for u in sentences(segment["text"])}
    listing = [{"id": f"f{i}", "claim": f["claim"], "quote": _flat(with_header(segment["text"], *span))}
               for i, f in enumerate(found, start=1) if isinstance(span := span_of(f, units), tuple)]
    return f"SECTION: {segment['title']}\n\nFACTS TO REVIEW:\n{json.dumps(listing, ensure_ascii=False, indent=1)}" if listing else None


def review(agent: Agent[Any, Reviews], segment: dict[str, Any], found: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ask the reviewer about the facts and apply its verdicts (apply_reviews)."""
    prompt = review_prompt(segment, found)
    return apply_reviews(found, llm.call("reviewer", agent, prompt, segment["segment_id"]).reviews if prompt else [])


def apply_reviews(found: list[dict[str, Any]], reviews: list[Review]) -> list[dict[str, Any]]:
    """A rewrite replaces the claim (the original is kept); a flag is recorded and the fact still goes to the
    audit; a fact the reviewer didn't answer is marked unreviewed."""
    verdicts = {r.id: r for r in reviews}
    reviewed = []
    for i, f in enumerate(found, start=1):
        r = verdicts.get(f"f{i}")
        if r is None:
            f = {**f, "review": {"verdict": "unreviewed"}}
        elif r.verdict == "rewrite" and r.claim.strip() and r.claim.strip() != f["claim"].strip():
            f = {**f, "claim": r.claim, "review": {"verdict": "rewrite", "original": f["claim"], "reason": r.reason}}
        elif r.verdict == "flag":
            f = {**f, "review": {"verdict": "flag", "reason": r.reason}}
        else:  # keep, or a rewrite that changes nothing
            f = {**f, "review": {"verdict": "keep"}}
        reviewed.append(f)
    return reviewed


def paragraph_ids(ids: list[str]) -> set[str]:
    """The planner's paragraph ids as the text numbers them ('[p3]' and ' p3' are 'p3')."""
    return {i.strip().strip("[]").strip() for i in ids}


def with_header(text: str, start: int, end: int) -> str:
    """The text from start to end, after its table's header rows when it is a table row."""
    header = table_header(text, start) if starts_with_values(text[start:end]) else None
    return f"{text[header[0] : header[1]]}\n{text[start:end]}" if header else text[start:end]


def still_valid(row: dict[str, Any], segment: dict[str, Any]) -> bool:
    """True if every kept fact of the row still passes the fact rules on the current segment text."""
    return all(kept_fact_problem(f, segment) is None for f in row.get("facts", []))


def run_facts(doc: str | None = None, max_facts: int | None = None, force: bool = False) -> None:
    """Facts for each segment, read in parts that remember the facts found so far. A segment's fact budget
    grows with its length (fact_budget). Given a whole chapter the model only takes facts from its first pages,
    so each part gets an even share of the budget (plus any shortfall of the parts before it)."""
    if max_facts is not None and max_facts < 1:
        raise BenchError("--max-facts must be at least 1")
    done = load_rows(FACTS)

    def limit(segment: dict[str, Any]) -> int:
        return fact_budget(segment, done.get(segment["segment_id"]), max_facts)

    planner, extractor = llm.agent(Topics, PLANNER_PROMPT), llm.agent(Pointers, EXTRACTOR_PROMPT)
    reviewer = llm.agent(Reviews, REVIEWER_PROMPT)

    def work(segment: dict[str, Any]) -> None:
        sid, text, units = segment["segment_id"], segment["text"], sentences(segment["text"])
        found: list[dict[str, Any]] = []
        cut, n_facts = parts(segment, PART_CHARS), limit(segment)
        for i, part in enumerate(cut):
            wanted = (i + 1) * n_facts // len(cut) - len(found)
            inside = [u for u in units if part["start"] <= u["start"] < part["end"]]
            if wanted <= 0 or not inside:
                continue
            label = f"{sid} part {i + 1}/{len(cut)}"
            plan = llm.call("planner", planner, planner_prompt(part, inside, text, found, wanted), label)
            for topic in plan.topics[:wanted]:
                chosen = [u for u in inside if u["id"].split(".")[0] in paragraph_ids(topic.paragraphs)]
                if not chosen:
                    continue  # the planner named paragraphs that aren't in this part
                heading = heading_path(segment, page_at(segment, chosen[0]["start"]))  # where the topic is, not where the part starts
                out = llm.call("extractor", extractor, extractor_prompt(heading, topic, chosen, text, found),
                               f"{label} ▸ {topic.kind} ▸ {topic.topic[:50]}")
                found += [{**f.model_dump(), "topic": topic.topic, "kind": topic.kind} for f in out.facts[:1]]
        kept, rejected = verify_facts(segment, review(reviewer, segment, found))
        done[sid] = {"segment_id": sid, "model": llm.settings().llm_model, "version": FACTS_VERSION,
                     "fingerprint": fingerprint(segment, n_facts), "max_facts": n_facts,
                     "max_facts_from": "option" if max_facts else done.get(sid, {}).get("max_facts_from", "length"),
                     "facts": kept, "rejected": rejected}
        write_jsonl(FACTS, list(done.values()))  # saved after every segment, so a stop loses little
        audit_line(sid, "facts", kept, rejected)

    rows_before = set(done)
    stale = todo(done, doc, force, lambda s: fingerprint(s, limit(s)), still_valid)
    if set(done) != rows_before:  # rows of sections that are gone: save that even if nothing else is redone
        write_jsonl(FACTS, list(done.values()))
    run_each(stale, work, "facts")
