"""The fact and question agents, and the two stages that run them one segment at a time.

Outputs hold one line per segment. A stage resumes where it stopped and redoes rows written by an older
version of its prompt or format (use --force to redo everything):
  data/facts.jsonl      {segment_id, model, version, facts: [...], rejected: [...]}
  data/questions.jsonl  {segment_id, model, version, questions: [... each with its evidence], rejected: [...]}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput

from bench import llm
from bench.config import DATA_DIR, MAX_FACTS, MAX_QUESTIONS, PART_CHARS
from bench.pdf import read_jsonl, write_jsonl
from bench.segments import load_segments, parts
from bench.verify import verify_facts, verify_questions

FACTS = DATA_DIR / "facts.jsonl"
QUESTIONS = DATA_DIR / "questions.jsonl"
FACTS_VERSION = "facts-v2"
QUESTIONS_VERSION = "questions-v4"  # bump when a prompt or the row format changes

FACT_PROMPT = """You extract facts from a document for a search benchmark. Each fact must be something \
a farmer or veterinarian could search for, and the search must be able to find it in the TEXT.

For each fact give:
- claim: one self-contained statement. Name the subject and every qualifier that makes it true \
(animal, breed, age or stage, condition, quantity and unit). Never write "it", "this" or "they".
- quote: one or two consecutive sentences copied exactly from the TEXT (same words, spelling and \
punctuation), at most 300 characters, that state the fact.

Prefer specific, checkable facts: numbers, doses, timings, conditions, causes and recommendations.
Quote only the TEXT, never the CONTEXT. Skip anything already listed under FACTS SO FAR.
Return at most the number of facts asked for, or an empty list if the TEXT has no new facts."""

QUESTION_PROMPT = """You write questions for a benchmark that checks whether a search engine finds the \
passage that answers them. You get verified facts from one document section, each with an id, a claim \
and the exact quote it comes from. The search covers a library of many documents, so every question must \
make sense on its own: name its subject and qualifiers (e.g. Aflatoxin, the animal, the feed, the stage), \
never "it" or "this".

Write at most the number of questions asked for, covering these types (at least one of each when the facts allow):
- direct: a natural question answered by one fact.
- paraphrased: like direct, but worded the way a farmer would ask, avoiding the quote's key words.
- keyword: a 2-5 word search query, the way people type into a search box.
- precise: asks for an exact number, dose, age, duration or limit from the quote (the answer holds that \
number), keeping the question's qualifiers.
- multi_fact: one natural question about a single topic whose full answer needs two different facts; \
list both ids.

For each question give its type, the question, the fact_ids that answer it, and answer: a short span \
copied exactly from one of those quotes (for multi_fact, one span from each fact's quote, joined by "; "). \
Never copy a whole quote into the question."""


class Fact(BaseModel):
    claim: str
    quote: str


class Facts(BaseModel):
    facts: list[Fact]


class Question(BaseModel):
    type: Literal["direct", "paraphrased", "keyword", "precise", "multi_fact"]
    question: str
    fact_ids: list[str]
    answer: str


class Questions(BaseModel):
    questions: list[Question]


def _load(path: Path) -> dict[str, dict[str, Any]]:
    return {row["segment_id"]: row for row in read_jsonl(path)} if path.exists() else {}


def _todo(done: dict[str, Any], doc: str | None, force: bool, version: str) -> list[dict[str, Any]]:
    """Segments used for questions that this stage hasn't done at its current version (all of them with
    --force). Rows of segments no longer used for questions are dropped from `done`."""
    segments = [s for s in load_segments() if s["use_for_questions"]]
    for gone in done.keys() - {s["segment_id"] for s in segments}:
        del done[gone]
    picked = [s for s in segments if not doc or doc.lower() in s["file_name"].lower()]
    return [s for s in picked if force or done.get(s["segment_id"], {}).get("version") != version]


def fact_prompt(part: dict[str, Any], so_far: list[str], wanted: int) -> str:
    memory = "\n".join(f"- {claim}" for claim in so_far) or "(none yet)"
    context = f"CONTEXT (end of the previous part, for understanding only):\n{part['context']}\n\n" if part["context"] else ""
    return f"SECTION: {part['heading']}\n\n{context}FACTS SO FAR:\n{memory}\n\nReturn at most {wanted} new facts.\n\nTEXT:\n{part['text']}"


def run_facts(doc: str | None = None, max_facts: int = MAX_FACTS, force: bool = False) -> None:
    """Facts for each segment, read in parts that remember the facts found so far.
    Given a whole chapter the model only takes facts from its first pages, so each part gets an
    even share of max_facts (plus any shortfall of the parts before it)."""
    done = _load(FACTS)
    agent = Agent(llm.model(), output_type=NativeOutput(Facts, strict=True), instructions=FACT_PROMPT)
    for segment in _todo(done, doc, force, FACTS_VERSION):
        found: list[dict[str, str]] = []
        cut = parts(segment, PART_CHARS)
        for i, part in enumerate(cut):
            wanted = (i + 1) * max_facts // len(cut) - len(found)
            if wanted <= 0:
                continue
            prompt = fact_prompt(part, [f["claim"] for f in found], wanted)
            result = agent.run_sync(prompt)
            llm.check_usage(result.usage.input_tokens, prompt, segment["segment_id"])
            found += [f.model_dump() for f in result.output.facts[:wanted]]
        kept, rejected = verify_facts(segment, found)
        done[segment["segment_id"]] = {"segment_id": segment["segment_id"], "model": llm.settings().llm_model,
                                       "version": FACTS_VERSION, "facts": kept, "rejected": rejected}
        write_jsonl(FACTS, list(done.values()))  # saved after every segment, so a stop loses little
        print(f"{segment['segment_id']}: {len(kept)} facts kept, {len(rejected)} rejected")


def run_questions(doc: str | None = None, max_questions: int = MAX_QUESTIONS, force: bool = False) -> None:
    """Questions for each segment, written only from its verified facts. Each question carries its
    evidence (quotes and their positions), so scoring never depends on a later facts run."""
    facts, done = _load(FACTS), _load(QUESTIONS)
    agent = Agent(llm.model(), output_type=NativeOutput(Questions, strict=True), instructions=QUESTION_PROMPT)
    for segment in _todo(done, doc, force, QUESTIONS_VERSION):
        verified = facts.get(segment["segment_id"], {}).get("facts", [])
        short = {f"f{i}": f for i, f in enumerate(verified, start=1)}  # short ids keep the prompt small
        kept, rejected = [], []
        if short:  # a segment without verified facts gets no questions (and loses any earlier ones)
            listing = json.dumps([{"id": k, "claim": f["claim"], "quote": f["quote"]} for k, f in short.items()], ensure_ascii=False)
            prompt = f"SECTION: {segment['title']}\n\nReturn at most {max_questions} questions.\n\nFACTS:\n{listing}"
            result = agent.run_sync(prompt)
            llm.check_usage(result.usage.input_tokens, prompt, segment["segment_id"])
            asked = [q.model_dump() for q in result.output.questions[:max_questions]]
            kept, rejected = verify_questions(asked, short)
        for n, q in enumerate(kept, start=1):
            linked = [short[i] for i in q["fact_ids"]]
            q["fact_ids"] = [f["fact_id"] for f in linked]
            q["evidence"] = [{k: f[k] for k in ("quote", "start", "end", "pages")} for f in linked]
            q["question_id"] = f"{segment['segment_id']}:q{n:02d}"
        done[segment["segment_id"]] = {"segment_id": segment["segment_id"], "model": llm.settings().llm_model,
                                       "version": QUESTIONS_VERSION, "questions": kept, "rejected": rejected}
        write_jsonl(QUESTIONS, list(done.values()))
        print(f"{segment['segment_id']}: {len(kept)} questions kept, {len(rejected)} rejected")
