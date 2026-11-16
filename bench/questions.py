"""The questions stage's parts: what the writers and the reviewer are told, the candidates code builds, and the rows.
bench/orchestrator.py runs the stage (`bench questions`), per group of PDFs (a PDF on its own, or PDFs grouped in
data/topics.json):
1. code lists candidates: which facts suit which question type (read from question_types.toml), pairs of facts that
   share a specific word (one section for types that need two facts, two sections for multi-hop), and absent terms
   from vocab.toml that the library never mentions, checked word by word;
2. the orchestrator plans the items: which type each fact or pair gets, and which absent term;
3. a writer per type writes one question per item; code attaches the item's facts and absent term, so a writer can't
   use the wrong fact or an absent term the library mentions;
4. a rejected draft goes back to its writer once, told why; the reviewer flags; the regex audit alone decides.

Output: one row per group in data/questions.jsonl {group, pdfs, version, fingerprint, plan, questions, rejected}, each
question with its evidence and a content id (uid); ids <section>:q01, and <first fact's section>:h01 for multi-hop.
"""

from __future__ import annotations

import json
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from bench.config import DATA_DIR, ROOT, BenchError
from bench.pdf import read_jsonl
from bench.verify import (
    ACRONYM,
    NUMBER,
    content_words,
    evidence_text,
    in_library,
    name_pattern,
    uid,
)

QUESTIONS = DATA_DIR / "questions.jsonl"
QUESTIONS_VERSION = "questions-v8"  # the row format (one row per group of PDFs); part of every row's fingerprint
VOCAB = ROOT / "vocab.toml"
PAIR_TERM_FACTS = 5  # a shared word pairs two facts only if at most this many facts of the group use it
PAIR_PARTNERS = 3  # pairs listed per fact and kind (one section, two sections), rarest shared word first

WRITER_PROMPT = """You write questions for a benchmark that checks whether a search engine finds the \
passage that answers them. You get verified facts, each with an id, a claim and the exact quote it comes \
from. The search covers a library of many documents that may give other values for other years, places, \
animals or schemes, so every question must make sense on its own: name its subject and every qualifier its \
answer depends on (e.g. Aflatoxin, the animal or class, the feed, the stage, the year, the country or state, \
the scale, the scheme). Never write "it", "this" or "the system". Never copy a whole quote into the \
question. Write in English.

Write questions of the TYPE below only. Ask about facts that the QUESTIONS ALREADY WRITTEN don't use, or \
about other parts of them; never ask the same thing again. For each give the question, the fact_ids it uses, and answer: the \
answer copied exactly from the quotes; when it has several parts (one per fact, or the items of a list), copy \
each part and join them with "; ". Leave absent_term empty unless the TYPE asks for it.

TYPE: """

ITEM_WRITER_PROMPT = """You write questions for a benchmark that checks whether a search engine finds the passage \
that answers them. Each ITEM names the verified facts to ask about (their ids, claims and the exact quotes they come \
from are under FACTS), and for some types an absent term and the word it replaces. The search covers a library of many \
documents that may give other values for other years, places, animals or schemes, so every question must make sense \
on its own: name its subject and every qualifier its answer depends on (e.g. Aflatoxin, the animal or class, the feed, \
the stage, the year, the country or state, the scale, the scheme). Never write "it", "this" or "the system". Never copy \
a whole quote into the question. Write in English.

Write exactly one question of the TYPE below for each ITEM, using only that item's facts. For each give item (its id), \
the question, and answer: the answer copied exactly from the quotes, with its unit; when it has several parts (one per \
fact, or the items of a list), copy each part and join them with "; ". Leave answer empty for an item with an absent term.

TYPE: """

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


class ItemDraft(BaseModel):
    item: str
    question: str
    answer: str


class ItemDrafts(BaseModel):
    questions: list[ItemDraft]


class Review(BaseModel):
    id: str
    verdict: Literal["keep", "rewrite", "flag"]
    question: str
    reason: str


class Reviews(BaseModel):
    reviews: list[Review]


def listing(facts: dict[str, dict[str, Any]], titles: dict[str, str] | None = None) -> str:
    rows = [{"id": k, **({"section": titles[f["fact_id"].rsplit(":", 1)[0]]} if titles else {}),
             "claim": f["claim"], "quote": " ".join(evidence_text(f).split()),
             **{k: f[k] for k in ("subject", "example", "brief") if k in f}} for k, f in facts.items()]
    return json.dumps(rows, ensure_ascii=False, indent=1)


def review_prompt(title: str, drafts: list[dict[str, Any]], facts: dict[str, dict]) -> str | None:
    """The question reviewer's input: every draft with its type, answer, fact ids and quotes (None if no drafts)."""
    if not drafts:
        return None
    listing = [{"id": f"d{i}", "type": d["type"], "question": d["question"], "answer": d["answer"] or "(none)",
                "fact_ids": d["fact_ids"], **({"absent_term": d["absent_term"]} if d.get("absent_term") else {}),
                "quotes": [" ".join(evidence_text(facts[k]).split()) for k in d["fact_ids"] if k in facts]}
               for i, d in enumerate(drafts, start=1)]
    return f"SECTION: {title}\n\nQUESTIONS TO REVIEW:\n{json.dumps(listing, ensure_ascii=False, indent=1)}"


def apply_reviews(drafts: list[dict[str, Any]], reviews: list[Review]) -> list[dict[str, Any]]:
    """A rewrite replaces the question (the original is kept); a flag is recorded and the draft still goes to
    the audit; a draft the reviewer didn't answer is marked unreviewed."""
    verdicts = {r.id: r for r in reviews}
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


def question_rows() -> list[dict[str, Any]]:
    """data/questions.jsonl in today's format, one row per group of PDFs; rows in an older format are left out (and redone)."""
    return [r for r in read_jsonl(QUESTIONS) if r.get("version") == QUESTIONS_VERSION] if QUESTIONS.exists() else []


def vocab(path: Path = VOCAB) -> dict[str, list[str]]:
    """The word lists that absent terms come from (vocab.toml): {list name: [terms]}, checked strictly."""
    try:
        raw = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise BenchError(f"{path.name}: {exc}") from exc
    if bad := [k for k, v in raw.items() if not isinstance(v, list) or not all(isinstance(t, str) and t.strip() for t in v)]:
        raise BenchError(f"{path.name}: {', '.join(bad)} must be lists of words")
    return raw


def needs(t: dict[str, Any]) -> str:
    """What an item of the type is written from, read from its fields: 'cross' (a pair of facts from two sections),
    'pair' (two facts of one section), 'absent' (a fact and an absent term) or 'fact'."""
    if (t["rules"].get("sections") or 1) >= 2:
        return "cross"
    if (t["rules"].get("facts") or [1])[0] >= 2:
        return "pair"
    return "absent" if t["source"] == "absent" else "fact"


def fits(t: dict[str, Any], fact: dict[str, Any]) -> bool:
    """A single fact suits the type: one larger than a sentence for source 'paragraph', a number in its quote for
    needs_number."""
    return ((t["source"] != "paragraph" or fact.get("size", "sentence") != "sentence")
            and (not t["rules"].get("needs_number") or bool(NUMBER.search(evidence_text(fact)))))


def section(fact: dict[str, Any]) -> str:
    return fact["fact_id"].rsplit(":", 1)[0]


def _words(fact: dict[str, Any], lists: dict[str, list[str]]) -> set[str]:
    claim = fact["claim"]
    return (content_words(claim) | {a.lower() for a in ACRONYM.findall(claim)}
            | {t.lower() for terms in lists.values() for t in terms if name_pattern(t).search(claim)})


def pairs(facts: dict[str, dict[str, Any]], lists: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Candidate pairs: two facts sharing a word that at most PAIR_TERM_FACTS facts of the group use (counted within
    the group, so a new PDF doesn't change another group's pairs), rarest shared word first, ties broken by the word
    and the facts' order; at most PAIR_PARTNERS per fact for each kind (one section, two sections)."""
    ids = list(facts)
    words = {k: _words(f, lists) for k, f in facts.items()}
    used = Counter(w for ws in words.values() for w in ws)
    found = []
    for i, a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            b = ids[j]
            shared = [w for w in words[a] & words[b] if used[w] <= PAIR_TERM_FACTS]
            if shared:
                rarest = min(shared, key=lambda w: (used[w], w))
                found.append(((used[rarest], rarest, i, j), a, b, rarest, section(facts[a]) != section(facts[b])))
    kept, partners = [], Counter()
    for _, a, b, word, cross in sorted(found, key=lambda p: p[0]):
        if partners[(a, cross)] < PAIR_PARTNERS and partners[(b, cross)] < PAIR_PARTNERS:
            partners[(a, cross)] += 1
            partners[(b, cross)] += 1
            kept.append({"a": a, "b": b, "word": word, "cross": cross})
    return [{"id": f"p{n}", **p} for n, p in enumerate(kept, start=1)]


def absent_terms(lists: dict[str, list[str]], corpus: str) -> dict[str, list[str]]:
    """Per word list, its terms none of whose words the library uses: the only absent terms a question may name."""
    return {name: [t for t in terms if not in_library(t, corpus)] for name, terms in lists.items()}


def named(fact: dict[str, Any], lists: dict[str, list[str]]) -> dict[str, str]:
    """The word lists a fact names a term of, each with the first such term: {list name: term}."""
    text = f"{fact['claim']}\n{evidence_text(fact)}"
    found = {}
    for name, terms in lists.items():
        if term := next((t for t in terms if name_pattern(t).search(text)), None):
            found[name] = term
    return found
