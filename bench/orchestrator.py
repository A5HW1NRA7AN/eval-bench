"""The orchestrator: per PDF, an orchestrator agent plans the template facts and questions and worker agents write
them. Code checks the plan and runs the workers, and the regex audit alone decides what is kept.

`bench templates run`, per PDF:
1. map (code): each used section with its headings, size, code's signal densities and opening lines;
2. categorise (orchestrator): the templates each section answers. Your data/topics.json override wins; a section
   the answer leaves out, or an unreadable answer, falls back to the density rules;
3. read (a reader per 12K part and template): every (intent, subject, paragraphs, detail) the part states. Code
   keeps an entry only if the audit could pass a fact from it: its subject named and a cue word of its intent in
   the text the audit reads (templates.near);
4. plan (orchestrator, per template): the entries worth a fact, one name per subject, and a brief where a worker
   needs what its passage doesn't show. An empty plan is kept; an unreadable one falls back to every entry;
5. workers: an extractor per pick, the fact reviewer and the audit; writers in the template's styles, one repair
   of a draft the audit rejects (told why), the question reviewer and the audit.
The orchestrator reads the full text of the sections it decides on when it fits ORCHESTRATOR_TEXT_CHARS (set per
model), else the map; a long map (MAP_CHARS of sections, PLAN_ENTRIES entries) is decided in consecutive groups of
sections. The calls of one
step run together (LLM_CONCURRENCY). Output: one row per PDF in data/templates.jsonl, ids <section>:tf01 (facts)
and <section>:t01 (questions).
"""

from __future__ import annotations

import inspect
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from bench import facts as fact_stage
from bench import llm
from bench import questions as question_stage
from bench.config import (
    PART_CHARS,
    REQUEST_TIMEOUT,
    BenchError,
    question_types,
    settings,
)
from bench.pdf import PAGES, read_jsonl, write_jsonl
from bench.segments import heading_path, load_segments, page_at, parts, sentences
from bench.templates import (
    CUE_BEFORE,
    ROWS,
    SUBJECT_BEFORE,
    TEMPLATES_VERSION,
    Template,
    active,
    by_rules,
    cue_pattern,
    inputs,
    is_on,
    load_topics,
    names,
    near,
    pdf_rows,
    row_problems,
    scores,
    subject_window,
    template_fact_problem,
    template_question_problem,
)
from bench.verify import (
    NUMBER,
    evidence_text,
    question_problem,
    reason_group,
    uid,
    verify_facts,
)

MAP_OPENING = 200  # characters of each section's opening shown in the map
MAP_CHARS = 40_000  # a longer map of sections is categorised in consecutive groups of sections
PLAN_ENTRIES = 100  # a plan covers at most this many entries, so its answer stays short; more are planned in groups
READ_PER_INTENT = 3  # entries a reader may give per intent in one part
BRIEF_CHARS = 300  # a brief is cut to this length
ORCHESTRATOR_SETTINGS = {"max_tokens": 8_000, "request_timeout": 3 * REQUEST_TIMEOUT}  # a long input, a whole plan out
QUESTION_PLAN_FACTS = 40  # a general question plan covers at most this many facts; more are planned in groups of sections
QUESTION_PLAN_PAIRS = 100  # pairs one plan call decides on; more are decided in consecutive groups
PER_FACT = 2  # one-fact general questions per fact at most, each of a different type
ABSENT_USES = 2  # an absent term is used for at most this many questions per PDF

QUESTION_PLAN_PROMPT = """You plan the general questions of a search benchmark for one document, or a group of \
documents. You get the question TYPES (with the most each may get, and what an item of it needs) and either the \
verified FACTS of some sections (each fact with its id, its size, whether its quote holds a number, the word lists it \
names, and its claim; each section with its QUOTA, how many questions of each type its facts allow) and the ABSENT \
TERMS (words of those lists that the library never mentions), or candidate PAIRS of facts that share a specific word, \
each with both claims. Plan the ITEMS to write, one question each:
- type: one of the TYPES.
- ref: a fact id (g7) for a type written from one fact; a pair id (p4) for a type that needs two facts.
- absent_term: for a type that needs one, an absent term of a list the fact names (it takes the place of that list's \
word in the question); otherwise empty.
- brief: only when the writer needs what the facts don't show (which detail to ask about); usually empty.
FACTS: fill every section's QUOTA, spread over its facts: at most two questions per fact, of different types. Prefer \
facts a user would really search for. Use an absent term only where the fact's answer depends on the word it replaces \
(a figure for India, a breed's yield) and the question stays one a user would ask; never for advice that holds \
whatever the word.
PAIRS: plan a pair, with the type for its kind (one section or two), only when its two facts serve one need a user \
would ask about in one question (a limit and what happens above it, a disease and its treatment, a feed's value and how \
much to feed). Two separate questions joined by "and" are not one question: plan fewer pairs rather than a forced one.
Name facts and pairs by their ids only."""

CATEGORISE_PROMPT = """You categorise the sections of one document for a search benchmark. TEMPLATES describe \
kinds of passage and what users ask about them. The MAP lists each section: its id, title, pages, size, table rows, \
how often each template's signal words occur per 1,000 words (counted by code, only a hint), its headings and its \
opening lines; the TEXT, when given, is the sections' full text. For each section give the templates whose \
questions it answers with specific information anywhere in it: none, one or several. Return every section once."""

READER_PROMPT = f"""You read one part of a document section for a search benchmark. The TEXT is in numbered \
paragraphs ([p3]); a table is a paragraph whose lines are rows of cells joined by " | ", its header rows first. \
For each INTENT below, list the subjects the TEXT gives specific information about, at most {READ_PER_INTENT} per \
intent, the most specific first. Skip what only the CONTEXT states. For each give:
- intent: the intent's name.
- subject: the SUBJECT it belongs to, the name a user would put in their question (a disease, a scheme, a \
measure), written exactly as the TEXT or the SECTION heading writes it; not the detail.
- paragraphs: the ids of the paragraphs that state it (usually one, at most three consecutive).
- detail: what they state about it, in at most 12 words.
Return an empty list if the TEXT answers none of the INTENTS."""

PLAN_PROMPT = f"""You plan the template questions of a search benchmark for one document. Readers listed ENTRIES: \
passages that state what users ask about (an intent) for a subject, each with a short detail. Each entry you keep \
becomes one fact, written by a worker who sees only its passage and your brief; the TEXT, when given, is the full \
text of these sections.
- Keep every entry that states a fact users would ask about. Drop an entry only if it repeats another entry's fact \
(keep the fuller one) or states nothing specific.
- Group the kept entries by subject: one name per subject across the document, as the text writes it, with its \
abbreviation if the text gives one (e.g. "Foot and Mouth Disease (FMD)"). Entries about other years, places, \
animals or figures of one subject stay separate entries.
- briefs: only where a worker needs what its passage doesn't show (what the subject is called elsewhere; which \
year, place, animal or scheme applies; which detail to take), at most {BRIEF_CHARS} characters each.
Name entries by their ids only (e.g. e7), never by their text. Return no subjects if no entry is worth a question."""

TABLE_NOTE = """

A table row (cells joined by " | ") counts as one sentence. For a figure in a table, point at the one row \
that holds it (first and last are that row); the table's header rows come first. Name the row's and the \
column's labels in the claim."""
EXTRACTOR_PROMPT = fact_stage.EXTRACTOR_PROMPT + TABLE_NOTE

WRITER_NOTE = ("Each fact answers something users ask about this kind of document. It names its subject, gives an "
               "example of such a question and may carry a brief of what to ask: write in that spirit, name the "
               "subject, and ask about each fact.")
QUESTION_REPAIR = ("\n\nA CHECK REJECTED THIS QUESTION:\n{question}\nanswer: {answer}\nreason: {reason}\n{hint}"
                   "Write it again so that it passes the check and still asks about the same fact.")
HINTS = {  # by reason_group: what to change, so a repair isn't a guess
    "answer not found in its facts' quotes": 'Copy the answer word for word from the quote; join several parts with "; ".',
    "too many of the question's words come from the quote": "Ask in your own words, as a farmer would; names stay.",
    "doesn't name its subject a term": "Name the subject in the question.",
    "question drops its quote's year": "Name the year the quote states (for example 2024-25).",
    "question copies its quote": "Ask about the quote; don't copy it.",
    "question contains an internal id": "Don't write fact ids such as f1 in the question.",
    "refers to a fact that isn't verified": "Use only the fact ids listed.",
    "the question doesn't name its absent term": "Name the absent term in the question, exactly as given.",
    "needs N-N words, has N": "Keep to the number of words the check names.",
    "needs N-N answer parts, has N": 'Give as many answer parts as the check names, each copied from a quote, joined with "; ".',
    "the answer needs a span from each fact": 'Copy one part of the answer from each fact\'s quote, joined with "; ".',
}
NO_REPAIR = {"same as a general question"}  # depends on another stage's rows, so a repair would make runs differ


class Subject(BaseModel):
    name: str
    entries: list[str] = Field(description="the ids of the entries kept under this subject, e.g. e7")


class Brief(BaseModel):
    entry: str = Field(description="the entry's id, e.g. e7")
    brief: str


class Plan(BaseModel):
    subjects: list[Subject]
    briefs: list[Brief]


def categories_schema(sections: list[str], templates_: list[str]) -> type[BaseModel]:
    """The categorise answer, its sections and templates limited to those it was asked about."""
    item = create_model("SectionTemplates", section=(Literal[tuple(sections)], ...),
                        templates=(list[Literal[tuple(templates_)]], ...))  # type: ignore[valid-type]
    return create_model("Categories", sections=(list[item], ...))  # type: ignore[valid-type]


def reader_schema(intents: list[str]) -> type[BaseModel]:
    """The reader's answer, its intent limited to the intents asked about."""
    entry = create_model("Entry", intent=(Literal[tuple(intents)], ...), subject=(str, ...),  # type: ignore[valid-type]
                         paragraphs=(list[str], ...), detail=(str, ...))
    return create_model("Entries", entries=(list[entry], ...))  # type: ignore[valid-type]


def cut(text: str, limit: int = BRIEF_CHARS) -> str:
    """The text on one line, cut at a word boundary to at most `limit` characters."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0]


def groups(items: list[Any], size: Callable[[Any], int], limit: int) -> list[list[Any]]:
    """The items in order, packed into consecutive groups of at most `limit` (an item bigger than that is alone)."""
    out: list[list[Any]] = []
    for item in items:
        if not out or (out[-1] and sum(map(size, out[-1])) + size(item) > limit):
            out.append([])
        out[-1].append(item)
    return out


def map_lines(short: str, segment: dict[str, Any], tmpls: dict[str, Template]) -> str:
    """One section in the map: id, title, pages, size, table rows, code's signal densities, headings, opening."""
    hints = scores(segment, tmpls)
    signals = ", ".join(f"{n} {s['density']:g}" for n, s in hints.items())
    numbers = next(iter(hints.values()))["numbers_per_word"] if hints else 0
    pages = f"pages {segment['pages'][0]}-{segment['pages'][-1]}" if segment["pages"] else "no pages"
    rows = sum(" | " in line for line in segment["text"].splitlines())
    heads = " · ".join(h["title"] for h in segment["headings"]) or "-"
    return (f"{short} | {segment['title']} | {pages} | {len(segment['text']):,} chars | {rows} table rows | "
            f"signals per 1,000 words: {signals} | numbers per word {numbers:g}\n   headings: {heads}\n"
            f"   opens: \"{' '.join(segment['text'][:MAP_OPENING].split())}\"")


def categorise_prompt(stem: str, items: list[tuple[str, dict[str, Any]]], tmpls: dict[str, Template],
                      budget: int) -> tuple[str, bool]:
    """The categorise input for some sections, and whether their full text fits the model's budget (then it's in)."""
    described = "\n".join(f"- {n}: {t.description} Users ask about: " + "; ".join(i.description for i in t.intents.values())
                          for n, t in tmpls.items())
    text = "\n\n".join(f"== {short} {s['title']} ==\n{s['text']}" for short, s in items)
    fits = len(text) <= budget
    return (f"DOCUMENT: {stem}\n\nTEMPLATES:\n{described}\n\nMAP:\n" + "\n".join(map_lines(short, s, tmpls) for short, s in items)
            + (f"\n\nTEXT:\n{text}" if fits else "")), fits


def categorise(segments: list[dict[str, Any]], tmpls: dict[str, Template], topics: dict[str, Any],
               failed: list[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """({segment_id: {templates, by}}, the views used): your override, else the orchestrator, else the rules."""
    on = {n: t for n, t in tmpls.items() if is_on(topics, n, t)}
    shorts = [(f"s{n}", s) for n, s in enumerate(segments, start=1)]
    answer: dict[str, list[str]] = {}
    views: list[str] = []
    if on and any(s["segment_id"] not in topics["sections"] for s in segments):
        packs = groups(shorts, lambda item: len(map_lines(item[0], item[1], on)), MAP_CHARS)
        jobs = []
        for k, pack in enumerate(packs, start=1):
            prompt, fits = categorise_prompt(Path(segments[0]["file_name"]).stem, pack, on, settings().orchestrator_text_chars)
            agent = llm.agent(categories_schema([short for short, _ in pack], list(on)), CATEGORISE_PROMPT)
            jobs.append(("orchestrator categorise", agent, prompt,
                         segments[0]["file_name"] + (f" group {k}/{len(packs)}" if len(packs) > 1 else "")))
            views.append("text" if fits else "map")
        for out in llm.call_all(jobs, tolerate=failed, model_settings=ORCHESTRATOR_SETTINGS):
            for c in out.sections if out else []:
                answer.setdefault(c.section, [n for n in on if n in c.templates])
    sections = {}
    for short, s in shorts:
        sid = s["segment_id"]
        sections[sid] = ({"templates": topics["sections"][sid], "by": "you"} if sid in topics["sections"] else
                         {"templates": answer[short], "by": "orchestrator"} if short in answer else
                         {"templates": by_rules(s, on), "by": "rules"})
    return sections, views


def reader_prompt(segment: dict[str, Any], tmpl: Template, intents: list[str], part: dict[str, Any],
                  inside: list[dict[str, Any]]) -> str:
    listed = "\n".join(f"- {n}: {tmpl.intents[n].description}" for n in intents)
    context = f"CONTEXT (end of the previous part, for understanding only):\n{part['context']}\n\n" if part["context"] else ""
    return (f"DOCUMENT TYPE: {tmpl.description}\nSUBJECT: {tmpl.subject or 'what the passage is about'}\nINTENTS:\n{listed}\n\n"
            f"SECTION: {part['heading']}\n\n{context}TEXT:\n{fact_stage.paragraph_listing(inside, segment['text'])}")


def read(segments: list[dict[str, Any]], on: dict[str, dict[str, list[str]]], tmpls: dict[str, Template],
         failed: list[str]) -> list[dict[str, Any]]:
    """Every entry the readers give, in document order with ids e1…; code marks the ones the audit could never
    pass (`dropped`, with why) and they get no further call."""
    jobs, asked, agents = [], [], {}
    for segment in segments:
        units = sentences(segment["text"])
        for name, intents in on.get(segment["segment_id"], {}).items():
            cut_parts = parts(segment, PART_CHARS)
            for i, part in enumerate(cut_parts):
                inside = [u for u in units if part["start"] <= u["start"] < part["end"]]
                if not inside:
                    continue
                if tuple(intents) not in agents:
                    agents[tuple(intents)] = llm.agent(reader_schema(intents), READER_PROMPT)
                jobs.append(("reader", agents[tuple(intents)], reader_prompt(segment, tmpls[name], intents, part, inside),
                             f"{segment['segment_id']} {name} part {i + 1}/{len(cut_parts)}"))
                asked.append((segment, name, inside))
    entries = []
    for (segment, name, inside), out in zip(asked, llm.call_all(jobs, tolerate=failed)):
        given: Counter[str] = Counter()
        for e in out.entries if out else []:
            given[e.intent] += 1
            if given[e.intent] > READ_PER_INTENT:
                continue
            wanted = fact_stage.paragraph_ids(e.paragraphs)
            chosen = [u for u in inside if u["id"].split(".")[0] in wanted]
            entry = {"segment_id": segment["segment_id"], "template": name, "intent": e.intent, "subject": " ".join(e.subject.split()),
                     "paragraphs": list(dict.fromkeys(u["id"].split(".")[0] for u in chosen)), "detail": " ".join(e.detail.split()),
                     "start": chosen[0]["start"] if chosen else None, "end": chosen[-1]["end"] if chosen else None}
            if not chosen:
                entry["dropped"] = "its paragraphs aren't in the part"
            elif not entry["subject"] or not names(entry["subject"], subject_window(segment, entry["start"], entry["end"])):
                entry["dropped"] = "its subject isn't named near it"
            elif not cue_pattern(tmpls[name].intents[e.intent].cues).search(near(segment, entry["start"], entry["end"], CUE_BEFORE)):
                entry["dropped"] = "no cue word of its intent near it"
            entries.append(entry)
    order = {s["segment_id"]: n for n, s in enumerate(segments)}
    entries.sort(key=lambda e: (order[e["segment_id"]], e["start"] if e["start"] is not None else -1, list(tmpls).index(e["template"]),
                                list(tmpls[e["template"]].intents).index(e["intent"]), e["subject"].lower()))
    return [{"id": f"e{n}", **e} for n, e in enumerate(entries, start=1)]


def plan_prompt(stem: str, tmpl: Template, pack: list[tuple[str, dict[str, Any], list[dict[str, Any]]]],
                budget: int) -> tuple[str, bool]:
    """The plan input for some sections' entries, and whether their full text fits the model's budget (then it's in)."""
    listed = "\n".join(f"- {n}: {i.description}" for n, i in tmpl.intents.items())
    lines = []
    for short, segment, mine in pack:
        pages = f"pages {segment['pages'][0]}-{segment['pages'][-1]}" if segment["pages"] else "no pages"
        lines.append(f"{short} {segment['title']} ({pages})")
        lines += [entry_line(e) for e in mine]
    text = "\n\n".join(f"== {short} {s['title']} ==\n{fact_stage.paragraph_listing(sentences(s['text']), s['text'])}"
                       for short, s, _ in pack)
    fits = len(text) <= budget
    return (f"DOCUMENT: {stem}\nDOCUMENT TYPE: {tmpl.description}\nINTENTS:\n{listed}\n\nENTRIES:\n" + "\n".join(lines)
            + (f"\n\nTEXT:\n{text}" if fits else "")), fits


def entry_line(e: dict[str, Any]) -> str:
    """An entry as the plan reads it: 'e7 prevention | mastitis | p3 | 640 chars | dry floors keep udders clean'."""
    return f"{e['id']} {e['intent']} | {e['subject']} | {', '.join(e['paragraphs'])} | {e['end'] - e['start']} chars | {e['detail']}"


def _key(text: str) -> str:
    return " ".join(text.strip().strip("[]").split()).lower()


def validate(answer: Plan, mine: list[dict[str, Any]], segments: dict[str, dict[str, Any]]) -> tuple[dict[str, dict], list[dict]]:
    """({entry id: pick}, what code can't use): each valid pick with its subject (the plan's name if the text names it
    near the entry, else the reader's) and its brief. An entry may be named by its id ('e7', '[e7]') or, as models
    sometimes do, by a copy of its line or its detail: all of these are text we wrote, so code maps them back."""
    by_id = {e["id"]: e for e in mine}
    copies: dict[str, str] = {}
    for e in mine:
        for text in (entry_line(e), entry_line(e).split(" ", 1)[1], e["detail"]):
            copies.setdefault(_key(text), e["id"])

    def entry_id(raw: str) -> str | None:
        start = re.match(r"\W*(e\d+)\b", raw)
        return start.group(1) if start and start.group(1) in by_id else copies.get(_key(raw))

    picks: dict[str, dict] = {}
    invalid = []
    for subject in answer.subjects:
        name = " ".join(subject.name.split())
        for raw in subject.entries:
            eid = entry_id(raw)
            if eid is None or eid in picks:
                invalid.append({"entry": raw[:80], "why": "no such entry" if eid is None else "listed twice"})
                continue
            e = by_id[eid]
            named = bool(name) and names(name, subject_window(segments[e["segment_id"]], e["start"], e["end"]))
            picks[eid] = {"subject": name if named else e["subject"], **({} if named or not name else {"proposed": name})}
    for b in answer.briefs:
        if (eid := entry_id(b.entry)) in picks and b.brief.strip():
            picks[eid]["brief"] = cut(b.brief)
    return picks, invalid


def plan(file_name: str, segments: list[dict[str, Any]], entries: list[dict[str, Any]], tmpls: dict[str, Template],
         agent: Any, failed: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Per template, the orchestrator's plan over the kept entries (in groups when the map is long). Sets each kept
    entry's `pick`: {by, subject, brief} or None. An unreadable plan, or one naming no valid entry, falls back to
    every entry (by code); an empty plan is kept."""
    by_id = {s["segment_id"]: s for s in segments}
    shorts = {s["segment_id"]: f"s{n}" for n, s in enumerate(segments, start=1)}
    kept = [e for e in entries if not e.get("dropped")]
    jobs, asked = [], []
    for name in dict.fromkeys(e["template"] for e in kept):
        mine = [e for e in kept if e["template"] == name]
        sections = [(shorts[sid], by_id[sid], [e for e in mine if e["segment_id"] == sid]) for sid in dict.fromkeys(e["segment_id"] for e in mine)]
        packs = groups(sections, lambda item: len(item[2]), PLAN_ENTRIES)
        for k, pack in enumerate(packs, start=1):
            prompt, fits = plan_prompt(Path(file_name).stem, tmpls[name], pack, settings().orchestrator_text_chars)
            jobs.append(("orchestrator plan", agent, prompt, f"{file_name} {name}" + (f" group {k}/{len(packs)}" if len(packs) > 1 else "")))
            asked.append((name, [e for _, _, es in pack for e in es], "text" if fits else "map"))
    plans: dict[str, list[dict[str, Any]]] = {}
    for (name, mine, view), answer in zip(asked, llm.call_all(jobs, tolerate=failed, model_settings=ORCHESTRATOR_SETTINGS)):
        picks, invalid = validate(answer, mine, by_id) if answer else ({}, [])
        by = "orchestrator" if answer and (picks or not invalid) else "code"
        if by == "code":
            picks = {e["id"]: {"subject": e["subject"]} for e in mine}
        for e in mine:
            e["pick"] = {"by": by, "brief": e["detail"], **picks[e["id"]]} if e["id"] in picks else None
        plans.setdefault(name, []).append({"by": by, "view": view, "entries": len(mine), "picked": len(picks), "invalid": invalid,
                                           "subjects": [s.model_dump() for s in answer.subjects] if answer else None})
    return plans


def extract(picks: list[dict[str, Any]], segments: dict[str, dict[str, Any]], tmpls: dict[str, Template], extractor: Any,
            failed: list[str]) -> list[dict[str, Any]]:
    """One extractor call per pick: the pick's paragraphs as numbered sentences, its brief, and the other picks in
    those paragraphs (so no pick depends on another's answer). Returns the facts before the audit."""
    units = {sid: sentences(s["text"]) for sid, s in segments.items()}
    jobs, asked = [], []
    for p in picks:
        segment, intent = segments[p["segment_id"]], tmpls[p["template"]].intents[p["intent"]]
        chosen = [u for u in units[p["segment_id"]] if u["id"].split(".")[0] in p["paragraphs"]]
        topic = fact_stage.Topic(topic=f"{intent.description} Subject: {p['pick']['subject']}.", kind=intent.kind, paragraphs=p["paragraphs"])
        others = [f"{o['intent']}: {o['pick']['subject']}" for o in picks
                  if o is not p and o["segment_id"] == p["segment_id"] and set(o["paragraphs"]) & set(p["paragraphs"])]
        prompt = fact_stage.extractor_prompt(heading_path(segment, page_at(segment, chosen[0]["start"])), topic, chosen,
                                             segment["text"], [], brief=p["pick"]["brief"], others=others)
        jobs.append(("extractor", extractor, prompt, f"{p['segment_id']} ▸ {p['intent']} ▸ {p['pick']['subject'][:40]}"))
        asked.append((p, topic, intent))
    found = []
    for (p, topic, intent), out in zip(asked, llm.call_all(jobs, tolerate=failed)):
        subject = p["pick"]["subject"]
        found += [{**f.model_dump(), "topic": topic.topic, "kind": intent.kind, "template": p["template"], "intent": p["intent"],
                   "subject": subject, "example": intent.examples[0].replace("{subject}", subject), "entry": p["id"],
                   "brief": p["pick"]["brief"], "segment_id": p["segment_id"]} for f in (out.facts[:1] if out else [])]
    return found


def audit_facts(found: list[dict[str, Any]], segments: dict[str, dict[str, Any]], tmpls: dict[str, Template], reviewer: Any,
                failed: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per section: the fact reviewer's flags (one batch of calls), then the regex audit (the fact rules and the
    intent audit). Kept facts are numbered <section>:tf01…"""
    sids = list(dict.fromkeys(f["segment_id"] for f in found))
    jobs, asked = [], []
    for sid in sids:
        if prompt := fact_stage.review_prompt(segments[sid], [f for f in found if f["segment_id"] == sid]):
            jobs.append(("reviewer", reviewer, prompt, f"{sid} templates"))
            asked.append(sid)
    reviews = dict(zip(asked, llm.call_all(jobs, tolerate=failed)))
    kept_all, rejected_all = [], []
    for sid in sids:
        segment = segments[sid]
        mine = [f for f in found if f["segment_id"] == sid]
        kept, rejected = verify_facts(segment, fact_stage.apply_reviews(mine, reviews[sid].reviews if reviews.get(sid) else []), prefix="tf")
        for f in list(kept):
            if problem := template_fact_problem(f, segment, tmpls):
                kept.remove(f)
                rejected.append({**f, "reason": problem, "decided_by": "audit"})
        for n, f in enumerate(kept, start=1):
            f["fact_id"] = f"{sid}:tf{n:02d}"
        kept_all += kept
        rejected_all += rejected
    return kept_all, rejected_all


def writer_prompt(segment: dict[str, Any], tmpl: Template, facts: dict[str, dict], drafts: list[dict[str, Any]], limit: int) -> str:
    memory = "\n".join(f"- [{q['type']}, {', '.join(q['fact_ids'])}] {q['question']}" for q in drafts) or "(none yet)"
    return (f"SECTION: {segment['title']}\nDOCUMENT TYPE: {tmpl.description}\n\n{WRITER_NOTE}\n\nQUESTIONS ALREADY WRITTEN:\n"
            f"{memory}\n\nWrite at most {limit} question{'s' if limit != 1 else ''}.\n\nFACTS:\n{question_stage.listing(facts)}")


def told_why(draft: dict[str, Any], problem: str) -> str:
    """What a repair adds to the writer's input: the rejected question, the check's reason and a hint."""
    hint = HINTS.get(reason_group(problem), "")
    return QUESTION_REPAIR.format(question=draft["question"], answer=draft["answer"] or "(none)", reason=problem,
                                  hint=f"{hint}\n" if hint else "")


def repair_prompt(segment: dict[str, Any], tmpl: Template, facts: dict[str, dict], draft: dict[str, Any], problem: str) -> str:
    """The writer's input again, for the draft's facts only, with the question the audit rejected and why."""
    used = {k: facts[k] for k in draft["fact_ids"] if k in facts} or facts
    return writer_prompt(segment, tmpl, used, [], 1) + told_why(draft, problem)


def settle(units: list[tuple[str, str, dict[str, dict]]], drafts: list[list[dict[str, Any]]], reviewer: Any,
           audit: Callable[[dict, int], str | None], repair: Callable[[int, dict, str], tuple], failed: list[str]) -> list[tuple[list, list]]:
    """The tail both question stages share, one batch of calls per step: the audit, one repair of each rejected draft
    (told why), the question reviewer's flags, and the final audit. units[k] is (label, title, facts by short id) and
    drafts[k] its drafts; audit(q, k) gives a problem or None; repair(k, draft, problem) gives (role, agent, prompt,
    redraft), redraft(answer, old) being the new draft or None. Returns (kept, rejected) per unit."""
    jobs, asked = [], []
    for k, (label, _, _) in enumerate(units):
        for j, d in enumerate(drafts[k]):
            if (problem := audit(d, k)) and reason_group(problem) not in NO_REPAIR:
                role, agent, prompt, redraft = repair(k, d, problem)
                jobs.append((role, agent, prompt, label))
                asked.append((k, j, problem, redraft))
    for (k, j, problem, redraft), out in zip(asked, llm.call_all(jobs, tolerate=failed)):
        old = drafts[k][j]
        new = redraft(out, old) if out else None
        drafts[k][j] = {**(new or old), "repair_of": problem, **({"original": old["question"]} if new else {})}
    jobs, asked = [], []
    for k, (label, title, facts) in enumerate(units):
        if prompt := question_stage.review_prompt(title, drafts[k], facts):
            jobs.append(("question reviewer", reviewer, prompt, label))
            asked.append(k)
    reviews = dict(zip(asked, llm.call_all(jobs, tolerate=failed)))
    settled = []
    for k in range(len(units)):
        kept, rejected = [], []
        for q in question_stage.apply_reviews(drafts[k], reviews[k].reviews if reviews.get(k) else []):
            if problem := audit(q, k):
                rejected.append({**q, "reason": problem, "decided_by": "audit"})
            else:
                kept.append(q)
        settled.append((kept, rejected))
    return settled


def write_questions(kept: list[dict[str, Any]], segments: dict[str, dict[str, Any]], tmpls: dict[str, Template],
                    kinds: dict[str, dict], workers: dict[str, Any], reviewer: Any, corpus: str, general: set[str],
                    failed: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per (section, template): a writer per style (each sees the drafts before it), then the shared tail (settle):
    one repair of every draft the audit rejects, the question reviewer's flags and the final audit."""
    units = []  # (section, template, its facts by short id)
    for sid in dict.fromkeys(f["segment_id"] for f in kept):
        section = [f for f in kept if f["segment_id"] == sid]
        for name in dict.fromkeys(f["template"] for f in section):
            units.append((sid, name, {f"f{i}": f for i, f in enumerate(section, start=1) if f["template"] == name}))
    drafts: list[list[dict[str, Any]]] = [[] for _ in units]
    for r in range(max((len(tmpls[name].styles) for _, name, _ in units), default=0)):
        jobs, asked = [], []
        for k, (sid, name, short) in enumerate(units):
            if r < len(styles_ := list(tmpls[name].styles.items())):
                style, per_fact = styles_[r]
                jobs.append((f"{style} writer", workers[style], writer_prompt(segments[sid], tmpls[name], short, drafts[k], per_fact * len(short)),
                             f"{sid} templates"))
                asked.append((k, style, per_fact * len(short)))
        for (k, style, limit), out in zip(asked, llm.call_all(jobs, tolerate=failed)):
            drafts[k] += [{"type": style, **d.model_dump()} for d in (out.questions[:limit] if out else [])]

    def audit(q: dict[str, Any], k: int) -> str | None:
        return template_question_problem(q, units[k][2], kinds, corpus, general)

    def repair(k: int, d: dict[str, Any], problem: str) -> tuple:
        sid, name, short = units[k]
        return (f"{d['type']} writer repair", workers[d["type"]], repair_prompt(segments[sid], tmpls[name], short, d, problem),
                lambda out, old: {**out.questions[0].model_dump(), "type": old["type"]} if out.questions else None)

    labels = [(f"{sid} templates", segments[sid]["title"], short) for sid, _, short in units]
    questions, rejected = [], []
    for (sid, name, short), (good, bad) in zip(units, settle(labels, drafts, reviewer, audit, repair, failed)):
        rejected += [{**x, "segment_id": sid, "template": name, "intent": (short.get(x["fact_ids"][0], {}) if x["fact_ids"] else {}).get("intent")}
                     for x in bad]
        good = [{**q, **{k: short[q["fact_ids"][0]][k] for k in ("template", "intent", "subject")}} for q in good]
        questions += [{**q, "segment_id": sid} for q in question_stage.finish(good, short, kinds, sid)]
    count: Counter[str] = Counter()
    for q in questions:
        count[q["segment_id"]] += 1
        q["question_id"] = f"{q['segment_id']}:t{count[q['segment_id']]:02d}"
    return questions, rejected


# The source of the prompt builders and run-time schemas is fingerprinted too, so a change to their wording redoes the rows.
BUILDERS = (fact_stage.paragraph_listing, fact_stage.extractor_prompt, fact_stage.review_prompt, question_stage.listing,
            question_stage.review_prompt, map_lines, categorise_prompt, categories_schema, reader_prompt, reader_schema,
            entry_line, plan_prompt, writer_prompt, repair_prompt)


def fingerprint(segments: list[dict[str, Any]], tmpls: dict[str, Template], topics: dict[str, Any], kinds: dict[str, dict]) -> str:
    """Everything that made a PDF's row: the prompts, their builders and the schemas, the switched-on templates and
    intents, the styles' definitions, the limits, and each section's text, headings, title and override. The model
    and the settings that follow it (LLM_CONCURRENCY, ORCHESTRATOR_TEXT_CHARS) are recorded on the row, not here."""
    on = {n: t.model_dump(exclude={"intents", "enabled"}) | {"intents": {i: t.intents[i].model_dump() for i in t.intents if is_on(topics, n, t, i)}}
          for n, t in tmpls.items() if is_on(topics, n, t)}
    told = {n: {k: t[k] for k in ("prompt", "rules", "review")} for n, t in kinds.items()}
    schemas = [m.model_json_schema() for m in (Plan, fact_stage.Pointers, fact_stage.Reviews, question_stage.Drafts, question_stage.Reviews)]
    texts = [CATEGORISE_PROMPT, READER_PROMPT, PLAN_PROMPT, EXTRACTOR_PROMPT, fact_stage.REVIEWER_PROMPT, question_stage.WRITER_PROMPT,
             question_stage.REVIEWER_PROMPT, WRITER_NOTE, QUESTION_REPAIR, HINTS, sorted(NO_REPAIR)]
    return uid(TEMPLATES_VERSION, texts, [inspect.getsource(f) for f in BUILDERS], schemas, on, told, ORCHESTRATOR_SETTINGS,
               PART_CHARS, MAP_CHARS, PLAN_ENTRIES, MAP_OPENING, READ_PER_INTENT, BRIEF_CHARS, CUE_BEFORE, SUBJECT_BEFORE,
               Path(segments[0]["file_name"]).stem,
               [(s["segment_id"], fact_stage.text_hash(s["text"]), s["headings"], s["title"], topics["sections"].get(s["segment_id"]))
                for s in segments])


def pdf_units(segments: list[dict[str, Any]] | None = None) -> list[tuple[str, list[dict[str, Any]]]]:
    """Each PDF (by file name) with its sections used for questions (of data/segments.jsonl unless given)."""
    pdfs: dict[str, list[dict[str, Any]]] = {}
    for s in load_segments() if segments is None else segments:
        if s["use_for_questions"]:
            pdfs.setdefault(s["file_name"], []).append(s)
    return list(pdfs.items())


def todo(done: dict[str, dict], doc: str | None, force: bool, current: Callable[[str, list], str],
         valid: Callable[[dict, list], bool], units: list[tuple[str, list]] | None = None) -> list[tuple[str, list[dict[str, Any]]]]:
    """The units (PDFs, or groups of PDFs; each with its used sections) whose row is missing, was made from other
    inputs, has failed calls, or holds items that fail today's audit; all of them with --force. Rows of units gone
    are dropped."""
    units = pdf_units() if units is None else units
    for gone in done.keys() - {name for name, _ in units}:
        del done[gone]
    picked = [(n, segs) for n, segs in units if not doc or any(doc.lower() in s["file_name"].lower() for s in segs)]
    if doc and not picked:
        raise BenchError(f"--doc {doc!r} matches no PDF with sections used for questions")

    def stale(name: str, segs: list[dict[str, Any]]) -> bool:
        row = done.get(name)
        return force or row is None or row["fingerprint"] != current(name, segs) or bool(row["failed_calls"]) or not valid(row, segs)

    return [(n, segs) for n, segs in picked if stale(n, segs)]


def question_numbers(kept: list[dict[str, Any]], rejected: list[dict[str, Any]], calls: Counter[str]) -> dict[str, Any]:
    """The numbers both question stages report: calls (and per kept question), kept, rejected and first-try
    rejections by rule, and repairs tried and fixed by rule."""
    repairs: dict[str, dict[str, int]] = {}
    for items, fixed in ((kept, 1), (rejected, 0)):
        for q in items:
            if q.get("repair_of"):
                tally = repairs.setdefault(reason_group(q["repair_of"]), {"tried": 0, "fixed": 0})
                tally["tried"] += 1
                tally["fixed"] += fixed
    return {"calls": dict(calls.most_common()), "calls_per_kept_question": round(sum(calls.values()) / max(len(kept), 1), 2),
            "questions": {"kept": len(kept), "rejected": dict(Counter(reason_group(x["reason"]) for x in rejected)),
                          "first_try_rejections": dict(Counter(reason_group(x.get("repair_of") or x["reason"]) for x in kept + rejected
                                                               if x.get("repair_of") or "reason" in x)),
                          "repairs": repairs}}


def generation_stats(rows: list[dict[str, Any]], general: list[dict[str, Any]]) -> dict[str, Any]:
    """What the template stage spent and kept, from its rows: changes to it are judged on these, never on retrieval.
    `general` is the general questions, to count template questions whose evidence overlaps theirs."""
    entries = [e for r in rows for e in r["entries"]]
    kept = [q for r in rows for q in r["questions"]]
    spans: dict[str, list[tuple[int, int]]] = {}
    for q in general:
        for ev in q["evidence"]:
            spans.setdefault(ev["segment_id"], []).append((ev["start"], ev["end"]))
    return {
        "pdfs": len(rows), "failed_calls": sum(len(r["failed_calls"]) for r in rows),
        **question_numbers(kept, [x for r in rows for x in r["rejected"]], sum((Counter(r["calls"]) for r in rows), Counter())),
        "sections_with_templates": dict(Counter(c["by"] for r in rows for c in r["sections"].values() if c["templates"])),
        "entries": len(entries), "entries_dropped": dict(Counter(e["dropped"] for e in entries if e.get("dropped"))),
        "picks": dict(Counter(e["pick"]["by"] for e in entries if e.get("pick"))),
        "not_picked": sum("pick" in e and e["pick"] is None for e in entries),
        "facts": {"kept": sum(len(r["facts"]) for r in rows),
                  "rejected": dict(Counter(reason_group(x["reason"]) for r in rows for x in r["rejected_facts"]))},
        "cells": len({(q["template"], q["intent"], q["subject"].lower()) for q in kept}),
        "overlap_with_general": sum(any(a < ev["end"] and b > ev["start"] for ev in q["evidence"] for a, b in spans.get(ev["segment_id"], []))
                                    for q in kept),
    }


def run_templates(doc: str | None = None, force: bool = False) -> None:
    """Template facts and questions for every PDF, planned by the orchestrator -> data/templates.jsonl."""
    tmpls, topics, kinds, corpus, general = inputs()
    done = {r["file_name"]: r for r in pdf_rows()}
    extractor, fact_reviewer = llm.agent(fact_stage.Pointers, EXTRACTOR_PROMPT), llm.agent(fact_stage.Reviews, fact_stage.REVIEWER_PROMPT)
    writers = {n: llm.agent(question_stage.Drafts, question_stage.WRITER_PROMPT + f"{n}: {t['prompt']}") for n, t in kinds.items()}
    described = "\n".join(f"- {n}: {t['review'] or ' '.join(t['prompt'].split())}" for n, t in kinds.items())
    question_reviewer = llm.agent(question_stage.Reviews, question_stage.REVIEWER_PROMPT + described)
    planner = llm.agent(Plan, PLAN_PROMPT)

    def current(name: str, segs: list[dict[str, Any]]) -> str:
        return fingerprint(segs, tmpls, topics, kinds)

    def valid(row: dict[str, Any], segs: list[dict[str, Any]]) -> bool:
        return not row_problems(row, {s["segment_id"]: s for s in segs}, tmpls, kinds, corpus, general)

    def work(unit: tuple[str, list[dict[str, Any]]]) -> None:
        (file_name, segs), before, failed = unit, Counter(llm.CALLS), []
        by_id = {s["segment_id"]: s for s in segs}
        sections, views = categorise(segs, tmpls, topics, failed)
        entries = read(segs, {sid: active(s, tmpls, topics, sections) for sid, s in by_id.items()}, tmpls, failed)
        plans = plan(file_name, segs, entries, tmpls, planner, failed)
        found = extract([e for e in entries if e.get("pick")], by_id, tmpls, extractor, failed)
        kept, rejected_facts = audit_facts(found, by_id, tmpls, fact_reviewer, failed)
        questions, rejected = write_questions(kept, by_id, tmpls, kinds, writers, question_reviewer, corpus, general, failed)
        s = settings()
        done[file_name] = {"file_name": file_name, "model": s.llm_model, "version": TEMPLATES_VERSION, "fingerprint": current(file_name, segs),
                           "concurrency": s.concurrency, "orchestrator_text_chars": s.orchestrator_text_chars,
                           "calls": dict(llm.CALLS - before), "failed_calls": failed, "categorise_views": views,
                           "sections": sections, "entries": entries, "plans": plans, "facts": kept,
                           "rejected_facts": rejected_facts, "questions": questions, "rejected": rejected}
        write_jsonl(ROWS, list(done.values()))  # saved after every PDF, so a stop loses little
        fact_stage.audit_line(file_name, "template facts", kept, rejected_facts)
        fact_stage.audit_line(file_name, "template questions", questions, rejected)

    saved = [r.get("file_name") for r in read_jsonl(ROWS)] if ROWS.exists() else []
    stale = todo(done, doc, force, current, valid)
    if saved != list(done):  # rows in an older format, or of PDFs that are gone
        write_jsonl(ROWS, list(done.values()))
    fact_stage.run_each(stale, work, "templates run", name=lambda unit: unit[0])
    general_questions = [q for row in question_stage.question_rows() for q in row["questions"]]
    numbers = generation_stats(list(done.values()), general_questions)
    llm.console.print(f"  [bold]templates[/] ▸ {numbers['pdfs']} PDFs: {numbers['entries']} entries "
                      f"({sum(numbers['entries_dropped'].values())} dropped by code), {sum(numbers['picks'].values())} picked, "
                      f"{numbers['facts']['kept']} facts and {numbers['questions']['kept']} questions kept; "
                      f"{sum(r['fixed'] for r in numbers['questions']['repairs'].values())} of "
                      f"{sum(r['tried'] for r in numbers['questions']['repairs'].values())} repairs passed; "
                      f"{numbers['calls_per_kept_question']} calls per kept question"
                      + (f"; [yellow]{numbers['failed_calls']} calls gave no readable answer and are retried on the next run[/]"
                         if numbers["failed_calls"] else ""))


# General questions (`bench questions`): per group of PDFs, the orchestrator plans which question each fact or pair gets.

def question_units(topics: dict[str, Any], segments: list[dict[str, Any]] | None = None) -> list[tuple[str, list[dict[str, Any]]]]:
    """Each group of PDFs whose facts multi-hop questions may pair, with its sections used for questions: the groups in
    data/topics.json, and every other PDF on its own (named by its file)."""
    member = {pdf: name for name, pdfs in topics["groups"].items() for pdf in pdfs}
    units: dict[str, list[dict[str, Any]]] = {}
    for file_name, segs in pdf_units(segments):
        units.setdefault(member.get(file_name, file_name), []).extend(segs)
    return list(units.items())


def material(segs: list[dict[str, Any]], facts_rows: dict[str, dict], lists: dict[str, list[str]],
             absent: dict[str, list[str]]) -> dict[str, Any]:
    """A group's verified facts with ids g1… in document order, their sections, and what code builds for the plan:
    pairs of facts sharing a specific word, the word lists each fact names, and the absent terms of those lists."""
    sections = [s for s in segs if facts_rows.get(s["segment_id"], {}).get("facts")]
    facts = {f"g{n}": f for n, f in enumerate((f for s in sections for f in facts_rows[s["segment_id"]]["facts"]), start=1)}
    named = {g: question_stage.named(f, lists) for g, f in facts.items()}
    return {"sections": sections, "facts": facts, "pdf_of": {s["segment_id"]: s["file_name"] for s in segs},
            "pairs": question_stage.pairs(facts, lists), "named": named,
            "absent": {lst: absent[lst] for lst in sorted({lst for found in named.values() for lst in found}) if absent.get(lst)}}


def question_plan_schema(types: list[str]) -> type[BaseModel]:
    """The question plan's answer, its type limited to the active question types."""
    item = create_model("PlannedQuestion", type=(Literal[tuple(types)], ...),  # type: ignore[valid-type]
                        ref=(str, Field(description="a fact id (g7), or a pair id (p4) for a type that needs two facts")),
                        absent_term=(str, ...), brief=(str, ...))
    return create_model("QuestionPlan", items=(list[item], ...))  # type: ignore[valid-type]


def question_plan_prompt(name: str, types: dict[str, dict], mat: dict[str, Any], sections: list[dict[str, Any]] | None = None,
                         pairs: list[dict[str, Any]] | None = None) -> str:
    """The plan's input: the types, then either some sections' facts with their quotas and the absent terms (one-fact
    types), or some candidate pairs with both facts' claims (types that need two facts)."""
    def need(t: dict[str, Any]) -> str:
        return ({"pair": " [a pair of one section]", "cross": " [a pair of two sections]",
                 "absent": " [a fact that names a word list, and one of its absent terms]"}.get(question_stage.needs(t), "")
                + (" [a fact longer than a sentence]" if t["source"] == "paragraph" else "")
                + (" [a fact whose quote holds a number]" if t["rules"].get("needs_number") else ""))

    def quota(mine: list[str]) -> str:  # per one-fact type, the questions the section's facts allow (at most its count)
        allowed = {n: min(t["count"], sum(any(mat["absent"].get(lst) for lst in mat["named"][g]) if question_stage.needs(t) == "absent"
                                          else question_stage.fits(t, mat["facts"][g]) for g in mine))
                   for n, t in types.items() if question_stage.needs(t) in ("fact", "absent")}
        return " | QUOTA: " + ", ".join(f"{n} {k}" for n, k in allowed.items() if k) if any(allowed.values()) else ""

    head = f"DOCUMENTS: {name}\n\nTYPES:\n" + "\n".join(
        f"- {n} (at most {t['count']} per {'PDF' if question_stage.needs(t) == 'cross' else 'section'}){need(t)}: {' '.join(t['prompt'].split())}"
        for n, t in types.items())
    if pairs is not None:
        titles, lines = {s["segment_id"]: s["title"] for s in mat["sections"]}, []
        for p in pairs:
            a, b = mat["facts"][p["a"]], mat["facts"][p["b"]]
            where = "; ".join(dict.fromkeys(titles[question_stage.section(f)] for f in (a, b)))
            lines.append(f"{p['id']} {'two sections' if p['cross'] else 'one section'} ({where}) | shared: {p['word']}\n"
                         f"  {p['a']}: {a['claim']}\n  {p['b']}: {b['claim']}")
        return f"{head}\n\nPAIRS:\n" + "\n".join(lines)
    lines = []
    for s in sections or []:
        mine = [g for g, f in mat["facts"].items() if question_stage.section(f) == s["segment_id"]]
        lines += [f"{s['segment_id']} {s['title']}{quota(mine)}"] if mine else []
        for g in mine:
            f = mat["facts"][g]
            marks = [f.get("size", "sentence"), *(["number"] if NUMBER.search(evidence_text(f)) else []),
                     *(f"names {term} ({lst})" for lst, term in mat["named"][g].items() if lst in mat["absent"])]
            lines.append(f"{g} | {' | '.join(marks)} | {f['claim']}")
    absent = "\n".join(f"- {lst}: {', '.join(terms)}" for lst, terms in mat["absent"].items())
    return head + (f"\n\nABSENT TERMS (the library never mentions them):\n{absent}" if absent else "") + "\n\nFACTS:\n" + "\n".join(lines)


class Tally:
    """What a group's plan has taken so far, so its caps hold across its calls: each type's count per section (per PDF,
    by the first fact, for pairs across sections), PER_FACT one-fact questions per fact of different types, ABSENT_USES
    per absent term and PDF, and no item twice."""

    def __init__(self, types: dict[str, dict], mat: dict[str, Any]) -> None:
        self.types, self.mat = types, mat
        self.count: Counter[tuple] = Counter()
        self.per_fact: Counter[str] = Counter()
        self.fact_types: defaultdict[str, set[str]] = defaultdict(set)
        self.seen: set[tuple] = set()

    def take(self, t: str, facts: list[str], absent: str = "") -> str | None:
        """Take the item if the caps allow it (None), else say why not."""
        need, sid = question_stage.needs(self.types[t]), question_stage.section(self.mat["facts"][facts[0]])
        pdf, one = self.mat["pdf_of"][sid], need in ("fact", "absent")
        unit, term, key = ("type", pdf if need == "cross" else sid, t), ("absent", pdf, absent.lower()), (t, tuple(facts), absent.lower())
        why = ("planned twice" if key in self.seen else
               "over the type's count" if self.count[unit] >= self.types[t]["count"] else
               "the fact has its questions" if one and (self.per_fact[facts[0]] >= PER_FACT or t in self.fact_types[facts[0]]) else
               "the absent term is used up" if absent and self.count[term] >= ABSENT_USES else None)
        if why is None:
            self.seen.add(key)
            self.count[unit] += 1
            if one:
                self.per_fact[facts[0]] += 1
                self.fact_types[facts[0]].add(t)
            if absent:
                self.count[term] += 1
        return why


def check_items(answer: Any, types: dict[str, dict], mat: dict[str, Any], tally: Tally) -> tuple[list[dict], list[dict]]:
    """The plan's items that code accepts, in order, and the rest with why: a type planned in this call, a ref to a fact
    or pair that suits it (a leading id is enough), an absent term of a list the fact names, and the caps (tally)."""
    pairs = {p["id"]: p for p in mat["pairs"]}
    items, invalid = [], []
    for raw in answer.items:
        t, found = types.get(raw.type), re.match(r"\W*([gp]\d+)\b", raw.ref)
        ref, need = found.group(1) if found else "", question_stage.needs(t) if t else ""
        facts: list[str] = []
        absent = replaces = ""
        if t is None:
            why = "a type this call doesn't plan"
        elif need in ("pair", "cross"):
            p = pairs.get(ref)
            why = "no such pair" if p is None else "a pair of the wrong kind" if p["cross"] != (need == "cross") else None
            facts = [p["a"], p["b"]] if p else []
        elif ref not in mat["facts"]:
            why = "no such fact"
        elif not question_stage.fits(t, mat["facts"][ref]):
            why = "the fact doesn't suit the type"
        else:
            why, facts = None, [ref]
            if need == "absent":
                wanted = " ".join(raw.absent_term.split()).lower()
                hit = next(((word, a) for lst, word in mat["named"][ref].items() for a in mat["absent"].get(lst, []) if a.lower() == wanted), None)
                if hit:
                    replaces, absent = hit
                else:
                    why = "the absent term isn't one for this fact"
        if why := why or tally.take(raw.type, facts, absent):
            invalid.append({"type": raw.type, "ref": raw.ref[:40], "why": why})
        else:
            items.append({"type": raw.type, "facts": facts, "absent_term": absent, "replaces": replaces,
                          "brief": cut(raw.brief) if raw.brief.strip() else "", "by": "orchestrator"})
    return items, invalid


def code_items(types: dict[str, dict], mat: dict[str, Any], tally: Tally, pack: list[dict[str, Any]]) -> list[dict]:
    """Code's plan for some sections: per one-fact type, the eligible facts with the fewest questions so far (then in
    document order), and absent terms in list order, up to the caps. No pairs: code can't judge whether two facts make
    one question."""
    items: list[dict] = []
    order = {g: n for n, g in enumerate(mat["facts"])}

    def add(t: str, facts: list[str], absent: str = "", replaces: str = "") -> None:
        if tally.take(t, facts, absent) is None:
            items.append({"type": t, "facts": facts, "absent_term": absent, "replaces": replaces, "brief": "", "by": "code"})

    for s in pack:
        mine = [g for g, f in mat["facts"].items() if question_stage.section(f) == s["segment_id"]]
        for name, t in types.items():
            need = question_stage.needs(t)
            if need == "absent":
                for g in mine:
                    for lst, word in mat["named"][g].items():
                        for term in mat["absent"].get(lst, []):
                            add(name, [g], term, word)
            elif need == "fact":
                for g in sorted((g for g in mine if question_stage.fits(t, mat["facts"][g])), key=lambda g: (tally.per_fact[g], order[g])):
                    add(name, [g])
    return items


def plan_questions(name: str, mat: dict[str, Any], types: dict[str, dict], planner: Any,
                   failed: list[str]) -> tuple[list[dict], list[dict], list[dict]]:
    """The group's items: one-fact types planned in groups of sections (QUESTION_PLAN_FACTS facts per call), then the
    types that need two facts, from the candidate pairs (QUESTION_PLAN_PAIRS per call). A call that gives nothing usable
    is planned by code (one-fact types only), and so is a section the plan leaves out. Returns (items i1…, invalid
    items with why, a record per call)."""
    tally, facts_in = Tally(types, mat), Counter(question_stage.section(f) for f in mat["facts"].values())
    one = {n: t for n, t in types.items() if question_stage.needs(t) in ("fact", "absent")}
    two = {n: t for n, t in types.items() if question_stage.needs(t) in ("pair", "cross")}
    kinds = {question_stage.needs(t) for t in two.values()}
    candidates = [p for p in mat["pairs"] if ("cross" if p["cross"] else "pair") in kinds]
    calls: list[tuple[dict, list | None, list | None]] = [
        (one, pack, None) for pack in groups(mat["sections"], lambda s: facts_in[s["segment_id"]], QUESTION_PLAN_FACTS)] if one else []
    calls += [(two, None, candidates[k : k + QUESTION_PLAN_PAIRS]) for k in range(0, len(candidates), QUESTION_PLAN_PAIRS)]
    jobs = [("orchestrator plan", planner, question_plan_prompt(name, call_types, mat, pack, chunk), f"{name} plan {k}/{len(calls)}")
            for k, (call_types, pack, chunk) in enumerate(calls, start=1)]
    items, invalid, record = [], [], []
    for (call_types, pack, chunk), answer in zip(calls, llm.call_all(jobs, tolerate=failed, model_settings=ORCHESTRATOR_SETTINGS)):
        mine, bad = check_items(answer, call_types, mat, tally) if answer else ([], [])
        by = "orchestrator" if answer and (mine or not bad) else "code"
        if by == "code":
            mine = code_items(call_types, mat, tally, pack) if pack is not None else []
        elif pack is not None:
            covered = {question_stage.section(mat["facts"][it["facts"][0]]) for it in mine}
            mine += code_items(call_types, mat, tally, [s for s in pack if s["segment_id"] not in covered])
        items += mine
        invalid += bad
        record.append(({"sections": [s["segment_id"] for s in pack]} if pack is not None else {"pairs": [p["id"] for p in chunk or []]})
                      | {"by": by, "items": len(mine), "invalid": len(bad)})
    items = [{"id": f"i{n}", **it, "fact_ids": [mat["facts"][g]["fact_id"] for g in it["facts"]]} for n, it in enumerate(items, start=1)]
    return items, invalid, record


def item_prompt(title: str, items: list[dict[str, Any]], mat: dict[str, Any]) -> str:
    """A writer's input: each item with its facts' ids, any absent term (and the word it replaces) and brief, then those
    facts (with their sections' titles when they come from several sections)."""
    used = {g: mat["facts"][g] for it in items for g in it["facts"]}
    titles = {s["segment_id"]: s["title"] for s in mat["sections"]}
    several = len({question_stage.section(f) for f in used.values()}) > 1
    lines = [f"- {it['id']}: facts {', '.join(it['facts'])}"
             + (f" | absent term: {it['absent_term']} (in place of {it['replaces']})" if it["absent_term"] else "")
             + (f" | brief: {it['brief']}" if it["brief"] else "") for it in items]
    return f"SECTION: {title}\n\nITEMS:\n" + "\n".join(lines) + f"\n\nFACTS:\n{question_stage.listing(used, titles if several else None)}"


def drafts_from(answer: Any, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The writer's questions as drafts of their items: code attaches each item's type, facts and absent term. An item
    answered twice keeps its first question; with one item, an answer that names no item is taken as its."""
    by_id, drafts = {it["id"]: it for it in batch}, []
    for d in answer.questions if answer else []:
        found = re.match(r"\W*(i\d+)\b", d.item)
        it = by_id.get(found.group(1)) if found else batch[0] if len(batch) == 1 else None
        if it is None or any(x["item"] == it["id"] for x in drafts):
            continue
        drafts.append({"type": it["type"], "question": d.question, "answer": "" if it["absent_term"] else d.answer,
                       "fact_ids": list(it["facts"]), "absent_term": it["absent_term"], "item": it["id"]})
    return drafts


def write_items(name: str, items: list[dict[str, Any]], mat: dict[str, Any], types: dict[str, dict], writers: dict[str, Any],
                reviewer: Any, corpus: str, failed: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One writer call per type and section (per type for the pairs across sections), all in one batch, then the shared
    tail (settle): one repair, the reviewer's flags, the final audit. Ids are <section>:q01, and <first fact's
    section>:h01 for pairs across sections; every item records its outcome (a question id, why it was rejected, or
    unfilled)."""
    def unit_of(it: dict[str, Any]) -> str:
        return "" if question_stage.needs(types[it["type"]]) == "cross" else question_stage.section(mat["facts"][it["facts"][0]])

    titles = {s["segment_id"]: s["title"] for s in mat["sections"]}
    batches: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for it in items:
        batches.setdefault((it["type"], unit_of(it)), []).append(it)
    keys = list(dict.fromkeys(u for _, u in batches))
    jobs = [(f"{t} writer", writers[t], item_prompt(titles.get(u, "several sections"), batch, mat), f"{name} {u or 'multi-hop'}")
            for (t, u), batch in batches.items()]
    drafts: dict[str, list[dict[str, Any]]] = {u: [] for u in keys}
    for ((_, u), batch), answer in zip(batches.items(), llm.call_all(jobs, tolerate=failed)):
        drafts[u] += drafts_from(answer, batch)
    by_id = {it["id"]: it for it in items}
    units = [(f"{name} {u or 'multi-hop'}", titles.get(u, "several sections"), mat["facts"]) for u in keys]

    def audit(q: dict[str, Any], k: int) -> str | None:
        return question_problem(q, mat["facts"], types[q["type"]]["rules"], corpus)

    def repair(k: int, d: dict[str, Any], problem: str) -> tuple:
        return (f"{d['type']} writer repair", writers[d["type"]], item_prompt(units[k][1], [by_id[d["item"]]], mat) + told_why(d, problem),
                lambda answer, old: {**old, "question": answer.questions[0].question,
                                     "answer": "" if old["absent_term"] else answer.questions[0].answer} if answer.questions else None)

    questions, rejected = [], []
    for u, (kept, bad) in zip(keys, settle(units, [drafts[u] for u in keys], reviewer, audit, repair, failed)):
        done = question_stage.finish(kept, mat["facts"], types, u or "multi-hop")
        if u:
            questions += [{**q, "segment_id": u} for q in done]
        else:
            count: Counter[str] = Counter()
            for q in done:
                first = q["fact_ids"][0].rsplit(":", 1)[0]
                count[first] += 1
                q["question_id"] = f"{first}:h{count[first]:02d}"
                q["across_pdfs"] = len({mat["pdf_of"][ev["segment_id"]] for ev in q["evidence"]}) > 1
            questions += done
        rejected += [{**x, "segment_id": u or None, "fact_ids": [mat["facts"][g]["fact_id"] for g in x["fact_ids"] if g in mat["facts"]]}
                     for x in bad]
    outcome = {q["item"]: q["question_id"] for q in questions} | {x["item"]: x["reason"] for x in rejected}
    for it in items:
        it["outcome"] = outcome.get(it["id"], "unfilled")
    return questions, rejected


# The source of the builders and of the plan's checks is fingerprinted too, so a change to them redoes the rows.
Q_BUILDERS = (question_stage.listing, question_stage.review_prompt, question_stage.needs, question_stage.fits,
              question_stage.pairs, question_stage.named, question_stage.absent_terms, material, question_plan_schema,
              question_plan_prompt, Tally, check_items, code_items, item_prompt, drafts_from, told_why)


def question_fingerprint(name: str, mat: dict[str, Any], types: dict[str, dict]) -> str:
    """Everything that made a group's row: the prompts, their builders, the plan's checks and the schemas, the question
    types in full, the limits, the member PDFs and their sections, and the group's facts and candidates (pairs, named
    word lists, absent terms). The model and LLM_CONCURRENCY are recorded on the row, not here."""
    return uid(question_stage.QUESTIONS_VERSION,
               [QUESTION_PLAN_PROMPT, question_stage.ITEM_WRITER_PROMPT, question_stage.REVIEWER_PROMPT, QUESTION_REPAIR, HINTS, sorted(NO_REPAIR)],
               [inspect.getsource(f) for f in Q_BUILDERS], [m.model_json_schema() for m in (question_stage.ItemDrafts, question_stage.Reviews)],
               types, ORCHESTRATOR_SETTINGS, BRIEF_CHARS, QUESTION_PLAN_FACTS, QUESTION_PLAN_PAIRS, PER_FACT, ABSENT_USES, question_stage.PAIR_TERM_FACTS,
               question_stage.PAIR_PARTNERS, name, sorted(set(mat["pdf_of"].values())), [(s["segment_id"], s["title"]) for s in mat["sections"]],
               {g: f["uid"] for g, f in mat["facts"].items()}, mat["pairs"], mat["named"], mat["absent"])


def question_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What the general question stage planned, spent and kept, from its rows: judged on these, never on retrieval."""
    items, kept = [it for r in rows for it in r["plan"]["items"]], [q for r in rows for q in r["questions"]]
    used = Counter(f for q in kept for f in q["fact_ids"])
    hops = [q for q in kept if "across_pdfs" in q]
    return {
        "groups": len(rows), "failed_calls": sum(len(r["failed_calls"]) for r in rows),
        **question_numbers(kept, [x for r in rows for x in r["rejected"]], sum((Counter(r["calls"]) for r in rows), Counter())),
        "items": dict(Counter(it["type"] for it in items)), "items_by": dict(Counter(it["by"] for it in items)),
        "invalid": dict(Counter(x["why"] for r in rows for x in r["plan"]["invalid"])),
        "unfilled": dict(Counter(it["type"] for it in items if it.get("outcome") == "unfilled")),
        "kept_by_type": dict(Counter(q["type"] for q in kept)),
        "sections_covered": len({q["segment_id"] for q in kept if q.get("segment_id")}),
        "facts_used": len(used), "most_questions_on_one_fact": max(used.values(), default=0),
        "absent_terms": sorted({q["absent_term"] for q in kept if q.get("absent_term")}),
        "multi_hop": {"within one PDF": sum(not q["across_pdfs"] for q in hops), "across PDFs": sum(q["across_pdfs"] for q in hops)},
    }


def run_questions(doc: str | None = None, force: bool = False) -> None:
    """General questions for every group of PDFs, planned by the orchestrator -> data/questions.jsonl."""
    types, topics, lists = question_types(), load_topics(), question_stage.vocab()
    corpus = "\n".join(p["text"] for p in read_jsonl(PAGES)) if PAGES.exists() else ""
    facts_rows, absent = fact_stage.load_rows(fact_stage.FACTS), question_stage.absent_terms(lists, corpus)
    done = {r["group"]: r for r in question_stage.question_rows()}
    writers = {n: llm.agent(question_stage.ItemDrafts, question_stage.ITEM_WRITER_PROMPT + f"{n}: {t['prompt']}") for n, t in types.items()}
    described = "\n".join(f"- {n}: {t['review'] or ' '.join(t['prompt'].split())}" for n, t in types.items())
    reviewer = llm.agent(question_stage.Reviews, question_stage.REVIEWER_PROMPT + described)
    planner = llm.agent(question_plan_schema(list(types)), QUESTION_PLAN_PROMPT) if types else None
    materials: dict[str, dict[str, Any]] = {}

    def mat_of(name: str, segs: list[dict[str, Any]]) -> dict[str, Any]:
        if name not in materials:
            materials[name] = material(segs, facts_rows, lists, absent)
        return materials[name]

    def current(name: str, segs: list[dict[str, Any]]) -> str:
        return question_fingerprint(name, mat_of(name, segs), types)

    def valid(row: dict[str, Any], segs: list[dict[str, Any]]) -> bool:
        facts = {f["fact_id"]: f for s in segs for f in facts_rows.get(s["segment_id"], {}).get("facts", [])}
        return all(q["type"] in types and question_problem(q, facts, types[q["type"]]["rules"], corpus) is None for q in row["questions"])

    def work(unit: tuple[str, list[dict[str, Any]]]) -> None:
        (name, segs), before, failed = unit, Counter(llm.CALLS), []
        mat = mat_of(name, segs)
        items, invalid, record = plan_questions(name, mat, types, planner, failed) if types else ([], [], [])
        questions, rejected = write_items(name, items, mat, types, writers, reviewer, corpus, failed)
        s = settings()
        done[name] = {"group": name, "pdfs": sorted(set(mat["pdf_of"].values())), "version": question_stage.QUESTIONS_VERSION,
                      "model": s.llm_model, "fingerprint": current(name, segs), "concurrency": s.concurrency,
                      "calls": dict(llm.CALLS - before), "failed_calls": failed,
                      "plan": {"calls": record, "items": items, "invalid": invalid}, "questions": questions, "rejected": rejected}
        write_jsonl(question_stage.QUESTIONS, list(done.values()))  # saved after every group, so a stop loses little
        fact_stage.audit_line(name, "questions", questions, rejected)

    saved = [r.get("group") for r in read_jsonl(question_stage.QUESTIONS)] if question_stage.QUESTIONS.exists() else []
    stale = todo(done, doc, force, current, valid, question_units(topics))
    if saved != list(done):  # rows in an older format, or of groups that are gone
        write_jsonl(question_stage.QUESTIONS, list(done.values()))
    fact_stage.run_each(stale, work, "questions", name=lambda unit: unit[0])
    numbers = question_stats(list(done.values()))
    repairs = numbers["questions"]["repairs"].values()
    llm.console.print(f"  [bold]questions[/] ▸ {numbers['groups']} groups: {sum(numbers['items'].values())} items planned "
                      f"{numbers['items_by']}, {sum(numbers['invalid'].values())} invalid; {numbers['questions']['kept']} questions kept, "
                      f"{sum(numbers['unfilled'].values())} items unfilled; {sum(r['fixed'] for r in repairs)} of "
                      f"{sum(r['tried'] for r in repairs)} repairs passed; {numbers['calls_per_kept_question']} calls per kept question"
                      + (f"; [yellow]{numbers['failed_calls']} calls gave no readable answer and are retried on the next run[/]"
                         if numbers["failed_calls"] else ""))
