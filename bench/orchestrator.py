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
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from bench import facts as fact_stage
from bench import llm
from bench import questions as question_stage
from bench.config import PART_CHARS, REQUEST_TIMEOUT, BenchError, settings
from bench.pdf import read_jsonl, write_jsonl
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
    names,
    near,
    pdf_rows,
    row_problems,
    scores,
    subject_window,
    template_fact_problem,
    template_question_problem,
)
from bench.verify import reason_group, uid, verify_facts

MAP_OPENING = 200  # characters of each section's opening shown in the map
MAP_CHARS = 40_000  # a longer map of sections is categorised in consecutive groups of sections
PLAN_ENTRIES = 100  # a plan covers at most this many entries, so its answer stays short; more are planned in groups
READ_PER_INTENT = 3  # entries a reader may give per intent in one part
BRIEF_CHARS = 300  # a brief is cut to this length
ORCHESTRATOR_SETTINGS = {"max_tokens": 8_000, "request_timeout": 3 * REQUEST_TIMEOUT}  # a long input, a whole plan out

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
                         {"templates": by_rules(s, tmpls), "by": "rules"})
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


def repair_prompt(segment: dict[str, Any], tmpl: Template, facts: dict[str, dict], draft: dict[str, Any], problem: str) -> str:
    """The writer's input again, for the draft's facts only, with the question the audit rejected and why."""
    used = {k: facts[k] for k in draft["fact_ids"] if k in facts} or facts
    hint = HINTS.get(reason_group(problem), "")
    return writer_prompt(segment, tmpl, used, [], 1) + QUESTION_REPAIR.format(
        question=draft["question"], answer=draft["answer"] or "(none)", reason=problem, hint=f"{hint}\n" if hint else "")


def write_questions(kept: list[dict[str, Any]], segments: dict[str, dict[str, Any]], tmpls: dict[str, Template],
                    kinds: dict[str, dict], workers: dict[str, Any], reviewer: Any, corpus: str, general: set[str],
                    failed: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per (section, template): a writer per style (each sees the drafts before it), one repair of every draft the
    audit rejects, the question reviewer's flags and the final audit. Each step is one batch of calls."""
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
    jobs, asked = [], []
    for k, (sid, name, short) in enumerate(units):
        for j, d in enumerate(drafts[k]):
            if (problem := template_question_problem(d, short, kinds, corpus, general)) and reason_group(problem) not in NO_REPAIR:
                jobs.append((f"{d['type']} writer repair", workers[d["type"]], repair_prompt(segments[sid], tmpls[name], short, d, problem),
                             f"{sid} templates"))
                asked.append((k, j, problem))
    for (k, j, problem), out in zip(asked, llm.call_all(jobs, tolerate=failed)):
        old = drafts[k][j]
        new = out.questions[0].model_dump() if out and out.questions else None
        drafts[k][j] = {**(new or old), "type": old["type"], "repair_of": problem, **({"original": old["question"]} if new else {})}
    jobs, asked = [], []
    for k, (sid, _, short) in enumerate(units):
        if prompt := question_stage.review_prompt(segments[sid]["title"], drafts[k], short):
            jobs.append(("question reviewer", reviewer, prompt, f"{sid} templates"))
            asked.append(k)
    reviews = dict(zip(asked, llm.call_all(jobs, tolerate=failed)))
    questions, rejected = [], []
    for k, (sid, name, short) in enumerate(units):
        good = []
        for q in question_stage.apply_reviews(drafts[k], reviews[k].reviews if reviews.get(k) else []):
            first = short.get(q["fact_ids"][0], {}) if q["fact_ids"] else {}
            if problem := template_question_problem(q, short, kinds, corpus, general):
                rejected.append({**q, "reason": problem, "decided_by": "audit", "segment_id": sid, "template": name, "intent": first.get("intent")})
            else:
                good.append({**q, "template": first["template"], "intent": first["intent"], "subject": first["subject"]})
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


def todo(done: dict[str, dict], doc: str | None, force: bool, current: Callable[[list], str],
         valid: Callable[[dict, list], bool]) -> list[list[dict[str, Any]]]:
    """The PDFs (each a list of its used sections) whose row is missing, was made from other inputs, has failed
    calls, or holds items that fail today's audit; all of them with --force. Rows of PDFs gone are dropped."""
    pdfs: dict[str, list[dict[str, Any]]] = {}
    for s in load_segments():
        if s["use_for_questions"]:
            pdfs.setdefault(s["file_name"], []).append(s)
    for gone in done.keys() - pdfs.keys():
        del done[gone]
    picked = {n: segs for n, segs in pdfs.items() if not doc or doc.lower() in n.lower()}
    if doc and not picked:
        raise BenchError(f"--doc {doc!r} matches no PDF with sections used for questions")

    def stale(name: str, segs: list[dict[str, Any]]) -> bool:
        row = done.get(name)
        return force or row is None or row["fingerprint"] != current(segs) or bool(row["failed_calls"]) or not valid(row, segs)

    return [segs for name, segs in picked.items() if stale(name, segs)]


def generation_stats(rows: list[dict[str, Any]], general: list[dict[str, Any]]) -> dict[str, Any]:
    """What the template stage spent and kept, from its rows: changes to it are judged on these, never on retrieval.
    `general` is the general questions, to count template questions whose evidence overlaps theirs."""
    entries = [e for r in rows for e in r["entries"]]
    kept, rejected = [q for r in rows for q in r["questions"]], [x for r in rows for x in r["rejected"]]
    calls = sum((Counter(r["calls"]) for r in rows), Counter())
    spans: dict[str, list[tuple[int, int]]] = {}
    for q in general:
        for ev in q["evidence"]:
            spans.setdefault(ev["segment_id"], []).append((ev["start"], ev["end"]))
    repairs: dict[str, dict[str, int]] = {}
    for items, fixed in ((kept, 1), (rejected, 0)):
        for q in items:
            if q.get("repair_of"):
                tally = repairs.setdefault(reason_group(q["repair_of"]), {"tried": 0, "fixed": 0})
                tally["tried"] += 1
                tally["fixed"] += fixed
    return {
        "pdfs": len(rows), "failed_calls": sum(len(r["failed_calls"]) for r in rows),
        "calls": dict(calls.most_common()), "calls_per_kept_question": round(sum(calls.values()) / max(len(kept), 1), 2),
        "sections_with_templates": dict(Counter(c["by"] for r in rows for c in r["sections"].values() if c["templates"])),
        "entries": len(entries), "entries_dropped": dict(Counter(e["dropped"] for e in entries if e.get("dropped"))),
        "picks": dict(Counter(e["pick"]["by"] for e in entries if e.get("pick"))),
        "not_picked": sum("pick" in e and e["pick"] is None for e in entries),
        "facts": {"kept": sum(len(r["facts"]) for r in rows),
                  "rejected": dict(Counter(reason_group(x["reason"]) for r in rows for x in r["rejected_facts"]))},
        "questions": {"kept": len(kept), "rejected": dict(Counter(reason_group(x["reason"]) for x in rejected)),
                      "first_try_rejections": dict(Counter(reason_group(x.get("repair_of") or x["reason"]) for x in kept + rejected
                                                           if x.get("repair_of") or "reason" in x)),
                      "repairs": repairs},
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

    def current(segs: list[dict[str, Any]]) -> str:
        return fingerprint(segs, tmpls, topics, kinds)

    def valid(row: dict[str, Any], segs: list[dict[str, Any]]) -> bool:
        return not row_problems(row, {s["segment_id"]: s for s in segs}, tmpls, kinds, corpus, general)

    def work(segs: list[dict[str, Any]]) -> None:
        file_name, before, failed = segs[0]["file_name"], Counter(llm.CALLS), []
        by_id = {s["segment_id"]: s for s in segs}
        sections, views = categorise(segs, tmpls, topics, failed)
        entries = read(segs, {sid: active(s, tmpls, topics, sections) for sid, s in by_id.items()}, tmpls, failed)
        plans = plan(file_name, segs, entries, tmpls, planner, failed)
        found = extract([e for e in entries if e.get("pick")], by_id, tmpls, extractor, failed)
        kept, rejected_facts = audit_facts(found, by_id, tmpls, fact_reviewer, failed)
        questions, rejected = write_questions(kept, by_id, tmpls, kinds, writers, question_reviewer, corpus, general, failed)
        s = settings()
        done[file_name] = {"file_name": file_name, "model": s.llm_model, "version": TEMPLATES_VERSION, "fingerprint": current(segs),
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
    fact_stage.run_each(stale, work, "templates run", name=lambda segs: segs[0]["file_name"])
    general_questions = [q for row in fact_stage.load_rows(question_stage.QUESTIONS).values() for q in row["questions"]]
    numbers = generation_stats(list(done.values()), general_questions)
    llm.console.print(f"  [bold]templates[/] ▸ {numbers['pdfs']} PDFs: {numbers['entries']} entries "
                      f"({sum(numbers['entries_dropped'].values())} dropped by code), {sum(numbers['picks'].values())} picked, "
                      f"{numbers['facts']['kept']} facts and {numbers['questions']['kept']} questions kept; "
                      f"{sum(r['fixed'] for r in numbers['questions']['repairs'].values())} of "
                      f"{sum(r['tried'] for r in numbers['questions']['repairs'].values())} repairs passed; "
                      f"{numbers['calls_per_kept_question']} calls per kept question"
                      + (f"; [yellow]{numbers['failed_calls']} calls gave no readable answer and are retried on the next run[/]"
                         if numbers["failed_calls"] else ""))
