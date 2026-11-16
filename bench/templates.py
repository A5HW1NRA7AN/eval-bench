"""Topic templates: what users ask about one kind of document (templates/<name>.toml), which sections each
template applies to, and the audit of the facts and questions written for them.

A template lists intents (e.g. symptoms, vaccination schedule), each with cue words and, for figure intents,
an answer shape. Which templates a section answers is decided by the orchestrator (bench/orchestrator.py), which
saves its decision on each PDF's row; your overrides and on/off switches live in data/topics.json, and code's
density rules (signal words per 1,000 words, numbers per word) are its hints and its fallback.
`bench templates coverage` reports, with no LLM, which intents each section states and which of them the
facts and questions cover.

The intent audit (regex, after the fact rules): a cue word in the text the audit reads around the quote (`near`:
the headings of its pages and up to CUE_BEFORE characters before it), a figure intent's answer shape, and the
subject named in or up to SUBJECT_BEFORE characters before the quote, its headings or the document title.
A template question must pass its style's rules, name its subject and not repeat a general question.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from bench import questions as question_stage
from bench.config import DATA_DIR, ROOT, BenchError, question_types
from bench.pdf import PAGES, read_jsonl, write_text
from bench.segments import heading_path, load_segments, page_at
from bench.verify import (
    SHAPES,
    content_words,
    data_row,
    has_shape,
    kept_fact_problem,
    name_pattern,
    plain,
    question_problem,
)

TEMPLATES_DIR = ROOT / "templates"
TOPICS = DATA_DIR / "topics.json"
COVERAGE = DATA_DIR / "coverage.md"
ROWS = DATA_DIR / "templates.jsonl"
TEMPLATES_VERSION = "templates-v2"  # the row format (one row per PDF); part of every row's fingerprint
HEADING_CHARS = 80  # a short line just above a quote counts as its heading (e.g. "SYMPTOMS:")
CUE_BEFORE = 600  # a template fact's cue may be this many characters before it ("Symptoms:" above a list)
SUBJECT_BEFORE = 1200  # its subject may be named this far before it (a disease's heading above its blocks)
NUMBER_TOKEN = re.compile(r"\d[\d,.]*")


class Intent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    kind: Literal["figure", "explanation"]
    description: str = Field(min_length=1)
    cues: list[str] = Field(min_length=1)
    shape: list[str] = []  # figure intents: the quote must hold an answer of one of these shapes
    examples: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _shapes(self) -> Intent:
        if unknown := set(self.shape) - set(SHAPES):
            raise ValueError(f"unknown shape: {', '.join(sorted(unknown))} (known: {', '.join(SHAPES)})")
        if self.kind == "figure" and not self.shape:
            raise ValueError("a figure intent needs a shape")
        return self


class Template(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    description: str = Field(min_length=1)
    subject: str = ""
    signals: list[str] = Field(min_length=1)
    min_density: float = 6.0  # signal words per 1,000 words for the density rule
    min_numbers_per_word: float = 0.0
    styles: dict[str, int] = {"direct": 1, "paraphrased": 1}  # question types (question_types.toml) and questions per fact
    intents: dict[str, Intent] = Field(min_length=1)

    @model_validator(mode="after")
    def _examples(self) -> Template:
        for name, intent in self.intents.items():
            if self.subject and (missing := [e for e in intent.examples if "{subject}" not in e]):
                raise ValueError(f"intents.{name}: examples need {{subject}}: {missing[0]!r}")
        return self


def templates(folder: Path = TEMPLATES_DIR) -> dict[str, Template]:
    """Every template in templates/, checked strictly: an unknown key, shape or kind is an error."""
    found = {}
    for path in sorted(folder.glob("*.toml")):
        try:
            found[path.stem] = Template.model_validate(tomllib.loads(path.read_text()))
        except tomllib.TOMLDecodeError as exc:
            raise BenchError(f"templates/{path.name}: {exc}") from exc
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'template'}: {e['msg']}" for e in exc.errors())
            raise BenchError(f"templates/{path.name}: {problems}") from exc
    return found


def cue_pattern(cues: list[str]) -> re.Pattern[str]:
    """Plain cue words, matched case-insensitively at the start of a word, spaces matching any whitespace."""
    return re.compile(r"(?<![A-Za-z])(?:" + "|".join(r"\s+".join(map(re.escape, c.split())) for c in cues) + ")", re.I)


def load_topics() -> dict[str, Any]:
    """data/topics.json, your choices for this data folder: {"switches": {"template" or "template.intent": bool},
    "sections": {segment_id: [templates]}, "groups": {name: [PDF file names]}} (PDFs whose facts multi-hop questions
    may pair; a PDF in no group is paired only within itself)."""
    if not TOPICS.exists():
        return {"switches": {}, "sections": {}, "groups": {}}
    try:
        topics = json.loads(TOPICS.read_text())
    except ValueError as exc:
        raise BenchError(f"data/topics.json isn't valid JSON ({exc}); fix or delete it") from exc
    return {"switches": topics.get("switches", {}), "sections": topics.get("sections", {}), "groups": topics.get("groups", {})}


def save_topics(topics: dict[str, Any]) -> None:
    write_text(TOPICS, json.dumps(topics, indent=1, ensure_ascii=False, sort_keys=True))


def is_on(topics: dict[str, Any], tmpl: str, template: Template, intent: str | None = None) -> bool:
    """A template (or one of its intents) is on by its TOML default unless data/topics.json switches it."""
    on = topics["switches"].get(tmpl, template.enabled)
    if intent is not None:
        on = on and topics["switches"].get(f"{tmpl}.{intent}", template.intents[intent].enabled)
    return on


def scores(segment: dict[str, Any], tmpls: dict[str, Template]) -> dict[str, dict[str, float]]:
    """For each template: its signal words per 1,000 words in the section, and the section's numbers per word."""
    text = " ".join([segment["title"], *(h["title"] for h in segment["headings"]), segment["text"]])
    words = max(len(text.split()), 1)
    numbers = len(NUMBER_TOKEN.findall(text)) / words
    return {name: {"density": round(1000 * len(cue_pattern(t.signals).findall(text)) / words, 1),
                   "numbers_per_word": round(numbers, 3)} for name, t in tmpls.items()}


def by_rules(segment: dict[str, Any], tmpls: dict[str, Template]) -> list[str]:
    """The density rules: every template whose thresholds the section meets."""
    return [name for name, s in scores(segment, tmpls).items()
            if s["density"] >= tmpls[name].min_density and s["numbers_per_word"] >= tmpls[name].min_numbers_per_word]


def assigned(segment: dict[str, Any], tmpls: dict[str, Template], topics: dict[str, Any],
             saved: dict[str, dict[str, Any]] | None = None) -> tuple[list[str], str]:
    """(templates, by): your override from data/topics.json, else the orchestrator's saved decision (`saved`,
    see categorised), else the density rules."""
    sid = segment["segment_id"]
    if sid in topics["sections"]:
        return topics["sections"][sid], "you"
    if saved and sid in saved:
        return saved[sid]["templates"], saved[sid]["by"]
    return by_rules(segment, tmpls), "rules"


def heading_above(text: str, start: int) -> str:
    """The last non-empty line before `start` if it looks like a heading: short, and not ending a sentence."""
    lines = [line.strip() for line in text[max(0, start - 400) : start].splitlines() if line.strip()]
    return lines[-1] if lines and len(lines[-1]) <= HEADING_CHARS and not lines[-1].endswith(".") else ""


def near(segment: dict[str, Any], start: int, end: int, before: int) -> str:
    """What the audit reads around text[start:end]: the heading path of every page it covers, then the text from
    `before` characters ahead of it to its end."""
    pages = [page_at(segment, start)] + [s["page"] for s in segment["page_spans"] if s["start"] < end and s["end"] > start]
    return "\n".join([*dict.fromkeys(heading_path(segment, p) for p in pages), segment["text"][max(0, start - before) : end]])


def subject_window(segment: dict[str, Any], start: int, end: int) -> str:
    """Where a subject must be named for text[start:end]: the document's and the section's titles, and near()."""
    return "\n".join([Path(segment["file_name"]).stem, segment["title"], near(segment, start, end, SUBJECT_BEFORE)])


def intent_problem(fact: dict[str, Any], text: str, intent: Intent, before: str = "") -> str | None:
    """Why a fact doesn't speak to the intent, by code, or None: a cue word must be in its quote, its table header,
    the heading line just above it or (for a fact written for the intent) `before`: the text the audit reads
    around it (near); a figure intent's answer shape must be in the quote (a table row's values take their unit
    from the header)."""
    window = "\n".join([before, heading_above(text, fact.get("header_start", fact["start"])), fact.get("header", ""), fact["quote"]])
    if not cue_pattern(intent.cues).search(window):
        return "no cue word of its intent in or just before the quote"
    if intent.kind == "figure" and not (has_shape(intent.shape, fact["quote"]) or (fact.get("header") and data_row(fact["quote"]))):
        return f"no {' or '.join(intent.shape)} in the quote"
    return None


def fact_intents(fact: dict[str, Any], text: str, tmpl: Template) -> list[str]:
    """The intents a fact speaks to: its own intent if a template wrote it, else every intent it passes (intent_problem)."""
    if "intent" in fact:
        return [fact["intent"]] if fact["intent"] in tmpl.intents else []
    return [name for name, intent in tmpl.intents.items() if intent_problem(fact, text, intent) is None]


def subject_names(subject: str) -> list[str]:
    """'Foot and Mouth Disease (FMD)' -> itself, 'Foot and Mouth Disease' and 'FMD'."""
    names = [subject.strip(), re.sub(r"\s*\([^)]*\)", "", subject).strip(), *re.findall(r"\(([^)]{2,30})\)", subject)]
    return [n for n in dict.fromkeys(names) if len(n) >= 2]


def mentions(subject: str, question: str) -> bool:
    """True if a question names its subject, whole or by one of its content words ('wool output' for 'wool
    production', 'calves' for 'health practices for calves'): a farmer words it their own way."""
    return names(subject, question) or any(name_pattern(w).search(question) for w in content_words(subject))


def names(subject: str, text: str) -> bool:
    """True if the text names the subject (or the name or abbreviation inside it): any case, singular or plural."""
    return any(name_pattern(n).search(text) for n in subject_names(subject))


def coverage(segments: list[dict[str, Any]], facts_rows: dict[str, dict], question_rows: dict[str, dict],
             tmpls: dict[str, Template], topics: dict[str, Any], template_rows: dict[str, dict] | None = None,
             saved: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """One row per (section, template, switched-on intent) with a status computed by code:
    covered (a kept question), facts only, stated but missing (cue words occur, no fact), not stated.
    General facts are tagged by code (fact_intents); template facts (template_rows, by section) count for their
    own intent. `saved` is the orchestrator's categorisation (categorised)."""
    rows = []
    for segment in (s for s in segments if s["use_for_questions"]):
        sid, text = segment["segment_id"], segment["text"]
        names, by = assigned(segment, tmpls, topics, saved)
        topical = (template_rows or {}).get(sid, {})
        facts = facts_rows.get(sid, {}).get("facts", []) + topical.get("facts", [])
        questions = question_rows.get(sid, {}).get("questions", []) + topical.get("questions", [])
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        for name in names:
            if name not in tmpls or not is_on(topics, name, tmpls[name]):
                continue
            tagged = {f["fact_id"]: fact_intents(f, text, tmpls[name]) for f in facts if f.get("template", name) == name}
            for intent_name, intent in tmpls[name].intents.items():
                if not is_on(topics, name, tmpls[name], intent_name):
                    continue
                cue = cue_pattern(intent.cues)
                stated = sum(bool(cue.search(p)) for p in paragraphs)
                with_facts = [fid for fid, intents in tagged.items() if intent_name in intents]
                with_questions = [q for q in questions if set(q["fact_ids"]) & set(with_facts)]
                status = ("covered" if with_questions else "facts only" if with_facts else
                          "stated but missing" if stated else "not stated")
                rows.append({"segment_id": sid, "file_name": segment["file_name"], "title": segment["title"], "by": by,
                             "template": name, "intent": intent_name, "paragraphs_with_cues": stated,
                             "facts": len(with_facts), "questions": len(with_questions), "status": status})
    return rows


MARKS = {"covered": "✓", "facts only": "◐", "stated but missing": "○", "not stated": "·"}


def coverage_report(rows: list[dict[str, Any]], tmpls: dict[str, Template]) -> str:
    """Markdown: per template, a section × intent matrix of statuses, then totals."""
    lines = ["# Template coverage", "",
             "Computed by code, with no LLM: an intent is *stated* in a section when its cue words occur there; a fact "
             "speaks to it when a cue sits in or just before its quote (and, for figure intents, the quote holds an "
             "answer of the intent's shape). Facts written for a template (`bench templates run`) count for their own "
             "intent.", "",
             "✓ covered by a kept question · ◐ facts only · ○ stated but missing · · not stated", ""]
    for name, tmpl in tmpls.items():
        mine = [r for r in rows if r["template"] == name]
        if not mine:
            lines += [f"## {name}", "", "No section is assigned to this template (or it is switched off).", ""]
            continue
        intents = list(dict.fromkeys(r["intent"] for r in mine))
        lines += [f"## {name}", "", f"{tmpl.description}", "",
                  "| section | by | " + " | ".join(intents) + " |", "|---" * (len(intents) + 2) + "|"]
        for sid in dict.fromkeys(r["segment_id"] for r in mine):
            cells = {r["intent"]: r for r in mine if r["segment_id"] == sid}
            title = next(r["title"] for r in mine if r["segment_id"] == sid)
            lines.append(f"| {sid} ({title[:40]}) | {cells[intents[0]]['by']} | "
                         + " | ".join(f"{MARKS[cells[i]['status']]} {cells[i]['facts']}f {cells[i]['questions']}q" for i in intents) + " |")
        totals = {status: sum(r["status"] == status for r in mine) for status in MARKS}
        lines += ["", "Totals: " + ", ".join(f"{n} {status}" for status, n in totals.items()), ""]
    return "\n".join(lines)


def active(segment: dict[str, Any], tmpls: dict[str, Template], topics: dict[str, Any],
           saved: dict[str, dict[str, Any]] | None = None) -> dict[str, list[str]]:
    """The section's switched-on templates (assigned), each with its switched-on intents."""
    names_, _ = assigned(segment, tmpls, topics, saved)
    out = {n: [i for i in tmpls[n].intents if is_on(topics, n, tmpls[n], i)] for n in names_ if n in tmpls and is_on(topics, n, tmpls[n])}
    return {n: intents for n, intents in out.items() if intents}


def styles(tmpls: dict[str, Template]) -> dict[str, dict]:
    """The question types the templates write in (question_types.toml), checked: each must read facts."""
    types, used = question_types(active_only=False), {s for t in tmpls.values() for s in t.styles}
    if bad := sorted(s for s in used if s not in types or types[s]["source"] != "facts"):
        raise BenchError(f"templates name styles that aren't question types with source \"facts\": {', '.join(bad)}")
    return {s: types[s] for s in sorted(used)}


def template_fact_problem(fact: dict[str, Any], segment: dict[str, Any], tmpls: dict[str, Template]) -> str | None:
    """Why a template fact fails the intent audit (after the fact rules), or None."""
    tmpl = tmpls.get(fact["template"])
    if tmpl is None or fact["intent"] not in tmpl.intents:
        return "its template or intent no longer exists"
    anchor = fact.get("header_start", fact["start"])
    if problem := intent_problem(fact, segment["text"], tmpl.intents[fact["intent"]], near(segment, anchor, fact["end"], CUE_BEFORE)):
        return problem
    around = subject_window(segment, anchor, fact["end"]) + "\n" + fact.get("header", "")
    return None if names(fact["subject"], around) else f"subject \"{fact['subject']}\" isn't named in or just before the quote, its headings or the document title"


def template_question_problem(q: dict[str, Any], facts: dict[str, dict], kinds: dict[str, dict], corpus: str,
                              general: set[str]) -> str | None:
    """Why a template question fails the audit, or None: its style's rules, its subject named, and new."""
    if q["type"] not in kinds:
        return f"style {q['type']} is no longer used"
    if problem := question_problem(q, facts, kinds[q["type"]]["rules"], corpus):
        return problem
    subject = facts[q["fact_ids"][0]]["subject"]
    if not mentions(subject, q["question"]):
        return f"doesn't name its subject \"{subject}\""
    return "same as a general question" if plain(q["question"]) in general else None


def inputs() -> tuple[dict[str, Template], dict[str, Any], dict[str, dict], str, set[str]]:
    """(templates, topics, styles, the library's text, the general questions) that rows are made and audited with."""
    tmpls = templates()
    general = {plain(q["question"]) for row in question_stage.question_rows() for q in row["questions"]}
    corpus = "\n".join(p["text"] for p in read_jsonl(PAGES)) if PAGES.exists() else ""  # only absent terms need it
    return tmpls, load_topics(), styles(tmpls), corpus, general


def pdf_rows() -> list[dict[str, Any]]:
    """data/templates.jsonl in today's format, one row per PDF; rows in an older format are left out (and redone)."""
    return [r for r in read_jsonl(ROWS) if r.get("version") == TEMPLATES_VERSION] if ROWS.exists() else []


def by_section(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Each section's facts and questions, kept and rejected, gathered from rows that span several sections (template
    rows per PDF, question rows per group); items with no single section (multi-hop questions) are left out."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in rows:
        for field in ("facts", "questions", "rejected_facts", "rejected"):
            for item in r.get(field, []):
                if item.get("segment_id"):
                    out.setdefault(item["segment_id"], {"facts": [], "questions": [], "rejected_facts": [], "rejected": []})[field].append(item)
    return out


def categorised(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The saved categorisation of every section: {segment_id: {"templates": [...], "by": orchestrator|rules|you}}."""
    return {sid: c for r in rows for sid, c in r["sections"].items()}


def row_problems(row: dict[str, Any], segments: dict[str, dict[str, Any]], tmpls: dict[str, Template],
                 kinds: dict[str, dict], corpus: str, general: set[str]) -> list[str]:
    """Kept facts and questions of a PDF's row that no longer pass the audit, as 'id (reason)'."""
    by_id = {f["fact_id"]: f for f in row["facts"]}
    found = []
    for f in row["facts"]:
        segment = segments.get(f["segment_id"])
        if problem := ("its section no longer exists" if segment is None
                       else kept_fact_problem(f, segment) or template_fact_problem(f, segment, tmpls)):
            found.append(f"{f['fact_id']} ({problem})")
    return found + [f"{q['question_id']} ({p})" for q in row["questions"]
                    if (p := template_question_problem(q, by_id, kinds, corpus, general))]


def current_questions() -> list[dict[str, Any]]:
    """Template questions whose template and intent are switched on for their section now: what evaluate scores."""
    rows = pdf_rows()
    if not rows:
        return []
    tmpls, topics, saved = templates(), load_topics(), categorised(rows)
    on = {s["segment_id"]: active(s, tmpls, topics, saved) for s in load_segments() if s["use_for_questions"]}
    return [q for r in rows for q in r["questions"] if q["intent"] in on.get(q["segment_id"], {}).get(q["template"], [])]
