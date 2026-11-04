"""Topic templates: what users ask about one kind of document (templates/<name>.toml), and which sections each
template applies to.

A template lists intents (e.g. symptoms, vaccination schedule), each with cue words and, for figure intents,
an answer shape. Code categorises every section by itself: a template applies when its signal words are dense
enough in the section (and, if the template asks, the section is dense with numbers). Your overrides and
on/off switches live in data/topics.json. `bench templates coverage` reports, with no LLM, which intents each
section states and which of them today's facts and questions already cover.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from bench.config import DATA_DIR, ROOT, BenchError
from bench.pdf import write_text
from bench.verify import SHAPES, has_shape

TEMPLATES_DIR = ROOT / "templates"
TOPICS = DATA_DIR / "topics.json"
COVERAGE = DATA_DIR / "coverage.md"
HEADING_CHARS = 80  # a short line just above a quote counts as its heading (e.g. "SYMPTOMS:")
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
    min_density: float = 6.0  # signal words per 1,000 words
    min_numbers_per_word: float = 0.0
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
    """data/topics.json: {"switches": {"template" or "template.intent": bool}, "sections": {segment_id: [templates]}}."""
    if not TOPICS.exists():
        return {"switches": {}, "sections": {}}
    try:
        topics = json.loads(TOPICS.read_text())
    except ValueError as exc:
        raise BenchError(f"data/topics.json isn't valid JSON ({exc}); fix or delete it") from exc
    return {"switches": topics.get("switches", {}), "sections": topics.get("sections", {})}


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


def assigned(segment: dict[str, Any], tmpls: dict[str, Template], topics: dict[str, Any]) -> tuple[list[str], str]:
    """(templates, by): your override from data/topics.json, else every template whose thresholds the section meets."""
    if segment["segment_id"] in topics["sections"]:
        return topics["sections"][segment["segment_id"]], "you"
    found = [name for name, s in scores(segment, tmpls).items()
             if s["density"] >= tmpls[name].min_density and s["numbers_per_word"] >= tmpls[name].min_numbers_per_word]
    return found, "rules"


def heading_above(text: str, start: int) -> str:
    """The last non-empty line before `start` if it looks like a heading: short, and not ending a sentence."""
    lines = [line.strip() for line in text[max(0, start - 400) : start].splitlines() if line.strip()]
    return lines[-1] if lines and len(lines[-1]) <= HEADING_CHARS and not lines[-1].endswith(".") else ""


def fact_intents(fact: dict[str, Any], text: str, tmpl: Template) -> list[str]:
    """The intents a fact speaks to, by code: a cue in its quote or in the heading line just above it, and for
    figure intents an answer of the intent's shape in the quote."""
    window = heading_above(text, fact["start"]) + "\n" + fact["quote"]
    return [name for name, intent in tmpl.intents.items()
            if cue_pattern(intent.cues).search(window) and (intent.kind != "figure" or has_shape(intent.shape, fact["quote"]))]


def coverage(segments: list[dict[str, Any]], facts_rows: dict[str, dict], question_rows: dict[str, dict],
             tmpls: dict[str, Template], topics: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per (section, template, switched-on intent) with a status computed by code:
    covered (a kept question), facts only, stated but missing (cue words occur, no fact), not stated."""
    rows = []
    for segment in (s for s in segments if s["use_for_questions"]):
        sid, text = segment["segment_id"], segment["text"]
        names, by = assigned(segment, tmpls, topics)
        facts = facts_rows.get(sid, {}).get("facts", [])
        questions = question_rows.get(sid, {}).get("questions", [])
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        for name in names:
            if name not in tmpls or not is_on(topics, name, tmpls[name]):
                continue
            tagged = {f["fact_id"]: fact_intents(f, text, tmpls[name]) for f in facts}
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
             "answer of the intent's shape). Today's facts and questions were not aimed at intents, so this is the "
             "baseline that intent-aimed extraction will be measured against.", "",
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
