"""Topic templates: what users ask about one kind of document (templates/<name>.toml), and which sections each
template applies to.

A template lists intents (e.g. symptoms, vaccination schedule), each with cue words and, for figure intents,
an answer shape. Code categorises every section by itself: a template applies when its signal words are dense
enough in the section (and, if the template asks, the section is dense with numbers). Your overrides and
on/off switches live in data/topics.json. `bench templates coverage` reports, with no LLM, which intents each
section states and which of them the facts and questions cover.

`bench templates run` writes questions aimed at the intents, per section and alongside the general ones:
1. code finds the parts of a section where a switched-on intent's cue words occur (other parts cost no call);
2. intent planner: picks topics in a part, each an intent and its subject (code drops a subject the section
   never names);
3. the facts stage's extractor and reviewer write and review one fact per topic; a table row can be a fact;
4. the regex audit: the fact rules, plus a cue word in the quote (or its table header, or the heading line
   above), a figure intent's answer shape, and the subject named in the quote or its headings;
5. writers in the template's styles (types from question_types.toml), then the question reviewer and the
   audit: the style's rules, the subject named in the question, and not the same as a general question.
Output: one line per section in data/templates.jsonl {segment_id, templates, fingerprint, facts,
rejected_facts, questions, rejected}; ids <section>:tf01 (facts) and <section>:t01 (questions).
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    model_validator,
)

from bench import facts as fact_stage
from bench import llm
from bench import questions as question_stage
from bench.config import DATA_DIR, PART_CHARS, ROOT, BenchError, question_types
from bench.pdf import PAGES, read_jsonl, write_jsonl, write_text
from bench.segments import heading_path, load_segments, page_at, parts, sentences
from bench.verify import (
    SHAPES,
    content_words,
    data_row,
    has_shape,
    kept_fact_problem,
    plain,
    question_problem,
    uid,
    verify_facts,
)

TEMPLATES_DIR = ROOT / "templates"
TOPICS = DATA_DIR / "topics.json"
COVERAGE = DATA_DIR / "coverage.md"
ROWS = DATA_DIR / "templates.jsonl"
TEMPLATES_VERSION = "templates-v1"  # the row format; part of every row's fingerprint
TOPICS_PER_INTENT = 2  # topics the planner may pick per intent in one part
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
    min_density: float = 6.0  # signal words per 1,000 words
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


def intent_problem(fact: dict[str, Any], text: str, intent: Intent, before: str = "") -> str | None:
    """Why a fact doesn't speak to the intent, by code, or None: a cue word must be in its quote, its table header,
    the heading line just above it or (for a fact written for the intent) `before`: its TOC headings and the text
    just before it; a figure intent's answer shape must be in the quote (a table row's values take their unit
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


def _name_pattern(name: str) -> re.Pattern[str]:
    """A name in any case, singular or plural, its words apart by spaces or hyphens ('Aflatoxins' finds 'aflatoxin')."""
    words = [re.escape(w[:-1] if w.endswith("s") and len(w) > 3 else w) + "(?:e?s)?" for w in re.findall(r"[\w.&'’]+", name)]
    return re.compile(r"(?<!\w)" + r"[\s\-‐–]+".join(words) + r"(?!\w)", re.I)


def mentions(subject: str, question: str) -> bool:
    """True if a question names its subject, whole or by one of its content words ('wool output' for 'wool
    production', 'calves' for 'health practices for calves'): a farmer words it their own way."""
    return names(subject, question) or any(_name_pattern(w).search(question) for w in content_words(subject))


def names(subject: str, text: str) -> bool:
    """True if the text names the subject (or the name or abbreviation inside it): any case, singular or plural."""
    return any(_name_pattern(n).search(text) for n in subject_names(subject))


def coverage(segments: list[dict[str, Any]], facts_rows: dict[str, dict], question_rows: dict[str, dict],
             tmpls: dict[str, Template], topics: dict[str, Any], template_rows: dict[str, dict] | None = None) -> list[dict[str, Any]]:
    """One row per (section, template, switched-on intent) with a status computed by code:
    covered (a kept question), facts only, stated but missing (cue words occur, no fact), not stated.
    General facts are tagged by code (fact_intents); template facts count for their own intent."""
    rows = []
    for segment in (s for s in segments if s["use_for_questions"]):
        sid, text = segment["segment_id"], segment["text"]
        names, by = assigned(segment, tmpls, topics)
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


PLANNER_PROMPT = """You plan a search benchmark for one kind of document. The TEXT is one part of a document \
section, in numbered paragraphs ([p3]); a table is a paragraph whose lines are rows of cells joined by " | ", \
its header rows first. For each INTENT below that the TEXT answers with specific information, pick topics it \
states, each about a different subject or detail. Skip what is under FACTS SO FAR and the CONTEXT. For each give:
- intent: the intent's name.
- subject: the SUBJECT the topic belongs to, the name a user would put in their question (a disease, a scheme, \
a measure), written exactly as the TEXT or the SECTION heading writes it; not the detail the topic covers.
- paragraphs: the ids of the paragraphs that state it (usually one, at most three consecutive).
Return at most the number of topics asked for, or an empty list if the TEXT answers none of the INTENTS."""

TABLE_NOTE = """

A table row (cells joined by " | ") counts as one sentence. For a figure in a table, point at the one row \
that holds it (first and last are that row); the table's header rows come first. Name the row's and the \
column's labels in the claim."""
EXTRACTOR_PROMPT = fact_stage.EXTRACTOR_PROMPT + TABLE_NOTE


def plan_schema(intents: list[str]) -> type[BaseModel]:
    """The planner's answer, its intent limited to the intents asked for."""
    topic = create_model("IntentTopic", intent=(Literal[tuple(intents)], ...), subject=(str, ...), paragraphs=(list[str], ...))
    return create_model("IntentTopics", topics=(list[topic], ...))  # type: ignore[valid-type]


def active(segment: dict[str, Any], tmpls: dict[str, Template], topics: dict[str, Any]) -> dict[str, list[str]]:
    """The section's switched-on templates (by the rules or by you), each with its switched-on intents."""
    names_, _ = assigned(segment, tmpls, topics)
    out = {n: [i for i in tmpls[n].intents if is_on(topics, n, tmpls[n], i)] for n in names_ if n in tmpls and is_on(topics, n, tmpls[n])}
    return {n: intents for n, intents in out.items() if intents}


def styles(tmpls: dict[str, Template]) -> dict[str, dict]:
    """The question types the templates write in (question_types.toml), checked: each must read facts."""
    types, used = question_types(active_only=False), {s for t in tmpls.values() for s in t.styles}
    if bad := sorted(s for s in used if s not in types or types[s]["source"] != "facts"):
        raise BenchError(f"templates name styles that aren't question types with source \"facts\": {', '.join(bad)}")
    return {s: types[s] for s in sorted(used)}


def fingerprint(segment: dict[str, Any], on: dict[str, list[str]], tmpls: dict[str, Template], kinds: dict[str, dict]) -> str:
    """Everything that made a row: the prompts and schemas, the switched-on templates and intents, the styles'
    definitions, and the section's text and headings."""
    told = {n: {k: t[k] for k in ("prompt", "rules", "review")} for n, t in kinds.items()}
    used = {n: tmpls[n].model_dump(exclude={"intents", "enabled"}) | {"intents": {i: tmpls[n].intents[i].model_dump() for i in on[n]}}
            for n in sorted(on)}
    schemas = [m.model_json_schema() for m in (fact_stage.Pointers, fact_stage.Reviews, question_stage.Drafts, question_stage.Reviews)]
    return uid(TEMPLATES_VERSION, PLANNER_PROMPT, EXTRACTOR_PROMPT, fact_stage.REVIEWER_PROMPT, question_stage.WRITER_PROMPT,
               question_stage.REVIEWER_PROMPT, schemas, used, told, TOPICS_PER_INTENT, PART_CHARS,
               fact_stage.text_hash(segment["text"]), segment["headings"], segment["title"])


def template_fact_problem(fact: dict[str, Any], segment: dict[str, Any], tmpls: dict[str, Template]) -> str | None:
    """Why a template fact fails the intent audit (after the fact rules), or None."""
    tmpl = tmpls.get(fact["template"])
    if tmpl is None or fact["intent"] not in tmpl.intents:
        return "its template or intent no longer exists"
    text, anchor = segment["text"], fact.get("header_start", fact["start"])
    path = heading_path(segment, page_at(segment, fact["start"]))
    if problem := intent_problem(fact, text, tmpl.intents[fact["intent"]], path + "\n" + text[max(0, anchor - CUE_BEFORE) : anchor]):
        return problem
    near = "\n".join([Path(segment["file_name"]).stem, path, segment["title"], text[max(0, anchor - SUBJECT_BEFORE) : anchor],
                      fact.get("header", ""), fact["quote"]])
    return None if names(fact["subject"], near) else f"subject \"{fact['subject']}\" isn't named in or just before the quote, its headings or the document title"


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
    general = {plain(q["question"]) for row in fact_stage.load_rows(question_stage.QUESTIONS).values() for q in row["questions"]}
    corpus = "\n".join(p["text"] for p in read_jsonl(PAGES)) if PAGES.exists() else ""  # only absent terms need it
    return tmpls, load_topics(), styles(tmpls), corpus, general


def row_problems(row: dict[str, Any], segment: dict[str, Any], tmpls: dict[str, Template], kinds: dict[str, dict],
                 corpus: str, general: set[str]) -> list[str]:
    """Kept facts and questions of a row that no longer pass the audit, as 'id (reason)'."""
    by_id = {f["fact_id"]: f for f in row["facts"]}
    found = [f"{f['fact_id']} ({p})" for f in row["facts"]
             if (p := kept_fact_problem(f, segment) or template_fact_problem(f, segment, tmpls))]
    return found + [f"{q['question_id']} ({p})" for q in row["questions"]
                    if (p := template_question_problem(q, by_id, kinds, corpus, general))]


def run_templates(doc: str | None = None, force: bool = False) -> None:
    """Template facts and questions for every section with a switched-on template -> data/templates.jsonl."""
    tmpls, topics, kinds, corpus, general = inputs()
    done = fact_stage.load_rows(ROWS)
    extractor, fact_reviewer = llm.agent(fact_stage.Pointers, EXTRACTOR_PROMPT), llm.agent(fact_stage.Reviews, fact_stage.REVIEWER_PROMPT)
    writers = {n: llm.agent(question_stage.Drafts, question_stage.WRITER_PROMPT + f"{n}: {t['prompt']}") for n, t in kinds.items()}
    described = "\n".join(f"- {n}: {t['review'] or ' '.join(t['prompt'].split())}" for n, t in kinds.items())
    question_reviewer = llm.agent(question_stage.Reviews, question_stage.REVIEWER_PROMPT + described)

    def current(segment: dict[str, Any]) -> str:
        return fingerprint(segment, active(segment, tmpls, topics), tmpls, kinds)

    def valid(row: dict[str, Any], segment: dict[str, Any]) -> bool:
        return not row_problems(row, segment, tmpls, kinds, corpus, general)

    def work(segment: dict[str, Any]) -> None:
        sid, on = segment["segment_id"], active(segment, tmpls, topics)
        found = [f for name, intents in on.items() for f in plan_and_extract(segment, name, tmpls[name], intents, extractor)]
        kept, rejected = verify_facts(segment, fact_stage.review(fact_reviewer, segment, found) if found else [], prefix="tf")
        for f in list(kept):
            if problem := template_fact_problem(f, segment, tmpls):
                kept.remove(f)
                rejected.append({**f, "reason": problem, "decided_by": "audit"})
        for n, f in enumerate(kept, start=1):
            f["fact_id"] = f"{sid}:tf{n:02d}"
        questions, rejected_questions = [], []
        for name in on:
            mine = {f"f{i}": f for i, f in enumerate(kept, start=1) if f["template"] == name}
            good, bad = write_questions(segment, tmpls[name], mine, kinds, writers, question_reviewer, corpus, general)
            questions += good
            rejected_questions += bad
        for n, q in enumerate(questions, start=1):
            q["question_id"] = f"{sid}:t{n:02d}"
        done[sid] = {"segment_id": sid, "model": llm.settings().llm_model, "version": TEMPLATES_VERSION,
                     "fingerprint": current(segment), "templates": on, "facts": kept, "rejected_facts": rejected,
                     "questions": questions, "rejected": rejected_questions}
        write_jsonl(ROWS, list(done.values()))
        if on:
            fact_stage.audit_line(sid, "template facts", kept, rejected)
            fact_stage.audit_line(sid, "template questions", questions, rejected_questions)

    rows_before = set(done)
    stale = fact_stage.todo(done, doc, force, current, valid)
    if set(done) != rows_before:
        write_jsonl(ROWS, list(done.values()))
    fact_stage.run_each(stale, work, "templates run")


def plan_and_extract(segment: dict[str, Any], name: str, tmpl: Template, intents: list[str], extractor: Any) -> list[dict[str, Any]]:
    """One template's facts in a section, before the audit: per part where its intents' cue words occur, the
    planner's topics (a subject the section never names is dropped), then one extracted fact per topic."""
    sid, text, units = segment["segment_id"], segment["text"], sentences(segment["text"])
    known = "\n".join([segment["title"], *(h["title"] for h in segment["headings"]), text])
    found: list[dict[str, Any]] = []
    for i, part in enumerate(cut := parts(segment, PART_CHARS)):
        cued = [n for n in intents if cue_pattern(tmpl.intents[n].cues).search(part["heading"] + "\n" + part["text"])]
        inside = [u for u in units if part["start"] <= u["start"] < part["end"]]
        if not cued or not inside:
            continue  # no call for a part that states none of the intents
        wanted = TOPICS_PER_INTENT * len(cued)
        listed = "\n".join(f"- {n}: {tmpl.intents[n].description}" for n in cued)
        prompt = (f"DOCUMENT TYPE: {tmpl.description}\nSUBJECT: {tmpl.subject or 'what the passage is about'}\nINTENTS:\n{listed}\n\n"
                  + fact_stage.planner_prompt(part, inside, text, found, wanted))
        label = f"{sid} {name} part {i + 1}/{len(cut)}"
        plan = llm.call("intent planner", llm.agent(plan_schema(cued), PLANNER_PROMPT), prompt, label)
        for topic in plan.topics[:wanted]:
            chosen = [u for u in inside if u["id"].split(".")[0] in fact_stage.paragraph_ids(topic.paragraphs)]
            if not chosen or not topic.subject.strip() or not names(topic.subject, known):
                continue  # paragraphs outside this part, or a subject the section never names
            intent = tmpl.intents[topic.intent]
            pick = fact_stage.Topic(topic=f"{intent.description} Subject: {topic.subject}.", kind=intent.kind, paragraphs=topic.paragraphs)
            heading = heading_path(segment, page_at(segment, chosen[0]["start"]))
            out = llm.call("extractor", extractor, fact_stage.extractor_prompt(heading, pick, chosen, text, found),
                           f"{label} ▸ {topic.intent} ▸ {topic.subject[:40]}")
            example = intent.examples[0].replace("{subject}", topic.subject)
            found += [{**f.model_dump(), "topic": pick.topic, "kind": intent.kind, "template": name, "intent": topic.intent,
                       "subject": topic.subject, "example": example} for f in out.facts[:1]]
    return found


def write_questions(segment: dict[str, Any], tmpl: Template, facts: dict[str, dict], kinds: dict[str, dict], writers: dict[str, Any],
                    reviewer: Any, corpus: str, general: set[str]) -> tuple[list[dict], list[dict]]:
    """Questions in the template's styles about one template's facts in a section, reviewed and audited."""
    if not facts:
        return [], []
    sid, title = segment["segment_id"], segment["title"]
    drafts: list[dict[str, Any]] = []
    for style, per_fact in tmpl.styles.items():
        memory = "\n".join(f"- [{q['type']}, {', '.join(q['fact_ids'])}] {q['question']}" for q in drafts) or "(none yet)"
        prompt = (f"SECTION: {title}\nDOCUMENT TYPE: {tmpl.description}\n\nEach fact answers something users ask about this "
                  "kind of document. It names its subject and gives an example of such a question: write in that spirit, "
                  f"name the subject, and ask about each fact.\n\nQUESTIONS ALREADY WRITTEN:\n{memory}\n\n"
                  f"Write at most {per_fact * len(facts)} questions.\n\nFACTS:\n{question_stage.listing(facts)}")
        out = llm.call(f"{style} writer", writers[style], prompt, f"{sid} templates")
        drafts += [{"type": style, **d.model_dump()} for d in out.questions[: per_fact * len(facts)]]
    kept, rejected = [], []
    for q in question_stage.review(reviewer, title, drafts, facts, f"{sid} templates"):
        if problem := template_question_problem(q, facts, kinds, corpus, general):
            rejected.append({**q, "reason": problem, "decided_by": "audit"})
        else:
            first = facts.get(q["fact_ids"][0], {})
            kept.append({**q, "template": first["template"], "intent": first["intent"], "subject": first["subject"]})
    return question_stage.finish(kept, facts, kinds, sid), rejected


def current_questions() -> list[dict[str, Any]]:
    """Template questions whose template and intent are switched on for their section now: what evaluate scores."""
    if not ROWS.exists():
        return []
    tmpls, topics = templates(), load_topics()
    segments = {s["segment_id"]: s for s in load_segments() if s["use_for_questions"]}
    on = {sid: active(s, tmpls, topics) for sid, s in segments.items()}
    return [q for row in read_jsonl(ROWS) for q in row["questions"] if q["intent"] in on.get(row["segment_id"], {}).get(q["template"], [])]
