"""Regex audit (no AI): every fact's quote and every question's answer must exist in the PDF text.

A fact points at a range of numbered sentences and its quote is cut from the segment text there, so it
can't be retyped wrongly; the audit checks the range and that the claim says nothing its quote doesn't.
An answer is matched in its quotes exactly except for whitespace (line breaks, double spaces), quote style
and dash style, which PDFs and models write inconsistently. Anything else that differs is rejected.
"""

from __future__ import annotations

import re
from typing import Any

from bench.pdf import NUMBER as NUMERIC_TOKEN
from bench.segments import sentences

MAX_QUOTE_CHARS = 1200  # at most a paragraph: one test-bed chunk
PRONOUN_START = re.compile(r"^(it|this|these|they|those|its|their)\b", re.I)
ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]+\b")  # BIS, FSSAI, B1, M1
DIGITS = re.compile(r"(?<![A-Za-z\d])\d+(?:[.,]\d+)*")  # not the 1 of B1: codes are names, checked as acronyms
# A date or year in a claim ("March 31, 2024", "2024-25") is a qualifier from the section, not a quoted value.
DATE = re.compile(r"\b(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+)?"
                  r"(?:19|20)\d\d(?:\s*[-–]\s*(?:19|20)?\d\d)?\b", re.I)
WORD_NUMBERS = {w: str(n) for n, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
YEAR_SPAN = re.compile(r"\b(?:19|20)\d\d-\d\d\b")  # a reporting year such as 2024-25
# Characters that PDFs and models write in several forms.
VARIANTS = {"'": "['‘’`]", "‘": "['‘’`]", "’": "['‘’`]", '"': "[\"“”]", "“": "[\"“”]", "”": "[\"“”]",
            "-": "[-‐‑–—]", "‐": "[-‐‑–—]", "–": "[-‐‑–—]", "—": "[-‐‑–—]"}
STOPWORDS = {"what", "which", "when", "where", "does", "should", "with", "from", "that", "this", "have", "their",
             "they", "into", "about", "after", "before", "much", "many", "used", "your", "there", "than"}
NUMBER = re.compile(r"\d|\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
                    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|"
                    r"ninety|hundred|thousand|lakhs?|crores?|million|billion|dozen|half|twice|double)\b", re.I)


def pattern(snippet: str) -> re.Pattern[str]:
    """A regex for the snippet that tolerates whitespace, quote and dash variants."""
    parts = ["".join(VARIANTS.get(c, re.escape(c)) for c in token) for token in snippet.split()]
    # A word broken across lines ("milk-\nproduction") may have whitespace after its dash.
    joined = r"\s+".join(parts)
    return re.compile(joined.replace("[-‐‑–—]", r"[-‐‑–—]\s*"))


def find(snippet: str, text: str) -> tuple[int, int] | None:
    match = pattern(snippet).search(text) if snippet.strip() else None
    return (match.start(), match.end()) if match else None


def pages_of(segment: dict[str, Any], start: int, end: int) -> list[int]:
    hit = [s["page"] for s in segment["page_spans"] if s["start"] < end and s["end"] > start]
    return [hit[0], hit[-1]] if hit else []


def table_row(quote: str) -> bool:
    """Mostly numbers: a bare table row, which says nothing without its headers."""
    tokens = quote.split()
    return sum(bool(NUMERIC_TOKEN.match(t)) for t in tokens) >= len(tokens) / 2


def span_of(fact: dict[str, Any], units: dict[str, dict[str, Any]]) -> tuple[int, int] | str:
    """(start, end) of the sentences from `first` to `last`, or why they can't be cut."""
    first, last = units.get(fact["first"]), units.get(fact["last"])
    if not first or not last:
        return "points at a sentence that isn't in the section"
    return (first["start"], last["end"]) if first["start"] <= last["start"] else "its last sentence comes before its first"


def size_of(fact: dict[str, Any], units: dict[str, dict[str, Any]]) -> str:
    """sentence, passage (several sentences) or paragraph (one or more whole paragraphs)."""
    if fact["first"] == fact["last"]:
        return "sentence"
    para, n = fact["last"].split(".s")
    whole = fact["first"].endswith(".s1") and f"{para}.s{int(n) + 1}" not in units
    return "paragraph" if whole else "passage"


def numbers(text: str) -> set[str]:
    """Numbers in digits ("1,000" as "1000") and small number words ("four" as "4")."""
    return ({n.replace(",", "") for n in DIGITS.findall(text)}
            | {WORD_NUMBERS[w] for w in re.findall(r"[a-z]+", text.lower()) if w in WORD_NUMBERS})


def claim_problem(claim: str, quote: str, segment: dict[str, Any]) -> str | None:
    """Why the claim says more than its quote supports, or None. Values must be in the quote; dates,
    years and acronyms are qualifiers, so they must be somewhere in the section."""
    if PRONOUN_START.match(claim):
        return "claim starts with a pronoun"
    if extra := sorted({n.replace(",", "") for n in DIGITS.findall(DATE.sub(" ", claim))} - numbers(quote)):
        return f"claim has a number its quote lacks: {', '.join(extra)}"
    known = " ".join([segment["text"], segment["title"], segment["file_name"], *(h["title"] for h in segment["headings"])])
    if unknown := sorted({y for y in re.findall(r"(?:19|20)\d\d", " ".join(DATE.findall(claim))) if y not in known}):
        return f"claim names a year the section never mentions: {', '.join(unknown)}"
    if unknown := sorted({a for a in ACRONYM.findall(claim) if a not in known}):
        return f"claim names {', '.join(unknown)}, which the section never mentions"
    return None


def verify_facts(segment: dict[str, Any], facts: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Cut each fact's quote from its sentence range and keep it if it passes every rule.
    Returns (kept, rejected); kept facts record their quote, position, pages and size."""
    units = {s["id"]: s for s in sentences(segment["text"])}
    kept, rejected = [], []
    for fact in facts:
        claim, span = fact["claim"].strip(), span_of(fact, units)
        quote = segment["text"][span[0] : span[1]] if isinstance(span, tuple) else ""
        overlap = next((k["fact_id"] for k in kept if isinstance(span, tuple)
                        and min(span[1], k["end"]) - max(span[0], k["start"]) > min(span[1] - span[0], k["end"] - k["start"]) / 2), None)
        reason = ("empty claim" if not claim else
                  span if isinstance(span, str) else
                  f"quote longer than {MAX_QUOTE_CHARS} chars" if len(quote) > MAX_QUOTE_CHARS else
                  "quote is a table row, not a sentence" if table_row(quote) else
                  claim_problem(claim, quote, segment) or (f"overlaps {overlap}" if overlap else None))
        if reason:
            rejected.append({**fact, "quote": quote, "reason": reason})
            continue
        fact_id = f"{segment['segment_id']}:f{len(kept) + 1:02d}"
        kept.append({"fact_id": fact_id, **fact, "claim": claim, "quote": quote, "start": span[0], "end": span[1],
                     "pages": pages_of(segment, *span), "size": size_of(fact, units)})
    return kept, rejected


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOPWORDS}


def _term(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term.strip())}(?!\w)", re.I)


def _plain(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def question_problem(q: dict[str, Any], facts: dict[str, dict[str, Any]], rules: dict[str, Any],
                     corpus: str | None = None) -> str | None:
    """Why a question fails the audit, or None if it passes: the general rules, then its type's rules
    (question_types.toml). `corpus` is the whole library's text, for the absent_term rule."""
    linked = [facts.get(i) for i in q["fact_ids"]]
    if not linked or None in linked:
        return "refers to a fact that isn't verified"
    quotes = [f["quote"] for f in linked]
    # The answer is one span, or several joined by "; "; each must be in a quote. Unanswerable ones have none.
    pieces = [p.strip() for p in q.get("answer", "").split(";") if p.strip()]
    sources = [{n for n, quote in enumerate(quotes) if find(p, quote)} for p in pieces]
    if rules.get("absent_term"):
        term = q.get("absent_term", "").strip()
        if not term or not _term(term).search(q["question"]):
            return "the question doesn't name its absent term"
        if corpus is not None and _term(term).search(corpus):
            return f"\"{term}\" is in the library, so the question may be answerable"
    elif not pieces or not all(sources):
        return "answer not found in its facts' quotes"
    if any(find(quote, q["question"]) for quote in quotes):
        return "question copies its quote"
    if any(YEAR_SPAN.search(quote) for quote in quotes) and not re.search(r"\b(?:19|20)\d\d", q["question"]):
        return "question drops its quote's year"
    n_words, n_facts = len(q["question"].split()), len(set(q["fact_ids"]))
    if "words" in rules and not rules["words"][0] <= n_words <= rules["words"][1]:
        return f"needs {rules['words'][0]}-{rules['words'][1]} words, has {n_words}"
    if "facts" in rules and not rules["facts"][0] <= n_facts <= rules["facts"][1]:
        return f"needs {rules['facts'][0]}-{rules['facts'][1]} facts, uses {n_facts}"
    if "pieces" in rules and not rules["pieces"][0] <= len(pieces) <= rules["pieces"][1]:
        return f"needs {rules['pieces'][0]}-{rules['pieces'][1]} answer parts, has {len(pieces)}"
    if rules.get("span_per_fact") and len(set().union(*sources)) < len(quotes):
        return "the answer needs a span from each fact"
    if rules.get("needs_number") and not NUMBER.search(q["answer"]):
        return "the answer has no number"
    if "sections" in rules and len({f["fact_id"].rsplit(":", 1)[0] for f in linked}) < rules["sections"]:
        return f"needs facts from {rules['sections']} sections"
    if "max_quote_words" in rules:
        asked, source = content_words(q["question"]), content_words(" ".join(quotes))
        if asked and len(asked & source) > rules["max_quote_words"] * len(asked):
            return "too many of the question's words come from the quote"
    if rules.get("rewrites_source") and _plain(q["question"]) == _plain(q.get("source_question", "")):
        return "the rewrite is the same as its source"
    return None


def verify_questions(questions: list[dict[str, Any]], facts: dict[str, dict[str, Any]], types: dict[str, dict],
                     corpus: str | None = None) -> tuple[list[dict], list[dict]]:
    """Keep questions that pass the audit, each by its type's rules. Returns (kept, rejected)."""
    kept, rejected = [], []
    for q in questions:
        problem = question_problem(q, facts, types[q["type"]].get("rules", {}), corpus)
        if problem:
            rejected.append({**q, "reason": problem})
        else:
            kept.append(q)
    return kept, rejected
