"""Regex audit (no AI): every fact's quote and every question's answer must exist in the PDF text.

A fact points at a range of numbered sentences and its quote is cut from the segment text there, so it
can't be retyped wrongly; the audit checks the range and that the claim says nothing its quote doesn't.
An answer is matched in its quotes exactly except for whitespace (line breaks, double spaces), quote style
and dash style, which PDFs and models write inconsistently. Anything else that differs is rejected.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from itertools import pairwise
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
YEAR_SPAN = re.compile(r"\b(?:19|20)\d\d[-–]\d\d\b")  # a reporting year such as 2024-25 (hyphen or en dash)
# Characters that PDFs and models write in several forms.
VARIANTS = {"'": "['‘’`]", "‘": "['‘’`]", "’": "['‘’`]", '"': "[\"“”]", "“": "[\"“”]", "”": "[\"“”]",
            "-": "[-‐‑–—]", "‐": "[-‐‑–—]", "–": "[-‐‑–—]", "—": "[-‐‑–—]"}
STOPWORDS = {"what", "which", "when", "where", "does", "should", "with", "from", "that", "this", "have", "their",
             "they", "into", "about", "after", "before", "much", "many", "used", "your", "there", "than"}
# Questions are English only for now: no Indian-script characters and no romanized-Hindi words.
INDIC_SCRIPT = re.compile(r"[\u0900-\u0DFF]")  # Devanagari to Sinhala, Gujarati included
HINGLISH = {"kitna", "kitni", "kitne", "kya", "kyu", "kyon", "kaise", "kaisa", "kaun", "kaunsa", "kaunse", "kaunsi",
            "kab", "kahan", "hai", "hain", "hota", "hoti", "hote", "chahiye", "karna", "karne", "karein", "dena",
            "mein", "nahi", "nahin", "bhi", "aur", "doodh", "bhains", "gaay", "bakri", "chara", "wala", "wali", "jyada"}
# Named answer shapes that templates refer to (templates never hold raw regex).
_NUM = r"\d[\d,]*(?:\.\d+)?"
_SPELLED = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_TIME_UNIT = r"(?:hours?|hrs?|days?|weeks?|months?|years?)"
SHAPES = {
    "money": re.compile(rf"(?:₹|\bRs\.?|\bINR\b|\brupees?\b)\s*{_NUM}|{_NUM}\s*(?:lakhs?|crores?)\b|\b(?:lakhs?|crores?)\s+(?:of\s+)?rupees?", re.I),
    "percent": re.compile(rf"{_NUM}\s*(?:%|per\s?cent\b|percent\b)", re.I),
    "ratio": re.compile(r"(?<![\d:.])\d{1,3}\s?:\s?\d{1,3}(?![\d:])(?!\s*(?:am|pm|hrs?)\b)", re.I),  # 60:40, not 10:30 am
    "age": re.compile(rf"(?:{_NUM}|{_SPELLED})(?:\s*(?:-|–|to)\s*(?:{_NUM}|{_SPELLED}))?\s*{_TIME_UNIT}\b", re.I),
    "interval": re.compile(r"\b(?:every|once|twice|annually|yearly|monthly|weekly|daily|six[- ]monthly|half[- ]yearly|booster)\b", re.I),
    "quantity": re.compile(rf"{_NUM}\s*(?:million|lakh|crore|thousand|billion)?\s*(?:tonnes?|tons?|kg|kilograms?|litres?|liters?|grams?|gm|g|heads?|eggs|hectares?|ha)\b", re.I),
    "rate": re.compile(rf"{_NUM}\s*(?:kg|g|gm|grams?|litres?|liters?|ml|l)\s*(?:/|per)\s*(?:day|animal|capita|head|year|lactation)", re.I),
    "organisation": re.compile(r"\b[A-Z][A-Z&]{1,}[A-Z0-9]*\b|\b(?:Board|Bank|Ministry|Department|Council|Corporation|Federation|Agency)\b"),
}


def reason_group(reason: str) -> str:
    """One name per audit rule: '"goats" is in the library…' and 'needs 2-5 words, has 9' lose their specifics."""
    rule = re.sub(r"^overlaps .*", "overlaps a kept fact", reason.split(":")[0])
    return re.sub(r"\d+", "N", re.sub(r'"[^"]*"', "a term", rule))


def has_shape(names: list[str], text: str) -> bool:
    """True if the text holds an answer of at least one of the named shapes."""
    return any(SHAPES[n].search(text) for n in names)


ID_TOKEN = re.compile(r"\b[dfgipqx]\d{1,3}(?:_\d+)?\b")  # the short ids agents see (d8, f3, g7, p4, i2, d8_1), case-sensitive
NUMBER = re.compile(r"\d|\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
                    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|"
                    r"ninety|hundred|thousand|lakhs?|crores?|million|billion|dozen|half|twice|double)\b", re.I)


def pattern(snippet: str) -> re.Pattern[str]:
    """A regex for the snippet that tolerates whitespace, quote and dash variants."""
    parts = ["".join(VARIANTS.get(c, re.escape(c)) for c in token) for token in snippet.split()]
    # A word broken across lines ("milk-\nproduction") may have whitespace after its dash.
    joined = r"\s+".join(parts)
    return re.compile(joined.replace("[-‐‑–—]", r"[-‐‑–—]\s*"))


def find(snippet: str, text: str, whole: bool = False) -> tuple[int, int] | None:
    """(start, end) of the snippet in the text, or None. With `whole`, it must not start or end inside a
    larger word or number ("1 kg" doesn't match inside "11 kg")."""
    if not snippet.strip():
        return None
    regex = pattern(snippet)
    if whole:
        first, last = snippet.strip()[0], snippet.strip()[-1]
        regex = re.compile((r"(?<![\w.])(?<!\d[-‐‑–—])" if first.isalnum() else "") + regex.pattern  # not in "3-5 days"
                           + (r"(?!\w)(?![.,]\d)" if last.isalnum() else ""))  # not "10" of "10.5"
    match = regex.search(text)
    return (match.start(), match.end()) if match else None


def uid(*parts: Any) -> str:
    """A short content id: the same content always gets the same id, across runs."""
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:12]


def pages_of(segment: dict[str, Any], start: int, end: int) -> list[int]:
    hit = [s["page"] for s in segment["page_spans"] if s["start"] < end and s["end"] > start]
    return [hit[0], hit[-1]] if hit else []


def table_row(quote: str) -> bool:
    """Mostly numbers: a bare table row, which says nothing without its headers."""
    tokens = quote.split()
    return sum(bool(NUMERIC_TOKEN.match(t)) for t in tokens) >= len(tokens) / 2


def _numeric_cell(cell: str) -> bool:
    """A value: every token a number ('239.30', '-0.50', 'NIL'); a year ('2023-24') is a label."""
    tokens = cell.split()
    return bool(tokens) and all(NUMERIC_TOKEN.match(t) for t in tokens) and not re.fullmatch(r"(?:19|20)\d\d(?:[-–]\d\d)?", cell)


def data_row(line: str) -> bool:
    """A table line (cells joined by " | ") with values in at least half of its cells."""
    cells = [c.strip() for c in line.split("|") if c.strip()]
    values = sum(_numeric_cell(c) for c in cells)
    return " | " in line and values > 0 and values * 2 >= len(cells)


HEADER_LINES = 6  # a table's title and header rows (statistical tables often have a title row and three header rows)


def table_header(text: str, start: int) -> tuple[int, int] | None:
    """(start, end) of the header rows of the table holding the row at `start`: the table lines of its paragraph
    before the first row of values (at most HEADER_LINES), or None if the table has no such rows."""
    para_start = text.rfind("\n\n", 0, start) + 2 if "\n\n" in text[:start] else 0
    first = header_end = None
    for line in re.finditer(r"[^\n]+", text[para_start:start]):
        if " | " not in line.group():
            if first is None:
                continue  # a caption line before the table
            break
        if data_row(line.group()):
            break
        first = para_start + line.start() if first is None else first
        header_end = para_start + line.end()
    if first is None or text[first:header_end].count("\n") >= HEADER_LINES:
        return None
    return first, header_end


def evidence_text(fact: dict[str, Any]) -> str:
    """A fact's quote, after its table's header rows when the quote is a table row."""
    return f"{fact['header']}\n{fact['quote']}" if fact.get("header") else fact["quote"]


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


CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")  # a garbled model answer ('5\x00C' for '5°C'); text from the PDFs has none


def claim_problem(claim: str, quote: str, segment: dict[str, Any]) -> str | None:
    """Why the claim says more than its quote supports, or None. Values must be in the quote; dates,
    years and acronyms are qualifiers, so they must be somewhere in the section."""
    if CONTROL.search(claim):
        return "claim has a control character (a garbled model answer)"
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


def kept_fact_problem(fact: dict[str, Any], segment: dict[str, Any]) -> str | None:
    """Why a kept fact no longer passes the fact rules (the rules or the segment text changed), or None."""
    quote, text = fact["quote"], segment["text"]
    if text[fact["start"] : fact["end"]] != quote:
        return "quote is no longer at its recorded position"
    if fact.get("header") and text[fact["header_start"] : fact["header_end"]] != fact["header"]:
        return "table header is no longer at its recorded position"
    return table_problem(quote, fact["start"], text) or (
        f"quote longer than {MAX_QUOTE_CHARS} chars" if len(quote) > MAX_QUOTE_CHARS else
        claim_problem(fact["claim"], evidence_text(fact), segment))


def starts_with_values(quote: str) -> bool:
    """True if the quote's first line is a table row of values (so its meaning is in header rows above it)."""
    return data_row(quote.strip().split("\n", 1)[0])


def table_problem(quote: str, start: int, text: str) -> str | None:
    """Why a quote can't stand as a fact for being table text, or None: table rows need their header rows, in
    the quote or above it in the same table; bare numbers without headers say nothing."""
    if " | " in quote:
        return "table rows without header rows" if starts_with_values(quote) and table_header(text, start) is None else None
    return "quote is a table row, not a sentence" if table_row(quote) else None


def verify_facts(segment: dict[str, Any], facts: list[dict[str, Any]], prefix: str = "f") -> tuple[list[dict], list[dict]]:
    """Cut each fact's quote from its sentence range and keep it if it passes every rule.
    Returns (kept, rejected); kept facts record their quote, position, pages and size."""
    units = {s["id"]: s for s in sentences(segment["text"])}
    kept, rejected = [], []
    for fact in facts:
        claim, span = fact["claim"].strip(), span_of(fact, units)
        quote = segment["text"][span[0] : span[1]] if isinstance(span, tuple) else ""
        overlap = next((k["fact_id"] for k in kept if isinstance(span, tuple)
                        and min(span[1], k["end"]) - max(span[0], k["start"]) > min(span[1] - span[0], k["end"] - k["start"]) / 2), None)
        header = table_header(segment["text"], span[0]) if isinstance(span, tuple) and starts_with_values(quote) else None
        table = {"header": segment["text"][header[0] : header[1]], "header_start": header[0], "header_end": header[1]} if header else {}
        reason = ("empty claim" if not claim else
                  span if isinstance(span, str) else
                  table_problem(quote, span[0], segment["text"]) or
                  (f"quote longer than {MAX_QUOTE_CHARS} chars" if len(quote) > MAX_QUOTE_CHARS else None) or
                  claim_problem(claim, evidence_text({"quote": quote, **table}), segment) or
                  (f"overlaps {overlap}" if overlap else None))
        if reason:
            rejected.append({**fact, "quote": quote, "reason": reason, "decided_by": "audit"})
            continue
        fact_id = f"{segment['segment_id']}:{prefix}{len(kept) + 1:02d}"
        kept.append({"fact_id": fact_id, "uid": uid(segment["segment_id"], span[0], span[1], claim, quote), **fact,
                     "claim": claim, "quote": quote, "start": span[0], "end": span[1], **table,
                     "pages": pages_of(segment, *span), "size": size_of(fact, units)})
    return kept, rejected


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOPWORDS}


@lru_cache(maxsize=4096)
def name_pattern(name: str) -> re.Pattern[str]:
    """A name in any case, singular or plural, its words apart by spaces or hyphens ('Aflatoxins' finds 'aflatoxin')."""
    words = [re.escape(w[:-1] if w.endswith("s") and len(w) > 3 else w) + "(?:e?s)?" for w in re.findall(r"[\w.&'’]+", name)]
    return re.compile(r"(?<!\w)" + r"[\s\-‐–]+".join(words) + r"(?!\w)", re.I)


FUNCTION_WORDS = frozenset(["a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of", "on", "or", "the", "to", "with"])


@lru_cache(maxsize=2)
def library_words(corpus: str) -> frozenset[str]:
    """Every word of the library's text in lower case, and every two adjacent words joined ('Blue Tongue': bluetongue)."""
    words = re.findall(r"[a-z]+", corpus.lower())
    return frozenset(words) | frozenset(a + b for a, b in pairwise(words))


def in_library(term: str, corpus: str) -> list[str]:
    """The words of a term that the library uses, in any case, singular or plural; function words don't count.
    A term is absent from the library only when this is empty ('sheep and lambs' is not: sheep is there)."""
    words = library_words(corpus)

    def forms(w: str) -> set[str]:
        return {w, w + "s", w + "es", w.removesuffix("s"), w.removesuffix("es"), *([w[:-1] + "ves"] if w.endswith("f") else [])}  # calf: calves

    return [w for w in re.findall(r"[a-z]+", term.lower()) if w not in FUNCTION_WORDS and forms(w) & words]


def plain(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def not_english(text: str) -> str | None:
    """What makes the text not plain English (an Indian-script letter or romanized-Hindi words), or None."""
    if match := INDIC_SCRIPT.search(text):
        return match.group()
    words = sorted(set(re.findall(r"[a-z]+", text.lower())) & HINGLISH)
    return ", ".join(words) or None


def question_problem(q: dict[str, Any], facts: dict[str, dict[str, Any]], rules: dict[str, Any],
                     corpus: str | None = None) -> str | None:
    """Why a question fails the audit, or None if it passes: the general rules, then its type's rules
    (question_types.toml). `corpus` is the whole library's text, for the absent_term rule."""
    if any(CONTROL.search(q.get(k, "")) for k in ("question", "answer", "absent_term")):
        return "has a control character (a garbled model answer)"
    linked = [facts.get(i) for i in q["fact_ids"]]
    if not linked or None in linked:
        return "refers to a fact that isn't verified"
    quotes = [evidence_text(f) for f in linked]  # a table row with its header rows
    # The answer is one span, or several joined by "; "; each must be in a quote. Unanswerable ones have none.
    pieces = [p.strip() for p in q.get("answer", "").split(";") if p.strip()]
    sources = [{n for n, quote in enumerate(quotes) if find(p, quote, whole=True)} for p in pieces]
    if rules.get("absent_term"):
        term = q.get("absent_term", "").strip()
        if not term or not name_pattern(term).search(q["question"]):
            return "the question doesn't name its absent term"
        if corpus is not None and in_library(term, corpus):
            return f"\"{term}\" is in the library, so the question may be answerable"
    elif not pieces or not all(sources):
        return "answer not found in its facts' quotes"
    if any(find(f["quote"], q["question"]) for f in linked):
        return "question copies its quote"
    if ids := sorted(t for t in set(ID_TOKEN.findall(q["question"]))  # "f1 crossbred" or "vitamin d3" from a quote is fine
                     if not any(re.search(rf"\b{t}\b", quote, re.I) for quote in quotes)):
        return f"question contains an internal id: {', '.join(ids)}"
    if rules.get("english_only", True) and (foreign := not_english(q["question"])):
        return f"question isn't plain English: {foreign}"
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
    return None
