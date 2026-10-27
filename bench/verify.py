"""Regex audit (no AI): every fact's quote and every question's answer must exist in the PDF text.

Matching is exact except for whitespace (line breaks, double spaces), quote style and dash style,
which PDFs and models write inconsistently. Anything else that differs is rejected.
"""

from __future__ import annotations

import re
from typing import Any

MAX_QUOTE_CHARS = 300
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


def verify_facts(segment: dict[str, Any], facts: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    """Keep facts whose quote is in the segment text; record where it is. Returns (kept, rejected)."""
    kept, rejected, seen = [], [], set()
    for fact in facts:
        quote, claim = fact["quote"].strip(), fact["claim"].strip()
        span = find(quote, segment["text"])
        reason = ("empty claim" if not claim else
                  f"quote longer than {MAX_QUOTE_CHARS} chars" if len(quote) > MAX_QUOTE_CHARS else
                  "quote not found in the PDF text" if span is None else
                  "duplicate quote" if span in seen else None)
        if reason:
            rejected.append({**fact, "reason": reason})
            continue
        seen.add(span)
        fact_id = f"{segment['segment_id']}:f{len(kept) + 1:02d}"
        kept.append({"fact_id": fact_id, "claim": claim, "quote": quote, "start": span[0], "end": span[1],
                     "pages": pages_of(segment, *span)})
    return kept, rejected


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOPWORDS}


def question_problem(q: dict[str, Any], facts: dict[str, dict[str, Any]]) -> str | None:
    """Why a question fails the audit, or None if it passes."""
    linked = [facts.get(i) for i in q["fact_ids"]]
    if not linked or None in linked:
        return "refers to a fact that isn't verified"
    quotes = [f["quote"] for f in linked]
    # The answer is one span, or one span per fact joined by "; "; each must be in a quote.
    pieces = [p.strip() for p in q["answer"].split(";") if p.strip()]
    sources = [{n for n, quote in enumerate(quotes) if find(p, quote)} for p in pieces]
    if not pieces or not all(sources):
        return "answer not found in its facts' quotes"
    if any(find(quote, q["question"]) for quote in quotes):
        return "question copies its quote"
    words = len(q["question"].split())
    if q["type"] == "keyword" and not 2 <= words <= 5:
        return "keyword query must be 2-5 words"
    if q["type"] == "precise" and not NUMBER.search(q["answer"]):
        return "precise answer has no number"
    if q["type"] == "multi_fact" and len(set(q["fact_ids"])) < 2:
        return "multi_fact needs two facts"
    if q["type"] == "multi_fact" and (len(pieces) < 2 or len(set().union(*sources)) < 2):
        return "multi_fact answer needs a span from each fact"
    if q["type"] == "paraphrased":
        asked, source = content_words(q["question"]), content_words(" ".join(quotes))
        if asked and len(asked & source) > len(asked) / 2:
            return "most of the paraphrase's words come from the quote"
    return None


def verify_questions(questions: list[dict[str, Any]], facts: dict[str, dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Keep questions that pass the audit. Returns (kept, rejected)."""
    kept, rejected = [], []
    for q in questions:
        problem = question_problem(q, facts)
        if problem:
            rejected.append({**q, "reason": problem})
        else:
            kept.append(q)
    return kept, rejected
