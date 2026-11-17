"""Settings from .env, file locations, the pipeline's default limits and the question types."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ROOT = Path(__file__).resolve().parent.parent
WORKSPACES = ROOT / "workspaces"
# BENCH_WORKSPACE picks a folder of its own PDFs and data (workspaces/<name>/); unset, the repo's pdfs/ and data/.
WORKSPACE = os.environ.get("BENCH_WORKSPACE", "").strip()
PDF_DIR = WORKSPACES / WORKSPACE / "pdfs" if WORKSPACE else ROOT / "pdfs"
DATA_DIR = WORKSPACES / WORKSPACE / "data" if WORKSPACE else ROOT / "data"

QUESTION_TYPES = ROOT / "question_types.toml"

FACT_CHARS = 3_000  # a segment gets one fact per this many characters of text...
MIN_FACTS = 3  # ...but at least this many...
MAX_FACTS = 15  # ...and at most this many
PART_CHARS = 12_000  # the fact agents read a segment in parts this size, so its facts cover all of it
OUTPUT_TOKENS = 4096  # cap on each LLM answer
REQUEST_TIMEOUT = 300  # seconds: one deadline per LLM request, across the SDK's retries
MAX_RETRIES = 2  # retries of a failed LLM request by the OpenAI SDK
MAX_FAILURES = 3  # a stage stops after this many sections fail in a row (e.g. the endpoint is down)
# Per model, each overridable in .env: LLM calls at once (LLM_CONCURRENCY) and how much PDF text one orchestrator
# call may read beside its map (ORCHESTRATOR_TEXT_CHARS; 0 = the map only). The 31B was probed on 2026-11-12:
# 300,000 characters (67,663 tokens) in 15 s, and 8 calls at once 4.3x faster than one at a time.
PROVIDERS = {"gemma": {"concurrency": 8, "orchestrator_text_chars": 300_000},  # self-hosted 31B, the primary model
             "ollama": {"concurrency": 1, "orchestrator_text_chars": 0}}  # local 12B fallback: Ollama runs one call at a time


class BenchError(Exception):
    """A problem the user can fix: the CLI prints it as one line and exits with code 1."""


class Rules(BaseModel):
    """The audit rules a question type can use (bench/verify.py). An unknown rule is an error."""

    model_config = ConfigDict(extra="forbid")
    words: tuple[int, int] | None = None  # words in the question
    facts: tuple[int, int] | None = None  # distinct facts used
    pieces: tuple[int, int] | None = None  # answer spans joined by "; "
    span_per_fact: bool = False  # each fact gives an answer span
    needs_number: bool = False  # the answer holds a number, in digits or words
    max_quote_words: float | None = None  # share of the question's content words that may come from its quotes
    sections: int | None = None  # facts from at least this many sections
    absent_term: bool = False  # names a term found nowhere in the library
    english_only: bool = True  # no Indian-script or romanized-Hindi words (on for every type unless set false)


class QuestionType(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    count: int = Field(ge=0)
    source: Literal["facts", "paragraph", "absent", "library"]
    prompt: str = Field(min_length=1)
    review: str = ""  # what the question reviewer checks for this type
    rules: Rules = Rules()

    @model_validator(mode="after")
    def _library(self) -> QuestionType:
        if (self.source == "library") != ((self.rules.sections or 1) >= 2):
            raise ValueError("source 'library' and rules.sections >= 2 go together: facts from several sections")
        return self


def question_types(path: Path = QUESTION_TYPES, active_only: bool = True) -> dict[str, dict]:
    """The question types in question_types.toml, checked strictly: an unknown key, rule or source is an
    error. By default only active types (enabled, count > 0) are returned, so a switched-off type costs no call."""
    try:
        raw = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise BenchError(f"{path.name}: {exc}") from exc
    types = {}
    for name, table in raw.items():
        try:
            t = QuestionType.model_validate(table)
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'type'}: {e['msg']}" for e in exc.errors())
            raise BenchError(f"{path.name}: [{name}] {problems}") from exc
        types[name] = {**t.model_dump(exclude={"rules"}), "rules": t.rules.model_dump(exclude_none=True)}
    return {n: t for n, t in types.items() if t["enabled"] and t["count"] > 0} if active_only else types


@dataclass(frozen=True)
class Settings:
    llm_url: str
    llm_key: str = field(repr=False)
    llm_model: str
    reasoning: str  # sent as reasoning_effort; "none" turns Gemma 4's thinking off
    ollama_url: str
    embed_model: str
    qdrant_url: str
    concurrency: int = 1  # LLM calls at once
    orchestrator_text_chars: int = 0  # PDF text one orchestrator call may read beside its map


@lru_cache(maxsize=1)
def settings() -> Settings:
    load_dotenv(ROOT / ".env")
    env = os.environ.get
    provider = env("LLM_PROVIDER", "ollama")
    if provider not in PROVIDERS:
        raise BenchError(f"LLM_PROVIDER must be one of {tuple(PROVIDERS)}, got {provider!r}")
    profile = PROVIDERS[provider]
    try:
        concurrency = int(env("LLM_CONCURRENCY") or profile["concurrency"])
        text_chars = int(env("ORCHESTRATOR_TEXT_CHARS") or profile["orchestrator_text_chars"])
    except ValueError as exc:
        raise BenchError(f"LLM_CONCURRENCY and ORCHESTRATOR_TEXT_CHARS must be whole numbers ({exc})") from exc
    if concurrency < 1 or text_chars < 0:
        raise BenchError("LLM_CONCURRENCY must be at least 1 and ORCHESTRATOR_TEXT_CHARS at least 0")
    ollama_url = env("OLLAMA_URL", "http://localhost:11434/v1")
    if provider == "ollama":
        url, key, model = ollama_url, "ollama", env("OLLAMA_MODEL", "gemma4:12b")  # Ollama ignores the key
    else:
        url, key, model = env("GEMMA_URL", ""), env("GEMMA_API_KEY", ""), env("GEMMA_MODEL", "gemma-4-31b-it")
        if not url:
            raise BenchError("LLM_PROVIDER=gemma needs GEMMA_URL (the endpoint's /v1 URL) in .env")
    return Settings(
        llm_url=url,
        llm_key=key,
        llm_model=model,
        reasoning=env("LLM_REASONING", "none"),
        ollama_url=ollama_url,
        embed_model=env("EMBED_MODEL", "bge-m3"),
        qdrant_url=env("QDRANT_URL", "http://127.0.0.1:6333"),
        concurrency=concurrency,
        orchestrator_text_chars=text_chars,
    )
