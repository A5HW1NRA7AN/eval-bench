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
PDF_DIR = ROOT / "pdfs"
DATA_DIR = ROOT / "data"

QUESTION_TYPES = ROOT / "question_types.toml"

FACT_CHARS = 3_000  # a segment gets one fact per this many characters of text...
MIN_FACTS = 3  # ...but at least this many...
MAX_FACTS = 15  # ...and at most this many
PART_CHARS = 12_000  # the fact agent reads a segment in parts this size, so its facts cover all of it
OUTPUT_TOKENS = 4096  # cap on each LLM answer
REQUEST_TIMEOUT = 300  # seconds: one deadline per LLM request, across the SDK's retries
MAX_RETRIES = 2  # retries of a failed LLM request by the OpenAI SDK
MAX_FAILURES = 3  # a stage stops after this many sections fail in a row (e.g. the endpoint is down)
PROVIDERS = ("ollama", "gemma")


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
    rewrites_source: bool = False  # differs from the question it rewrites
    english_only: bool = True  # no Indian-script or romanized-Hindi words (on for every type unless set false)


class QuestionType(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    enabled: bool = True
    count: int = Field(ge=0)
    source: Literal["facts", "paragraph", "absent", "question", "library"]
    prompt: str = Field(min_length=1)
    review: str = ""  # what the question reviewer checks for this type
    rules: Rules = Rules()
    from_: list[str] = Field(default=[], alias="from")  # source "question": the types it rewrites

    @model_validator(mode="after")
    def _from_needed(self) -> QuestionType:
        if self.source == "question" and not self.from_:
            raise ValueError("source 'question' needs `from`: a list of other types")
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
        if unknown := set(t.from_) - set(raw) | ({name} & set(t.from_)):
            raise BenchError(f"{path.name}: [{name}] `from` names no other type: {', '.join(sorted(unknown))}")
        types[name] = {**t.model_dump(by_alias=True, exclude={"rules"}), "rules": t.rules.model_dump(exclude_none=True)}
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


@lru_cache(maxsize=1)
def settings() -> Settings:
    load_dotenv(ROOT / ".env")
    env = os.environ.get
    provider = env("LLM_PROVIDER", "ollama")
    if provider not in PROVIDERS:
        raise BenchError(f"LLM_PROVIDER must be one of {PROVIDERS}, got {provider!r}")
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
    )
