"""Settings from .env, file locations and the pipeline's default limits."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "pdfs"
DATA_DIR = ROOT / "data"

QUESTION_TYPES = ROOT / "question_types.toml"

MAX_FACTS = 10  # per segment
PART_CHARS = 12_000  # the fact agent reads a segment in parts this size, so its facts cover all of it
OUTPUT_TOKENS = 4096  # cap on each LLM answer
PROVIDERS = ("ollama", "gemma")
SOURCES = ("facts", "paragraph", "absent", "question", "library")  # what a question type's writer reads
RULES = {"words", "facts", "pieces", "span_per_fact", "needs_number", "max_quote_words", "sections", "absent_term",
         "rewrites_source"}  # the audit rules a question type can use (bench/verify.py)


def question_types(path: Path = QUESTION_TYPES) -> dict[str, dict]:
    """The question types in question_types.toml, after checking each has a count, a known source, a
    prompt and only known rules."""
    types = tomllib.loads(path.read_text())
    for name, t in types.items():
        unknown = set(t.get("rules", {})) - RULES
        bad_from = t.get("source") == "question" and (not t.get("from") or not set(t["from"]) <= set(types) - {name})
        problem = ("count must be a whole number" if not isinstance(t.get("count"), int) or t["count"] < 0 else
                   f"source must be one of {', '.join(SOURCES)}" if t.get("source") not in SOURCES else
                   "prompt is missing" if not str(t.get("prompt", "")).strip() else
                   f"unknown rules: {', '.join(sorted(unknown))}" if unknown else
                   "source question needs `from`: a list of other types" if bad_from else None)
        if problem:
            raise SystemExit(f"{path.name}: [{name}] {problem}")
    return types


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
        raise SystemExit(f"LLM_PROVIDER must be one of {PROVIDERS}, got {provider!r}")
    ollama_url = env("OLLAMA_URL", "http://localhost:11434/v1")
    if provider == "ollama":
        url, key, model = ollama_url, "ollama", env("OLLAMA_MODEL", "gemma4:12b")  # Ollama ignores the key
    else:
        url, key, model = env("GEMMA_URL", ""), env("GEMMA_API_KEY", ""), env("GEMMA_MODEL", "gemma-4-31b-it")
        if not url:
            raise SystemExit("LLM_PROVIDER=gemma needs GEMMA_URL (the endpoint's /v1 URL) in .env")
    return Settings(
        llm_url=url,
        llm_key=key,
        llm_model=model,
        reasoning=env("LLM_REASONING", "none"),
        ollama_url=ollama_url,
        embed_model=env("EMBED_MODEL", "bge-m3"),
        qdrant_url=env("QDRANT_URL", "http://127.0.0.1:6333"),
    )
