"""The LLM behind every agent: local Ollama or the remote Gemma 4 endpoint, through one code path.

Every agent call goes through `call`, which shows it live in the terminal and appends it to data/trace.jsonl.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from bench.config import DATA_DIR, MAX_RETRIES, OUTPUT_TOKENS, REQUEST_TIMEOUT, settings

TRACE = DATA_DIR / "trace.jsonl"
RUN_ID = f"{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"  # tells runs apart in the trace
console = Console(highlight=False)
VERBOSE = False  # `bench --verbose`: also print each prompt and output

# Ollama's /v1 rejects forced tool calls and never shows the JSON schema to the model, so every Agent
# is built with NativeOutput (native JSON-schema output) and this profile writes the schema into the instructions.
PROFILE = OpenAIModelProfile(
    native_output_requires_schema_in_instructions=True,
    openai_chat_supports_multiple_system_messages=False,
    openai_chat_supports_max_completion_tokens=False,
)
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # keep CLI output to our own lines


def model() -> OpenAIChatModel:
    s = settings()
    model_settings = OpenAIChatModelSettings(temperature=0.0, seed=7, max_tokens=OUTPUT_TOKENS,
                                             request_timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES)
    if s.reasoning:
        model_settings["openai_reasoning_effort"] = s.reasoning
    provider = OpenAIProvider(base_url=s.llm_url, api_key=s.llm_key)
    return OpenAIChatModel(s.llm_model, provider=provider, profile=PROFILE, settings=model_settings)


def check_usage(input_tokens: int, sent_text: str, label: str) -> None:
    """Warn when the model read far less than we sent: a sign the prompt was truncated."""
    if input_tokens and input_tokens < 0.5 * len(sent_text) / 4.5:
        console.print(f"  [yellow]! {escape(label)}: model read {input_tokens} tokens for {len(sent_text)} chars; was the prompt truncated?[/]")


def _summary(output: BaseModel) -> str:
    """'6 facts' for an output holding one list, else the output's field names."""
    lists = {k: v for k, v in output if isinstance(v, list)}
    return ", ".join(f"{len(v)} {k}" for k, v in lists.items()) or ", ".join(type(output).model_fields)


def _trace(record: dict[str, Any]) -> None:
    TRACE.parent.mkdir(parents=True, exist_ok=True)
    with TRACE.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def call(role: str, agent: Agent[Any, Any], prompt: str, label: str, **kwargs: Any) -> Any:
    """Run one agent call: a spinner while it runs, one line when done, and a record in the trace.
    A failed call is traced with its error and re-raised."""
    started = time.monotonic()
    base = {"time": datetime.now().isoformat(timespec="seconds"), "run": RUN_ID, "role": role, "label": label,
            "model": settings().llm_model}
    try:
        with console.status(f"[cyan]{role}[/] ▸ {escape(label)} …"):
            result = agent.run_sync(prompt, **kwargs)
    except Exception as exc:
        _trace({**base, "seconds": round(time.monotonic() - started, 2), "prompt": prompt,
                "error": f"{type(exc).__name__}: {exc}"})
        raise
    seconds, usage, response = time.monotonic() - started, result.usage, result.response
    check_usage(usage.input_tokens, prompt, label)
    console.print(f"  [cyan]{role}[/] ▸ {escape(label)} → {_summary(result.output)} "
                  f"[dim]({seconds:.1f}s, {usage.input_tokens:,} in / {usage.output_tokens:,} out)[/]")
    output = result.output.model_dump()
    if VERBOSE:
        console.print(Panel(prompt, title=f"{role} prompt", border_style="dim"))
        console.print(Panel(json.dumps(output, indent=1, ensure_ascii=False), title=f"{role} output", border_style="cyan"))
    _trace({**base, "served_model": response.model_name, "finish_reason": response.finish_reason,
            "requests": usage.requests, "seconds": round(seconds, 2), "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens, "prompt": prompt, "output": output})
    return result.output
