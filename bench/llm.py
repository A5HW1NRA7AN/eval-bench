"""The LLM behind every agent: local Ollama or the remote Gemma 4 endpoint, through one code path.

Every agent call goes through `call_all` (`call` for one), which shows it live in the terminal and appends it to
data/trace.jsonl. Answers are cached in data/cache/ by everything that shapes them (model, settings, instructions,
schema, prompt), so a re-run, or a run after a change, only asks the model what it hasn't been asked before. The
calls of one batch run together, up to LLM_CONCURRENCY at once, and their answers come back in job order.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput, PromptedOutput
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from bench.config import DATA_DIR, MAX_RETRIES, OUTPUT_TOKENS, REQUEST_TIMEOUT, settings
from bench.pdf import write_text

TRACE = DATA_DIR / "trace.jsonl"
CACHE = DATA_DIR / "cache"
USE_CACHE = True  # `bench --no-cache`: ask the model again (answers are still saved)
RUN_ID = f"{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"  # tells runs apart in the trace
CALLS: Counter[str] = Counter()  # calls per role in this process, cached ones included
console = Console(highlight=False)
VERBOSE = False  # `bench --verbose`: also print each prompt and output
Job = tuple[str, Agent[Any, Any], str, str]  # (role, agent, prompt, label)

# Ollama's /v1 rejects forced tool calls and never shows the JSON schema to the model, so every Agent
# is built with NativeOutput (native JSON-schema output) and this profile writes the schema into the instructions.
PROFILE = OpenAIModelProfile(
    native_output_requires_schema_in_instructions=True,
    openai_chat_supports_multiple_system_messages=False,
    openai_chat_supports_max_completion_tokens=False,
    supports_json_object_output=False,  # the fallback below must be free text: JSON mode is guided decoding too
)
# The Gemma endpoint's guided decoding garbles non-ASCII it writes as \uXXXX escapes ('5\x00C' for '5°C', or
# '\u000b' until the token limit for 'µg'). Such an answer is asked again as free text that holds the JSON.
GARBLED = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # keep CLI output to our own lines


def model() -> OpenAIChatModel:
    s = settings()
    model_settings = OpenAIChatModelSettings(temperature=0.0, seed=7, max_tokens=OUTPUT_TOKENS,
                                             request_timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES)
    if s.reasoning:
        model_settings["openai_reasoning_effort"] = s.reasoning
    provider = OpenAIProvider(base_url=s.llm_url, api_key=s.llm_key)
    return OpenAIChatModel(s.llm_model, provider=provider, profile=PROFILE, settings=model_settings)


def agent(output: type[BaseModel], instructions: str) -> Agent[Any, Any]:
    """An agent that answers in `output`'s JSON schema. It remembers what it was built with, for the cache key."""
    built = Agent(model(), output_type=NativeOutput(output, strict=True), instructions=instructions)
    built.bench_spec = (output, instructions)  # type: ignore[attr-defined]
    return built


def cache_key(agent: Agent[Any, Any], prompt: str, model_settings: dict[str, Any] | None) -> str:
    """Everything that shapes an answer; deadlines and retries don't."""
    output, instructions = agent.bench_spec  # type: ignore[attr-defined]
    shaping = {k: v for k, v in {**(agent.model.settings or {}), **(model_settings or {})}.items()
               if k not in ("request_timeout", "max_retries")}
    parts = [agent.model.model_name, json.dumps(shaping, sort_keys=True, default=str), instructions,
             json.dumps(output.model_json_schema(), sort_keys=True), prompt]
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def check_usage(input_tokens: int, sent_text: str, label: str) -> None:
    """Warn when the model read far less than we sent: a sign the prompt was truncated."""
    if input_tokens and input_tokens < 0.5 * len(sent_text) / 4.5:
        console.print(f"  [yellow]! {escape(label)}: model read {input_tokens} tokens for {len(sent_text)} chars; was the prompt truncated?[/]")


def _summary(output: BaseModel) -> str:
    """'6 facts' for an output holding one list, else the output's field names."""
    lists = {k: v for k, v in output if isinstance(v, list)}
    return ", ".join(f"{len(v)} {k}" for k, v in lists.items()) or ", ".join(type(output).model_fields)


def _strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        value = list(value.values())
    return [t for v in value for t in _strings(v)] if isinstance(value, list) else [value] if isinstance(value, str) else []


def _trace(record: dict[str, Any]) -> None:
    TRACE.parent.mkdir(parents=True, exist_ok=True)
    with TRACE.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _cached(agent: Agent[Any, Any], prompt: str, model_settings: dict[str, Any] | None) -> tuple[Path, dict[str, Any] | None, Any]:
    """Where the answer is cached, and the cached record and answer when there is a usable one (an unreadable
    or garbled entry is asked again)."""
    key = cache_key(agent, prompt, model_settings)
    path = CACHE / key[:2] / f"{key}.json"
    if USE_CACHE and path.exists():
        try:
            hit = json.loads(path.read_text())
            output = agent.bench_spec[0].model_validate(hit["output"])  # type: ignore[attr-defined]
            if not any(GARBLED.search(text) for text in _strings(hit["output"])):
                return path, hit, output
        except (ValueError, KeyError):
            pass
    return path, None, None


async def _ask(agent: Agent[Any, Any], prompt: str, kwargs: dict[str, Any]) -> tuple[Any, str]:
    """Native JSON-schema output first; a garbled or unparseable answer is asked again as free text holding the JSON."""
    try:
        result, mode = await agent.run(prompt, **kwargs), "native"
        garbled = any(GARBLED.search(text) for text in _strings(result.output.model_dump()))
    except UnexpectedModelBehavior:
        garbled = True
    if garbled:
        output, instructions = agent.bench_spec  # type: ignore[attr-defined]
        fallback = Agent(agent.model, output_type=PromptedOutput(output), instructions=instructions)
        result, mode = await fallback.run(prompt, **kwargs), "prompted"
    return result, mode


async def _live(job: Job, base: dict[str, Any], path: Path, kwargs: dict[str, Any]) -> Any:
    """Ask the model, then print, cache and trace the answer. A failed call is traced with its error and re-raised."""
    role, agent, prompt, label = job
    base, started = {**base, "time": datetime.now().isoformat(timespec="seconds")}, time.monotonic()  # when it really starts
    try:
        result, mode = await _ask(agent, prompt, kwargs)
    except Exception as exc:
        _trace({**base, "seconds": round(time.monotonic() - started, 2), "prompt": prompt, "error": f"{type(exc).__name__}: {exc}"})
        raise
    seconds, usage, response = time.monotonic() - started, result.usage, result.response
    check_usage(usage.input_tokens, prompt, label)
    console.print(f"  [cyan]{role}[/] ▸ {escape(label)} → {_summary(result.output)} "
                  f"[dim]({seconds:.1f}s, {usage.input_tokens:,} in / {usage.output_tokens:,} out)[/]")
    output = result.output.model_dump()
    if VERBOSE:
        console.print(Panel(prompt, title=f"{role} prompt", border_style="dim"))
        console.print(Panel(json.dumps(output, indent=1, ensure_ascii=False), title=f"{role} output", border_style="cyan"))
    write_text(path, json.dumps({"served_model": response.model_name, "output": output}, ensure_ascii=False))
    _trace({**base, "mode": mode, "served_model": response.model_name, "finish_reason": response.finish_reason,
            "requests": usage.requests, "seconds": round(seconds, 2), "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens, "prompt": prompt, "output": output})
    return result.output


_LOOP: asyncio.AbstractEventLoop | None = None


def _run(coro: Any) -> Any:
    """Run a batch on one event loop kept for the process, so agents keep their connections between batches.
    On Ctrl-C the batch is cancelled and cleaned up before the interrupt goes on."""
    global _LOOP
    if _LOOP is None or _LOOP.is_closed():
        _LOOP = asyncio.new_event_loop()
    task = _LOOP.create_task(coro)
    try:
        return _LOOP.run_until_complete(task)
    except BaseException:
        if not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                _LOOP.run_until_complete(task)
        raise


def call_all(jobs: list[Job], tolerate: list[str] | None = None, **kwargs: Any) -> list[Any]:
    """Run agent calls, up to LLM_CONCURRENCY at once: cached answers first, the rest from the model (identical
    calls are asked once and share the answer). Answers come back in job order. After a failed call (e.g. the
    endpoint is down) no new call starts, and once the running ones finish (and are cached) the first error is
    re-raised; with `tolerate`, an answer that stays unparseable is None instead, and its label is added to that list."""
    answers: list[Any] = [None] * len(jobs)
    live: list[tuple[int, dict[str, Any], Path]] = []
    asked: dict[Path, int] = {}  # cache file -> the job that asks for it
    twins: list[tuple[int, dict[str, Any], int, Path]] = []  # (job, its trace record, the identical job that asks, cache file)
    for n, (role, agent, prompt, label) in enumerate(jobs):
        CALLS[role] += 1
        base = {"time": datetime.now().isoformat(timespec="seconds"), "run": RUN_ID, "role": role, "label": label,
                "model": settings().llm_model}
        path, hit, output = _cached(agent, prompt, kwargs.get("model_settings"))
        if hit is None and path in asked:
            twins.append((n, base, asked[path], path))
        elif hit is None:
            asked[path] = n
            live.append((n, base, path))
        else:
            console.print(f"  [cyan]{role}[/] ▸ {escape(label)} → {_summary(output)} [dim](cached)[/]")
            _trace({**base, "cached": path.stem, "served_model": hit.get("served_model"), "seconds": 0, "prompt": prompt,
                    "output": hit["output"]})
            answers[n] = output
    if not live:
        return answers
    gate, outage = asyncio.Semaphore(settings().concurrency), []

    async def one(n: int, base: dict[str, Any], path: Path) -> Any:
        async with gate:
            if outage:  # an earlier call failed for good: don't start more
                raise outage[0]
            try:
                return await _live(jobs[n], base, path, kwargs)
            except UnexpectedModelBehavior:
                raise
            except Exception as exc:
                outage.append(exc)
                raise

    async def batch() -> list[Any]:
        return await asyncio.gather(*(one(*item) for item in live), return_exceptions=True)

    role, _, _, label = jobs[live[0][0]]
    roles = ", ".join(dict.fromkeys(jobs[n][0] for n, _, _ in live))
    with console.status(f"[cyan]{role}[/] ▸ {escape(label)} …" if len(live) == 1 else f"{len(live)} calls ({escape(roles)}) …"):
        outcomes = dict(zip((n for n, _, _ in live), _run(batch())))
    for n, base, first, path in twins:  # identical to a call in this batch: its answer, traced like a cached one
        outcomes[n] = outcomes[first]
        if not isinstance(outcomes[n], BaseException):
            _trace({**base, "cached": path.stem, "seconds": 0, "prompt": jobs[n][2], "output": outcomes[n].model_dump()})
    errors = []
    for n, out in sorted(outcomes.items()):
        if isinstance(out, UnexpectedModelBehavior) and tolerate is not None:
            tolerate.append(jobs[n][3])
            console.print(f"  [yellow]! {escape(jobs[n][0])} ▸ {escape(jobs[n][3])}: no readable answer; left empty, "
                          f"and retried on the next run[/]")
        elif isinstance(out, BaseException):
            errors.append(out)
        else:
            answers[n] = out
    if errors:
        raise errors[0]
    return answers


def call(role: str, agent: Agent[Any, Any], prompt: str, label: str, **kwargs: Any) -> Any:
    """Run one agent call (see call_all)."""
    return call_all([(role, agent, prompt, label)], **kwargs)[0]
