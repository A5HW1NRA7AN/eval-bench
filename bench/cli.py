"""`bench` command line: one command per pipeline stage, run in order."""

from __future__ import annotations

import fcntl
import functools
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import typer

from bench import config
from bench.config import MAX_FACTS, BenchError

app = typer.Typer(no_args_is_help=True, add_completion=False)
DOC = typer.Option(None, help="Only PDFs whose file name contains this text")
FORCE = typer.Option(False, help="Redo segments that are already done")


@contextmanager
def _writing(command: str) -> Iterator[None]:
    """One writing command at a time per data folder: a second one stops instead of corrupting the files."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (config.DATA_DIR / ".lock").open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.seek(0)
            raise BenchError(f"another bench command is writing data/ ({f.read().strip() or 'unknown'}); "
                             "wait for it to finish") from None
        f.seek(0)
        f.truncate()
        f.write(f"pid {os.getpid()}: bench {command}")
        f.flush()
        try:
            yield
        finally:
            f.seek(0)
            f.truncate()
            fcntl.flock(f, fcntl.LOCK_UN)


def _command(writes: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """A CLI command: a BenchError becomes one red line and exit code 1; `writes` takes the data lock."""
    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                if not writes:
                    return fn(*args, **kwargs)
                with _writing(fn.__name__):
                    return fn(*args, **kwargs)
            except BenchError as exc:
                typer.secho(f"✗ {exc}", fg="red", err=True)
                raise typer.Exit(1) from None
        return wrapper
    return decorate


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Also print each agent's prompt and output")) -> None:
    """Search-quality benchmark. Every agent call is shown as it runs and saved to data/trace.jsonl."""
    from bench import llm

    llm.VERBOSE = verbose


@app.command()
@_command(writes=True)
def segment(refresh: bool = typer.Option(False, help="Re-extract page text from the PDFs")) -> None:
    """1. Cut the PDFs into segments along their tables of contents -> data/segments.jsonl."""
    from bench.segments import run_segments

    run_segments(refresh=refresh)


@app.command()
@_command(writes=True)
def facts(doc: str = DOC, max_facts: int = typer.Option(None, help=f"Facts per segment (default: each segment's last value, {MAX_FACTS} for new ones)"),
          force: bool = FORCE) -> None:
    """2. Extract facts per segment and keep those whose quote is in the PDF -> data/facts.jsonl."""
    from bench.facts import run_facts

    run_facts(doc=doc, max_facts=max_facts, force=force)


@app.command()
@_command(writes=True)
def questions(doc: str = DOC, force: bool = FORCE) -> None:
    """3. Write each question type in question_types.toml from the verified facts, keep those that pass the
    audit -> data/questions.jsonl."""
    from bench.questions import run_questions

    run_questions(doc=doc, force=force)


@app.command()
@_command(writes=True)
def index() -> None:
    """4. Build the local Qdrant test bed from every segment -> data/chunks.jsonl."""
    from bench.testbed import run_index

    run_index()


@app.command()
@_command(writes=True)
def evaluate() -> None:
    """5. Search the test bed with every question and score it -> data/report.md."""
    from bench.checks import print_checks
    from bench.evaluate import run_evaluate

    if not print_checks(["questions", "index"]):
        raise BenchError("the questions or the test bed failed a check above: fix that before scoring")
    run_evaluate()


@app.command()
@_command(writes=True)
def run() -> None:
    """Run every stage in order, checking each stage's output before the next one starts."""
    from bench.checks import print_checks
    from bench.evaluate import run_evaluate
    from bench.export import run_export
    from bench.facts import run_facts
    from bench.questions import run_questions
    from bench.segments import run_segments
    from bench.testbed import run_index

    stages = {"segment": run_segments, "facts": run_facts, "questions": run_questions, "index": run_index,
              "evaluate": run_evaluate}
    for name, stage in stages.items():
        typer.echo(f"\n== {name}")
        stage()
        if not print_checks([name]):
            raise BenchError(f"stopped: the {name} output failed a check")
    run_export()


@app.command()
@_command(writes=True)
def export() -> None:
    """Write every stage's output, the scores and the agent calls to one workbook -> data/bench.xlsx."""
    from bench.export import run_export

    run_export()


@app.command()
@_command()
def check(stage: str = typer.Argument(None, help="segment, facts, questions, index or evaluate (all if left out)")) -> None:
    """Check every stage's saved output without the LLM: is it complete, verbatim and in step?"""
    from bench.checks import STAGES, print_checks

    if stage and stage not in STAGES:
        raise typer.BadParameter(f"choose one of {', '.join(STAGES)}")
    if not print_checks([stage] if stage else None):
        raise typer.Exit(1)


@app.command()
@_command()
def serve(port: int = typer.Option(8000, help="Port on 127.0.0.1")) -> None:
    """Browse every stage's output as JSON (for Postman or a browser) on http://127.0.0.1:<port>."""
    from bench.serve import run_server

    run_server(port)


if __name__ == "__main__":
    app()
