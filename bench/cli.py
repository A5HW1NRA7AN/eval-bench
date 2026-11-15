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
from bench.config import FACT_CHARS, MAX_FACTS, MIN_FACTS, BenchError

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
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Also print each agent's prompt and output"),
         no_cache: bool = typer.Option(False, "--no-cache", help="Ask the model again instead of using cached answers")) -> None:
    """Search-quality benchmark. Every agent call is shown as it runs and saved to data/trace.jsonl;
    answers are cached in data/cache/."""
    from bench import llm

    llm.VERBOSE, llm.USE_CACHE = verbose, not no_cache


@app.command()
@_command(writes=True)
def segment(refresh: bool = typer.Option(False, help="Re-extract page text from the PDFs")) -> None:
    """1. Cut the PDFs into segments along their tables of contents -> data/segments.jsonl."""
    from bench.segments import run_segments

    run_segments(refresh=refresh)


@app.command()
@_command(writes=True)
def facts(doc: str = DOC, max_facts: int = typer.Option(None, help=f"Facts per segment (default: one per {FACT_CHARS:,} characters, {MIN_FACTS}-{MAX_FACTS})"),
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
def run(no_templates: bool = typer.Option(False, "--no-templates", help="Skip the template questions stage")) -> None:
    """Run every stage in order, checking each stage's output before the next one starts."""
    from bench.checks import print_checks
    from bench.evaluate import run_evaluate
    from bench.export import run_export
    from bench.facts import run_facts
    from bench.orchestrator import run_templates
    from bench.questions import run_questions
    from bench.segments import run_segments
    from bench.testbed import run_index

    stages = {"segment": run_segments, "facts": run_facts, "questions": run_questions,
              **({} if no_templates else {"templates": run_templates}), "index": run_index, "evaluate": run_evaluate}
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
@_command(writes=True)
def testset() -> None:
    """Write the test set to hand to another team -> data/testset.json and data/testset.csv."""
    from bench.export import run_testset

    run_testset()


@app.command()
@_command()
def check(stage: str = typer.Argument(None, help="segment, facts, questions, templates, index or evaluate (all if left out)")) -> None:
    """Check every stage's saved output without the LLM: is it complete, verbatim and in step?"""
    from bench.checks import STAGES, print_checks

    if stage and stage not in STAGES:
        raise typer.BadParameter(f"choose one of {', '.join(STAGES)}")
    if not print_checks([stage] if stage else None):
        raise typer.Exit(1)


templates_app = typer.Typer(no_args_is_help=True, help="Topic templates: what users ask about one kind of document (templates/).")
app.add_typer(templates_app, name="templates")


@templates_app.command("run")
@_command(writes=True)
def templates_run(doc: str = DOC, force: bool = FORCE) -> None:
    """Per PDF, the orchestrator plans facts and questions for each section's switched-on intents and workers
    write them -> data/templates.jsonl."""
    from bench.orchestrator import run_templates

    run_templates(doc=doc, force=force)


@templates_app.command("list")
@_command()
def templates_list() -> None:
    """Each template and its intents (on or off), and the sections it applies to (by the orchestrator, the rules or you)."""
    from bench.segments import load_segments
    from bench.templates import (
        assigned,
        categorised,
        is_on,
        load_topics,
        pdf_rows,
        templates,
    )

    tmpls, topics, saved = templates(), load_topics(), categorised(pdf_rows())
    for name, t in tmpls.items():
        intents = ", ".join(i if is_on(topics, name, t, i) else f"{i} (off)" for i in t.intents)
        typer.echo(f"{name} [{'on' if is_on(topics, name, t) else 'off'}]: {intents}")
    typer.echo("")
    for segment in (s for s in load_segments() if s["use_for_questions"]):
        names, by = assigned(segment, tmpls, topics, saved)
        typer.echo(f"  {segment['segment_id']}: {', '.join(names) or '-'} ({by})")


@templates_app.command("assign")
@_command(writes=True)
def templates_assign(section: str = typer.Argument(..., help="A section id, or part of one (e.g. a PDF's name)"),
                     names: str = typer.Argument(..., help="Templates, comma-separated; 'none' for no template; 'auto' to let the orchestrator decide again")) -> None:
    """Override the orchestrator's categorisation for the matching sections."""
    from bench.segments import load_segments
    from bench.templates import load_topics, save_topics, templates

    tmpls, topics = templates(), load_topics()
    chosen = [] if names == "none" else [n.strip() for n in names.split(",") if n.strip()]
    if names != "auto" and (unknown := set(chosen) - set(tmpls)):
        raise BenchError(f"unknown template: {', '.join(sorted(unknown))} (templates: {', '.join(tmpls)})")
    matched = [s["segment_id"] for s in load_segments() if s["use_for_questions"] and section.lower() in s["segment_id"].lower()]
    if not matched:
        raise BenchError(f"no section used for questions matches {section!r}")
    for sid in matched:
        if names == "auto":
            topics["sections"].pop(sid, None)
        else:
            topics["sections"][sid] = chosen
    save_topics(topics)
    typer.echo(f"{len(matched)} section(s): {'categorised by the orchestrator' if names == 'auto' else ', '.join(chosen) or 'no template'}")


def _switch(name: str, intent: str | None, on: bool) -> None:
    from bench.templates import load_topics, save_topics, templates

    tmpls, topics = templates(), load_topics()
    if name not in tmpls or (intent and intent not in tmpls[name].intents):
        raise BenchError(f"no such template or intent: {name}{'.' + intent if intent else ''}")
    topics["switches"][f"{name}.{intent}" if intent else name] = on
    save_topics(topics)
    typer.echo(f"{name}{'.' + intent if intent else ''} switched {'on' if on else 'off'}")


@templates_app.command("on")
@_command(writes=True)
def templates_on(name: str, intent: str = typer.Argument(None)) -> None:
    """Switch a template (or one of its intents) on for this data folder."""
    _switch(name, intent, True)


@templates_app.command("off")
@_command(writes=True)
def templates_off(name: str, intent: str = typer.Argument(None)) -> None:
    """Switch a template (or one of its intents) off for this data folder: no calls, nothing deleted."""
    _switch(name, intent, False)


@templates_app.command("coverage")
@_command(writes=True)
def templates_coverage() -> None:
    """Which intents each section states, and which today's facts and questions cover (code only) -> data/coverage.md."""
    from collections import Counter

    from bench.facts import FACTS, load_rows
    from bench.pdf import write_text
    from bench.questions import QUESTIONS
    from bench.segments import load_segments
    from bench.templates import (
        COVERAGE,
        by_section,
        categorised,
        coverage,
        coverage_report,
        load_topics,
        pdf_rows,
        templates,
    )

    tmpls, template_rows = templates(), pdf_rows()
    rows = coverage(load_segments(), load_rows(FACTS), load_rows(QUESTIONS), tmpls, load_topics(), by_section(template_rows),
                    categorised(template_rows))
    write_text(COVERAGE, coverage_report(rows, tmpls))
    for name in tmpls:
        counts = Counter(r["status"] for r in rows if r["template"] == name)
        typer.echo(f"{name}: {', '.join(f'{n} {s}' for s, n in counts.items()) or 'no section assigned'}")
    typer.echo(f"wrote {COVERAGE}")


@app.command()
@_command()
def serve(port: int = typer.Option(8000, help="Port on 127.0.0.1")) -> None:
    """Browse every stage's output as JSON (for Postman or a browser) on http://127.0.0.1:<port>."""
    from bench.serve import run_server

    run_server(port)


if __name__ == "__main__":
    app()
