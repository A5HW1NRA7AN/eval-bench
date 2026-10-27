"""`bench` command line: one command per pipeline stage, run in order."""

from __future__ import annotations

import typer

from bench.config import MAX_FACTS, MAX_QUESTIONS

app = typer.Typer(no_args_is_help=True, add_completion=False)
DOC = typer.Option(None, help="Only PDFs whose file name contains this text")
FORCE = typer.Option(False, help="Redo segments that are already done")


@app.command()
def segment(refresh: bool = typer.Option(False, help="Re-extract page text from the PDFs")) -> None:
    """1. Cut the PDFs into segments along their tables of contents -> data/segments.jsonl."""
    from bench.segments import run_segments

    run_segments(refresh=refresh)


@app.command()
def facts(doc: str = DOC, max_facts: int = typer.Option(MAX_FACTS, help="Facts per segment"), force: bool = FORCE) -> None:
    """2. Extract facts per segment and keep those whose quote is in the PDF -> data/facts.jsonl."""
    from bench.agents import run_facts

    run_facts(doc=doc, max_facts=max_facts, force=force)


@app.command()
def questions(doc: str = DOC, max_questions: int = typer.Option(MAX_QUESTIONS, help="Questions per segment"),
              force: bool = FORCE) -> None:
    """3. Write questions from the verified facts and keep those that pass the audit -> data/questions.jsonl."""
    from bench.agents import run_questions

    run_questions(doc=doc, max_questions=max_questions, force=force)


@app.command()
def index() -> None:
    """4. Build the local Qdrant test bed from every segment -> data/chunks.jsonl."""
    from bench.testbed import run_index

    run_index()


@app.command()
def evaluate() -> None:
    """5. Search the test bed with every question and score it -> data/report.md."""
    from bench.evaluate import run_evaluate

    run_evaluate()


if __name__ == "__main__":
    app()
