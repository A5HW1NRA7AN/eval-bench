import pytest

from bench import llm


@pytest.fixture(autouse=True)
def trace_in_tmp(tmp_path, monkeypatch):
    """Agent calls in tests are traced to a temporary file, never to data/trace.jsonl."""
    monkeypatch.setattr(llm, "TRACE", tmp_path / "trace.jsonl")
