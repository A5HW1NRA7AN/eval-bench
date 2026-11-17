"""Workspaces, `bench group`, and the UI's helpers (no Streamlit needed: the app imports it only when it runs)."""

import json
import os
import subprocess
import sys

from typer.testing import CliRunner

from bench import cli, config, templates, ui


def test_a_workspace_has_its_own_folders_and_collection_and_none_changes_nothing():
    probe = "from bench import config, testbed; print(config.PDF_DIR, config.DATA_DIR, testbed.COLLECTION)"
    run = lambda extra: subprocess.run([sys.executable, "-c", probe], env={**os.environ, **extra}, capture_output=True, text=True, check=True).stdout.split()
    env = {k: v for k, v in os.environ.items() if k != "BENCH_WORKSPACE"}
    plain = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True).stdout.split()
    assert plain == [str(config.ROOT / "pdfs"), str(config.ROOT / "data"), "bench_docs"]
    assert run({"BENCH_WORKSPACE": "demo"}) == [str(config.ROOT / "workspaces/demo/pdfs"), str(config.ROOT / "workspaces/demo/data"), "bench_docs_demo"]


def test_bench_group_saves_known_pdfs_once(isolated):
    for name in ("dairy-book.pdf", "nutrition.pdf", "stats.pdf"):
        (config.PDF_DIR / name).write_bytes(b"")
    runner = CliRunner()
    assert runner.invoke(cli.app, ["group", "feeding", "dairy,nutrition"]).exit_code == 0
    assert json.loads(templates.TOPICS.read_text())["groups"] == {"feeding": ["dairy-book.pdf", "nutrition.pdf"]}
    result = runner.invoke(cli.app, ["group", "other", "nutrition"])
    assert result.exit_code == 1 and "already in another group" in result.output
    assert "matches 0 PDFs" in runner.invoke(cli.app, ["group", "x", "nope"]).output
    assert runner.invoke(cli.app, ["group", "feeding", "none"]).exit_code == 0 and templates.load_topics()["groups"] == {}


def test_ui_helpers_highlight_safely_and_see_the_lock(isolated):
    assert ui.marked("a <b> c", [(2, 5)]) == "<div style='white-space:pre-wrap;font-size:0.9em'>a <mark>&lt;b&gt;</mark> c</div>"
    assert ui.busy() is None
    with cli._writing("facts"):
        assert "facts" in ui.busy()  # the command holding the lock, as the sidebar shows it
