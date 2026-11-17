"""Workspaces, `bench group`, and the UI's helpers (no Streamlit needed: the app imports it only when it runs)."""

import json
import os
import subprocess
import sys

import typer
from typer.testing import CliRunner

from bench import cli, config, templates, ui


def _python(code: str, workspace: str | None = None) -> list[str]:
    """Run code in a fresh process (module paths are read at import) and return its printed lines."""
    env = {k: v for k, v in os.environ.items() if k != "BENCH_WORKSPACE"} | ({"BENCH_WORKSPACE": workspace} if workspace else {})
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.splitlines()


def test_a_workspace_has_its_own_folders_and_collection_and_none_changes_nothing():
    probe = "from bench import config, testbed; print(config.PDF_DIR); print(config.DATA_DIR); print(testbed.COLLECTION)"
    assert _python(probe) == [str(config.ROOT / "pdfs"), str(config.ROOT / "data"), "bench_docs"]
    assert _python(probe, "demo") == [str(config.ROOT / "workspaces/demo/pdfs"), str(config.ROOT / "workspaces/demo/data"), "bench_docs_demo"]


def test_switching_workspace_in_the_ui_points_every_module_at_it():
    probe = ("import sys; from pathlib import Path; from bench import config, ui, export\n"
             "def stale(root): return sorted(f'{m}.{k}' for m, mod in list(sys.modules.items()) if m.startswith('bench.') "
             "for k, v in vars(mod).items() if isinstance(v, Path) and (v == root or root in v.parents))\n"
             "ui.use('demo'); print(stale(config.ROOT / 'data')); ui.use('main'); print(stale(config.ROOT / 'workspaces' / 'demo'))")
    assert _python(probe) == ["[]", "[]"]  # no module keeps the other folder's paths


def test_every_pipeline_button_names_a_cli_command():
    commands = typer.main.get_command(cli.app).commands
    for args in ui.STAGES.values():
        group = commands[args[0]]
        assert len(args) == 1 or args[1] in group.commands, args


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
    assert ui.marked("abcdef", [(3, 5), (1, 4)]).count("<mark>") == 1  # overlapping quotes are one highlight
    assert ui.busy() is None
    with cli._writing("facts"):
        assert ui.busy() == f"pid {os.getpid()}: bench facts"  # the command holding the lock, as the sidebar shows it
    trace = config.DATA_DIR / "trace.jsonl"
    trace.write_text("".join(json.dumps({"n": n}) + "\n" for n in range(5)) + '{"n": 5, "cut')  # a line still being written
    assert [r["n"] for r in ui.tail_rows(trace, 3)] == [2, 3, 4]
