"""`bench ui`: a Streamlit app over the whole pipeline, per workspace. It reads data/ with the stages' own loaders and
makes every change by running a `bench` command (a subprocess under the same data lock, logged to data/ui/), so it
can't corrupt a run or change what a row's fingerprint covers. Streamlit is an optional extra: uv run --extra ui bench ui.
"""

from __future__ import annotations

import fcntl
import html
import importlib
import os
import pkgutil
import re
import shlex
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import bench
from bench import config

STAGES = {"1 Segment": ["segment"], "2 Facts": ["facts"], "3 Questions": ["questions"], "Template questions": ["templates", "run"],
          "4 Index": ["index"], "5 Evaluate": ["evaluate"], "Run everything": ["run"], "Export workbook": ["export"],
          "Test set": ["testset"], "Template coverage": ["templates", "coverage"]}
TAKES_DOC = {"facts", "questions", "templates run"}  # commands with --doc
TAKES_FORCE = {"questions", "templates run"}  # --force from the UI; facts are left out: redoing them restales everything after
TRACE_TAIL = 500  # agent calls shown, newest first (the trace only grows)


def workspaces() -> list[str]:
    return ["main"] + sorted(p.name for p in config.WORKSPACES.glob("*") if (p / "pdfs").is_dir())


def use(name: str) -> None:
    """Point every bench module at a workspace ('main' is the repo's pdfs/ and data/): set BENCH_WORKSPACE, reload
    config, then every other bench module twice, so each picks up the paths of the modules it imports from."""
    wanted = "" if name == "main" else name
    if config.WORKSPACE == wanted and "bench.export" in sys.modules:
        return
    if wanted:
        os.environ["BENCH_WORKSPACE"] = wanted
    else:
        os.environ.pop("BENCH_WORKSPACE", None)
    importlib.reload(config)
    others = [m.name for m in pkgutil.iter_modules(bench.__path__) if m.name not in ("config", "ui", "__main__")]
    for _ in range(2):
        for module in others:
            importlib.reload(importlib.import_module(f"bench.{module}"))


def busy() -> str | None:
    """The bench command holding this workspace's data lock ("pid N: bench ..."), or None."""
    lock = config.DATA_DIR / ".lock"
    if not lock.exists():
        return None
    with lock.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.seek(0)
            return f.read().strip() or "a bench command"
        fcntl.flock(f, fcntl.LOCK_UN)
    return None


def stop(running: str) -> None:
    """Interrupt the running bench command's process group, as Ctrl-C would: it saves what it finished and frees the lock."""
    if found := re.match(r"pid (\d+)", running):
        os.killpg(os.getpgid(int(found.group(1))), signal.SIGINT)


def env(provider: str) -> dict[str, str]:
    return {**os.environ, "LLM_PROVIDER": provider, "NO_COLOR": "1", "COLUMNS": "160", "PYTHONUNBUFFERED": "1"}


def start(args: list[str], provider: str) -> Path:
    """Run `bench <args>` in the background, in a process group of its own; its output (and exit code) goes to
    data/ui/<time>-<command>.log."""
    folder = config.DATA_DIR / "ui"
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / f"{datetime.now():%Y%m%d-%H%M%S}-{'-'.join(a for a in args if not a.startswith('-'))[:40]}.log"
    command = shlex.join([sys.executable, "-m", "bench", *args])
    with log.open("w") as f:
        f.write(f"$ bench {shlex.join(args)}  (LLM_PROVIDER={provider}, workspace {config.WORKSPACE or 'main'})\n")
        f.flush()
        subprocess.Popen(["/bin/sh", "-c", f'{command}; echo "[exit $?]"'], cwd=config.ROOT, env=env(provider),
                         stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    return log


def now(args: list[str]) -> tuple[bool, str]:
    """Run a quick `bench <args>` (a settings change, a score) and wait: (ok, its output)."""
    done = subprocess.run([sys.executable, "-m", "bench", *args], cwd=config.ROOT, env=env(os.environ.get("LLM_PROVIDER", "ollama")),
                          capture_output=True, text=True, timeout=300, check=False)
    return done.returncode == 0, (done.stdout + done.stderr).strip()


def marked(text: str, spans: list[tuple[int, int]]) -> str:
    """The text as HTML with the spans highlighted (overlapping spans merged)."""
    merged: list[list[int]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out, at = [], 0
    for a, b in merged:
        out += [html.escape(text[at:a]), f"<mark>{html.escape(text[a:b])}</mark>"]
        at = b
    out.append(html.escape(text[at:]))
    return "<div style='white-space:pre-wrap;font-size:0.9em'>" + "".join(out) + "</div>"


def tail_rows(path: Path, n: int) -> list[dict[str, Any]]:
    """The last n rows of a JSONL file, read from its end (the trace is tens of MB and only grows)."""
    import json
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(size - 4_000_000, 0))
        lines = f.read().decode(errors="replace").splitlines()[1 if size > 4_000_000 else 0 :]
    rows = []
    for line in lines[-n - 1 :]:
        try:
            rows.append(json.loads(line))
        except ValueError:  # a line cut at the start of the read, or still being written
            continue
    return rows[-n:]


def endpoint_up() -> bool:
    """Whether the self-hosted 31B (GEMMA_URL in .env) answers now; it is down at night. The key is never shown."""
    import httpx
    from dotenv import dotenv_values
    found = dotenv_values(config.ROOT / ".env")
    url, key = os.environ.get("GEMMA_URL") or found.get("GEMMA_URL"), os.environ.get("GEMMA_API_KEY") or found.get("GEMMA_API_KEY")
    try:
        return bool(url) and httpx.get(url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="eval-bench", layout="wide")
    up = st.cache_data(ttl=60, show_spinner=False)(endpoint_up)()
    with st.sidebar:
        st.title("eval-bench")
        names = workspaces()
        wanted = st.session_state.get("ws", "main")
        name = st.selectbox("Workspace", names, index=names.index(wanted) if wanted in names else 0)
        st.session_state["ws"] = name
        with st.expander("New workspace"):
            new = st.text_input("Name (letters, digits, - and _)")
            if st.button("Create") and new:
                if not new.replace("-", "").replace("_", "").isalnum() or new == "main":
                    st.error("Use letters, digits, - and _ only")
                else:
                    for sub in ("pdfs", "data"):
                        (config.WORKSPACES / new / sub).mkdir(parents=True, exist_ok=True)
                    st.session_state["ws"] = new
                    st.rerun()
        use(name)
        st.caption(f"31B endpoint: {'up' if up else 'down (it is off at night)'}")
        provider = st.radio("Model for LLM stages", ["gemma", "ollama"], index=0 if up else 1,
                            help="gemma: the self-hosted 31B (primary); ollama: the local 12B fallback. Rows record their model.")
        running = busy()
        st.caption(f"PDFs: `{config.PDF_DIR.relative_to(config.ROOT)}` · data: `{config.DATA_DIR.relative_to(config.ROOT)}`")
        if running:
            st.warning(f"Running: {running}")
            if st.button("Stop it"):
                stop(running)
                st.rerun()
        else:
            st.success("Idle")

    from bench.pdf import read_jsonl

    @st.cache_data(max_entries=32, show_spinner=False)
    def rows_at(path: str, mtime: float) -> list[dict[str, Any]]:  # cached until the file changes
        return read_jsonl(Path(path), missing_ok=True)

    def rows(path: Path) -> list[dict[str, Any]]:
        return rows_at(str(path), path.stat().st_mtime if path.exists() else 0.0)

    tabs = st.tabs(["Overview", "Pipeline", "PDFs", "Explore", "Results", "Search", "Configure", "Agent calls", "Compare"])
    for tab, page in zip(tabs, (overview, pipeline, pdfs, explore, results, search, configure, agent_calls, compare)):
        with tab:
            page(st, rows, running=running, provider=provider)


def overview(st: Any, rows: Any, **_: Any) -> None:
    import pandas as pd

    from bench import checks, evaluate, facts, questions, segments, templates, testbed
    segs = rows(segments.SEGMENTS)
    cols = st.columns(6)
    cols[0].metric("PDFs", len(list(config.PDF_DIR.glob("*.pdf"))))
    cols[1].metric("Sections", len(segs), f"{sum(s['use_for_questions'] for s in segs)} for questions", delta_color="off")
    cols[2].metric("Facts", sum(len(r["facts"]) for r in rows(facts.FACTS)))
    cols[3].metric("General questions", sum(len(r["questions"]) for r in questions.question_rows()))
    cols[4].metric("Template questions", sum(len(r["questions"]) for r in templates.pdf_rows()))
    cols[5].metric("Chunks", len(rows(testbed.CHUNKS)))
    if st.button("Run all checks"):
        st.dataframe(pd.DataFrame([{"stage": s, "ok": "✓" if ok else "✗", "check": n, "detail": d}
                                   for s, cs in checks.all_checks().items() for n, ok, d in cs]), hide_index=True, width="stretch")
    if evaluate.REPORT.exists():
        st.markdown(evaluate.REPORT.read_text())
    else:
        st.info("No report yet: run the pipeline (Pipeline tab).")


def pipeline(st: Any, rows: Any, running: str | None, provider: str, **_: Any) -> None:
    from bench import evaluate, export
    st.write("Each button runs the `bench` command in the background, under the data lock: one writing command at a time.")
    doc = st.text_input("Only PDFs whose file name contains (facts, questions, template questions)")
    force = st.checkbox("Force: redo rows that are already current (questions and template questions)")
    cols = st.columns(5)
    for k, (label, args) in enumerate(STAGES.items()):
        if cols[k % 5].button(label, disabled=bool(running), width="stretch"):
            command = " ".join(args)
            extra = (["--doc", doc] if doc and command in TAKES_DOC else []) + (["--force"] if force and command in TAKES_FORCE else [])
            st.session_state["log"] = str(start([*args, *extra], provider))
            st.rerun()
    logs = sorted((config.DATA_DIR / "ui").glob("*.log"), reverse=True)
    if logs:
        log_name = st.selectbox("Log", [p.name for p in logs], index=next((i for i, p in enumerate(logs) if str(p) == st.session_state.get("log")), 0))

        @st.fragment(run_every=3)
        def tail(log: Path) -> None:  # an argument, not a closure: Streamlit reruns the fragment with the same arguments
            st.code("\n".join(log.read_text(errors="replace").splitlines()[-200:]) or "(starting)", language=None)

        tail(config.DATA_DIR / "ui" / log_name)
    st.subheader("Downloads")
    cols = st.columns(4)
    for col, path, label in ((cols[0], export.WORKBOOK, "Workbook (.xlsx)"), (cols[1], export.TESTSET_JSON, "Test set (.json)"),
                             (cols[2], export.TESTSET_CSV, "Test set (.csv)"), (cols[3], evaluate.REPORT, "Report (.md)")):
        if path.exists():
            col.download_button(label, path.read_bytes(), file_name=path.name)
        else:
            col.caption(f"{label}: not made yet")


def pdfs(st: Any, rows: Any, running: str | None, **_: Any) -> None:
    import pandas as pd

    from bench import pdf, segments
    found, pages, segs = sorted(config.PDF_DIR.glob("*.pdf")), rows(pdf.PAGES), rows(segments.SEGMENTS)
    st.dataframe(pd.DataFrame([{"pdf": p.name, "MB": round(p.stat().st_size / 1e6, 1), "pages": sum(x["file_name"] == p.name for x in pages),
                                "sections": sum(s["file_name"] == p.name for s in segs)} for p in found]), hide_index=True, width="stretch")
    if pages:
        cols = st.columns([3, 1])
        name = cols[0].selectbox("Page text: PDF", sorted({x["file_name"] for x in pages}))
        numbers = [x["page"] for x in pages if x["file_name"] == name]
        number = cols[1].number_input("Page", min(numbers), max(numbers), min(numbers))
        page = next((x for x in pages if x["file_name"] == name and x["page"] == number), None)
        if page:
            st.caption(f"Printed page {page.get('printed')}; status: {page.get('status', 'garbled' if page.get('garbled') else 'text')}")
            left, right = st.columns(2)
            left.text_area("Clean text (what the stages read)", page["text"], height=300)
            right.text_area("Raw text (as extracted)", page.get("raw_text", ""), height=300)
    if not config.WORKSPACE:
        st.caption("New PDFs are best tried in a workspace: on main, a new PDF changes which terms the library never mentions, "
                   "so every group's general questions would be planned again.")
    uploads = st.file_uploader("Add PDFs to this workspace", type="pdf", accept_multiple_files=True, disabled=bool(running))
    if uploads and st.button("Save PDFs"):
        config.PDF_DIR.mkdir(parents=True, exist_ok=True)
        for up in uploads:
            (config.PDF_DIR / Path(up.name).name).write_bytes(up.getvalue())
        st.success(f"Saved {len(uploads)}. Run 1 Segment (Pipeline tab) to read them in.")


def explore(st: Any, rows: Any, **_: Any) -> None:
    import pandas as pd

    from bench import facts, questions, segments, templates
    used = [s for s in rows(segments.SEGMENTS) if s["use_for_questions"]]
    if not used:
        st.info("No sections yet: run 1 Segment.")
        return
    pdf_name = st.selectbox("PDF", sorted({s["file_name"] for s in used}))
    mine = [s for s in used if s["file_name"] == pdf_name]
    sid = st.selectbox("Section", [s["segment_id"] for s in mine], format_func=lambda x: f"{x} · {next(s['title'] for s in mine if s['segment_id'] == x)}")
    section = next(s for s in mine if s["segment_id"] == sid)
    own = {r["segment_id"]: r for r in rows(facts.FACTS)}.get(sid, {"facts": [], "rejected": []})
    tsec = templates.by_section(templates.pdf_rows()).get(sid, {"facts": [], "questions": [], "rejected": [], "rejected_facts": []})
    general = questions.question_rows()
    gq = [q for r in general for q in r["questions"] if q.get("segment_id") == sid or any(ev["segment_id"] == sid for ev in q["evidence"])]
    gx = [x for r in general for x in r["rejected"] if (x.get("segment_id") or (x["fact_ids"][0].rsplit(":", 1)[0] if x["fact_ids"] else "")) == sid]
    qs = [{**q, "set": "general"} for q in gq] + [{**q, "set": "template"} for q in tsec["questions"]]
    left, right = st.columns([3, 2])
    with right:
        st.markdown(f"**Facts** ({len(own['facts'])} general, {len(tsec['facts'])} template)")
        st.dataframe(pd.DataFrame([{"fact_id": f["fact_id"], "claim": f["claim"]} for f in own["facts"] + tsec["facts"]]), hide_index=True, width="stretch")
        st.markdown(f"**Questions** ({len(qs)})")
        pick = st.dataframe(pd.DataFrame([{"id": q["question_id"], "set": q["set"], "type": q["type"], "question": q["question"],
                                           "answer": q.get("answer") or q.get("absent_term")} for q in qs]),
                            hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row")
        picked = qs[pick.selection.rows[0]] if pick.selection.rows else None
        rejected = own["rejected"] + gx + tsec["rejected_facts"] + tsec["rejected"]
        with st.expander(f"Rejected by the regex audit ({len(rejected)})"):
            st.dataframe(pd.DataFrame([{"what": x.get("question") or x.get("claim"), "reason": x["reason"]} for x in rejected]),
                         hide_index=True, width="stretch")
    with left:
        spans = ([(ev["start"], ev["end"]) for ev in picked["evidence"] if ev["segment_id"] == sid] if picked else
                 [(f["start"], f["end"]) for f in own["facts"] + tsec["facts"]])
        st.caption("Highlighted: the selected question's evidence" if picked else "Highlighted: every fact's quote (select a question to see its evidence)")
        st.markdown(marked(section["text"], spans), unsafe_allow_html=True)


def results(st: Any, rows: Any, **_: Any) -> None:
    import pandas as pd

    from bench import evaluate, testbed
    found = rows(evaluate.RESULTS)
    if not found:
        st.info("No results yet: run 5 Evaluate.")
        return
    kinds = sorted({r["type"] for r in found})
    cols = st.columns(4)
    kind = cols[0].multiselect("Types", kinds, default=kinds)
    misses = cols[1].checkbox("Misses only (not in the top 10)")
    sets = cols[2].radio("Set", ["general", "template", "both"], horizontal=True)
    split = cols[3].radio("Split", ["both", "dev", "held-out"], horizontal=True)
    shown = [r for r in found if r["type"] in kind and (not misses or (r.get("answerable", True) and not r.get("hit@10")))
             and (sets == "both" or evaluate.set_of(r) == sets) and split in ("both", r.get("split"))]
    pick = st.dataframe(pd.DataFrame([{"id": r["question_id"], "type": r["type"], "split": r.get("split"), "question": r["question"],
                                       "first rank": r.get("first_rank"), **{f"hit@{k}": r.get(f"hit@{k}") for k in evaluate.KS}} for r in shown]),
                        hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row")
    if pick.selection.rows:
        r = shown[pick.selection.rows[0]]
        chunks = {c["chunk_id"]: c for c in rows(testbed.CHUNKS)}
        st.markdown("**Expected (evidence)**")
        for quote in r["expected"] or ["(none: the library can't answer this)"]:
            st.info(quote)
        st.markdown("**What the search returned**")
        for n, (cid, score) in enumerate(r["ranked"][:5], start=1):
            c = chunks.get(cid, {})
            st.write(f"{n}. `{score:.3f}` {c.get('file_name', cid)} p{c.get('pages', ['?'])[0]}: {' '.join(c.get('text', '').split())[:400]}")


def search(st: Any, rows: Any, **_: Any) -> None:
    from bench import testbed
    query = st.text_input("Search the test bed now")
    k = st.slider("Results", 1, 20, 5)
    if not query:
        return
    try:
        chunks = {c["chunk_id"]: c for c in rows(testbed.CHUNKS)}
        vector_hits = testbed.search(testbed.qdrant(), testbed.embed([query])[0], k)
        word_hits = testbed.Keyword(list(chunks.values())).search(query, k)
    except Exception as exc:  # noqa: BLE001 - no index yet, or Qdrant/Ollama not running: shown, not raised
        st.error(f"{type(exc).__name__}: {exc}")
        return
    for col, label, hits in zip(st.columns(2), ("Embedding search (Qdrant)", "Keyword search (BM25)"), (vector_hits, word_hits)):
        col.markdown(f"**{label}**")
        for n, (cid, score) in enumerate(hits, start=1):
            c = chunks[cid]
            col.write(f"{n}. `{score:.3f}` {c['file_name']} p{c['pages'][0]}: {' '.join(c['text'].split())[:300]}")


def configure(st: Any, rows: Any, running: str | None, **_: Any) -> None:
    import pandas as pd

    from bench import questions, segments, templates
    try:
        tmpls, topics = templates.templates(), templates.load_topics()
    except config.BenchError as exc:
        st.error(str(exc))
        return

    def change(args: list[str]) -> None:
        st.session_state["said"] = now(args)

    st.subheader("Templates and intents")
    st.caption("Switches apply to this workspace (data/topics.json). The next Template questions run redoes what they affect.")
    for tname, t in tmpls.items():
        on = templates.is_on(topics, tname, t)
        cols = st.columns([1, 4])
        if cols[0].toggle(tname, value=on, key=f"t-{tname}", disabled=bool(running)) != on:
            change(["templates", "on" if not on else "off", tname])
            st.rerun()
        with cols[1].expander(t.description[:90]):
            for iname in t.intents:
                ion = templates.is_on(topics, tname, t, iname)
                if st.toggle(iname, value=ion, key=f"i-{tname}-{iname}", disabled=bool(running) or not on) != ion:
                    change(["templates", "on" if not ion else "off", tname, iname])
                    st.rerun()
    st.subheader("A section's templates")
    used = [s["segment_id"] for s in rows(segments.SEGMENTS) if s["use_for_questions"]]
    if used:
        cols = st.columns([2, 3, 1])
        sid = cols[0].selectbox("Section", used)
        current = topics["sections"].get(sid)
        names = cols[1].multiselect("Templates (empty: none)", list(tmpls), default=current or [],
                                    help="Your override. 'Let the orchestrator decide' removes it.")
        if cols[2].button("Save", disabled=bool(running)):
            change(["templates", "assign", sid, ",".join(names) or "none"])
        if current is not None and cols[2].button("Let the orchestrator decide", disabled=bool(running)):
            change(["templates", "assign", sid, "auto"])
    st.subheader("Groups of PDFs for multi-hop questions")
    st.caption("Facts of PDFs in one group may be paired across PDFs; a PDF in no group is paired only within itself.")
    if topics["groups"]:
        st.dataframe(pd.DataFrame([{"group": g, "pdfs": ", ".join(ps)} for g, ps in topics["groups"].items()]), hide_index=True)
    cols = st.columns([2, 4, 1])
    gname = cols[0].text_input("Group")
    members = cols[1].multiselect("PDFs (none: remove the group)", [p.name for p in sorted(config.PDF_DIR.glob("*.pdf"))],
                                  default=topics["groups"].get(gname, []))
    if cols[2].button("Save group", disabled=bool(running) or not gname):
        change(["group", gname, ",".join(members) or "none"])
    if said := st.session_state.pop("said", None):
        (st.success if said[0] else st.error)(said[1] or "done")
    with st.expander("Question types (question_types.toml) and word lists (vocab.toml): edit the files"):
        st.code(config.QUESTION_TYPES.read_text(), language="toml")
        st.code(questions.VOCAB.read_text(), language="toml")


def agent_calls(st: Any, rows: Any, **_: Any) -> None:
    import pandas as pd

    from bench import llm
    trace = list(reversed(tail_rows(llm.TRACE, TRACE_TAIL)))
    st.caption(f"The latest {len(trace)} agent calls in data/trace.jsonl, newest first; select one to see its prompt and output.")
    pick = st.dataframe(pd.DataFrame([{"time": t["time"], "role": t["role"], "label": t["label"], "model": t["model"], "seconds": t["seconds"],
                                       "in": t.get("input_tokens"), "out": t.get("output_tokens"), "error": t.get("error")} for t in trace]),
                        hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row")
    if pick.selection.rows:
        call = trace[pick.selection.rows[0]]
        left, right = st.columns(2)
        left.text_area("Prompt", call.get("prompt", ""), height=400)
        right.json(call.get("output") or {"error": call.get("error")})


def compare(st: Any, rows: Any, **_: Any) -> None:
    from bench import evaluate
    kept = ["current", *reversed(evaluate.runs())]
    cols = st.columns(2)
    a = cols[0].selectbox("Run A", kept, index=min(1, len(kept) - 1))
    b = cols[1].selectbox("Run B", kept, index=0)
    if st.button("Compare"):
        try:
            st.markdown(evaluate.compare(a, b))
        except config.BenchError as exc:
            st.error(str(exc))
    st.subheader("Score another search system")
    st.caption("A JSONL file with one line per question of the test set (Pipeline → Test set): "
               '{"uid": "...", "passages": ["text", ...]}, best first. A passage counts when it holds the evidence quote '
               f"(or a run of {evaluate.WINDOW} of its words) and one of the answer's spans; a question with no line is a miss.")
    upload = st.file_uploader("Results file", type=["jsonl"])
    if upload and st.button("Score it"):
        target = config.DATA_DIR / "ui" / Path(upload.name).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(upload.getvalue())
        ok, out = now(["score", str(target)])
        (st.markdown if ok else st.error)(out)


if __name__ == "__main__":
    main()
