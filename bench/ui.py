"""`bench ui`: a Streamlit app over the whole pipeline, per workspace. It reads data/ with the stages' own loaders and
makes every change by running a `bench` command (a subprocess under the same data lock, logged to data/ui/), so it
can't corrupt a run or change what a row's fingerprint covers. Streamlit is an optional extra: uv run --extra ui bench ui.
"""

from __future__ import annotations

import fcntl
import html
import importlib
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from bench import config

ORDER = ["config", "pdf", "segments", "verify", "llm", "facts", "questions", "templates", "testbed", "orchestrator",
         "evaluate", "checks", "export"]  # bench modules in import order, reloaded when the workspace changes
STAGES = {"1 Segment": ["segment"], "2 Facts": ["facts"], "3 Questions": ["questions"], "Template questions": ["templates", "run"],
          "4 Index": ["index"], "5 Evaluate": ["evaluate"], "Run everything": ["run"], "Export workbook": ["export"],
          "Test set": ["testset"], "Template coverage": ["templates", "coverage"]}
TAKES_DOC = {"facts", "questions", "templates run"}


def workspaces() -> list[str]:
    return ["main"] + sorted(p.name for p in config.WORKSPACES.glob("*") if (p / "pdfs").is_dir())


def use(name: str) -> None:
    """Point every bench module at a workspace ('main' is the repo's pdfs/ and data/): set BENCH_WORKSPACE and reload."""
    wanted = "" if name == "main" else name
    if config.WORKSPACE == wanted and "bench.export" in sys.modules:
        return
    if wanted:
        os.environ["BENCH_WORKSPACE"] = wanted
    else:
        os.environ.pop("BENCH_WORKSPACE", None)
    for module in ORDER:
        importlib.reload(importlib.import_module(f"bench.{module}"))


def rows(path: Path) -> list[dict[str, Any]]:
    from bench.pdf import read_jsonl
    return read_jsonl(path) if path.exists() else []


def busy() -> str | None:
    """The bench command holding this workspace's data lock, or None."""
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


def env(provider: str) -> dict[str, str]:
    return {**os.environ, "LLM_PROVIDER": provider, "NO_COLOR": "1", "COLUMNS": "160", "PYTHONUNBUFFERED": "1"}


def start(args: list[str], provider: str) -> Path:
    """Run `bench <args>` in the background; its output (and exit code) goes to data/ui/<time>-<command>.log."""
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
    """Run a quick `bench <args>` (a config change) and wait: (ok, its output)."""
    done = subprocess.run([sys.executable, "-m", "bench", *args], cwd=config.ROOT, env=env(os.environ.get("LLM_PROVIDER", "ollama")),
                          capture_output=True, text=True, timeout=120, check=False)
    return done.returncode == 0, (done.stdout + done.stderr).strip()


def marked(text: str, spans: list[tuple[int, int]]) -> str:
    """The text as HTML with the spans highlighted."""
    out, at = [], 0
    for a, b in sorted(spans):
        if a >= at:
            out += [html.escape(text[at:a]), f"<mark>{html.escape(text[a:b])}</mark>"]
            at = b
    out.append(html.escape(text[at:]))
    return "<div style='white-space:pre-wrap;font-size:0.9em'>" + "".join(out) + "</div>"


def main() -> None:
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="eval-bench", layout="wide")
    with st.sidebar:
        st.title("eval-bench")
        names = workspaces()
        name = st.selectbox("Workspace", names, index=names.index(st.session_state.get("ws", "main")) if st.session_state.get("ws", "main") in names else 0)
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
        provider = st.radio("Model for LLM stages", ["gemma", "ollama"], help="gemma: the self-hosted 31B (primary); ollama: the local 12B fallback")
        running = busy()
        st.caption(f"PDFs: `{config.PDF_DIR.relative_to(config.ROOT)}` · data: `{config.DATA_DIR.relative_to(config.ROOT)}`")
        if running:
            st.warning(f"Running: {running}")
        else:
            st.success("Idle")

    from bench import (
        checks,
        evaluate,
        export,
        facts,
        llm,
        pdf,
        questions,
        segments,
        templates,
        testbed,
    )

    tabs = st.tabs(["Overview", "Pipeline", "PDFs", "Explore", "Results", "Search", "Configure", "Agent calls"])

    with tabs[0]:
        segs, fact_rows, general, topical = rows(segments.SEGMENTS), rows(facts.FACTS), questions.question_rows(), templates.pdf_rows()
        cols = st.columns(6)
        cols[0].metric("PDFs", len(list(config.PDF_DIR.glob("*.pdf"))))
        cols[1].metric("Sections", len(segs), f"{sum(s['use_for_questions'] for s in segs)} for questions", delta_color="off")
        cols[2].metric("Facts", sum(len(r["facts"]) for r in fact_rows))
        cols[3].metric("General questions", sum(len(r["questions"]) for r in general))
        cols[4].metric("Template questions", sum(len(r["questions"]) for r in topical))
        cols[5].metric("Chunks", len(rows(testbed.CHUNKS)))
        if st.button("Run all checks"):
            st.dataframe(pd.DataFrame([{"stage": s, "ok": "✓" if ok else "✗", "check": n, "detail": d}
                                       for s, cs in checks.all_checks().items() for n, ok, d in cs]), hide_index=True, width="stretch")
        if evaluate.REPORT.exists():
            st.markdown(evaluate.REPORT.read_text())
        else:
            st.info("No report yet: run the pipeline (Pipeline tab).")

    with tabs[1]:
        st.write("Each button runs the `bench` command in the background, under the data lock; one writing command at a time.")
        doc = st.text_input("Only PDFs whose file name contains (facts, questions, template questions)")
        force = st.checkbox("Force: redo rows that are already current")
        cols = st.columns(5)
        for k, (label, args) in enumerate(STAGES.items()):
            if cols[k % 5].button(label, disabled=bool(running), width="stretch"):
                extra = (["--doc", doc] if doc and " ".join(args) in TAKES_DOC else []) + (["--force"] if force and " ".join(args) in {"questions", "templates run"} else [])
                st.session_state["log"] = str(start([*args, *extra], provider))
                st.rerun()
        logs = sorted((config.DATA_DIR / "ui").glob("*.log"), reverse=True)
        if logs:
            chosen = st.selectbox("Log", [p.name for p in logs], index=next((i for i, p in enumerate(logs) if str(p) == st.session_state.get("log")), 0))

            @st.fragment(run_every=3)
            def tail() -> None:
                text = (config.DATA_DIR / "ui" / chosen).read_text(errors="replace")
                st.code("\n".join(text.splitlines()[-200:]) or "(starting)", language=None)

            tail()
        st.subheader("Downloads")
        cols = st.columns(4)
        for col, path, label in ((cols[0], export.WORKBOOK, "Workbook (.xlsx)"), (cols[1], export.TESTSET_JSON, "Test set (.json)"),
                                 (cols[2], export.TESTSET_CSV, "Test set (.csv)"), (cols[3], evaluate.REPORT, "Report (.md)")):
            if path.exists():
                col.download_button(label, path.read_bytes(), file_name=path.name)
            else:
                col.caption(f"{label}: not made yet")

    with tabs[2]:
        found = sorted(config.PDF_DIR.glob("*.pdf"))
        pages = rows(pdf.PAGES)
        st.dataframe(pd.DataFrame([{"pdf": p.name, "MB": round(p.stat().st_size / 1e6, 1),
                                    "pages": sum(x["file_name"] == p.name for x in pages),
                                    "sections": sum(s["file_name"] == p.name for s in segs)} for p in found]), hide_index=True, width="stretch")
        uploads = st.file_uploader("Add PDFs to this workspace", type="pdf", accept_multiple_files=True, disabled=bool(running))
        if uploads and st.button("Save PDFs"):
            config.PDF_DIR.mkdir(parents=True, exist_ok=True)
            for up in uploads:
                (config.PDF_DIR / Path(up.name).name).write_bytes(up.getvalue())
            st.success(f"Saved {len(uploads)}. Run 1 Segment (Pipeline tab) to read them in.")

    with tabs[3]:
        used = [s for s in segs if s["use_for_questions"]]
        if not used:
            st.info("No sections yet: run 1 Segment.")
        else:
            pdf_name = st.selectbox("PDF", sorted({s["file_name"] for s in used}))
            mine = [s for s in used if s["file_name"] == pdf_name]
            sid = st.selectbox("Section", [s["segment_id"] for s in mine], format_func=lambda x: f"{x} · {next(s['title'] for s in mine if s['segment_id'] == x)}")
            section = next(s for s in mine if s["segment_id"] == sid)
            own = {r["segment_id"]: r for r in fact_rows}.get(sid, {"facts": [], "rejected": []})
            tsec = templates.by_section(topical).get(sid, {"facts": [], "questions": [], "rejected": [], "rejected_facts": []})
            gq = [q for r in general for q in r["questions"] if q.get("segment_id") == sid or any(ev["segment_id"] == sid for ev in q["evidence"])]
            qs = [{**q, "set": "general"} for q in gq] + [{**q, "set": "template"} for q in tsec["questions"]]
            left, right = st.columns([3, 2])
            with right:
                st.markdown(f"**Facts** ({len(own['facts'])} general, {len(tsec['facts'])} template)")
                st.dataframe(pd.DataFrame([{"fact_id": f["fact_id"], "claim": f["claim"]} for f in own["facts"] + tsec["facts"]]), hide_index=True, width="stretch")
                st.markdown(f"**Questions** ({len(qs)})")
                pick = st.dataframe(pd.DataFrame([{"id": q["question_id"], "set": q["set"], "type": q["type"], "question": q["question"],
                                                   "answer": q.get("answer") or q.get("absent_term")} for q in qs]),
                                    hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row")
                chosen = qs[pick.selection.rows[0]] if pick.selection.rows else None
                with st.expander(f"Rejected by the regex audit ({len(own['rejected']) + len(tsec['rejected_facts']) + len(tsec['rejected'])})"):
                    st.dataframe(pd.DataFrame([{"what": x.get("question") or x.get("claim"), "reason": x["reason"]}
                                               for x in own["rejected"] + tsec["rejected_facts"] + tsec["rejected"]]), hide_index=True, width="stretch")
            with left:
                spans = [(ev["start"], ev["end"]) for ev in chosen["evidence"] if ev["segment_id"] == sid] if chosen else \
                        [(f["start"], f["end"]) for f in own["facts"] + tsec["facts"]]
                st.caption("Highlighted: the selected question's evidence" if chosen else "Highlighted: every fact's quote (select a question to see its evidence)")
                st.markdown(marked(section["text"], spans), unsafe_allow_html=True)

    with tabs[4]:
        results = rows(evaluate.RESULTS)
        if not results:
            st.info("No results yet: run 5 Evaluate.")
        else:
            kinds = sorted({r["type"] for r in results})
            cols = st.columns(3)
            kind = cols[0].multiselect("Types", kinds, default=kinds)
            misses = cols[1].checkbox("Misses only (not in the top 10)")
            sets = cols[2].radio("Set", ["general", "template", "both"], horizontal=True)
            shown = [r for r in results if r["type"] in kind and (not misses or (r.get("answerable", True) and not r.get("hit@10")))
                     and (sets == "both" or ("template" in r) == (sets == "template"))]
            pick = st.dataframe(pd.DataFrame([{"id": r["question_id"], "type": r["type"], "question": r["question"], "first rank": r.get("first_rank"),
                                               **{f"hit@{k}": r.get(f"hit@{k}") for k in evaluate.KS}} for r in shown]),
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

    with tabs[5]:
        query = st.text_input("Search the test bed now")
        k = st.slider("Results", 1, 20, 5)
        if query:
            try:
                chunks = {c["chunk_id"]: c for c in testbed.load_chunks()}
                vector_hits = testbed.search(testbed.qdrant(), testbed.embed([query])[0], k)
                word_hits = testbed.Keyword(list(chunks.values())).search(query, k)
                for col, label, hits in zip(st.columns(2), ("Embedding search (Qdrant)", "Keyword search (BM25)"), (vector_hits, word_hits)):
                    col.markdown(f"**{label}**")
                    for n, (cid, score) in enumerate(hits, start=1):
                        c = chunks[cid]
                        col.write(f"{n}. `{score:.3f}` {c['file_name']} p{c['pages'][0]}: {' '.join(c['text'].split())[:300]}")
            except Exception as exc:  # noqa: BLE001 - no index yet, or Qdrant/Ollama not running: shown, not raised
                st.error(f"{type(exc).__name__}: {exc}")

    with tabs[6]:
        try:
            tmpls, topics = templates.templates(), templates.load_topics()
        except config.BenchError as exc:
            st.error(str(exc))
            tmpls, topics = {}, {"switches": {}, "sections": {}, "groups": {}}

        def change(args: list[str]) -> None:
            ok, out = now(args)
            st.session_state["said"] = (ok, out)

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
        used = [s["segment_id"] for s in segs if s["use_for_questions"]]
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
        st.dataframe(pd.DataFrame([{"group": g, "pdfs": ", ".join(ps)} for g, ps in topics["groups"].items()]) if topics["groups"] else None)
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

    with tabs[7]:
        trace = rows(llm.TRACE)
        st.caption(f"{len(trace)} agent calls in data/trace.jsonl, newest first")
        st.dataframe(pd.DataFrame([{"time": t["time"], "role": t["role"], "label": t["label"], "model": t["model"], "seconds": t["seconds"],
                                    "in": t.get("input_tokens"), "out": t.get("output_tokens"), "error": t.get("error")}
                                   for t in reversed(trace[-500:])]), hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
