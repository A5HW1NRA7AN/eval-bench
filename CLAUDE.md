# eval-bench: notes for Claude

A search-quality benchmark. Agents write facts and questions from PDFs, a regex audit verifies them, and a local Qdrant test bed is scored. See `README.md` for how it works and how to run it.

## Rules

- **Git belongs to the user.** Never run `git commit`, `push`, `amend`, `reset`, `rebase` or `init`; read-only git is fine. At milestones, tell the user it's a good time to commit and give the exact command. Never add AI attribution anywhere.
- **Keep it minimal.** Use a flat `bench/` package and add a file only when no existing module fits. Comments are one line, or a short sentence where needed. Delete code, options and docs that a change makes unnecessary.
- **Secrets stay in `.env`.** Never print or commit keys. `.env.example` holds no secrets (keys are left blank).
- **Verification is regex, not AI.** A fact's quote must match the PDF text exactly, except for whitespace, quote style and dash style. A question's answer must match its facts' quotes.
- **Never tune the benchmark to the search.** Questions are never kept or dropped because of retrieval results.
- **LLM access:**
  - Local Ollama (`gemma4:12b`, embeddings `bge-m3`) is the default.
  - The remote Gemma 4 endpoint (`LLM_PROVIDER=gemma`) is down after ~9 pm IST.
  - Ollama's `/v1` endpoint needs `NativeOutput` (it rejects forced tool calls), which is passed wherever an Agent is built, and `reasoning_effort: "none"`, which is set in `bench/llm.py`.
  - Bump `FACTS_VERSION` or `QUESTIONS_VERSION` (`bench/agents.py`) when a prompt or the row format changes.
- The shared Amul Qdrant is out of scope. Only the local test bed (`docker compose up -d`) is used.
