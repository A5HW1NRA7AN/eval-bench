# eval-bench

A benchmark for **search quality**: how well does our search find the passage that answers a question?

LLM agents read our PDFs, pull out facts and write questions about them. A regex audit (no AI) checks that every fact and answer really is in the PDF. The verified questions are then run against a search system, and we score whether the right passage came back. Today that system is a **local Qdrant test bed** built from the same PDFs.

## How it works

```
pdfs/ ──segment──► segments: one per top-level entry of each PDF's table of contents
         ├──facts────► fact agent reads each segment in parts of a few pages ──regex audit──► facts.jsonl
         ├──questions► question agent writes questions from verified facts   ──regex audit──► questions.jsonl
         ├──index────► test bed: segments chunked (≤1200 chars) → bge-m3 → local Qdrant
         └──evaluate─► each question searched; a hit = a chunk overlapping its verified quote → report.md
```

1. **segment**: page text comes from PyMuPDF's layout engine, in reading order, with tables written one row per line under their headers. Running headers, page numbers, stray glyphs and junk OCR pages are removed, and every page gets a status (text, image only, garbled, broken font, blank). An agent reads each PDF's printed table of contents. Its entries are checked against the TOC text and kept in the TOC's order, then located in the body to find the page offset. An entry printed without a page number starts at the page where its title stands as a heading, or is dropped if there is none. Each top-level entry becomes a segment (one longer than 30,000 characters is cut at its sections), and every page belongs to exactly one segment. Junk OCR pages are left out of the text. Not used for questions, though still indexed:
   - front matter;
   - boilerplate sections (preface, abbreviations…);
   - the back matter: everything from the first annexure, references, glossary or index after the body;
   - the covers: up to 4 unnumbered pages after the last printed page number;
   - segments with less than 500 characters of text.

   A PDF without a usable TOC becomes one segment, minus its covers. A TOC is unusable when fewer than 3 entries are placed, or fewer than half of its titles are found where the TOC says.
2. **facts**: each segment is read in parts of about 12,000 characters (a few pages), cut at paragraph breaks, by three agents:
   - The **planner** sees the part as numbered paragraphs and picks searchable topics, each either a figure (one value) or an explanation (a procedure, list, comparison or rule).
   - The **extractor** sees each topic's paragraphs as numbered sentences. It writes a self-contained claim and points at the sentences that state it: one for a figure, usually the whole paragraph for an explanation. Code cuts the quote there, so it is never retyped.
   - The **reviewer** reads every fact with its quote and keeps it, fixes its claim or flags it, with a reason. A flag is an AI verdict: it is recorded and reported, but only the regex audit decides what is kept.
   - Each part carries its heading path, the end of the previous part and the facts found so far. The segment's fact limit is shared evenly across its parts: given a whole chapter at once, the model took all its facts from the first fifth.
3. **questions**: one writer agent per question type writes questions from each section's verified facts. A question reviewer keeps, rewords or flags each one (flags are reported as AI verdicts), and the regex audit alone decides. Questions are English only for now. The types live in **`question_types.toml`** (count, source, prompt and audit rules), so adding a type needs no code:
   - `direct`, `paraphrased` (other words), `keyword` (2–5 words), `precise` (a number with its qualifiers), `multi_fact` (two facts);
   - `comparison`: several answer parts from one paragraph-sized fact;
   - `noisy`: a kept question retyped the way farmers type (switched off for now: questions are English only);
   - `unanswerable`: a qualifier changed to something the library never mentions;
   - `multi_hop`: facts from two sections or PDFs, written once for the whole library.
4. **index**: every segment is cut into chunks without overlap or gaps, embedded with `bge-m3` and stored in Qdrant.
5. **evaluate**: each question is embedded and searched. Gold chunks are the ones that overlap its evidence quote, so scoring needs no judging. The report shows hit@1/3/5/10 and MRR by question type, plus every miss with what came back instead.

**The regex audit:**
- **Facts:** the quote is cut by code from the sentence range the agent points at, so it is always the PDF's own text. The range must exist and be at most 1,200 characters. A table row is a sentence of its own: rows of values are kept with their table's header rows (cut by code and recorded with the fact), and bare numbers without headers are rejected. The claim can't start with a pronoun or have a value its quote lacks. Any date, year or acronym it names must appear in the section. A fact that overlaps a kept one by more than half is dropped.
- **Questions:** each must refer to verified facts. Its answer is a single span, or several joined by `; `, and each span must appear in their quotes. No question may copy its quote, and a question whose quote states a reporting year (e.g. 2024-25) must name a year. Each type then has its own rules in `question_types.toml`, for example:
  - keyword queries are 2–5 words;
  - precise answers contain a number, in digits or as a number word (e.g. `four`, `twice`, `lakh`);
  - multi-fact and multi-hop questions take an answer span from each fact, and multi-hop facts come from two sections;
  - at most half of a paraphrase's content words may come from its quote;
  - an unanswerable question names a term found nowhere in the library;
  - a noisy question differs from the question it rewrites;
  - every question is plain English (no Indian script or romanized-Hindi words) and carries no internal id such as `d8_1`.
- **Rejected items** are kept in the output with the reason. Every kept fact and question has a content id (`uid`), so results can't be matched to the wrong question after a re-run.

## Setup

Needs [uv](https://docs.astral.sh/uv/), Docker and [Ollama](https://ollama.com).

```
ollama pull gemma4:12b        # writes facts and questions locally (~8 GB)
ollama pull bge-m3            # embeddings for the test bed (~1.2 GB)
cp .env.example .env          # then fill in GEMMA_URL and GEMMA_API_KEY to use the remote Gemma 4
uv sync --locked
docker compose up -d          # local Qdrant on 127.0.0.1:6333
```

Put the PDFs in `pdfs/`.

## Run

The entry point is the `bench` command (`bench/cli.py`; `python -m bench` works too). `bench run` runs every stage in order and checks each stage's output before the next one starts:

```
uv run bench run              # segment → facts → questions → index → evaluate, with checks
```

Or one stage at a time:

```
uv run bench segment          # new or removed PDFs are picked up; --refresh re-reads them all
uv run bench facts            # one fact per 3,000 characters (3-15 per segment); --max-facts N, --doc <part of a file name>, --force
uv run bench questions        # --doc, --force; types and counts in question_types.toml
uv run bench index
uv run bench evaluate
```

## Check and inspect

```
uv run bench check            # checks every stage's saved output without the LLM (or: bench check facts)
uv run bench export           # every stage, the scores and the agent calls in one workbook: data/bench.xlsx
uv run bench serve            # a read-only JSON API over every stage's output on http://127.0.0.1:8000
uv run bench --verbose facts  # any stage, also printing each agent's prompt and answer
```

Every agent call is shown live in the terminal: what it worked on, what it returned, its time and tokens. Each audit verdict is shown too, with the reason for every rejection. Every call is also saved to `data/trace.jsonl`, and its answer is cached in `data/cache/`, so a re-run (or a run after a change) only asks the model what it hasn't been asked before. `bench --no-cache <stage>` asks again.

Import `postman/eval-bench.postman_collection.json` into Postman. It holds requests for three things:
- the local Qdrant: browse chunks, filter by PDF or segment, and run a two-step semantic search;
- every stage's output, through `bench serve`;
- the local Ollama models.

The `facts` and `questions` stages save after every segment (atomically, so a stop never corrupts a file). Each row carries a **fingerprint** of everything that made it (prompts, output schemas, question types, settings and the text or facts it was written from). A run skips rows whose fingerprint is current and redoes the rest, plus rows whose kept items no longer pass the audit, so editing a prompt or `question_types.toml` redoes exactly what it affects. The model is recorded but not fingerprinted: to regenerate with another model, use `--force`. A section that fails (e.g. the endpoint is down) is left to do while the stage carries on; after 3 failures in a row the stage stops, and a re-run resumes. Only one writing command runs at a time per `data/` folder. `evaluate` first checks the questions and test bed, and stops if a question's evidence no longer matches the indexed text.

**Troubleshooting.** `✗ another bench command is writing data/`: wait for the other run. `N sections failed in a row`: the LLM endpoint is probably down (the remote Gemma 4 endpoint is up until about 9 pm IST); re-run later and it resumes. `Qdrant is not reachable`: `docker compose up -d`. `bench index` only rebuilds a local Qdrant.

**Switching to Gemma 4:** set `LLM_PROVIDER=gemma` in `.env`, or prefix a single command (`LLM_PROVIDER=gemma uv run bench questions`). The remote endpoint is only up in the daytime.

## Topic templates

A template describes what users ask about one *kind* of document, whatever the PDF: `templates/animal_care.toml`, `templates/govt_scheme.toml` and `templates/statistical_report.toml`. Each lists **intents** (e.g. symptoms, vaccination schedule; eligibility, benefit; growth, share). Every intent has:
- plain **cue words**, matched at the start of a word ("vaccin" matches vaccine and vaccination);
- for a figure intent, an **answer shape** (money, percent, ratio, age, interval, quantity, rate, organisation), named patterns defined in `bench/verify.py`. Templates never hold raw regex;
- example questions with `{subject}`.

**Sections are categorised by code, by themselves.** A template applies to a section when its signal words are dense enough there, and, for statistics, when the section is dense with numbers. A book can mix topics, so each section is categorised on its own. You can override this, and switch templates or single intents on and off for your data folder; these choices are stored in `data/topics.json`:

```
uv run bench templates list                          # each template, its intents, and the sections it applies to
uv run bench templates assign <section or pdf> animal_care,govt_scheme   # or: none, auto
uv run bench templates off animal_care symptoms      # on/off for a template or one intent
uv run bench templates run                           # facts and questions aimed at each section's intents (--doc, --force)
uv run bench templates coverage                      # data/coverage.md: which intents each section states and covers
uv run bench check templates                         # the files and data/topics.json are valid
```

The coverage report needs no LLM. An intent is *stated* in a section when its cue words occur there. A fact speaks to an intent when a cue is in its quote or in the heading line just above it, and, for figure intents, the quote holds an answer of the intent's shape. Each (section, intent) is marked covered (a kept question), facts only, stated but missing, or not stated.

**Template questions** (`bench templates run`, part of `bench run` unless `--no-templates`) are written alongside the general ones. In each part of a section where a switched-on intent's cue words occur (other parts cost no call), an intent planner picks topics, each an intent and its subject (e.g. a disease, a scheme, a measure). The fact extractor and reviewer write and review one fact per topic; a table row can be a fact. The regex audit adds three rules to the fact rules: a cue word in the quote, its table header, the heading line above or its TOC headings; a figure intent's answer shape; and the subject named in the quote, its headings or the document title. Questions are written in the template's `styles` (question types, `direct` and `paraphrased` by default), with the intent's example question as a guide, and must also name their subject and differ from every general question. They are scored in their own report section (`Topic templates`, per intent), so the general headline stays comparable. Output: `data/templates.jsonl`; ids `<section>:tf01` (facts) and `<section>:t01` (questions).

To add a document type, add a TOML file with the same fields; `bench check templates` reports any mistake.

## Outputs (`data/`, not committed)

| File | What it holds |
|---|---|
| `pages.jsonl` | cleaned text of every PDF page, with its status and table count |
| `ingest.jsonl` | per PDF: page statuses, tables, how the TOC was used (agreement, dropped titles), segment sizes |
| `segments.jsonl` | segments with their text, page map and TOC headings |
| `facts.jsonl` | per segment: verified facts (with position and pages) and rejected ones (with reasons) |
| `questions.jsonl` | per segment: verified questions, each with its evidence (quotes and positions), and rejected ones |
| `chunks.jsonl` | the test-bed chunks stored in Qdrant |
| `results.jsonl`, `report.md` | per-question results (the top 100 chunks) and the scored report, per type and per document |
| `runs/<run>/` | each `bench evaluate`: the facts, questions, results and report it scored, `manifest.json` (code, models, test set, PDFs) and `stats.json` (audit rejections by rule, AI flags, scores) |
| `cache/` | every LLM answer, keyed by model, settings, instructions, schema and prompt |
| `topics.json`, `coverage.md` | your template overrides and switches; the template coverage report |
| `templates.jsonl` | per section: template facts and questions (with template, intent and subject) and rejected ones |
| `trace.jsonl` | every agent call: role, prompt, output, tokens, seconds |
| `bench.xlsx` | all of the above as readable sheets (`bench export`) |

## Layout

```
bench/      cli.py · config.py · llm.py · pdf.py · segments.py · facts.py · questions.py · verify.py · testbed.py · evaluate.py
            checks.py (bench check) · export.py (bench export) · serve.py (bench serve)
tests/      offline tests (no LLM or Qdrant needed): uv run pytest
postman/    the Postman collection
question_types.toml   the question types: count, source, prompt and audit rules per type
templates/            topic templates: intents, cue words, answer shapes and examples per document type
```
