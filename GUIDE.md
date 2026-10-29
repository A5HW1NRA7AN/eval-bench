# eval-bench: a guided tour

This guide explains how the benchmark works from end to end, how to run it, how to check and inspect every stage, and why it is built the way it is. Read it with the code open: every concept names the file and function where it lives. The commands are safe to try; nothing here changes shared systems.

Contents:
1. [The problem](#1-the-problem)
2. [The pipeline at a glance](#2-the-pipeline-at-a-glance)
3. [One fact's journey, end to end](#3-one-facts-journey-end-to-end)
4. [Each stage in detail](#4-each-stage-in-detail)
5. [Running it](#5-running-it)
6. [Checking and inspecting](#6-checking-and-inspecting)
7. [Reading the report](#7-reading-the-report)
8. [Code map](#8-code-map)
9. [Why it is built this way: the decision log](#9-why-it-is-built-this-way-the-decision-log)
10. [Where we are and what's next](#10-where-we-are-and-whats-next)

---

## 1. The problem

We want to measure **search quality**: when someone asks a question, does the search system return the passage that answers it?

The metric this replaces is `search_quality` in OpenAgriNet's `oan-evaluation` (`metrics/llm_metrics.py`). It scores 1.0 whenever the word "search" or "query" appears in the agent's turns. That shows the agent searched, not that it found the right thing. A real metric needs **ground truth**: questions paired with the exact passage that answers each one.

Our approach builds that ground truth from the documents themselves:
1. An LLM reads the PDFs and writes **facts**: a self-contained claim plus a **verbatim quote** from the PDF.
2. An LLM writes **questions** whose answer is in those quotes.
3. A **regex audit** (no AI) proves every quote really is in the PDF and every answer really is in its quotes. Nothing unverified gets in.
4. Each question is sent to a **search system**. We score whether the returned text contains the question's evidence.

Any search system can then be graded the same way, because the grading only needs the question, its evidence quote and the returned text.

**Why a local test bed?** We first probed the Amul dev Qdrant. It holds only per-farmer memory collections, not these PDFs, so it could not answer questions about them. We built our own Qdrant index of the same 5 PDFs to develop and validate the benchmark. Real targets, starting with the agent's search, are the next step ([section 10](#10-where-we-are-and-whats-next)).

---

## 2. The pipeline at a glance

```
pdfs/ ─► segment ─► facts ─► questions ─► index ─► evaluate ─► report.md
         (TOC)     (agent)   (agent)     (Qdrant)  (search + score)
                      ▲          ▲
                 regex audit  regex audit
```

| Stage | Input | What happens | Output (`data/`) | Checked by | Look at it |
|---|---|---|---|---|---|
| `segment` | PDFs | Clean page text; an LLM parses each table of contents; cut each PDF into one segment per top-level TOC entry | `pages.jsonl`, `toc/`, `segments.jsonl` | `bench check segment` | `/pages`, `/toc`, `/segments` |
| `facts` | question segments | The fact agent reads each segment in parts and writes facts (claim + verbatim quote); the audit keeps only verified ones | `facts.jsonl` | the audit, then `bench check facts` | `/facts` |
| `questions` | verified facts | The question agent writes 5 types of questions; the audit keeps only verified ones; each question carries its evidence | `questions.jsonl` | the audit, then `bench check questions` | `/questions` |
| `index` | all segments | Chunk every segment (≤1200 chars, lossless), embed with bge-m3, store in the local Qdrant | `chunks.jsonl`, Qdrant `bench_docs` | `bench check index` | Postman folder 1 |
| `evaluate` | questions + index | Search each question; score where its evidence lands | `results.jsonl`, `report.md` | `bench check evaluate` | `/results`, `/report` |

The `/…` paths are endpoints of `bench serve` ([section 6](#6-checking-and-inspecting)).

---

## 3. One fact's journey, end to end

Here is one real fact followed through every stage. You can repeat each step in Postman or with `bench serve`.

**1. Page.** On page 25 of `Aflatoxin in feed and milk.pdf` (printed page 24) the FAQ says:

> It has been reported that Aflatoxins appear in milk within 12 hours following oral administration of Aflatoxin B1. The peak concentration in milk occurs after 24 hrs. Clearance is also very rapid, since Aflatoxin disappears from the milk four days after cessation of oral administration.

Cleaning removes the running header and the page number. Compare `text` with `raw_text` at `/pages?doc=Aflatoxin&page=25`.

**2. Segment.** The TOC agent read the PDF's contents page. The code checked the entries against the TOC text and found where each starts (PDF page = printed page + 1 in this PDF). "PART – III : Frequently Asked Questions" became segment `aflatoxin-in-feed-and-milk-03`, covering pages 23-26 (`/segments/aflatoxin-in-feed-and-milk-03`).

**3. Fact.** The segment is only 3,596 characters, so the fact agent read it in one part, with its heading path as context. It returned:
- claim: *"Aflatoxins appear in milk within 12 hours of oral administration of Aflatoxin B1, peaking at 24 hours and disappearing four days after administration ceases."*
- quote: the three sentences above.

The audit found the quote verbatim in the segment text at characters 2162-2453 (page 25) and kept it as `aflatoxin-in-feed-and-milk-03:f08` (`/facts?segment=aflatoxin-in-feed-and-milk-03`).

**4. Question.** From the segment's verified facts, the question agent wrote, among others:
- `direct`: *"How long does it take for Aflatoxins to peak in milk after oral administration of Aflatoxin B1?"*
- answer: `after 24 hrs`

The audit confirmed `after 24 hrs` is inside the quote, and the question doesn't copy the quote. The question was saved with its **evidence**: the quote and its position (`/questions?doc=Aflatoxin`).

**5. Index.** Segment `-03` was cut into chunks. The quote sits in chunk `aflatoxin-in-feed-and-milk-03:c003`, which was embedded with bge-m3 and stored in Qdrant. In Postman, run "Chunks of one segment" with `segment_id = aflatoxin-in-feed-and-milk-03`.

**6. Search and score.** `evaluate` embedded the question and asked Qdrant for the 10 nearest chunks:

| rank | chunk | score | text starts |
|---|---|---|---|
| 1 | `aflatoxin-…-01:c012` | 0.805 | "Diaz et al. (2004) have indicated that Aflatoxins appear in milk within 12 hours…" |
| 2 | `aflatoxin-…-03:c003` | 0.788 | "It has been reported that Aflatoxins appear in milk within 12 hours…" |
| 3 | `aflatoxin-…-03:c002` | 0.709 | "Bureau of Indian Standards (BIS) in India has set…" |

The gold chunk is the one that overlaps the evidence quote: rank 2. So this question scores hit@1 = 0, hit@3 = 1 and reciprocal rank 0.5.

**This example also shows the benchmark's main known limitation.** The rank-1 chunk comes from the main text (Part I), which states the same fact. A person would call it a correct answer, but only the linked passage counts. So scores are a **lower bound** when a fact appears in more than one place. A fairer rule is listed under "what's next".

---

## 4. Each stage in detail

### 4.1 `segment`: pages → table of contents → segments

Code: `bench/pdf.py`, `bench/segments.py`.

**Clean the pages** (`pdf.extract`). PyMuPDF reads each page. We then remove:
- **running headers and footers**: lines repeated at the top or bottom of 4+ pages (`running_lines`, `clean_page`);
- **printed page numbers**, but only numbers that advance with the page, so table values survive;
- **stray glyphs** such as `yy` and `□`.

Pages whose text layer is OCR junk are flagged `garbled` and left out of the text. Each page keeps its `printed` page number (`printed_number`), which later identifies the covers.

**Parse the table of contents** (`find_toc_pages`, `read_toc`). The TOC pages are found by look: "Contents" or "Index" plus many lines ending in page numbers. The **TOC agent** (an LLM, prompt `TOC_PROMPT`) turns them into JSON entries: number, title, printed page, level. LLMs make mistakes here, so nothing is trusted until the code has checked it:
- `keep_valid`:
  - every title must actually appear in the TOC text, or it was invented;
  - entries are put back in the TOC text's order (the model once moved Part labels below their sections);
  - each TOC line may be used once;
  - page numbers must never go backwards.
- `locate`:
  - the **page offset** (PDF page minus printed page) is the most common offset over all titles found in the body;
  - each entry then starts at its own title within 2 pages of printed page + offset;
  - an entry with no page number starts at the page where its title stands as a heading line, searched no further than the next entry that has a page.

The parse is cached in `data/toc/`, so re-runs need no LLM.

**Cut segments** (`segment_pdf`). Each top-level TOC entry becomes one segment, and every page with text belongs to exactly one segment. Some segments are **indexed but never used for questions**:
- front matter;
- boilerplate titles (preface, abbreviations…), matched at the start of the title only;
- the back matter: everything from the first annexure, references, glossary or index after the body;
- the covers: up to 4 unnumbered pages after the last printed page number;
- segments with less than 500 characters of text.

A PDF without a usable TOC becomes one segment.

### 4.2 `facts`: the fact agent

Code: `bench/agents.py` (`FACT_PROMPT`, `run_facts`), `bench/verify.py` (`verify_facts`).

The fact agent reads each question segment in **parts of about 12,000 characters** (`parts`, `PART_CHARS`), cut at paragraph breaks. Each part carries three kinds of context:
- the **heading path** ("PART-I > Section IV > …");
- the **end of the previous part** (800 characters, marked as context only);
- the **facts found so far** (its memory), so it doesn't repeat itself.

The segment's limit of 10 facts is **shared evenly across its parts**, plus any shortfall from earlier parts. Each fact is:
- `claim`: one self-contained statement that names its subject and every qualifier (animal, stage, dose, unit, year, place, scheme);
- `quote`: one or two sentences copied exactly from the text, at most 300 characters, never a bare table row.

**The audit** (`verify_facts`) keeps a fact only if its quote is found in the segment text. Only whitespace, quote style (' ’) and dash style (- –) may differ (`pattern`). It must also be at most 300 characters, not mostly numbers (`table_row`) and not a duplicate. Kept facts record their exact position and pages. Rejected ones are kept with the reason.

### 4.3 `questions`: the question agent

Code: `bench/agents.py` (`QUESTION_PROMPT`, `run_questions`), `bench/verify.py` (`question_problem`).

The question agent gets a segment's verified facts and writes up to 8 questions of five types:

| Type | What it tests | Example shape |
|---|---|---|
| `direct` | the plain case | "What is the maximum level of Aflatoxin B1 allowed in dairy feed in India?" |
| `paraphrased` | meaning, not words | the way a farmer would say it, avoiding the quote's key words |
| `keyword` | search-box queries | "Aflatoxin heat stability" (2-5 words) |
| `precise` | exact values with qualifiers | a dose, age, duration or limit |
| `multi_fact` | finding two passages | one topic whose full answer needs two facts |

Every question must make sense on its own in a library of many documents, so it names its subject and qualifiers.

**The audit** (`question_problem`) rejects a question when:
- it refers to an unverified fact;
- an answer span is not in its facts' quotes (multi-fact answers are spans joined by `; `);
- it copies its quote;
- it breaks its type's rule:
  - a keyword query isn't 2-5 words;
  - a precise answer has no number (digits or a number word);
  - a multi-fact question doesn't use two facts;
  - more than half of a paraphrase's words come from the quote;
- its quote states a reporting year (e.g. 2024-25) that the question doesn't name.

A kept question stores its **evidence**: the quotes and their positions. Scoring depends only on the question itself, never on a later facts run.

### 4.4 `index`: the local test bed

Code: `bench/testbed.py`.

Every segment, including the ones not used for questions (they are realistic distractors), is cut into **chunks of at most 1,200 characters** (`chunk`). The cuts prefer paragraph breaks, then sentence ends, then spaces. Chunks never overlap and never drop text, so joined back together they give the segment text exactly. Each chunk is embedded with **bge-m3** through Ollama and stored in the Qdrant collection `bench_docs`. The payload holds the chunk id, segment, file, pages and **text**.

### 4.5 `evaluate`: search and score

Code: `bench/evaluate.py`.

Each question is embedded the same way, and Qdrant returns the 10 nearest chunks. The **gold chunks** are the chunks that overlap the question's evidence quote, so scoring needs no judgement. For each question we record:
- **first rank**: where the first gold chunk appears (1-10, or none);
- **hit@k**: whether a gold chunk is in the top k (k = 1, 3, 5, 10);
- **rr**: 1 / first rank, or 0 (MRR is its average);
- **all_facts@10**: for multi-fact questions, whether every fact's chunk was found.

Before scoring, `check_evidence` confirms each evidence quote is still where it was recorded. If an earlier stage was re-run without the later ones, evaluation stops instead of scoring stale data.

Results are never used to keep, drop or rewrite questions. That would tune the benchmark to the system it grades.

---

## 5. Running it

**Setup once:**
```
ollama pull gemma4:12b && ollama pull bge-m3
cp .env.example .env            # fill GEMMA_URL and GEMMA_API_KEY to use the remote Gemma 4
uv sync
docker compose up -d            # local Qdrant on 127.0.0.1:6333
```
Put the PDFs in `pdfs/`.

**The entry point** is the `bench` command (`bench/cli.py`; `python -m bench` also works). To run everything, with a check after each stage:
```
uv run bench run
```
It stops at the first stage whose output fails a check, and tells you which one.

**One stage at a time:**
```
uv run bench segment            # --refresh re-reads every PDF
uv run bench facts              # --doc Aflatoxin  --max-facts 10  --force
uv run bench questions          # --doc Aflatoxin  --max-questions 8  --force
uv run bench index
uv run bench evaluate
```

**Resuming.** `facts` and `questions` save after every segment and skip segments already done at the current prompt version. A stopped run picks up where it left off. Changing a prompt means bumping `FACTS_VERSION` or `QUESTIONS_VERSION` in `bench/agents.py`; the next run then redoes the old rows. `--force` redoes everything.

**Which LLM writes.** Local Ollama `gemma4:12b` is the default. The remote Gemma 4 31B is better and much faster (a full run takes about 10 minutes instead of about 70), but it is only up in the daytime. Set `LLM_PROVIDER=gemma` in `.env`, or prefix one command: `LLM_PROVIDER=gemma uv run bench questions`.

---

## 6. Checking and inspecting

### Verification happens at four levels

1. **While generating:** the regex audit accepts or rejects every fact and question, and every rejection is saved with its reason.
2. **After each stage:** `bench check` re-checks the saved output without the LLM. `bench run` does this after every stage.
3. **By eye:** `bench serve` and Postman show every agent's output.
4. **Independently:** reviewer agents audited the questions blind to search results ([section 9](#9-why-it-is-built-this-way-the-decision-log)).

### `bench check`

```
uv run bench check              # all stages
uv run bench check questions    # one stage
```

| Stage | What it checks | If it fails |
|---|---|---|
| segment | every PDF is extracted; TOC caches are current; every page with text is in exactly one segment; page positions point at the right text | `bench segment` |
| facts | every question segment has facts at the current version; every quote is still verbatim at its position; quote rules hold | `bench facts` (or `--force`) |
| questions | every segment with facts has questions at the current version; every question still passes the audit; evidence is still in place | `bench questions` (or `--force`) |
| index | chunks rebuild every segment's text exactly; Qdrant holds every chunk | `bench index` (start Qdrant first) |
| evaluate | every current question is scored; the report exists | `bench evaluate` |

### `bench serve`: every stage's output as JSON

```
uv run bench serve              # http://127.0.0.1:8000, local only, read-only
```

| Endpoint | Shows |
|---|---|
| `/` | the endpoint list |
| `/pages?doc=Aflatoxin&page=25` | one page: clean text, raw text, printed number |
| `/toc` | the TOC agent's parses |
| `/segments?doc=Aflatoxin` · `/segments/<id>` | segments; one segment with its text and headings |
| `/facts?doc=…` · `/facts?segment=<id>` | the fact agent's output: verified and rejected facts |
| `/questions?doc=…&type=paraphrased` | the question agent's output, with evidence |
| `/results?misses=1` | per-question scores (misses only) |
| `/report` | the report (Markdown) |
| `/checks` | the stage checks |
| `/search?q=…&k=5` | search the test bed now |

### Postman

Import `postman/eval-bench.postman_collection.json` (File → Import). Its variables (`qdrant`, `bench`, `ollama`, `doc`, `segment_id`, `question`…) are set for this machine, so nothing needs filling in.

1. **Qdrant: the local test bed.** Collection info, count, browse chunks (each run of "Browse the next chunks" pages on), chunks of one PDF or one segment, and one chunk by id.
   - **Semantic search** shows how vector search works, in two steps. Step 1 sends `{{question}}` to Ollama's bge-m3 and saves the 1,024-number vector. Step 2 asks Qdrant for the 5 chunks whose vectors are closest. This is exactly what `bench evaluate` does.
2. **Pipeline outputs.** Every `bench serve` endpoint. Start `uv run bench serve` first.
3. **Ollama.** Installed and loaded models.

Change `question`, `doc` or `segment_id` in the collection's Variables tab to explore.

---

## 7. Reading the report

`data/report.md` (or `/report`) has one table: hit@1/3/5/10 and MRR overall and per question type, with n in every row. Then it lists every **miss**: a question whose evidence was not in the top 10, the evidence it expected, and the 3 chunks that came back instead.

How to read it:
- **hit@1** is what a user sees first. **hit@10** is what an agent that reads 10 results can still use.
- **Paraphrased** questions are the hardest by design: they test meaning, not shared words.
- Scores on the local test bed are a **sanity baseline** from a near-ideal index (clean chunks of the very same PDFs). They prove the benchmark works. They don't measure the Amul search; that's the next step.
- When a fact is stated in more than one place, only the linked passage counts, so scores are a **lower bound** (see the example in section 3).

---

## 8. Code map

| File | Responsibility | Key functions |
|---|---|---|
| `bench/cli.py` | the `bench` command (entry point) | `run`, `segment`, `facts`, `questions`, `index`, `evaluate`, `check`, `serve` |
| `bench/config.py` | `.env` settings and limits | `settings`, `MAX_FACTS`, `MAX_QUESTIONS`, `PART_CHARS` |
| `bench/llm.py` | one model object for Ollama or Gemma 4 | `model`, `check_usage` |
| `bench/pdf.py` | clean page text | `extract`, `running_lines`, `clean_page`, `printed_number`, `garbled` |
| `bench/segments.py` | TOC → segments, segments → parts | `read_toc`, `keep_valid`, `locate`, `segment_pdf`, `parts` |
| `bench/agents.py` | the fact and question agents and their stages | `FACT_PROMPT`, `QUESTION_PROMPT`, `run_facts`, `run_questions` |
| `bench/verify.py` | the regex audit | `find`, `verify_facts`, `question_problem` |
| `bench/testbed.py` | chunks, embeddings, Qdrant | `chunk`, `embed`, `run_index`, `search` |
| `bench/evaluate.py` | scoring and the report | `gold_chunks`, `check_evidence`, `score`, `run_evaluate` |
| `bench/checks.py` | `bench check` | `check_segment`, `check_facts`, `check_questions`, `check_index`, `check_evaluate` |
| `bench/serve.py` | `bench serve` | `route` |
| `tests/` | offline tests, no LLM or Qdrant needed | `uv run pytest` |

---

## 9. Why it is built this way: the decision log

Each decision, why it was made, and what it was based on. Newest learnings are folded into the area they belong to.

### Principles

| Decision | Why |
|---|---|
| Ground truth is built from the PDFs: fact + verbatim quote + question | A retrieval metric needs to know the right passage; the documents are the only trustworthy source of it. |
| **Verification is regex, not AI** | An LLM checking another LLM can share its mistakes. A verbatim match is deterministic, explainable and cheap. If it isn't literally in the PDF, it isn't in the benchmark. |
| **Never keep, drop or rewrite a question because of search results** | Otherwise the benchmark is tuned to the system it grades, and its scores stop meaning anything. Quality is judged only by the audit and by reviewers who never see the scores. |
| A fixed pipeline, not an autonomous agent | A benchmark must be reproducible: the same PDFs and settings give the same test set. The agents do language work inside fixed stages. |
| Reproducibility settings: temperature 0, a fixed seed, cached TOC parses, versioned prompts | So reruns don't drift, and stale outputs are detected rather than mixed in. |
| English PDFs with a text layer only (v1) | Start simple; OCR and translation add error sources of their own. |
| Shared systems are read-only and out of scope; secrets live only in `.env` | The Amul dev servers are shared and hold personal farmer data. |
| A minimal repo: a flat `bench/` package, short comments, no dead code | Your ground rule; easier to read, review and hand over. |
| Git belongs to you; no AI attribution | Your rule; you commit and push. |

### Scope changes

| Stage | Decision | Why |
|---|---|---|
| Start (Oct 6-13) | Evaluate the Amul Qdrant with questions built from PDFs | The original plan. |
| First probe | The read-only probe found the Amul dev Qdrant holds only farmer-memory collections, not the PDFs | Questions about the PDFs couldn't be answered there; probing it further would touch personal data. |
| Restart (Oct 27) | Rebuild from scratch: agents + TOC segments + regex audit + **local Qdrant test bed** | Your restart. A local index of the same PDFs lets us build and validate the benchmark before pointing it at real systems. |
| Now | The benchmark must evaluate other searches too, starting with the agent's search | Your direction: the test bed is temporary. |

### Models and tools

| Decision | Why |
|---|---|
| PydanticAI agents with typed outputs (`Facts`, `Questions`, `TocEntries`) | Structured, validated JSON from the LLM, with no hand-written parsing. |
| Local Ollama `gemma4:12b` by default; remote Gemma 4 31B via `LLM_PROVIDER=gemma` | The remote endpoint is down after ~9 pm; the local model keeps work going. Same code path for both. |
| `NativeOutput` + the JSON schema written into the instructions + `reasoning_effort: "none"` | Ollama's API rejects forced tool calls and never shows the schema to the model; turning Gemma's "thinking" off makes it fast and keeps the output clean. |
| Gemma 4 31B for the current test set | Compared on the same segments, it was about 7x faster, kept more questions (221 vs 212) and copied fewer paraphrases (7 rejected vs 26). |
| bge-m3 embeddings via Ollama; Qdrant 1.19 in Docker | A strong, local, multilingual embedding model; Qdrant matches the Amul stack we first targeted. |
| PyMuPDF for text | Fast, keeps text blocks (paragraphs), gives per-page text for page numbers. |

### Segmentation

| Decision | Why | Evidence |
|---|---|---|
| Segments follow the printed table of contents, one per top-level entry | Your choice: the author's own structure gives meaningful units, and facts and questions can be capped per section. | — |
| The LLM parses the TOC, but the code validates everything | LLM TOC parses go wrong in recognisable ways. | Observed: Part labels moved below their sections; repeated and invented entries; levels promoted inconsistently. |
| Page offset by majority vote, each entry within 2 pages | Printed and PDF page numbers differ by a fixed offset per book (+8, +12, +1, +19…). | 4 of 5 PDFs have one clean offset. |
| Entries without page numbers are found by their heading line, searched only up to the next paged entry | Some TOCs list Glossary or Part labels without pages; an unbounded search could push every later entry to a wrong page. | Nutrition booklet, Aflatoxin Parts. |
| Every page in exactly one segment | No text is lost from the index or counted twice. | An early version copied page 91 into 5 segments. |
| Front matter, boilerplate, back matter and covers are indexed but not used for questions | They are realistic distractors for search but poor sources of questions (citations, contact details, tables without headers). | The Handbook's back cover (NDDB address) and Nutrition's table annex were feeding questions before the fix. |
| Back matter = everything from the first annexure/references/glossary/index after the body | The LLM's levels are unreliable; this rule doesn't depend on them. | Statistics' table sections were promoted to top level by the model. |
| Covers = up to 4 unnumbered pages after the last printed page number | Body pages carry printed numbers; covers don't. The limit stops a page-number detection failure from cutting the body. | All 5 PDFs end with 1-2 such pages. |
| Boilerplate words are matched at the start of a title only | "Temperature Humidity Index" is a chapter, not an index. | Found in review. |
| An independent check of every segment boundary by 5 reviewer agents | A second pair of eyes on the structure everything else depends on. | It found the table annex, the covers and a dropped TOC entry, which were fixed. A few 1-2 page boundary slips (e.g. a heading broken in the PDF's text layer) were left: no text is lost from questions. |

### Facts

| Decision | Why | Evidence |
|---|---|---|
| **Read each segment in ~12,000-character parts, with the fact limit shared evenly** | Given a whole chapter, the model took every fact from its beginning. Asking it to spread them, or marking the text into parts, didn't help. This changed your original plan to send whole segments; the evidence decided it. | Dairy ch.2: all facts in the first 18% → with parts, spread across 1-99%. |
| Memory across parts: heading path + end of the previous part + facts so far | Your choice. Context-aware parts that don't repeat facts. | — |
| Claims must be self-contained, with every qualifier | A fact like "give 2 kg" is useless or wrong without "to a pregnant buffalo in the last 2 months". This is the failure the Amul search_quality judge describes. | — |
| Quotes are verbatim, ≤300 characters | Verbatim makes them checkable; short makes the gold passage precise. | — |
| Bare table rows are rejected as quotes | A row of numbers has no meaning without its headers, so claims built on it guessed the year or unit. | The blind audit; the rule caught 12 rows on the next run. |

### Questions

| Decision | Why | Evidence |
|---|---|---|
| Five question types | Your choice: easy direct questions, paraphrases that test meaning, keyword queries like a search box, precise values, and multi-passage questions. | — |
| Multi-fact answers are spans joined by `; ` | A two-fact answer naturally has two parts; the first rule demanded one span and rejected every multi-fact question. | 0 of 3 passed before the fix. |
| Number words count as numbers in precise answers | "four days" is a precise answer. | Rejected before the fix. |
| Paraphrases may take at most half their words from the quote | Otherwise a "paraphrase" is just the quote again and tests nothing new. | — |
| Questions must name their subject and qualifiers; reporting years must be named | A question with no year or place has several correct answers in a multi-document library. | The blind audit: 28 of the 34 serious question problems were this kind of ambiguity. |
| Each question carries its evidence (quote + position) | Questions first referred to facts by position-based ids, so a later facts run silently changed their meaning. | A real crash (`KeyError`) when one segment's facts changed. |
| Versioned, resumable stages; stale rows redone or dropped; `evaluate` refuses stale evidence | Mixed or stale outputs must never be scored silently. | Found in review. |

### Test bed and scoring

| Decision | Why |
|---|---|
| Lossless chunks of ≤1,200 characters, paragraph-first | A common chunk size; lossless chunks make gold mapping exact. |
| Gold = chunks overlapping the evidence quote | Exact and judge-free on our own index. |
| hit@1/3/5/10, MRR, all-facts@10, misses listed with what came back | Standard retrieval metrics, plus the misses so a person can see *why* a search failed. |
| The chunk text is stored in the Qdrant payload | So you can read what is indexed in Postman, as real deployments do. |

### Quality reviews

| Review | Result | What changed |
|---|---|---|
| Segment boundaries (5 agents + skeptics) | 1 high, 2 medium, 6 low findings | Table annex, covers, hyphenated TOC titles, wrapped headings |
| Code review (docs, dead code) | 6 medium, 11 low findings | Version-aware resume, model named in the report, `--k` removed, new PDFs detected, doc fixes |
| Blind audit of 295 facts and 219 questions (judges never saw search results) | Facts: 1% unsupported. Questions: 15% with a serious problem, mostly ambiguity | Qualifier rules, table-row rule, year rule; the set was regenerated |
| Re-audit of the regenerated questions | Only 1 of 5 PDFs finished before a session limit (3 problems in 19) | To be re-run |

### Repository

| Decision | Why |
|---|---|
| Internal hostnames removed from committed files | The GitHub repo is public. |
| A fresh one-commit history | The old baseline described the Amul dev Qdrant and its farmer data, which shouldn't be published. |
| Your GitHub noreply email for commits | Commits link to your account without exposing a personal address. |

---

## 10. Where we are and what's next

**Now:**
- **Pipeline:** complete, with 22 offline tests and checks at every stage.
- **Test set:** 5 PDFs → 57 segments (33 used for questions) → 288 verified facts → 221 verified questions, written by Gemma 4 31B.
- **Test-bed scores:** hit@1 0.66, hit@10 0.97, MRR 0.77; paraphrased questions are the hardest (hit@1 0.41).

**Known limits:**
- A fact stated in two places counts only at its linked passage, so scores are a lower bound.
- About 1 question in 7 was ambiguous before the latest fixes; the re-audit will measure what's left.
- Only English PDFs with a text layer are supported.

**Next: evaluate real search systems.** We researched the Amul agent's code (read-only, public repos):
- The chat agent's tool `search_documents` sends compact keyword queries to a Beckn "seeker" service, or straight to a Marqo index (`amul-veterinary-index`), depending on the deployment.
- The chat API doesn't return the passages it retrieved, but the Langfuse traces do.

The plan:
1. **Targets.** A target is anything that takes a query and returns ranked text: the local test bed, a Marqo or Qdrant index, the seeker API, or the agent itself through its chat API plus traces.
2. **One scoring rule for all targets.** A hit means the returned text contains the evidence quote, with the same tolerant matching as the audit. Other systems chunk differently, so chunk ids can't be compared.
3. **Presence comes from a document list, not from search hits.** If a PDF isn't in a system's corpus, its questions are reported as "not indexed", not as search failures.

What we need from the Amul team:
- **Which search to test:** production chat (direct Marqo) or main (Beckn seeker)?
- **Read-only access** to one of: the Marqo endpoint, the seeker's `/search`, or the chat API plus Langfuse read keys.
- **The list of documents** in the agent's index, and whether our 5 PDFs are among them. If not, we generate questions from the PDFs that are.
- **Your approval** to widen the rule that keeps shared infrastructure out of scope.
