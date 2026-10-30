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

1. **segment**: page text is cleaned (running headers, page numbers, stray glyphs, junk OCR pages). An agent reads each PDF's printed table of contents. Its entries are checked against the TOC text and kept in the TOC's order, then located in the body to find the page offset. An entry printed without a page number starts at the page where its title stands as a heading, or is dropped if there is none. Each top-level entry becomes a segment, and every page belongs to exactly one segment. Junk OCR pages are left out of the text. Not used for questions, though still indexed:
   - front matter;
   - boilerplate sections (preface, abbreviations…);
   - the back matter: everything from the first annexure, references, glossary or index after the body;
   - the covers: up to 4 unnumbered pages after the last printed page number;
   - segments with less than 500 characters of text.

   A PDF without a usable TOC becomes one segment, minus its covers. A TOC is unusable when fewer than 3 entries are placed, or fewer than half of its titles are found where the TOC says.
2. **facts**: each segment is read in parts of about 12,000 characters (a few pages), cut at paragraph breaks, by three agents:
   - The **planner** sees the part as numbered paragraphs and picks searchable topics, each either a figure (one value) or an explanation (a procedure, list, comparison or rule).
   - The **extractor** sees each topic's paragraphs as numbered sentences. It writes a self-contained claim and points at the sentences that state it: one for a figure, usually the whole paragraph for an explanation. Code cuts the quote there, so it is never retyped.
   - The **reviewer** reads every fact with its quote and keeps it, fixes its claim or drops it, with a reason.
   - Each part carries its heading path, the end of the previous part and the facts found so far. The segment's fact limit is shared evenly across its parts: given a whole chapter at once, the model took all its facts from the first fifth.
3. **questions**: one writer agent per question type writes questions from each section's verified facts. A question reviewer keeps, rewords or drops each one, and the regex audit decides. The types live in **`question_types.toml`** (count, source, prompt and audit rules), so adding a type needs no code:
   - `direct`, `paraphrased` (other words), `keyword` (2–5 words), `precise` (a number with its qualifiers), `multi_fact` (two facts);
   - `comparison`: several answer parts from one paragraph-sized fact;
   - `noisy`: a kept question retyped the way farmers type (typos, Hinglish);
   - `unanswerable`: a qualifier changed to something the library never mentions;
   - `multi_hop`: facts from two sections or PDFs, written once for the whole library.
4. **index**: every segment is cut into chunks without overlap or gaps, embedded with `bge-m3` and stored in Qdrant.
5. **evaluate**: each question is embedded and searched. Gold chunks are the ones that overlap its evidence quote, so scoring needs no judging. The report shows hit@1/3/5/10 and MRR by question type, plus every miss with what came back instead.

**The regex audit:**
- **Facts:** the quote is cut by code from the sentence range the agent points at, so it is always the PDF's own text. The range must exist and be at most 1,200 characters, and the quote can't be a bare table row (mostly numbers). The claim can't start with a pronoun or have a value its quote lacks. Any date, year or acronym it names must appear in the section. A fact that overlaps a kept one by more than half is dropped.
- **Questions:** each must refer to verified facts. Its answer is a single span, or several joined by `; `, and each span must appear in their quotes. No question may copy its quote, and a question whose quote states a reporting year (e.g. 2024-25) must name a year. Each type then has its own rules in `question_types.toml`, for example:
  - keyword queries are 2–5 words;
  - precise answers contain a number, in digits or as a number word (e.g. `four`, `twice`, `lakh`);
  - multi-fact and multi-hop questions take an answer span from each fact, and multi-hop facts come from two sections;
  - at most half of a paraphrase's content words may come from its quote;
  - an unanswerable question names a term found nowhere in the library;
  - a noisy question differs from the question it rewrites.
- **Rejected items** are kept in the output with the reason.

## Setup

Needs [uv](https://docs.astral.sh/uv/), Docker and [Ollama](https://ollama.com).

```
ollama pull gemma4:12b        # writes facts and questions locally (~8 GB)
ollama pull bge-m3            # embeddings for the test bed (~1.2 GB)
cp .env.example .env          # then fill in GEMMA_URL and GEMMA_API_KEY to use the remote Gemma 4
uv sync
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
uv run bench facts            # --max-facts 10, --doc <part of a file name>, --force to redo
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

Every agent call is shown live in the terminal: what it worked on, what it returned, its time and tokens. Each audit verdict is shown too, with the reason for every rejection. Every call is also saved to `data/trace.jsonl`.

Import `postman/eval-bench.postman_collection.json` into Postman. It holds requests for three things:
- the local Qdrant: browse chunks, filter by PDF or segment, and run a two-step semantic search;
- every stage's output, through `bench serve`;
- the local Ollama models.

The `facts` and `questions` stages save after every segment. They skip segments already done at the current prompt version, so a stopped run picks up where it left off, and a bumped version redoes the old rows. After changing a stage, re-run the stages that follow it (`--force` where needed). `evaluate` stops if a question's evidence no longer matches the indexed text.

**Switching to Gemma 4:** set `LLM_PROVIDER=gemma` in `.env`, or prefix a single command (`LLM_PROVIDER=gemma uv run bench questions`). The remote endpoint is only up in the daytime.

## Outputs (`data/`, not committed)

| File | What it holds |
|---|---|
| `pages.jsonl` | cleaned text of every PDF page |
| `toc/<slug>.json` | the parsed table of contents, named after the PDF in lower case with dashes (cached, so re-runs skip the LLM) |
| `segments.jsonl` | segments with their text, page map and TOC headings |
| `facts.jsonl` | per segment: verified facts (with position and pages) and rejected ones (with reasons) |
| `questions.jsonl` | per segment: verified questions, each with its evidence (quotes and positions), and rejected ones |
| `chunks.jsonl` | the test-bed chunks stored in Qdrant |
| `results.jsonl`, `report.md` | per-question results and the scored report |
| `trace.jsonl` | every agent call: role, prompt, output, tokens, seconds |
| `bench.xlsx` | all of the above as readable sheets (`bench export`) |

## Layout

```
bench/      cli.py · config.py · llm.py · pdf.py · segments.py · facts.py · questions.py · verify.py · testbed.py · evaluate.py
            checks.py (bench check) · export.py (bench export) · serve.py (bench serve)
tests/      offline tests (no LLM or Qdrant needed): uv run pytest
postman/    the Postman collection
question_types.toml   the question types: count, source, prompt and audit rules per type
```
