# eval-bench

A benchmark for **search quality**: how well does our search find the passage that answers a question?

LLM agents read our PDFs, pull out facts and write questions about them. A regex audit (no AI) checks that every fact and answer really is in the PDF. The verified questions are then run against a search system, and we score whether the right passage came back. Today that system is a **local Qdrant test bed** built from the same PDFs.

**New here? Read [GUIDE.md](GUIDE.md).** It is a hands-on walkthrough of every stage, how to check and inspect each one, and why the project is built this way.

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
2. **facts**: the fact agent reads each segment in parts of about 12,000 characters (a few pages), cut at paragraph breaks.
   - Each part carries its heading path, the end of the previous part and the facts found so far. The agent keeps the segment's context and doesn't repeat itself.
   - The segment's fact limit is shared evenly across its parts. Given a whole chapter at once, the model took all its facts from the first fifth.
   - Every fact is a self-contained claim plus a verbatim quote.
3. **questions**: the question agent writes up to 8 questions per segment from the verified facts:
   - `direct`: a natural question about one fact;
   - `paraphrased`: the same idea in different words;
   - `keyword`: a 2–5 word search query;
   - `precise`: an exact number, dose, age, duration or limit, with its qualifiers;
   - `multi_fact`: needs two facts.
4. **index**: every segment is cut into chunks without overlap or gaps, embedded with `bge-m3` and stored in Qdrant.
5. **evaluate**: each question is embedded and searched. Gold chunks are the ones that overlap its evidence quote, so scoring needs no judging. The report shows hit@1/3/5/10 and MRR by question type, plus every miss with what came back instead.

**The regex audit:**
- **Facts:** the quote must appear in the segment text, ignoring only whitespace, quote style and dash style. It must be at most 300 characters and not a bare table row (mostly numbers). Duplicates are dropped.
- **Questions:** each must refer to verified facts. Its answer is a single span, or one span per fact joined by `; `, and each span must appear in their quotes. Type rules apply:
  - keyword queries are 2–5 words;
  - precise answers contain a number, in digits or as a number word (e.g. `four`, `twice`, `lakh`);
  - multi-fact questions use at least two facts, with answer spans from at least two of them;
  - at most half of a paraphrase's content words may come from its quote;
  - a question whose quote states a reporting year (e.g. 2024-25) must name a year;
  - no question may copy its quote.
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
uv run bench questions        # --max-questions 8, --doc, --force
uv run bench index
uv run bench evaluate
```

## Check and inspect

```
uv run bench check            # checks every stage's saved output without the LLM (or: bench check facts)
uv run bench serve            # a read-only JSON API over every stage's output on http://127.0.0.1:8000
```

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

## Layout

```
bench/      cli.py · config.py · llm.py · pdf.py · segments.py · agents.py · verify.py · testbed.py · evaluate.py
            checks.py (bench check) · serve.py (bench serve)
tests/      offline tests (no LLM or Qdrant needed): uv run pytest
postman/    the Postman collection
GUIDE.md    walkthrough and decision log
```
