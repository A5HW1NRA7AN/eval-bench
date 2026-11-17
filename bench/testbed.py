"""The local test bed: every segment chunked, embedded with Ollama and searched in Qdrant.

Chunks partition each segment's text exactly (no overlap, nothing dropped), so a fact's verified
position in the segment tells us which chunks hold it. Two baselines search the same chunks: keyword
search (BM25, in pure Python) and a hybrid that fuses it with the embedding search (reciprocal rank fusion).
"""

from __future__ import annotations

import math
import os
import re
import uuid
from collections import Counter, defaultdict
from typing import Any
from urllib.parse import urlparse

from openai import OpenAI
from qdrant_client import QdrantClient, models

from bench.config import DATA_DIR, WORKSPACE, BenchError, settings
from bench.pdf import read_jsonl, write_jsonl
from bench.segments import load_segments

CHUNKS = DATA_DIR / "chunks.jsonl"
COLLECTION = f"bench_docs_{WORKSPACE}" if WORKSPACE else "bench_docs"  # a Qdrant collection per workspace
CHUNK_CHARS = 1200
BATCH = 32
TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")  # words and numbers ("247.87", "1,500")
BM25_K1, BM25_B = 1.2, 0.75  # the usual BM25 settings
RRF_K = 60  # reciprocal rank fusion: score = sum of 1 / (RRF_K + rank) over the rankings (Cormack et al., 2009)
# Prefer a paragraph break, then a sentence end, then any space, in the second half of the window.
BOUNDARIES = (re.compile(r"\n\s*\n"), re.compile(r"[.!?]\s+"), re.compile(r"\s+"))


def chunk(text: str, size: int = CHUNK_CHARS) -> list[tuple[int, int]]:
    """Lossless (start, end) chunks of at most `size` characters."""
    spans, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            for boundary in BOUNDARIES:
                cuts = [m.end() for m in boundary.finditer(window) if m.end() >= size // 2]
                if cuts:
                    end = start + cuts[-1]
                    break
        spans.append((start, end))
        start = end
    return spans


def embed(texts: list[str]) -> list[list[float]]:
    """Vectors from Ollama's OpenAI-compatible embeddings endpoint, in input order."""
    s = settings()
    client = OpenAI(base_url=s.ollama_url, api_key="ollama")
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        vectors += [d.embedding for d in client.embeddings.create(model=s.embed_model, input=texts[i : i + BATCH]).data]
    return vectors


def qdrant() -> QdrantClient:
    return QdrantClient(url=settings().qdrant_url)


def run_index() -> None:
    """Chunk every segment (question segments and the rest), embed, and rebuild the collection."""
    rows: list[dict[str, Any]] = []
    for seg in load_segments():
        for n, (start, end) in enumerate(chunk(seg["text"]), start=1):
            pages = [s["page"] for s in seg["page_spans"] if s["start"] < end and s["end"] > start]
            rows.append({"chunk_id": f"{seg['segment_id']}:c{n:03d}", "segment_id": seg["segment_id"],
                         "file_name": seg["file_name"], "start": start, "end": end,
                         "pages": [pages[0], pages[-1]] if pages else [], "text": seg["text"][start:end]})
    url = settings().qdrant_url
    host = urlparse(url if "://" in url else f"http://{url}").hostname
    if host not in ("localhost", "127.0.0.1", "::1") and os.environ.get("BENCH_ALLOW_REMOTE_QDRANT") != "1":
        raise BenchError(f"`bench index` deletes and rebuilds '{COLLECTION}', so it only runs on a local Qdrant; "
                         f"QDRANT_URL points at {host or url} (set BENCH_ALLOW_REMOTE_QDRANT=1 to override)")
    vectors = embed([r["text"] for r in rows])
    client = qdrant()
    print(f"rebuilding '{COLLECTION}' on {settings().qdrant_url}")
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE))
    points = [models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, r["chunk_id"])), vector=v,
                                 payload={k: r[k] for k in ("chunk_id", "segment_id", "file_name", "pages", "text")})
              for r, v in zip(rows, vectors)]
    for i in range(0, len(points), 256):
        client.upsert(COLLECTION, points=points[i : i + 256], wait=True)
    write_jsonl(CHUNKS, rows)
    print(f"{COLLECTION}: {len(rows)} chunks from {len({r['segment_id'] for r in rows})} segments ({settings().embed_model})")


def search(client: QdrantClient, vector: list[float], k: int) -> list[tuple[str, float]]:
    """(chunk_id, score) of the k nearest chunks, best first. Exact search (no approximate index) and ties
    broken by chunk_id, so the same question always gets the same ranking."""
    hits = client.query_points(COLLECTION, query=vector, limit=k, with_payload=True,
                               search_params=models.SearchParams(exact=True)).points
    return sorted(((h.payload["chunk_id"], h.score) for h in hits), key=lambda hit: (-hit[1], hit[0]))


class Keyword:
    """BM25 keyword search over the test-bed chunks: the baseline the embedding search is compared with."""

    def __init__(self, chunks: list[dict[str, Any]]):
        self.ids = [c["chunk_id"] for c in chunks]
        self.docs = [Counter(TOKEN.findall(c["text"].lower())) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.mean = sum(self.lengths) / max(len(self.lengths), 1)
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, doc in enumerate(self.docs):
            for term in doc:
                self.postings[term].append(i)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - len(ids) + 0.5) / (len(ids) + 0.5)) for t, ids in self.postings.items()}

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """(chunk_id, score) of the k best chunks, best first, ties broken by chunk_id."""
        scores: dict[int, float] = defaultdict(float)
        for term in set(TOKEN.findall(query.lower())):
            for i in self.postings.get(term, []):
                tf = self.docs[i][term]
                scores[i] += self.idf[term] * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * self.lengths[i] / self.mean))
        return sorted(((self.ids[i], s) for i, s in scores.items()), key=lambda hit: (-hit[1], hit[0]))[:k]


def fuse(*rankings: list[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    """Reciprocal rank fusion of several rankings: the hybrid baseline. Only ranks count, not scores."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, start=1):
            scores[cid] += 1 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda hit: (-hit[1], hit[0]))[:k]


def load_chunks() -> list[dict[str, Any]]:
    if not CHUNKS.exists():
        raise BenchError("data/chunks.jsonl missing: run `bench index` first")
    return read_jsonl(CHUNKS)
