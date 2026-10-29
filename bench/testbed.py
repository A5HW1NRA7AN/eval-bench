"""The local test bed: every segment chunked, embedded with Ollama and searched in Qdrant.

Chunks partition each segment's text exactly (no overlap, nothing dropped), so a fact's verified
position in the segment tells us which chunks hold it.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient, models

from bench.config import DATA_DIR, settings
from bench.pdf import read_jsonl, write_jsonl
from bench.segments import load_segments

CHUNKS = DATA_DIR / "chunks.jsonl"
COLLECTION = "bench_docs"
CHUNK_CHARS = 1200
BATCH = 32
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
    vectors = embed([r["text"] for r in rows])
    client = qdrant()
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
    """(chunk_id, score) of the k nearest chunks, best first."""
    hits = client.query_points(COLLECTION, query=vector, limit=k, with_payload=True).points
    return [(h.payload["chunk_id"], h.score) for h in hits]


def load_chunks() -> list[dict[str, Any]]:
    if not CHUNKS.exists():
        raise SystemExit("data/chunks.jsonl missing: run `bench index` first")
    return read_jsonl(CHUNKS)
