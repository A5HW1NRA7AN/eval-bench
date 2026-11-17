import hashlib
import importlib
import json
import pkgutil
import re
from pathlib import Path

import pydantic_ai.models
import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from qdrant_client import QdrantClient

import bench
from bench import config, llm

# Every bench module, imported here before any test patches config: a module first imported mid-test would
# build its paths from the patched DATA_DIR and keep them after the test, and a module left out would write
# to the real data/ folder.
MODULES = [importlib.import_module(f"bench.{m.name}") for m in pkgutil.iter_modules(bench.__path__) if m.name != "__main__"]
SETTINGS = config.Settings(llm_url="http://llm.invalid/v1", llm_key="test", llm_model="test-model", reasoning="none",
                           ollama_url="http://ollama.invalid/v1", embed_model="fake-embed", qdrant_url="http://127.0.0.1:6333")


def fake_embed(texts: list[str]) -> list[list[float]]:
    """A deterministic bag-of-words vector, so texts that share words are close (no Ollama needed)."""
    vectors = []
    for text in texts:
        v = [0.0] * 64
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
        vectors.append(v if any(v) else [1.0] + [0.0] * 63)
    return vectors


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Every test gets its own data/ and pdfs/ folders, fixed settings (no .env), an in-memory Qdrant, fake
    embeddings, and no real model calls (FunctionModel still works)."""
    data, pdfs = tmp_path / "data", tmp_path / "pdfs"
    data.mkdir()
    pdfs.mkdir()
    real_data, real_pdfs = config.DATA_DIR, config.PDF_DIR  # before the loop patches config itself
    client = QdrantClient(":memory:")
    monkeypatch.setattr(pydantic_ai.models, "ALLOW_MODEL_REQUESTS", False)
    for module in MODULES:
        for attr, value in list(vars(module).items()):
            if isinstance(value, Path) and (value == real_data or real_data in value.parents):
                monkeypatch.setattr(module, attr, data / value.relative_to(real_data))
            elif isinstance(value, Path) and value == real_pdfs:
                monkeypatch.setattr(module, attr, pdfs)
        if hasattr(module, "settings"):
            monkeypatch.setattr(module, "settings", lambda: SETTINGS)
        if hasattr(module, "qdrant"):
            monkeypatch.setattr(module, "qdrant", lambda: client)
        if hasattr(module, "embed"):
            monkeypatch.setattr(module, "embed", fake_embed)
    return tmp_path


@pytest.fixture
def agents(monkeypatch):
    """agents(reply): every agent answers reply(instructions, prompt) as JSON, never from the cache. The prompt is the
    conversation's first, so an answer asked again after an unreadable one gets the same reply."""
    def use(reply):
        monkeypatch.setattr(llm, "USE_CACHE", False)
        monkeypatch.setattr(llm, "model", lambda: FunctionModel(lambda messages, info: ModelResponse(
            parts=[TextPart(json.dumps(reply(messages[-1].instructions, messages[0].parts[-1].content)))])))
    return use
