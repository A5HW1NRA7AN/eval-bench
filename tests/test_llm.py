"""bench/llm.py: unreadable answers asked again as free text, outages, identical calls asked once, and job order at
any concurrency. The trace and the response cache are tested with the workbook in test_checks.py."""

import dataclasses
import json
import threading
import time

import pytest
from pydantic import BaseModel
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from bench import llm


def test_a_garbled_or_unparseable_answer_is_asked_again_as_free_text(monkeypatch):
    class Out(BaseModel):
        claim: str

    modes = []

    def endpoint(messages, info):  # guided decoding garbles '°' (native mode); free text gets it right
        modes.append(info.model_request_parameters.output_mode)
        native = modes[-1] == "native"
        return ModelResponse(parts=[TextPart(json.dumps({"claim": "Cool milk to 5\x00C." if native else "Cool milk to 5°C."}))])

    monkeypatch.setattr(llm, "model", lambda: FunctionModel(endpoint))
    assert llm.call("tester", llm.agent(Out, "Write a claim."), "milk", "book-01").claim == "Cool milk to 5°C."
    assert modes == ["native", "prompted"] and json.loads(llm.TRACE.read_text())["mode"] == "prompted"
    llm.call("tester", llm.agent(Out, "Write a claim."), "milk", "book-01")
    assert len(modes) == 2  # the clean answer was cached


def test_an_outage_stops_new_calls_and_identical_calls_are_asked_once(monkeypatch):
    class Out(BaseModel):
        n: int

    asked = []

    def answer(messages, info):
        n = int(messages[-1].parts[-1].content)
        asked.append(n)
        if n == 0:
            raise ConnectionError("endpoint down")
        time.sleep(0.05)  # still running when the first call fails
        return ModelResponse(parts=[TextPart(json.dumps({"n": n}))])

    monkeypatch.setattr(llm, "model", lambda: FunctionModel(answer))
    monkeypatch.setattr(llm, "USE_CACHE", False)
    base = llm.settings()
    monkeypatch.setattr(llm, "settings", lambda: dataclasses.replace(base, concurrency=2))
    agent = llm.agent(Out, "Echo the number.")
    with pytest.raises(ConnectionError):
        llm.call_all([("echo", agent, str(n), f"job {n}") for n in range(6)])
    assert sorted(asked) == [0, 1]  # the first wave only: the other four never started
    asked.clear()
    monkeypatch.setattr(llm, "USE_CACHE", True)
    answers = llm.call_all([("echo", agent, "7", "job a"), ("echo", agent, "7", "job b")])
    assert [a.n for a in answers] == [7, 7] and asked == [7]  # one call for both
    assert [("cached" in json.loads(line)) for line in llm.TRACE.read_text().splitlines()[-2:]] == [False, True]


def test_answers_come_back_in_job_order_at_any_concurrency(monkeypatch):
    class Out(BaseModel):
        n: int

    running, peak, lock = [0], [0], threading.Lock()

    def answer(messages, info):  # later jobs finish first
        n = int(messages[-1].parts[-1].content)
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.04 * (5 - n))
        with lock:
            running[0] -= 1
        return ModelResponse(parts=[TextPart(json.dumps({"n": n}))])

    monkeypatch.setattr(llm, "model", lambda: FunctionModel(answer))
    monkeypatch.setattr(llm, "USE_CACHE", False)
    base, agent = llm.settings(), llm.agent(Out, "Echo the number.")
    for limit in (1, 4):
        monkeypatch.setattr(llm, "settings", lambda limit=limit: dataclasses.replace(base, concurrency=limit))
        peak[0] = 0
        assert [o.n for o in llm.call_all([("echo", agent, str(n), f"job {n}") for n in range(5)])] == [0, 1, 2, 3, 4]
        assert peak[0] == 1 if limit == 1 else 1 < peak[0] <= limit
