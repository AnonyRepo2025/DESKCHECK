"""Offline regression tests for bounded/checkpointed plain-repair simloop queries."""

import time
from types import ModuleType, SimpleNamespace

import pytest


from simagent import subagent as sub
from simagent import pipeline as sp
from simagent import plainrepair as plain


class _ConfiguredModel:
    def __init__(self, model_kwargs=None):
        self.config = SimpleNamespace(model_name="fake/model", model_kwargs=model_kwargs or {})
        self.fallback_calls = 0

    def query(self, _messages):
        self.fallback_calls += 1
        return {"content": "fallback"}


def _fake_litellm(completion):
    mod = ModuleType("litellm")
    mod.completion = completion
    mod.completion_cost = lambda **_kwargs: 0.0
    return mod


def test_direct_query_has_default_timeout_and_completion_cap(monkeypatch):
    seen = {}

    def completion(**kwargs):
        seen.update(kwargs)
        msg = SimpleNamespace(content="FUNCTION: f\nFILE: f.py", reasoning_content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)

    monkeypatch.setitem(__import__("sys").modules, "litellm", _fake_litellm(completion))
    model = _ConfiguredModel()
    query = sub._make_agent_query_fn(SimpleNamespace(model=model))
    assert query("locate") == "FUNCTION: f\nFILE: f.py"
    assert seen["timeout"] == sub.SUBAGENT_QUERY_TIMEOUT
    assert seen["max_tokens"] == sub.SUBAGENT_MAX_COMPLETION_TOKENS
    assert model.fallback_calls == 0


def test_timeout_does_not_retry_through_unbounded_model_fallback(monkeypatch):
    def completion(**_kwargs):
        raise TimeoutError("response body read timed out")

    monkeypatch.setitem(__import__("sys").modules, "litellm", _fake_litellm(completion))
    model = _ConfiguredModel()
    query = sub._make_agent_query_fn(SimpleNamespace(model=model))
    with pytest.raises(TimeoutError):
        query("simulate")
    assert model.fallback_calls == 0


def test_response_text_is_bounded(monkeypatch):
    monkeypatch.setattr(sub, "SUBAGENT_RESPONSE_CHAR_CAP", 200)
    text = "HEAD" + ("x" * 500) + "TAIL"
    bounded = sub._bounded_response(text)
    assert len(bounded) < len(text)
    assert bounded.startswith("HEAD") and bounded.endswith("TAIL")
    assert "TRUNCATED" in bounded


def test_budget_precheck_skips_query():
    old_cap, old_spent = sp.BUDGET.cap, sp.BUDGET.spent
    calls = []
    try:
        sp.BUDGET.reset(1.0)
        sp.BUDGET.add(1.0)
        assert plain._ask(lambda _p: calls.append(1), "prompt", "unit") == ""
        assert calls == []
    finally:
        sp.BUDGET.cap, sp.BUDGET.spent = old_cap, old_spent


def test_case_id_decoration_is_normalized():
    assert plain._normalize_case_id("<U5#1>") == "U5#1"
    parsed = plain._parse_cases(
        "=== CASE id=<U5#1> ===\nINPUT: x\nEXPECTED: y\n=== END CASE ===")
    assert parsed[0]["id"] == "U5#1"


def test_checkpoint_is_atomic_and_leaves_no_temp(tmp_path):
    path = tmp_path / "phase.json"
    plain._write_json_checkpoint(path, {"progress": {"units_completed": 4}})
    assert '"units_completed": 4' in path.read_text()
    assert not (tmp_path / "phase.json.tmp").exists()


def test_outer_wall_deadline_cannot_be_swallowed_as_exception(monkeypatch):
    monkeypatch.setattr(plain, "SIMLOOP_QUERY_WALL_TIMEOUT", 0.02)

    def provider_retry_loop(_prompt):
        # Simulate a provider that catches ordinary Exception around each attempt. The deadline's
        # BaseException must escape that loop and be converted to an empty/incomplete response.
        while True:
            try:
                time.sleep(1)
            except Exception:
                continue

    started = time.monotonic()
    assert plain._ask(provider_retry_loop, "prompt", "deadline") == ""
    assert time.monotonic() - started < 0.5
