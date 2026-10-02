"""Repository-independent regression coverage for evidence plumbing, not benchmark answers."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from simagent import evidence as evidence
from simagent import subagent as sub
from simagent import pipeline as sp
from simagent import plainrepair as plain


@pytest.mark.parametrize("text,rc", [
    ("ERROR: unrecognized arguments: --old\n", 4),
    ("no tests ran in 0.02s", 5), ("Killed", 137),
    ("3 passed in 0.02s\nKilled", 137), ("", 0),
    ("2 skipped in 0.03s", 0), ("3 passed in 0.02s", None),
    ("3 failed in 0.02s", 1),
])
def test_invalid_tests_never_pass(text, rc):
    assert evidence.pytest_evidence(text, rc).status == "invalid"


def test_affirmative_execution_and_failed_nodes():
    assert evidence.pytest_evidence("=== 3 passed in 0.01s ===", 0).status == "passed"
    r = evidence.pytest_evidence("FAILED tests/test_api.py::test_shape - AssertionError\n1 failed in 0.1s", 1)
    assert r.status == "failed" and r.failures == {"tests/test_api.py::test_shape"}


def test_runner_profile_reused_on_base_and_patch(monkeypatch):
    from simagent import validation as pv
    monkeypatch.setattr(pv, "python_profile", lambda *_: {"prefix": "python", "version": "3.9"})
    monkeypatch.setattr(sp, "_py_test_runner", lambda *_: "")
    monkeypatch.setattr(sub, "is_go", lambda: False)
    monkeypatch.setattr(sub, "is_js", lambda: False)
    class Env:
        commands = []
        def execute(self, payload, timeout=None):
            self.commands.append(payload["command"])
            if len(self.commands) == 1:
                return {"returncode": 4, "output": "ERROR: unrecognized arguments: --old\ninifile: /app/setup.cfg"}
            return {"returncode": 0, "output": "1 passed in 0.01s"}
    env = Env()
    assert sp._run_test_files(env, "/app", ["tests/test_api.py"])[0] == set()
    assert sp._run_test_files(env, "/app", ["tests/test_api.py"])[0] == set()
    assert env.commands[1] == env.commands[2]
    assert "-o addopts=''" in env.commands[2]


def test_backtick_source_grounding_does_not_accept_invented_source():
    finding = {"quotes": ["`return value.strip()`"], "basis": "Whitespace must be removed from both ends."}
    assert plain._ground_finding(finding, "    return value.strip()", finding["basis"])[0]
    assert not plain._ground_finding(finding, "return value.upper()", finding["basis"])[0]


def test_incomplete_review_retries_and_stays_incomplete():
    calls = []
    def query(prompt):
        calls.append(prompt)
        return sub.QueryReply("NO FINDINGS", "length")
    rows, steps, _, log = plain._run_reasoning_review(query, None, ctx="", ps="spec", instance={}, units=[], pack="", callers="", tests="")
    assert len(calls) == 4 and rows == []
    assert log["complete"] is False and log["incomplete"] == ["trace", "contract"]


def test_incomplete_review_recovery():
    replies = iter([sub.QueryReply("", "length"), sub.QueryReply("NO FINDINGS", "stop")])
    steps = []
    reply, complete = plain._ask_review(lambda _: next(replies), None, "prompt", "review:trace", steps)
    assert complete and len(steps) == 2 and reply == "NO FINDINGS"


def test_direct_calls_accounted_once_including_retry(monkeypatch):
    from test_simloop_reliability import _ConfiguredModel, _fake_litellm
    replies = iter([("", "length"), ("NO FINDINGS", "stop")])
    kwargs_seen = []
    def completion(**kwargs):
        kwargs_seen.append(kwargs)
        text, finish = next(replies)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, reasoning_content="not an answer"), finish_reason=finish)], usage=None)
    mod = _fake_litellm(completion)
    mod.completion_cost = lambda **_: 0.25
    monkeypatch.setitem(sys.modules, "litellm", mod)
    budget = sp._InstanceBudget(); budget.reset(3)
    monkeypatch.setattr(sp, "BUDGET", budget)
    model = sp._BudgetedModel(_ConfiguredModel())
    meter = plain._UsageMeter()
    qf = sub._make_agent_query_fn(SimpleNamespace(model=model), on_usage=meter.add)
    reply, complete = plain._ask_review(qf, meter, "prompt", "review:trace", [])
    assert complete and reply == "NO FINDINGS"
    assert budget.spent == 0.5 and meter.totals()["cost"] == 0.5
    assert len(budget.events) == 2
    assert kwargs_seen[1]["max_tokens"] > kwargs_seen[0]["max_tokens"]


def test_no_fallback_double_charge(monkeypatch):
    from test_simloop_reliability import _ConfiguredModel, _fake_litellm
    def completion(**kwargs):
        raise ValueError("unsupported parameter")
    monkeypatch.setitem(sys.modules, "litellm", _fake_litellm(completion))
    inner = _ConfiguredModel()
    inner.query = lambda _: {"content": "NO FINDINGS", "extra": {"cost": 0.4}}
    budget = sp._InstanceBudget(); budget.reset(3)
    monkeypatch.setattr(sp, "BUDGET", budget)
    meter = plain._UsageMeter()
    qf = sub._make_agent_query_fn(SimpleNamespace(model=sp._BudgetedModel(inner)), on_usage=meter.add)
    assert qf("prompt") == "NO FINDINGS"
    assert budget.spent == 0.4 and meter.totals()["cost"] == 0.4


def test_test_excerpt_retains_referenced_table(tmp_path):
    path = tmp_path / "test_parser.py"
    path.write_text("CASES = [('alpha', 'ALPHA')]\n\ndef test_normalize():\n    for raw, expected in CASES:\n        assert normalize(raw) == expected\n")
    out = subprocess.check_output([sys.executable, "-c", plain._TEST_EXCERPT_PY, str(path), json.dumps({"normalize": 1}), "4200"], text=True)
    assert "CASES = [('alpha', 'ALPHA')]" in out
    assert "def test_normalize" in out


def test_scope_keeps_explicit_prohibitions_and_allows_application_resources():
    base = "Do NOT modify tests, configuration, or packaging files."
    text = evidence.clarify_scope(base)
    assert base in text and "Honor any explicit" in text
    assert "application-owned" in text.lower()
    template = plain._plain_instance_template({"instance_template": base})
    assert evidence.SCOPE_NOTE in template


def test_verifier_receives_full_authoritative_contract():
    assert "{problem_statement}" in plain._REVIEW_VERIFY_PROMPT
    assert "{requirements}" in plain._REVIEW_VERIFY_PROMPT
    assert "{interface}" in plain._REVIEW_VERIFY_PROMPT


def test_removed_symbol_cannot_exempt_pass_to_pass_test():
    test_id = "tests/test_api.py::test_old_parser"
    part = sp._partition_regressions({"fail_to_pass": ["tests/test_api.py::test_new_parser"],
                                      "pass_to_pass": [test_id]}, [test_id],
                                     removed_symbols=["old_parser"])
    assert part["targets"] == [test_id]
    assert part["classes"][test_id] == "pass_to_pass"
