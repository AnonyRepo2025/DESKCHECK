"""JS/TS pipeline defects J1, J2, J5 (found in the Luna JS/TS batches, fixed 2026-09-16)."""

import pytest


from simagent import subagent as sub
from simagent import pipeline as sp


@pytest.fixture(autouse=True)
def _restore_lang():
    before = sub.LANG
    yield
    sub.set_lang(before)


# ---------------------------------------------------------------- J1 ----
@pytest.mark.parametrize("tid, expected", [
    ("test/user.js | User invites after invites checks should verify installation with no errors", True),   # mocha
    ("test/components/views/settings/Notifications-test.tsx | <Notifications /> | renders", True),            # jest
    ("test/tests/contacts/VCardImporterTest.ts | test with no charset but encoding", True),                   # ospec
    ("test/utils/EventUtils-test.ts", False),          # suite failed to load: the file id
    ("npm test | test execution", False),              # script runner: the whole run failed
    ("", False),
])
def test_j1_js_test_ids(tid, expected):
    sub.set_lang("js")
    assert sp._is_test_level_id(tid) is expected


def test_j1_python_and_go_unchanged():
    sub.set_lang("python")
    assert sp._is_test_level_id("tests/test_x.py::test_y") and not sp._is_test_level_id("tests/test_x.py")
    assert not sp._is_test_level_id("test/user.js | User x")      # the old `::` rule, exactly as before
    sub.set_lang("go")
    assert sp._is_test_level_id("TestParse/sub") and sp._is_test_level_id("BUILD:pkg/x_test.go")
    assert not sp._is_test_level_id("github.com/x/pkg")


# ---------------------------------------------------------------- J2 ----
def test_j2_retry_budget_for_every_non_python_language():
    big = (2 * sub.SUBAGENT_MAX_COMPLETION_TOKENS) or None
    for lang, want in (("python", None), ("go", big), ("js", big), ("ts", big)):
        sub.set_lang(lang)
        assert sp._audit_retry_max_tokens() == want, lang


class _Reply(str):
    def __new__(cls, text, finish_reason=None, truncated=False):
        obj = str.__new__(cls, text)
        obj.finish_reason, obj.truncated = finish_reason, truncated
        return obj


def test_j2_length_cut_is_retried_with_the_budget_and_recorded_as_truncated():
    seen = []

    def query(prompt, max_tokens=None):
        seen.append(max_tokens)
        if max_tokens is None:
            return _Reply("", finish_reason="length")          # reasoning ate the budget: empty reply
        return _Reply("[R1] VERDICT: SATISFIED\nEVIDENCE: `f()` returns 1\n", finish_reason="stop")

    report, calls = sp._complete_audit(query, [("R1", "f returns 1")], lambda pts: "audit " + pts[0][0],
                                       retry_max_tokens=16384)
    assert seen == [None, 16384]
    assert "[R1] VERDICT: SATISFIED" in report
    assert calls[0]["finish_reason"] == "length" and calls[0]["truncated"] is True    # was recorded False
    assert calls[1]["truncated"] is False


# ---------------------------------------------------------------- J5 ----
class _FindEnv:
    def __init__(self, output):
        self.output, self.commands = output, []

    def execute(self, action, timeout=None):
        self.commands.append(action["command"])
        return {"output": self.output, "returncode": 0}


def test_j5_e2e_specs_are_not_covering_tests():
    sub.set_lang("js")
    sp._JS_RUNNER.clear(); sp._JS_RUNNER.update({"kind": "jest", "word": "npx jest"})
    patch = ("diff --git a/src/components/views/messages/TextualBody.tsx b/src/components/views/messages/TextualBody.tsx\n"
             "--- a/src/components/views/messages/TextualBody.tsx\n+++ b/src/components/views/messages/TextualBody.tsx\n")
    found = ("./playwright/e2e/messages/messages.spec.ts\n./playwright/snapshots/messages/messages.spec.ts\n"
             "./cypress/e2e/messages.spec.ts\n./test/unit-tests/components/views/messages/TextualBody-test.tsx\n"
             "__MIRROR__\n")
    env = _FindEnv(found)
    tests = sp._js_find_covering_tests(env, "/repo", patch)
    assert tests == ["test/unit-tests/components/views/messages/TextualBody-test.tsx"], tests
    for d in ("playwright", "e2e", "cypress"):
        assert env.commands[0].count(f"-not -path '*/{d}/*'") == 2        # both find invocations exclude it
    assert not sp._is_js_e2e_path("test/unit-tests/hooks/useUserDirectory-test.tsx")
    assert sp._is_js_e2e_path("playwright/e2e/messages/messages.spec.ts")
