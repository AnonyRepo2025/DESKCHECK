"""Two defects found in the js_batch2 baseline-only RCA (2026-09-22). Run with the sweagent python:

    python -m test_audit_probe_and_trace   # one line per check, exits non-zero on failure

D1  the repair-review TRACE-quote parser read the RAW finding body with a bare `^TRACE:` regex, so a
    correctly-blocked finding whose label the model bolded (`**TRACE:**`) lost every quote and was
    dropped as ungrounded -- js_batch2 element-web-b007ea81, where the dropped finding named the
    defect exactly ("never computes any neighbor averages").
D2  the audit's execution probe was Python-only (`if not _sub.is_python(): return ""` plus a
    ```python-only fence and a python profile), so on JS/TS a VIOLATED requirement could never be
    execution-confirmed and never triggered a fix -- js_batch2 webclients-e9677f6c, where audit
    returned R6/R9 VIOLATED with a verbatim prediction of the grader error and stayed advisory.
"""
from __future__ import annotations

from simagent import pipeline as sp
from simagent import plainrepair as pr

_sub = sp._sub


class FakeEnv:
    def __init__(self, canned=None):
        self.cmds = []
        self.canned = canned or {}

    def execute(self, payload, timeout=None):
        cmd = payload["command"]
        self.cmds.append(cmd)
        for key, out in self.canned.items():
            if key in cmd:
                return {"output": out, "returncode": 0}
        return {"output": "", "returncode": 0}


def _finding(trace_label: str) -> str:
    return (
        "=== FINDING id=F1 ===\n"
        "UNIT: arraySmoothingResample\n"
        "KIND: BEHAVIOUR\n"
        "INPUT: [2,2,0,2,2,0,2,2,0], 4 points\n"
        "EXPECTED: [1, 1, 2, 1] -- each value the average of its two neighbours\n"
        "BASIS: requirement bullet 8\n"
        f"{trace_label}\n"
        "> while (smoothed.length > 2 * points) {\n"
        ">     newSmoothed.push(smoothed[i]);\n"
        "ACTUAL: returns [2, 0, 2, 2]; the code never computes a neighbour average\n"
        "=== END FINDING ===\n"
    )


# ---------------------------------------------------------------- D1
def check_trace_quotes_survive_a_bold_label():
    plain = pr._parse_findings(_finding("TRACE:"))
    bold = pr._parse_findings(_finding("**TRACE:**"))
    dashed = pr._parse_findings(_finding("- TRACE:"))
    assert len(plain) == len(bold) == len(dashed) == 1, (plain, bold, dashed)
    assert plain[0]["quotes"], plain
    for got in (bold, dashed):
        assert got[0]["quotes"] == plain[0]["quotes"], (got[0]["quotes"], plain[0]["quotes"])
        # the rest of the finding must survive the normalization too
        assert got[0]["expected"] == plain[0]["expected"] and got[0]["actual"] == plain[0]["actual"], got


def check_bold_label_finding_is_groundable():
    """The quotes are what grounding substring-checks; without them the finding is dropped."""
    f = pr._parse_findings(_finding("**TRACE:**"))[0]
    source = "function arraySmoothingResample(input, points) {\n  while (smoothed.length > 2 * points) {\n  }\n}"
    assert any(sp._ws_squash(q) in sp._ws_squash(source) for q in f["quotes"]), f["quotes"]


# ---------------------------------------------------------------- D2
def check_probe_fence_accepts_js_and_ts():
    assert sp._PROBE_FENCE_RE["python"].search("```python\nx=1\n```")
    for fence in ("js", "jsx", "javascript", "ts", "tsx", "typescript"):
        m = sp._PROBE_FENCE_RE["js"].search(f"prose\n```{fence}\nconsole.log('PROBE[R1]: x');\n```\n")
        assert m and "PROBE[R1]" in m.group(1), fence
    assert not sp._PROBE_FENCE_RE["js"].search("```python\nx=1\n```")


def check_probe_path_sits_next_to_a_patched_source_file():
    _sub.set_lang("js")          # the path is language-derived; it is only ever called on a JS run
    try:
        _check_probe_paths()
    finally:
        _sub.set_lang("python")


def _check_probe_paths():
    # webclients: tests sit BESIDE the source as `X.test.tsx`
    wc = ("--- a/packages/components/components/dialog/Dialog.tsx\n"
          "+++ b/packages/components/components/dialog/Dialog.tsx\n@@ -1 +1 @@\n+x\n")
    env = FakeEnv(canned={"git ls-files": "packages/components/components/dialog/Modal.test.tsx\n"
                                          "packages/other/deep/fixtures/x.test.ts\n"})
    assert sp._js_probe_path(env, "/app", wc)[1] == \
        "packages/components/components/dialog/_audit_probe.test.tsx", sp._js_probe_path(env, "/app", wc)
    # element-web: only `test/**/*-test.ts(x)` is collected -- the separator and the directory
    # both come from the sibling, and a .tsx source upgrades the extension so JSX parses
    ew = "--- a/src/utils/arrays.ts\n+++ b/src/utils/arrays.ts\n@@ -1 +1 @@\n+x\n"
    env = FakeEnv(canned={"git ls-files": "test/utils/arrays-test.ts\ntest/setup-test.ts\n"})
    assert sp._js_probe_path(env, "/app", ew)[1] == "test/utils/_audit_probe-test.ts", \
        sp._js_probe_path(env, "/app", ew)
    ew_jsx = "--- a/src/components/Room.tsx\n+++ b/src/components/Room.tsx\n@@ -1 +1 @@\n+x\n"
    assert sp._js_probe_path(env, "/app", ew_jsx)[1].endswith("_audit_probe-test.tsx"), \
        sp._js_probe_path(env, "/app", ew_jsx)
    # NodeBB: mocha, `test/*.js`
    nb = "--- a/src/posts/create.js\n+++ b/src/posts/create.js\n@@ -1 +1 @@\n+x\n"
    env = FakeEnv(canned={"git ls-files": "test/posts.test.js\n"})
    assert sp._js_probe_path(env, "/app", nb)[1] == "test/_audit_probe.test.js", \
        sp._js_probe_path(env, "/app", nb)
    # no test file anywhere -> fall back to the patched source's own directory
    bare = FakeEnv()
    assert sp._js_probe_path(bare, "/app", ew)[1] == "src/utils/_audit_probe.test.ts"
    assert sp._js_probe_path(bare, "/app/", wc)[1] == \
        "packages/components/components/dialog/_audit_probe.test.tsx"
    assert sp._js_probe_path(bare, "/app", "") is None
    # absolute path is repo_path + rel
    assert sp._js_probe_path(bare, "/app", ew)[0] == "/app/src/utils/_audit_probe.test.ts"


def check_js_probe_writes_runs_and_always_removes_the_file():
    _sub.set_lang("js")
    sp._JS_RUNNER.clear()
    sp._JS_RUNNER.update({"kind": "jest", "word": "test"})
    try:
        env = FakeEnv(canned={"npx jest": "console.log output\n  PROBE[R6]: \"<dialog>\"\nTests: 1 failed"})
        out = sp._js_probe_output(env, "/app", "it('p', () => { console.log('PROBE[R6]: x'); });",
                                  "--- a/src/a.ts\n+++ b/src/a.ts\n@@ -1 +1 @@\n+x\n")
        assert "PROBE[R6]" in out, out
        joined = "\n".join(env.cmds)
        assert "base64 -d > /app/src/_audit_probe.test.ts" in joined, env.cmds
        assert ".git/info/exclude" in joined, env.cmds          # never leaks into the patch
        assert env.cmds[-1].startswith("rm -f /app/src/_audit_probe.test.ts"), env.cmds[-1]
    finally:
        sp._JS_RUNNER.clear()
        _sub.set_lang("python")


def check_js_probe_removes_the_file_even_when_the_runner_raises():
    _sub.set_lang("js")
    sp._JS_RUNNER.clear()
    sp._JS_RUNNER.update({"kind": "jest", "word": "test"})

    class Boom(FakeEnv):
        def execute(self, payload, timeout=None):
            self.cmds.append(payload["command"])
            if "npx jest" in payload["command"]:
                raise RuntimeError("container died")
            return {"output": "", "returncode": 0}

    try:
        env = Boom()
        out = sp._js_probe_output(env, "/app", "x", "--- a/src/a.js\n+++ b/src/a.js\n@@ -1 +1 @@\n+x\n")
        assert "probe run failed" in out, out
        assert env.cmds[-1].startswith("rm -f "), env.cmds[-1]
    finally:
        sp._JS_RUNNER.clear()
        _sub.set_lang("python")


def check_whole_suite_runner_is_skipped():
    """`script` kind runs the entire suite (tutanota): too expensive for a throwaway probe."""
    _sub.set_lang("js")
    sp._JS_RUNNER.clear()
    sp._JS_RUNNER.update({"kind": "script", "entry": "npm test"})
    try:
        env = FakeEnv()
        assert sp._js_probe_output(env, "/app", "x", "--- a/src/a.js\n+++ b/src/a.js\n@@ -1 +1 @@\n+x\n") == ""
        assert env.cmds == [], env.cmds
    finally:
        sp._JS_RUNNER.clear()
        _sub.set_lang("python")


def check_reverdict_authors_a_js_probe_instead_of_bailing_out():
    """The whole point of D2: on JS the device used to return "" before authoring anything."""
    _sub.set_lang("js")
    sp._JS_RUNNER.clear()
    sp._JS_RUNNER.update({"kind": "jest", "word": "test"})
    prompts: list = []
    orig = _sub._make_agent_query_fn
    try:
        def fake_qf_factory(agent, on_usage=None):
            def qf(prompt):
                prompts.append(prompt)
                if len(prompts) == 1:
                    return "```ts\nit('p', () => { console.log('PROBE[R6]: rendered <dialog>'); });\n```"
                return "[R6] VERDICT: VIOLATED\nEVIDENCE: PROBE[R6] shows a native <dialog>"
            return qf

        _sub._make_agent_query_fn = fake_qf_factory
        env = FakeEnv(canned={"npx jest": "PROBE[R6]: rendered <dialog>\n"})
        out = sp._execution_grounded_reverdict(
            env, "/app", None, ["R6"],
            [("R6", "falls back to a compatible host element when HTMLDialogElement is limited")],
            "--- a/src/Dialog.tsx\n+++ b/src/Dialog.tsx\n@@ -1 +1 @@\n+x\n", "PATCHED SOURCE")
        assert len(prompts) == 2, prompts          # authored a probe AND re-asked
        assert "```ts" in prompts[0] and "console.log" in prompts[0], prompts[0][:400]
        assert "_audit_probe.test.ts" in prompts[0], prompts[0][:400]
        assert "VIOLATED" in out, out
        assert "rendered <dialog>" in prompts[1], prompts[1][:400]
    finally:
        _sub._make_agent_query_fn = orig
        sp._JS_RUNNER.clear()
        _sub.set_lang("python")


def check_go_still_returns_empty():
    """Go has no probe runner yet; it must stay advisory rather than half-run a JS probe."""
    _sub.set_lang("go")
    try:
        called: list = []
        orig = _sub._make_agent_query_fn
        _sub._make_agent_query_fn = lambda agent, on_usage=None: (lambda p: called.append(p) or "")
        try:
            out = sp._execution_grounded_reverdict(FakeEnv(), "/app", None, ["R1"], [("R1", "x")],
                                                   "--- a/a.go\n+++ b/a.go\n@@ -1 +1 @@\n+x\n", "pack")
        finally:
            _sub._make_agent_query_fn = orig
        assert out == "" and called == [], (out, called)
    finally:
        _sub.set_lang("python")


def main() -> int:
    checks = [v for k, v in sorted(globals().items()) if k.startswith("check_")]
    bad = 0
    for fn in checks:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{len(checks) - bad}/{len(checks)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
