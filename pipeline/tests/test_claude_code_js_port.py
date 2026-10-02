"""ccpipe JS/TS port checks (2026-09-20). Run with the sweagent python:

    python -m test_ccpipe_js_port      # prints one line per check, exits non-zero on failure

pytest is not installed in that environment, so the checks are plain asserts in __main__.
Each check pins one defect found auditing ccpipe against JavaScript/TypeScript instances.
"""
from __future__ import annotations

from simagent import pipeline as sp

from simagent.claude_code import validate_cc
from simagent.claude_code.phases import audit, localize, repair, repro_localize

_sub = sp._sub


class FakeEnv:
    """Records commands; returns a canned output per substring match."""

    def __init__(self, canned=None):
        self.cmds = []
        self.canned = canned or {}

    def execute(self, payload, timeout=None):
        cmd = payload["command"]
        self.cmds.append(cmd)
        for key, out in self.canned.items():
            if key in cmd:
                return {"output": out}
        return {"output": ""}


class FakeCtx:
    def __init__(self, env, repo_path="/app"):
        self.env = env
        self.repo_path = repo_path


JS_SRC = '''import x from "y";

export function parseList(input) {
  if (!Array.isArray(input)) {
    return [];
  }
  return input.map((s) => `${s}`);
}

const buildUrl = (base, path) => {
  const u = new URL(path, base);   // a brace in a comment }
  return u.toString();
};

export class Repo {
  constructor(db) {
    this.db = db;
  }

  async getStarred(userId, opts = {}) {
    const rows = await this.db.query("select 1 }");
    return rows.filter(Boolean);
  }
}
'''


def check_units_function():
    u = repair._js_units_from_source("a.ts", JS_SRC, {5})
    assert [r["symbol"] for r in u] == ["parseList"], u
    assert "return [];" in u[0]["code"]


def check_units_arrow_and_strings():
    u = repair._js_units_from_source("a.ts", JS_SRC, {12})
    assert [r["symbol"] for r in u] == ["buildUrl"], u
    # the brace inside the comment and the one inside the string must not end the block
    assert u[0]["code"].rstrip().endswith("};"), u[0]["code"]


def check_units_method_innermost():
    u = repair._js_units_from_source("a.ts", JS_SRC, {23})
    assert [r["symbol"] for r in u] == ["getStarred"], u


def check_units_default_param_brace():
    # the `opts = {}` default in the signature must not close the method block early (line 22 is
    # the last statement of getStarred; line 24 is the class's own closing brace)
    u = repair._js_units_from_source("a.ts", JS_SRC, {22})
    assert [r["symbol"] for r in u] == ["getStarred"], u
    assert "filter(Boolean)" in u[0]["code"]


def check_exists_uses_js_grammar():
    _sub.set_lang("js")
    env = FakeEnv({"grep": "12: export function parseList(input) {"})
    assert localize._exists(FakeCtx(env), "src/a.ts", "parseList") is True
    cmd = env.cmds[-1]
    assert "def " not in cmd and "function" in cmd, cmd
    # the Python branch's pattern would have matched nothing in a .ts file
    _sub.set_lang("python")
    env2 = FakeEnv()
    localize._exists(FakeCtx(env2), "src/a.py", "parse_list")
    assert "def" in env2.cmds[-1]


def check_exists_skips_js_test_paths():
    _sub.set_lang("js")
    assert localize._exists(FakeCtx(FakeEnv()), "src/a.test.ts", "parseList") is False
    _sub.set_lang("python")


def check_module_level_js_no_ast():
    _sub.set_lang("js")
    env = FakeEnv({"grep": "FOUND src/a.ts"})
    out = audit._module_level_in(FakeCtx(env), "Repo.getStarred", ["src/a.ts"])
    assert out.startswith("FOUND"), out
    assert "import ast" not in env.cmds[-1], env.cmds[-1]
    _sub.set_lang("python")


def check_repro_rule_js():
    _sub.set_lang("js")
    rule = repro_localize._repro_files_rule()
    assert ".test.js" in rule and "Never edit an existing test" in rule, rule
    _sub.set_lang("go")
    assert "_test.go" in repro_localize._repro_files_rule()
    _sub.set_lang("python")
    assert "test_repro_" in repro_localize._repro_files_rule()


def check_test_guard_js_patterns():
    assert "*.test.ts" in validate_cc._JS_TEST_GUARD_HOOK
    assert "_test.go)" not in validate_cc._JS_TEST_GUARD_HOOK
    assert "*_test.go)" in validate_cc._TEST_GUARD_HOOK


def check_lang_prompt_adaptation():
    _sub.set_lang("js")
    body = sp._maybe_lang(sp.EXPLORE_BODY)
    assert "pytest" not in body.lower() or "jest" in body.lower(), body[:400]
    _sub.set_lang("python")
    assert sp._maybe_lang(sp.EXPLORE_BODY) == sp.EXPLORE_BODY


# --- Fix 1: frozen first-failure spec tests -------------------------------------------------------------
import json as _json
import os as _os
import tempfile as _tempfile


def _stream(events):
    fd, path = _tempfile.mkstemp(suffix=".jsonl")
    with _os.fdopen(fd, "w") as fh:
        for kind, payload in events:
            if kind == "use":
                ev = {"type": "assistant", "message": {"content": [{"type": "tool_use", **payload}]}}
            else:
                ev = {"type": "user", "message": {"content": [{"type": "tool_result", "content": payload}]}}
            fh.write(_json.dumps(ev) + "\n")
    return path


class DiskEnv(FakeEnv):
    """FakeEnv whose `cat <file>` returns a given on-disk content and `git show` a HEAD content."""

    def __init__(self, disk=None, head=None):
        super().__init__()
        self.disk, self.head = disk or {}, head or {}

    def execute(self, payload, timeout=None):
        cmd = payload["command"]
        self.cmds.append(cmd)
        for rel, c in self.head.items():
            if "git show HEAD:" in cmd and rel in cmd:
                return {"output": c, "returncode": 0}
        if "git show HEAD:" in cmd:
            return {"output": "", "returncode": 1}
        if "__CCPIPE_NOFILE__" in cmd:
            for rel, c in self.disk.items():
                if f"-f {rel} ]" in cmd:
                    return {"output": c}
            return {"output": "__CCPIPE_NOFILE__\n"}
        return {"output": ""}


def check_frozen_first_failure_survives_test_bending():
    # ce554276 shape: test written, run fails, test edited until green -> the FAILING version is frozen
    v1 = "it('start once', () => { expect(start).toHaveBeenCalledTimes(1) })\n"
    v2 = "it('renamed', () => { expect(true).toBe(true) })\n"
    s = _stream([
        ("use", {"name": "Write", "input": {"file_path": "/app/test/pip-test.tsx", "content": v1}}),
        ("res", "Tests:       1 failed, 13 passed, 14 total"),
        ("use", {"name": "Edit", "input": {"file_path": "/app/test/pip-test.tsx", "old_string": v1, "new_string": v2}}),
        ("res", "Tests:       14 passed, 14 total"),
    ])
    fz = validate_cc.frozen_failing_tests(DiskEnv(disk={"test/pip-test.tsx": v2}), [s], "/app")
    assert fz["files"] == {"test/pip-test.tsx": v1}, fz
    assert "1 failed" in fz["output"]


def check_frozen_ignores_setup_failures():
    s = _stream([
        ("use", {"name": "Write", "input": {"file_path": "/app/src/a.test.ts", "content": "x"}}),
        ("res", "FAIL src/a.test.ts\n  Test suite failed to run\n  Cannot find module './b'\nTests:       1 failed"),
        ("res", "Tests:       3 passed, 3 total"),
    ])
    fz = validate_cc.frozen_failing_tests(DiskEnv(disk={"src/a.test.ts": "x"}), [s], "/app")
    assert fz["files"] == {}, fz


def check_frozen_edit_of_existing_test_uses_head():
    head = "describe('x', () => {\n  it('old', () => {})\n})\n"
    s = _stream([
        ("use", {"name": "Edit", "input": {"file_path": "/app/test/x.test.js", "old_string": "it('old', () => {})",
                                           "new_string": "it('new', () => { assert(0) })"}}),
        ("res", "  3 passing\n  1 failing"),
    ])
    frozen = head.replace("it('old', () => {})", "it('new', () => { assert(0) })")
    fz = validate_cc.frozen_failing_tests(DiskEnv(disk={"test/x.test.js": frozen}, head={"test/x.test.js": head}), [s], "/app")
    assert fz["files"] == {"test/x.test.js": frozen}, fz


def check_frozen_drops_bash_edited_file():
    # reconstructed final != disk (the file was later rewritten through Bash) -> unreliable, not frozen
    s = _stream([
        ("use", {"name": "Write", "input": {"file_path": "/app/a_test.py", "content": "assert 0\n"}}),
        ("res", "1 failed, 2 passed in 0.1s"),
    ])
    fz = validate_cc.frozen_failing_tests(DiskEnv(disk={"a_test.py": "assert 1  # sed -i\n"}), [s], "/app")
    assert fz["files"] == {} and fz["unreliable"] == ["a_test.py"], fz


def check_spec_fix_guard_protects_listed_paths():
    assert "protected.txt" in validate_cc._PATH_GUARD_HOOK and "exit 2" in validate_cc._PATH_GUARD_HOOK


def check_harness_fidelity_in_pipeline_reminder_only():
    from simagent.claude_code.phases import validate as _v
    assert "HARNESS FIDELITY" in validate_cc.CC_VALIDATE_REMINDER
    assert "HARNESS FIDELITY" not in _v.V2_BASELINE_REMINDER


def check_frozen_keeps_deleted_scratch_test():
    # ce554276/4c6b0d35 shape: the session deletes its failing scratch test at the end -> still frozen
    v1 = "it('x', () => { expect(1).toBe(2) })\n"
    s = _stream([
        ("use", {"name": "Write", "input": {"file_path": "/app/test/a-validation.test.tsx", "content": v1}}),
        ("res", "Tests:       1 failed, 3 passed, 4 total"),
    ])
    fz = validate_cc.frozen_failing_tests(DiskEnv(disk={}), [s], "/app")
    assert fz["files"] == {"test/a-validation.test.tsx": v1} and not fz["unreliable"], fz


def check_frozen_detects_nodebb_mocha_tests():
    # NodeBB mocha tests are test/*.js with no .test. suffix (8168c6c4, a5afad27 froze nothing)
    _sub.set_lang("js")
    v1 = "it('allow list', async () => { await assert.rejects(p) })\n"
    s = _stream([
        ("use", {"name": "Write", "input": {"file_path": "/app/test/chat-allow-validation.js", "content": v1}}),
        ("res", "  15 passing (794ms)\n  1 failing"),
    ])
    fz = validate_cc.frozen_failing_tests(DiskEnv(disk={"test/chat-allow-validation.js": v1}), [s], "/app")
    _sub.set_lang("python")
    assert fz["files"] == {"test/chat-allow-validation.js": v1}, fz


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
