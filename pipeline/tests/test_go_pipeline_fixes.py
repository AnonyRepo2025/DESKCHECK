"""Go batch1 defect fixes G1-G6 (2026-09-12): Go behaviour, plus Python paths left unchanged."""

import pytest

from simagent import subagent as _sub
from simagent import pipeline as sp
from simagent import plainrepair as pr


@pytest.fixture
def go():
    _sub.set_lang("go")
    yield
    _sub.set_lang("python")


@pytest.fixture
def python():
    _sub.set_lang("python")
    yield


class FakeEnv:
    """Records commands; ``handler(cmd) -> dict`` supplies each execute() result."""

    def __init__(self, handler):
        self.handler = handler
        self.cmds = []

    def execute(self, action, timeout=None):
        cmd = action["command"]
        self.cmds.append(cmd)
        return self.handler(cmd)


# ------------------------------------------------------------------ G3: per-test id rule
def test_is_test_level_id_python_keeps_the_double_colon_rule(python):
    for t in ["tests/a.py::test_x", "tests/a.py", "TestX", "github.com/x/y", ""]:
        assert sp._is_test_level_id(t) == ("::" in t)


def test_is_test_level_id_go(go):
    assert sp._is_test_level_id("TestLoad")
    assert sp._is_test_level_id("TestLoad/advanced_(YAML)")
    assert sp._is_test_level_id("ExampleFoo")
    assert sp._is_test_level_id("BUILD:core/players_test.go")
    assert not sp._is_test_level_id("github.com/navidrome/navidrome/core")


# ------------------------------------------------------------------ G3 + G6: Go test runner
REAL_BUILD_FAIL = """# github.com/mattn/go-sqlite3
sqlite3-binding.c: In function 'sqlite3SelectNew':
sqlite3-binding.c:128049:10: warning: function may return address of local variable [-Wreturn-local-addr]
# github.com/navidrome/navidrome/core [github.com/navidrome/navidrome/core.test]
core/players_test.go:38:13: p.Type undefined (type *model.Player has no field or method Type)
FAIL\tgithub.com/navidrome/navidrome/core [build failed]
ok  \tgithub.com/navidrome/navidrome/persistence\t0.217s
FAIL
"""


def test_go_runner_parses_subtests_and_drops_double_counted_package(go):
    out = ("--- FAIL: TestA (0.00s)\n    --- FAIL: TestA/sub (0.00s)\nFAIL\n"
           "FAIL\tgithub.com/x/p\t0.1s\nok  \tgithub.com/x/q\t0.2s\n")
    env = FakeEnv(lambda c: {"output": out, "returncode": 1})
    failed, _ = sp._run_test_files(env, "/app", ["./p", "./q"])
    assert failed == {"TestA/sub"}


@pytest.mark.parametrize("res", [{"output": "", "returncode": 1},
                                 {"output": "some shell error\n", "returncode": 2},
                                 {"output": "ok  \tx\t0.1s\n", "returncode": -1,
                                  "exception_info": "TimeoutExpired"}])
def test_go_runner_rejects_runs_that_cannot_be_trusted(go, res):
    failed, text = sp._run_test_files(FakeEnv(lambda c: res), "/app", ["./p"])
    assert failed is None and "INVALID" in text


def test_go_runner_sets_aside_test_files_that_no_longer_compile(go):
    second = ("--- FAIL: TestOther (0.00s)\nFAIL\tgithub.com/navidrome/navidrome/core\t0.3s\n"
              "ok  \tgithub.com/navidrome/navidrome/persistence\t0.2s\nFAIL\n")

    def handler(cmd):
        if "mv core/players_test.go core/players_test.go.pipeline_stale" in cmd:
            return {"output": second, "returncode": 1}
        if "mv -f core/players_test.go.pipeline_stale core/players_test.go" in cmd:
            return {"output": "", "returncode": 0}
        return {"output": REAL_BUILD_FAIL, "returncode": 1}

    env = FakeEnv(handler)
    failed, text = sp._run_test_files(env, "/app", ["./core", "./persistence"])
    assert failed == {"TestOther", "BUILD:core/players_test.go"}
    assert "set aside" in text
    assert any("mv -f core/players_test.go.pipeline_stale core/players_test.go" in c
               for c in env.cmds), "the set-aside file must always be restored"


def test_go_runner_keeps_package_id_when_source_does_not_build(go):
    out = ("# github.com/x/core\ncore/players.go:10:2: undefined: y\n"
           "FAIL\tgithub.com/x/core [build failed]\nFAIL\n")
    env = FakeEnv(lambda c: {"output": out, "returncode": 1})
    failed, _ = sp._run_test_files(env, "/app", ["./core"])
    assert failed == {"github.com/x/core"} and len(env.cmds) == 1


# ------------------------------------------------------------------ G6: baseline revert
def _stub_tree(monkeypatch, runs):
    monkeypatch.setattr(sp, "_tree_snapshot", lambda env, repo: "snap")
    monkeypatch.setattr(sp, "_tree_restore", lambda env, repo, snap: True)
    monkeypatch.setattr(sp, "_patch_new_files", lambda patch: [])
    monkeypatch.setattr(sp, "_run_test_files",
                        lambda env, repo, tf, *a, **k: runs.append(tf) or (set(), "ok"))


PATCH = "diff --git a/scanner/walk.go b/scanner/walk.go\n--- a/scanner/walk.go\n+++ b/scanner/walk.go\n"


def test_regression_baseline_go_refuses_an_unverified_revert(go, monkeypatch):
    runs = []
    _stub_tree(monkeypatch, runs)
    env = FakeEnv(lambda c: {"output": "", "returncode": 1})
    failed, msg = sp._regression_baseline(env, "/app", PATCH, ["./scanner"])
    assert failed is None and "not verified" in msg and not runs
    assert "git diff --quiet HEAD -- scanner/walk.go" in env.cmds[0]


def test_regression_baseline_python_unchanged(python, monkeypatch):
    runs = []
    _stub_tree(monkeypatch, runs)
    env = FakeEnv(lambda c: {"output": "", "returncode": 1})
    failed, _ = sp._regression_baseline(env, "/app", PATCH.replace(".go", ".py"), ["t.py"])
    assert failed == set() and runs == [["t.py"]]
    assert "git diff --quiet" not in env.cmds[0]


# ------------------------------------------------------------------ G1: vet on *_test.go
def _vet_env(after_output):
    calls = {"vet": 0}

    def handler(cmd):
        if "go vet" in cmd:
            calls["vet"] += 1
            return {"output": "" if calls["vet"] == 1 else after_output, "returncode": 0}
        return {"output": "", "returncode": 0}
    return FakeEnv(handler)


def test_go_vet_test_file_breakage_is_advisory(go, monkeypatch):
    _stub_tree(monkeypatch, [])
    env = _vet_env("scanner/walk_dir_tree_test.go:24:47: cannot use baseDir (variable of type "
                   "string) as fs.FS value in argument to walkDirTree\n")
    [fact] = sp._go_vet_facts(env, "/app", PATCH)
    assert fact["violated"] is False and fact.get("advisory") is True
    assert "ADVISORY" in fact["fact"]


def test_go_vet_non_test_issue_still_violates(go, monkeypatch):
    _stub_tree(monkeypatch, [])
    env = _vet_env("scanner/walk.go:3:1: fmt.Sprintf format %d has arg x of wrong type string\n")
    [fact] = sp._go_vet_facts(env, "/app", PATCH)
    assert fact["violated"] is True and "walk.go" in fact["fact"]


# ------------------------------------------------------------------ G2: Go units + callees
GO_SRC = """package core

import "fmt"

type Player struct {
\tID        string
\tUserAgent string
}

func (p *playersRepo) Register(ctx context.Context, id string) (*Player, error) {
\tif id != "" {
\t\treturn p.Get(id)
\t}
\treturn nil, fmt.Errorf("x")
}

func helper() int { return 1 }

var (
\tdefaultA = 1
\tdefaultB = 2
)
"""


def _line(src, text):
    return src.splitlines().index(text) + 1


def test_go_units_from_source_maps_changed_lines_to_declarations():
    changed = {_line(GO_SRC, "\tUserAgent string"), _line(GO_SRC, "\t\treturn p.Get(id)"),
               _line(GO_SRC, "func helper() int { return 1 }"), _line(GO_SRC, "\tdefaultB = 2")}
    units = pr._go_units_from_source("core/p.go", GO_SRC, changed)
    by = {u["symbol"]: u for u in units}
    assert set(by) == {"Player", "playersRepo.Register", "helper", "defaultA"}
    reg = by["playersRepo.Register"]["code"]
    assert reg.startswith("func (p *playersRepo) Register(") and reg.endswith("}")
    assert by["helper"]["code"] == "func helper() int { return 1 }"
    assert by["Player"]["code"].endswith("}")


def test_extract_units_go_reads_the_patched_file(go):
    patch = ("diff --git a/core/p.go b/core/p.go\n--- a/core/p.go\n+++ b/core/p.go\n"
             "@@ -12,1 +12,1 @@\n-\t\treturn nil\n+\t\treturn p.Get(id)\n"
             "diff --git a/core/p_test.go b/core/p_test.go\n--- a/core/p_test.go\n+++ b/core/p_test.go\n"
             "@@ -1,1 +1,1 @@\n-x\n+y\n")
    env = FakeEnv(lambda c: {"output": GO_SRC if "cat core/p.go" in c else "", "returncode": 0})
    units = pr._extract_units_go(env, "/app", patch, 20)
    assert [u["symbol"] for u in units] == ["playersRepo.Register"] and units[0]["id"] == "U1"
    assert not any("p_test.go" in c for c in env.cmds)


def test_go_call_names_skip_keywords_and_builtins():
    code = "x := make([]int, 0)\ny := strings.TrimPrefix(a, b)\nz := p.Get(id)\nif len(a) > 0 {}\n"
    assert pr._go_call_names(code) == ["TrimPrefix", "Get"]


def test_callee_definitions_go_greps_go_definitions(go):
    def handler(cmd):
        if "grep -rn" in cmd and "Get" in cmd:
            return {"output": "./core/repo.go:12:func (r *repo) Get(id string) (*Player, error) {\n"}
        if "sed -n '12,56p'" in cmd:
            return {"output": "func (r *repo) Get(id string) (*Player, error) {\n\treturn nil, nil\n}\n"}
        return {"output": ""}
    out = pr._callee_definitions(FakeEnv(handler), "/app",
                                 [{"symbol": "playersRepo.Register", "code": "return p.Get(id)"}])
    assert "core/repo.go:12 (Get)" in out


# ------------------------------------------------------------------ G4: audit grounding + retry
def test_quote_grounding_tolerates_gofmt_layout_only_on_go(python):
    corpus = sp._ws_squash("func f() {\n\tc := &managedConn{\n\t\tclock: clock,\n\t}\n}")
    quote = "c := &managedConn{clock: clock}"
    assert sp._quote_grounded(quote, corpus) is False       # Python: unchanged strictness
    _sub.set_lang("go")
    try:
        assert sp._quote_grounded(quote, corpus) is True
        assert sp._quote_grounded("c := &otherThing{clock: clock}", corpus) is False
    finally:
        _sub.set_lang("python")


def test_complete_audit_retries_a_truncated_reply_with_a_larger_budget():
    good = "[R1] VERDICT: SATISFIED\nEVIDENCE: `x = 1`\nSIMULATION: traced x through f; ok.\n"
    for budget, want in ((16384, {"max_tokens": 16384}), (None, {})):
        seen = []

        def query(prompt, **kw):
            seen.append(kw)
            return _sub.QueryReply("", "length") if len(seen) == 1 else _sub.QueryReply(good, "stop")
        report, _calls = sp._complete_audit(query, [("R1", "x is set")], lambda pts: "P",
                                            retry_max_tokens=budget)
        assert "[R1] VERDICT: SATISFIED" in report
        assert seen[0] == {} and seen[1] == want


# ------------------------------------------------------------------ G5: review budget
def test_review_first_attempt_gets_the_larger_budget(monkeypatch):
    seen = []

    def qf(prompt, **kw):
        seen.append(kw)
        return "NO FINDINGS"
    qf.supports_output_limit = True
    monkeypatch.setattr(pr, "_review_reply_complete", lambda reply, ids=None: True)
    reply, ok = pr._ask_review(qf, None, "P", "review:trace", [])
    assert ok and seen == [{"max_tokens": 2 * _sub.SUBAGENT_MAX_COMPLETION_TOKENS}]


# ------------------------------------------------------------------ G1/G3: stale test files
def test_partition_never_forwards_build_markers(go):
    part = sp._partition_regressions({}, ["BUILD:scanner/walk_dir_tree_test.go", "TestWalk"])
    assert part["targets"] == ["TestWalk"]
    assert part["build_skipped"] == ["BUILD:scanner/walk_dir_tree_test.go"]
    assert "BUILD:scanner/walk_dir_tree_test.go" not in part["f2p_skipped"]


def test_partition_python_unchanged(python):
    part = sp._partition_regressions({}, ["tests/a.py::test_x", "tests/b.py::test_y"])
    assert part["targets"] == ["tests/a.py::test_x", "tests/b.py::test_y"]
    assert part["build_skipped"] == [] and part["f2p_skipped"] == []
