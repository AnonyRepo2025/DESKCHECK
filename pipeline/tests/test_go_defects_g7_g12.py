"""Go defect assessment fixes (2026-09-16): G7 subtest contract class, G10 empty *_test.go files.

Python and JS paths must stay unchanged."""

import pytest

from simagent import subagent as _sub
from simagent import pipeline as sp


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
    def __init__(self, handler):
        self.handler = handler
        self.cmds = []

    def execute(self, action, timeout=None):
        self.cmds.append(action["command"])
        return self.handler(action["command"])


# ------------------------------------------------------------------ G7: Go subtest contract class
FLIPT = {"fail_to_pass": "['TestLoad', 'TestServeHTTP']", "pass_to_pass": "[]"}


def test_go_subtest_of_f2p_parent_is_task_owned(go):
    assert sp._classify_regression_test(FLIPT, "TestLoad/advanced_(ENV)") == "fail_to_pass"
    assert sp._classify_regression_test(FLIPT, "TestLoad/database_key/value") == "fail_to_pass"
    assert sp._classify_regression_test(FLIPT, "TestMarshalYAML/defaults") == "unknown"


def test_go_explicit_listing_and_nearest_ancestor_win(go):
    # vuls-0ec945d0 shape: parent in fail_to_pass, its subtests listed in pass_to_pass.
    inst = {"fail_to_pass": "['TestX', 'TestX/new']", "pass_to_pass": "['TestX/old', 'TestY']"}
    assert sp._classify_regression_test(inst, "TestX/old") == "pass_to_pass"
    assert sp._classify_regression_test(inst, "TestX/new/deeper") == "fail_to_pass"
    assert sp._classify_regression_test(inst, "TestY/case") == "pass_to_pass"
    assert sp._classify_regression_test(inst, "TestX/unlisted") == "fail_to_pass"


def test_go_partition_skips_stale_subtests_of_f2p_parent(go):
    part = sp._partition_regressions(FLIPT, ["TestLoad/advanced_(ENV)", "TestMarshalYAML/defaults"])
    assert part["f2p_skipped"] == ["TestLoad/advanced_(ENV)"]
    assert part["targets"] == ["TestMarshalYAML/defaults"]


def test_python_slash_ids_keep_the_old_classification(python):
    inst = {"fail_to_pass": "['tests/a.py::test_x']", "pass_to_pass": "[]"}
    assert sp._classify_regression_test(inst, "tests/a.py::test_x/sub") == "unknown"


# ------------------------------------------------------------------ G10: empty *_test.go files
def test_go_runner_deletes_empty_test_files_before_running(go):
    env = FakeEnv(lambda c: {"output": "--- FAIL: TestIsOvalDefAffected (0.00s)\nFAIL\n"
                                       "FAIL\tgithub.com/x/oval\t0.1s\n", "returncode": 1})
    failed, _ = sp._run_test_files(env, "/app", ["./oval"])
    assert failed == {"TestIsOvalDefAffected"}
    cmd = env.cmds[0]
    assert cmd.index("-size 0 -delete") < cmd.index("go test -count=1 ./oval")


def test_regression_guard_message_names_unlisted_tests():
    ev = sp._regression_guard_evidence({"fail_to_pass": "['TestA']", "pass_to_pass": "[]"},
                                       [{"regressions": ["TestB", "TestC"]}, {"regressions": []}])
    assert ev["reason"] == "executed compatibility evidence includes pass_to_pass or unlisted tests"
    assert not ev["restore_pre_regression"]


# ------------------------------------------------------------------ G9: strict Go refine gate
REAL_VIAHTTP_BEFORE = """ok  \tgithub.com/future-architect/vuls/gost\t0.018s
--- FAIL: TestViaHTTP (0.00s)
    serverapi_test.go:122: kernel version: expected 3.16.51-2, actual 
FAIL
FAIL\tgithub.com/future-architect/vuls/scanner\t0.025s
FAIL
"""
REAL_VIAHTTP_AFTER = """ok  \tgithub.com/future-architect/vuls/gost\t0.018s
time="2026-09-13T16:35:47Z" level=info msg="Open boltDB: /tmp/vuls-test-cache-11111111.db"
--- FAIL: TestViaHTTP (0.01s)
    serverapi_test.go:108: error: expected %!s(<nil>), actual: X-Vuls-Server-Name header is required
    serverapi_test.go:112: os family: expected centos, actual 
    serverapi_test.go:122: kernel version: expected 3.16.51-2, actual 
FAIL
FAIL\tgithub.com/future-architect/vuls/scanner\t0.031s
FAIL
"""


def test_go_failure_output_masks_volatile_values_and_nests_subtests():
    text = ("--- FAIL: TestA (0.00s)\n    a_test.go:5: parent log 0xc000123\n"
            "    --- FAIL: TestA/sub (0.12s)\n        a_test.go:9: got 3 want 4\n"
            "    a_test.go:12: parent again\nFAIL\n")
    out = sp._go_failure_output(text)
    assert out["TestA"] == {"a_test.go:N: parent log N": 1, "a_test.go:N: parent again": 1}
    assert out["TestA/sub"] == {"a_test.go:N: got N want N": 1}


def test_go_new_failure_output_finds_a_break_inside_an_already_failing_test():
    grown = sp._go_new_failure_output(REAL_VIAHTTP_BEFORE, REAL_VIAHTTP_AFTER, {"TestViaHTTP"})
    assert sorted(grown) == ["TestViaHTTP"] and len(grown["TestViaHTTP"]) == 2
    # the same failure re-run (different timings) is not growth
    assert sp._go_new_failure_output(REAL_VIAHTTP_BEFORE, REAL_VIAHTTP_BEFORE.replace("0.025", "0.030"),
                                     {"TestViaHTTP"}) == {}


@pytest.fixture
def go_repo(tmp_path):
    import subprocess
    repo = tmp_path / "repo"; repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "api.go").write_text("package api\n\nvar Org = \"example.com\"\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=f@example.invalid", "-c", "user.name=F",
                    "commit", "-qm", "fixture"], check=True)

    class LocalEnv:
        def execute(self, payload, timeout=60):
            r = subprocess.run(payload["command"], shell=True, capture_output=True, text=True, timeout=timeout)
            return dict(output=r.stdout + r.stderr, returncode=r.returncode)
    return repo, LocalEnv()


def _gate(monkeypatch, repo, env, instance, run, strict):
    monkeypatch.setattr(sp, "_mech_audit", lambda *a: [])
    monkeypatch.setattr(sp, "_import_smoke", lambda *a: (True, "ok"))
    monkeypatch.setattr(sp, "_find_covering_tests", lambda *a, **kw: ["./api"])
    monkeypatch.setattr(sp, "GATE_TEST_REGRESSION", True)
    monkeypatch.setattr(sp, "_run_test_files", lambda *a, **kw: run((repo / "api.go").read_text()))
    patch_before = sp._extract_patch(env, str(repo))
    before = sp._tree_snapshot(env, str(repo))
    (repo / "api.go").write_text("package api\n\nvar Org = \"mongo.example.com\"\n")
    return sp._gate_phase_mutation(env, str(repo), instance, None, "repair_refine", "Submitted",
                                   patch_before, 0, True, snap_before=before, check_tests_on_conflict=True,
                                   revert_on_test_regression=True, intended_symbols=["GenerateDatabaseKeys"],
                                   strict_test_regression=strict, log=lambda m: None)


TELEPORT = {"fail_to_pass": "['TestGenerateDatabaseKeys/mongodb_certificate']", "pass_to_pass": "[]"}


def _mongo_run(src):
    return ({"TestGenerateDatabaseKeys/mongodb_certificate"}, "") if "mongo." in src else (set(), "")


def test_strict_refine_gate_vetoes_an_f2p_listed_break_on_go(go, go_repo, monkeypatch):
    repo, env = go_repo
    _, rec, _, _ = _gate(monkeypatch, repo, env, TELEPORT, _mongo_run, strict=True)
    assert not rec["kept"] and "no stale exemption" in rec["why"]
    assert "example.com\"" in (repo / "api.go").read_text() and "mongo." not in (repo / "api.go").read_text()


def test_default_gate_keeps_the_stale_exemption(go, go_repo, monkeypatch):
    repo, env = go_repo
    _, rec, _, _ = _gate(monkeypatch, repo, env, TELEPORT, _mongo_run, strict=False)
    assert rec["kept"] and rec["executed_check"]["stale_suspects"]


def test_strict_refine_gate_vetoes_new_output_under_an_already_failing_test(go, go_repo, monkeypatch):
    repo, env = go_repo
    run = lambda src: ({"TestViaHTTP"}, REAL_VIAHTTP_AFTER if "mongo." in src else REAL_VIAHTTP_BEFORE)
    _, rec, _, _ = _gate(monkeypatch, repo, env, {}, run, strict=True)
    assert not rec["kept"] and rec["executed_check"]["newly_failing"] == ["TestViaHTTP"]


def test_strict_flag_is_inert_off_go(python, go_repo, monkeypatch):
    repo, env = go_repo
    _, rec, _, _ = _gate(monkeypatch, repo, env, TELEPORT, _mongo_run, strict=True)
    assert rec["kept"] and rec["executed_check"]["stale_suspects"]


# ------------------------------------------------------------------ G8: Go reasoned-finding trigger
from simagent import plainrepair as pr

DIFF = """diff --git a/scanner/serverapi.go b/scanner/serverapi.go
--- a/scanner/serverapi.go
+++ b/scanner/serverapi.go
@@ -10,3 +10,4 @@
 	serverName := header.Get("X-Vuls-Server-Name")
+	kernelVersion := header.Get("X-Vuls-Kernel-Version")
 	if toLocalFile && serverName == "" {
"""
SPEC = 'Produce an error if the server name is absent using "X-Vuls-Server-Name".'


def test_finding_traced_only_through_base_lines_is_not_a_trigger():
    f = {"quotes": ['if toLocalFile && serverName == "" {'],
         "expected": 'an error naming "X-Vuls-Server-Name"'}
    assert pr._reason_trigger_failures(f, DIFF, SPEC, ("touched", "anchored")) == ["touched"]
    f["quotes"].append('kernelVersion := header.Get("X-Vuls-Kernel-Version")')
    assert pr._reason_trigger_failures(f, DIFF, SPEC, ("touched", "anchored")) == []


def test_expected_must_be_stated_in_the_spec_not_just_the_basis():
    f = {"quotes": ['kernelVersion := header.Get("X-Vuls-Kernel-Version")'],
         "expected": "`Variable` interpolates trait `bar` for {{literal.foo}}"}
    assert pr._reason_trigger_failures(f, DIFF, SPEC, ("touched", "anchored")) == ["anchored"]
    assert pr._reason_trigger_failures(f, DIFF, SPEC, ("touched",)) == []
    assert pr._reason_trigger_failures(f, DIFF, SPEC, ()) == []
