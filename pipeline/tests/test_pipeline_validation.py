"""Offline checks of generic workflow fixes; fixtures contain no benchmark answers."""
import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from simagent import validation as pv
from simagent import pipeline as sp


class LocalEnv:
    def __init__(self, **env):
        self.environ = dict(os.environ, **env)
        self.commands = []

    def execute(self, payload, timeout=60):
        self.commands.append(payload['command'])
        result = subprocess.run(payload['command'], shell=True, env=self.environ,
                                capture_output=True, text=True, timeout=timeout)
        return dict(output=result.stdout + result.stderr, returncode=result.returncode)


def frozen(code, pid='behavior'):
    return dict(id=pid, code=code, sha256=hashlib.sha256(code.encode()).hexdigest())


@pytest.mark.parametrize('layout', ['lib', 'src'])
def test_checkout_imports_win_over_installed_package(tmp_path, layout):
    repo, installed = tmp_path / 'repo', tmp_path / 'installed'
    for root in [repo / layout, installed]:
        (root / 'pkg').mkdir(parents=True)
        (root / 'pkg/__init__.py').write_text('')
    (repo / layout / 'pkg/value.py').write_text('def value(): return 2\n')
    (repo / layout / 'pkg/api.py').write_text('from pkg.value import value\n')
    (installed / 'pkg/value.py').write_text('raise RuntimeError("wrong package")\n')
    env = LocalEnv(PYTHONPATH=str(installed))
    status, details = pv.import_smoke(env, str(repo), [('pkg.api', layout + '/pkg/api.py')])
    assert status is True, details
    assert json.loads(details)['modules'][0]['origin'] == str(repo / layout / 'pkg/api.py')
    profile = pv.python_profile(env, str(repo))
    assert pv.python_profile(env, str(repo)) is profile
    assert pv.run_probe(env, str(repo), frozen('from pkg.value import value\nassert value() == 2'))['status'] == 'passed'


@pytest.mark.parametrize('source,status', [
    ('import missing_third_party_package_xyz\n', None),
    ('import pkg.missing_local_module\n', False),
    ('raise AttributeError("circular import")\n', False),
    ('def broken(:\n', False),
])
def test_smoke_distinguishes_unavailable_and_broken_checkout(tmp_path, source, status):
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg/__init__.py').write_text('')
    (tmp_path / 'pkg/api.py').write_text(source)
    actual, detail = pv.import_smoke(LocalEnv(), str(tmp_path), [('pkg.api', 'pkg/api.py')])
    assert actual is status, detail


@pytest.mark.parametrize('code,status', [
    ('assert 1 == 1', 'passed'), ('assert 1 == 2', 'failed'),
    ('def test_unused():\n    assert False', 'invalid'),
    ('import missing_dependency_abc\nassert True', 'invalid'),
    ('assert True\nraise SystemExit(0)', 'invalid'),
])
def test_probe_requires_executed_assertions_and_completed_run(tmp_path, code, status):
    assert pv.run_probe(LocalEnv(), str(tmp_path), frozen(code))['status'] == status


def test_changed_frozen_probe_is_invalid(tmp_path):
    record = frozen('assert True')
    record['code'] = 'assert False'
    assert pv.run_probe(LocalEnv(), str(tmp_path), record)['status'] == 'invalid'


def test_oracle_must_be_independently_confirmed(tmp_path):
    spec = 'The function must return the integer two for this input.'
    path = tmp_path / 'probes.json'
    item = dict(id='two', basis=spec, expected='integer two', code='from api import value\nassert value() == 2')
    path.write_text(json.dumps(dict(probes=[item])))
    env = LocalEnv()
    records, rejected = pv.load_probes(env, str(path), spec, lambda _: '{"verdict":"UNCERTAIN","reason":"ambiguous"}')
    assert not records and rejected
    records, rejected = pv.load_probes(env, str(path), spec, lambda _: '{"verdict":"CONFIRMED","reason":"exact specified value"}')
    assert len(records) == 1 and not rejected
    records, rejected = pv.load_probes(env, str(path), spec,
        lambda _: sp._sub.QueryReply('{"verdict":"CONFIRMED","reason":"value"}', 'length'))
    assert not records and rejected


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    (repo / 'api.py').write_text('def value(): return 1\n')
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.email=fixture@example.invalid',
                    '-c', 'user.name=Fixture', 'commit', '-qm', 'fixture'], check=True)
    return repo, LocalEnv()


def test_replay_restores_candidate_and_records_failing_baseline(git_repo):
    repo, env = git_repo
    before = sp._tree_snapshot(env, str(repo))
    (repo / 'api.py').write_text('def value(): return 2\n')
    probe = frozen('from api import value\nassert value() == 2')
    result = sp._compare_behavior_probes(env, str(repo), before, [probe])
    assert result['restored'] and result['complete'], result
    assert result['improved'] == ['behavior'] and not result['regressions']
    assert (repo / 'api.py').read_text() == 'def value(): return 2\n'


def test_probe_source_mutation_invalidates_evidence_and_restores_candidate(git_repo):
    repo, env = git_repo
    before = sp._tree_snapshot(env, str(repo))
    (repo / 'api.py').write_text('def value(): return 2\n')
    probe = frozen('open("api.py", "w").write("# bad write\\n")\nassert True')
    result = sp._compare_behavior_probes(env, str(repo), before, [probe])
    assert result['restored'] and not result['complete'] and not result['improved']
    assert (repo / 'api.py').read_text() == 'def value(): return 2\n'


def gate_setup(monkeypatch):
    monkeypatch.setattr(sp, '_mech_audit', lambda *a: [])
    monkeypatch.setattr(sp, '_import_smoke', lambda *a: (True, 'fixture imports'))
    monkeypatch.setattr(sp, '_find_covering_tests', lambda *a, **kw: [])
    monkeypatch.setattr(sp, 'GATE_REQUIRE_IMPROVEMENT', True)
    monkeypatch.setattr(sp, 'GATE_TEST_REGRESSION', True)


def test_gate_keeps_executed_fix_without_covering_tests(git_repo, monkeypatch):
    repo, env = git_repo
    gate_setup(monkeypatch)
    patch_before = sp._extract_patch(env, str(repo))
    before = sp._tree_snapshot(env, str(repo))
    (repo / 'api.py').write_text('def value(): return 2\n')
    result = sp._gate_phase_mutation(env, str(repo), {}, SimpleNamespace(), 'validate',
        'Submitted', patch_before, 0, True, snap_before=before, require_improvement=True,
        revert_on_test_regression=True, check_tests_on_conflict=True, behavior_probes=[frozen('from api import value\nassert value() == 2')])
    assert result[1]['kept'], result[1]
    assert result[1]['behavior_probe_check']['improved'] == ['behavior']


def test_probe_improvement_cannot_override_new_test_failure(git_repo, monkeypatch):
    repo, env = git_repo
    gate_setup(monkeypatch)
    monkeypatch.setattr(sp, '_find_covering_tests', lambda *a, **kw: ['test_api.py'])
    monkeypatch.setattr(sp, '_run_test_files', lambda *a: (
        {'test_api.py::test_compatibility'} if 'return 2' in (repo / 'api.py').read_text() else set(), ''))
    before = sp._tree_snapshot(env, str(repo))
    (repo / 'api.py').write_text('def value(): return 2\n')
    result = sp._gate_phase_mutation(env, str(repo), {}, SimpleNamespace(), 'validate',
        'Submitted', '', 0, True, snap_before=before, require_improvement=True,
        revert_on_test_regression=True, check_tests_on_conflict=True, behavior_probes=[frozen('from api import value\nassert value() == 2')])
    assert not result[1]['kept'] and result[1]['reverted']
    assert result[1]['candidate_patch']
    assert (repo / 'api.py').read_text() == 'def value(): return 1\n'


def test_passing_probe_protects_against_later_reversion(git_repo, monkeypatch):
    repo, env = git_repo
    gate_setup(monkeypatch)
    before = sp._tree_snapshot(env, str(repo))
    (repo / 'api.py').write_text('def value(): return 2\n')
    result = sp._gate_phase_mutation(env, str(repo), {}, SimpleNamespace(), 'regression_block',
        'Submitted', '', 0, True, snap_before=before,
        behavior_probes=[frozen('from api import value\nassert value() == 1')])
    assert not result[1]['kept'] and result[1]['reverted']
    assert result[1]['behavior_probe_check']['regressions'] == ['behavior']


def test_all_checklist_clauses_are_retained():
    long_clause = 'The final requirement must preserve ' + ('every detail ' * 65) + 'including THE_END.'
    requirements = '\n'.join('- Requirement %d must work.' % i for i in range(1, 12)) + '\n- ' + long_clause
    contracts = [dict(name='f%d' % i, inputs='integer', outputs='value') for i in range(9)]
    points = sp._audit_checklist(dict(requirements=requirements), SimpleNamespace(signature_contracts=contracts))
    assert len(points) == 21 and dict(points)['R12'] == long_clause
    assert 'I9' in dict(points)
    assert [pt for batch in sp._audit_batches(points) for pt in batch] == points


def verdict(pid, outcome='UNVERIFIABLE'):
    return f'[{pid}] VERDICT: {outcome}\nEVIDENCE: Required code is absent from the pack.\n'


def test_partial_audit_retries_only_missing_points():
    prompts = []
    replies = iter([verdict('R1'), verdict('R2')])
    report, calls = sp._complete_audit(lambda p: (prompts.append(p) or next(replies)),
        [('R1', 'first'), ('R2', 'second')], lambda pts: repr(pts))
    assert 'R1' not in prompts[1] and 'R2' in prompts[1]
    coverage = sp._audit_coverage(report, [('R1', ''), ('R2', '')])
    assert coverage['complete'] and not coverage['passed']
    assert len(calls) == 2


def test_truncated_duplicate_and_unknown_verdicts_do_not_complete_audit():
    checklist = [('R1', 'first'), ('R2', 'second')]
    responses = iter([sp._sub.QueryReply(verdict('R1') + verdict('R2'), 'length'),
                      verdict('R1') + verdict('R1') + verdict('R99')])
    report, _ = sp._complete_audit(lambda _: next(responses), checklist, lambda pts: repr(pts))
    assert sp._audit_coverage(report, checklist)['missing'] == ['R1', 'R2']
    coverage = sp._audit_coverage(verdict('R1', 'CONFLICT') + verdict('R2'), checklist)
    assert coverage['complete'] and not coverage['passed']


def test_file_interface_is_not_a_function_obligation(monkeypatch):
    monkeypatch.setattr(sp._sub, 'is_src_path', lambda p: p.endswith('.py'))
    interface = 'Type: File\nName: operations\nPath: pkg/operations.py\nDescription: Public module.'
    assert pv.interface_files(interface) == [('pkg/operations.py', 'operations')]
    assert ('pkg/operations.py', 'operations') not in sp._iface_declared_sites(interface)


def test_mechanical_advisory_and_owner_checks(tmp_path):
    sample = tmp_path / 'api.py'
    sample.write_text('class Wrong:\n    def run(self): return 1\nclass Right: pass\ndef read():\n    payload = {"entry_points": {"a": 1}}\n    entry_points = payload["entry_points"]\n    return list(entry_points.items())\n')
    checks = [dict(kind='sentinel_method', path=str(sample), added=[7], risky=['items']),
              dict(kind='declared_member', path=str(sample), qualified='Right.run'),
              dict(kind='declared_member', path=str(sample), qualified='Wrong.run')]
    path = tmp_path / 'checks.json'
    path.write_text(json.dumps(checks))
    run = subprocess.run([sys.executable, '-c', sp._MECH_AUDIT_SCRIPT, str(path)],
                         text=True, capture_output=True, check=True)
    rows = json.loads(run.stdout.split('MECH_AUDIT_JSON:', 1)[1])
    assert rows[0]['advisory'] and not rows[0]['violated']
    assert rows[1]['violated'] and not rows[2]['violated']


def test_real_sentinel_protocol_failure_remains_executable(tmp_path):
    (tmp_path / 'objects.py').write_text(
        'class Thing:\n'
        '    def __getattr__(self, name): return lambda: None\n'
        'def entries(): return Thing()\n')
    probe = frozen('from objects import entries\nassert entries().items() is not None')
    assert pv.run_probe(LocalEnv(), str(tmp_path), probe)['status'] == 'failed'


def test_probe_wrong_import_origin_is_invalid(tmp_path):
    repo, elsewhere = tmp_path / 'repo', tmp_path / 'elsewhere'
    repo.mkdir()
    elsewhere.mkdir()
    (repo / 'api.py').write_text('VALUE = 1\n')
    (elsewhere / 'api.py').write_text('VALUE = 2\n')
    probe = frozen('import sys\nsys.path.insert(0, ' + repr(str(elsewhere)) + ')\nimport api\nassert api.VALUE == 2')
    result = pv.run_probe(LocalEnv(), str(repo), probe)
    assert result['status'] == 'invalid' and result['wrong_origins']


def test_invalid_probe_does_not_justify_an_edit(git_repo, monkeypatch):
    repo, env = git_repo
    gate_setup(monkeypatch)
    before = sp._tree_snapshot(env, str(repo))
    (repo / 'api.py').write_text('def value(): return 2\n')
    result = sp._gate_phase_mutation(env, str(repo), {}, SimpleNamespace(), 'validate',
        'Submitted', '', 0, True, snap_before=before, require_improvement=True,
        behavior_probes=[frozen('import missing_dependency_xyz\nassert True')])
    assert not result[1]['kept'] and result[1]['reverted']
    assert result[1]['evidence_status'] == 'unverified'
    assert result[1]['candidate_patch'] and result[1]['baseline_patch'] == ''


def test_audit_recovery_preserves_completed_points_across_batches():
    points = [('R%d' % i, 'requirement %d' % i) for i in range(1, 15)]
    calls = []
    def query(prompt):
        ids = json.loads(prompt)
        calls.append(ids)
        return ''.join(verdict(pid) for pid in ids)
    report, _ = sp._complete_audit(query, points,
        lambda batch: json.dumps([pid for pid, _ in batch]))
    assert list(map(len, calls)) == [6, 6, 2]
    assert sp._audit_coverage(report, points)['covered'] == 14
