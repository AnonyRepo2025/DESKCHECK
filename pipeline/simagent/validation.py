"""Checkout execution profiles and frozen, specification-grounded validation probes.

All shell execution is delegated to the instance environment. This module neither
uses benchmark tests nor changes the evaluated repository's configuration.
"""
import ast
import base64
import hashlib
import json
import os
import re
import shlex
import uuid


_PROFILES = {}


def interface_files(text):
    """Explicit file/module declarations are resources, not function obligations."""
    try:
        decoded = json.loads(text)
        if isinstance(decoded, str):
            text = decoded
    except (ValueError, TypeError):
        pass
    text = (text or '').replace('\\n', '\n')
    result = []
    for match in re.finditer(r'(?im)^\s*(?:[-*]\s*)?Type:\s*`?(File|Module|Ansible Module)`?\s*$', text):
        tail = re.split(r'(?im)^\s*(?:[-*]\s*)?Type:', text[match.end():], maxsplit=1)[0]
        name = re.search(r'(?im)^\s*Name:\s*`?([\w.]+)', tail)
        path = re.search(r'(?im)^\s*(?:Path|Location):\s*`?([\w./-]+\.py)', tail)
        if name and path:
            result.append((path.group(1), name.group(1)))
    return list(dict.fromkeys(result))


def python_profile(env, repo):
    """Freeze the environment's Python interpreter and checkout import roots once."""
    key = (id(env), repo)
    cached = _PROFILES.get(key)
    if cached and cached[0] is env:
        return cached[1]
    script = ('import json,sys; print("PROFILE_JSON:" + json.dumps('
              '{"python": sys.executable, "version": "%d.%d" % sys.version_info[:2]}))')
    profile = None
    try:
        command = '(python -c {0} || python3 -c {0})'.format(shlex.quote(script))
        result = env.execute({'command': command}, timeout=30)
        match = re.search(r'^PROFILE_JSON:(.*)$', result.get('output') or '', re.M)
        if result.get('returncode') == 0 and match:
            profile = json.loads(match.group(1))
            if not os.path.isabs(profile.get('python', '')):
                profile = None
    except Exception:
        pass
    if profile is None:
        return None  # unavailable is not a successful profile, and may be retried
    # Nonexistent roots are harmless; retaining them also supports new source packages.
    roots = [repo + '/lib', repo + '/src', repo]
    profile['roots'] = roots
    profile['prefix'] = ('env QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 '
                         'PYTHONPYCACHEPREFIX=/tmp/pipeline_pycache_' + uuid.uuid4().hex + ' PYTHONPATH=' +
                         shlex.quote(':'.join(roots)) + ' ' + shlex.quote(profile['python']))
    _PROFILES[key] = (env, profile)
    return profile


def module_name(path):
    parts = path[:-3].split('/')
    if parts[0] in ('lib', 'src'):
        parts = parts[1:]
    if parts[-1] == '__init__':
        parts.pop()
    return '.'.join(parts)


_SMOKE_SCRIPT = r'''
import importlib, json, os, sys, traceback
repo, targets, changed_paths = json.loads(sys.argv[1])
changed_paths = {os.path.realpath(os.path.join(repo, p)) for p in changed_paths}
roots = [os.path.realpath(os.path.join(repo, p)) for p in ('lib', 'src', '')]
sys.path[:] = roots + [p for p in sys.path if p and os.path.realpath(p) not in roots]
def local_origin(name, module):
    rel = name.replace('.', '/')
    expected = [os.path.realpath(os.path.join(root, rel + suffix))
                for root in roots for suffix in ('.py', '/__init__.py')
                if os.path.isfile(os.path.join(root, rel + suffix))]
    if not expected:
        return True
    actual = getattr(module, '__file__', None)
    return bool(actual and os.path.realpath(actual) in expected)
rows = []
for name, path in targets:
    error = None
    missing_dependency = False
    attributable = path is not None
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        error = traceback.format_exc()
        attributable = attributable or any(os.path.realpath(frame.filename) in changed_paths
                                            for frame in traceback.extract_tb(exc.__traceback__))
        if isinstance(exc, ModuleNotFoundError):
            missing = (exc.name or '').replace('.', '/')
            top = missing.split('/')[0]
            missing_dependency = not any(os.path.exists(os.path.join(root, top)) or
                                         os.path.isfile(os.path.join(root, top + '.py'))
                                         for root in roots)
    wrong = {n: getattr(m, '__file__', None) for n, m in list(sys.modules.items())
             if m is not None and not local_origin(n, m)}
    if error:
        # Missing packages outside this checkout invalidate this invocation.
        status = 'invalid' if wrong or missing_dependency or not attributable else 'failed'
        rows.append(dict(module=name, status=status, error=error, wrong_origins=wrong))
    else:
        actual = getattr(mod, '__file__', None)
        expected = os.path.realpath(os.path.join(repo, path)) if path else None
        valid = not wrong and (not expected or actual and os.path.realpath(actual) == expected)
        rows.append(dict(module=name, status='passed' if valid else 'invalid',
                         origin=actual, expected=expected, wrong_origins=wrong))
print('SMOKE_JSON:' + json.dumps(rows))
'''


def import_smoke(env, repo, targets):
    """Return True/False/None: valid pass, attributable failure, unavailable evidence."""
    if not targets:
        return None, 'SMOKE_INVALID: no importable changed modules'
    profile = python_profile(env, repo)
    if not profile:
        return None, 'SMOKE_INVALID: Python execution profile unavailable'
    rows = []
    # Separate processes prevent one import from making another pass by accident.
    for name, path in targets:
        args = json.dumps([repo, [[name, path]], [p for _, p in targets if p is not None]])
        command = ('cd ' + shlex.quote(repo) + ' && ' + profile['prefix'] + ' -c ' +
                   shlex.quote(_SMOKE_SCRIPT) + ' ' + shlex.quote(args))
        try:
            result = env.execute({'command': command}, timeout=60)
            match = re.search(r'^SMOKE_JSON:(.*)$', result.get('output') or '', re.M)
            if result.get('returncode') != 0 or not match:
                rows.append(dict(module=name, status='invalid', error='smoke did not complete'))
            else:
                rows.extend(json.loads(match.group(1)))
        except Exception as exc:
            rows.append(dict(module=name, status='invalid', error=str(exc)))
    # A known checkout failure remains a failure even if another target is unavailable.
    status = (False if any(r['status'] == 'failed' for r in rows) else
              None if any(r['status'] == 'invalid' for r in rows) else True)
    return status, json.dumps(dict(profile=profile, modules=rows), ensure_ascii=False)


def probe_instructions(path, profile):
    return '''
=== REPLAYABLE VALIDATION EVIDENCE ===
For each Python behavior probe, preserve it in the JSON manifest at %s.
Format: {"probes": [{"id": "short_unique_id", "basis": "exact specification quotation",
"expected": "observable result and why the full specification requires it",
"code": "self-contained Python script with executable assert statements"}]}.
Use imports from the actual checkout. Run top-level assertions (or call functions
containing them); do not merely define tests, print expected results, use pytest.main,
or exit successfully without assertions. Use temporary directories for fixture writes.
Do not modify repository source/configuration from a probe or detect a test runner,
mock identity, snapshot, or probe harness. Do not mock the behavior under test.
Keep inputs and expected values independent of the candidate's output. Include
conflicting clauses rather than choosing an unsupported expected value. Probes will
be independently reviewed, frozen, and replayed unchanged on before/after snapshots.
An assertion failure followed by a pass can justify an edit; an unavailable run cannot.
Record already-passing probes too, to protect them against later edits. Do not submit
the manifest or probes as production code. Keep the manifest outside the repository.
Execution profile: %s
''' % (path, profile['prefix'] if profile else 'unavailable; report the evidence gap')


def load_probes(env, path, specification, query):
    """Freeze code before replay and require independent semantic oracle confirmation."""
    records, rejected = [], []
    try:
        raw = env.execute({'command': 'test ! -f {0} || head -c 100001 {0}'.format(shlex.quote(path))}, timeout=30)
        text = raw.get('output') or ''
        if not text.strip():
            return records, rejected
        if len(text) > 100000:
            raise ValueError('probe manifest exceeds 100000 characters')
        items = json.loads(text)['probes']
        if not isinstance(items, list):
            raise ValueError('probes must be a list')
    except Exception as exc:
        return [], [dict(reason='invalid manifest: ' + str(exc))]
    normalized = ' '.join(specification.split())
    seen = set()
    for index, item in enumerate(items):
        try:
            pid, basis, expected, code = (item[k] for k in ('id', 'basis', 'expected', 'code'))
            if not all(isinstance(s, str) and s.strip() for s in (pid, basis, expected, code)):
                raise ValueError('all fields must be nonempty strings')
            if pid in seen or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', pid):
                raise ValueError('duplicate or invalid id')
            seen.add(pid)
            if index >= 12:
                raise ValueError('probe review budget reached; remaining probes unverified')
            if len(code) > 12000 or len(basis) < 24 or ' '.join(basis.split()) not in normalized:
                raise ValueError('oversized code or ungrounded specification quotation')
            parsed = ast.parse(code)
            if not any(isinstance(n, ast.Assert) for n in ast.walk(parsed)):
                raise ValueError('no executable assertions')
            prompt = ('Independently review this proposed validation probe. The code under test '
                      'is NOT the expected-value oracle. Consider the ENTIRE specification, '
                      'including contradictory clauses and input restrictions. Confirm only if '
                      'the assertions follow from it, exercise real production behavior through '
                      'normal imports, do not substitute the implementation, write production '
                      'files, or inspect tests/mocks/harness/snapshot identity. Otherwise reject '
                      'or mark uncertain. Return JSON only: {"verdict":"CONFIRMED|REJECTED|'
                      'UNCERTAIN", "reason":"..."}.\nSPECIFICATION:\n' + specification +
                      '\nPROBE:\n' + json.dumps(item))
            reply = query(prompt)
            if getattr(reply, 'truncated', False) or getattr(reply, 'finish_reason', None) in ('length', 'tool_calls'):
                raise ValueError('oracle review incomplete')
            review = json.loads(str(reply).strip())
            if review.get('verdict') != 'CONFIRMED' or not review.get('reason'):
                raise ValueError('oracle not confirmed: ' + str(review))
            records.append(dict(id=pid, basis=basis, expected=expected, code=code,
                                sha256=hashlib.sha256(code.encode()).hexdigest(), review=review))
        except Exception as exc:
            rejected.append(dict(id=item.get('id') if isinstance(item, dict) else None, reason=str(exc)))
    return records, rejected


_PROBE_SCRIPT = r'''
import ast, base64, json, os, sys, traceback
source = base64.b64decode(sys.argv[1]).decode()
roots = os.environ['PYTHONPATH'].split(':')
sys.path[:] = roots + [p for p in sys.path if p and os.path.realpath(p) not in roots]
count = [0]
class Assertions(ast.NodeTransformer):
    def visit_Assert(self, node):
        # Preserve lazy assertion-message evaluation by leaving the assert intact.
        tick = ast.Expr(ast.Call(ast.Name('_pipeline_tick', ast.Load()), [], []))
        return [ast.copy_location(tick, node), node]
def tick():
    count[0] += 1
try:
    tree = ast.fix_missing_locations(Assertions().visit(ast.parse(source)))
    exec(compile(tree, '<frozen-validation-probe>', 'exec'),
         {'__name__': '__main__', '_pipeline_tick': tick})
    result = dict(status='passed' if count[0] else 'invalid', assertions=count[0])
except AssertionError:
    result = dict(status='failed' if count[0] else 'invalid', assertions=count[0], error=traceback.format_exc())
except BaseException:
    result = dict(status='invalid', assertions=count[0], error=traceback.format_exc())
wrong = {}
for name, module in list(sys.modules.items()):
    rel = name.replace('.', '/')
    expected = [os.path.realpath(os.path.join(root, rel + suffix))
                for root in roots for suffix in ('.py', '/__init__.py')
                if os.path.isfile(os.path.join(root, rel + suffix))]
    actual = getattr(module, '__file__', None)
    if expected and (not actual or os.path.realpath(actual) not in expected):
        wrong[name] = actual
if wrong:
    result = dict(status='invalid', reason='wrong import origins', wrong_origins=wrong)
print('PROBE_JSON:' + json.dumps(result))
'''


def run_probe(env, repo, probe):
    profile = python_profile(env, repo)
    if not profile or hashlib.sha256(probe['code'].encode()).hexdigest() != probe['sha256']:
        return dict(status='invalid', reason='unavailable profile or changed frozen probe')
    encoded = base64.b64encode(probe['code'].encode()).decode()
    command = ('cd ' + shlex.quote(repo) + ' && ' + profile['prefix'] + ' -c ' +
               shlex.quote(_PROBE_SCRIPT) + ' ' + shlex.quote(encoded))
    try:
        result = env.execute({'command': command}, timeout=60)
        matches = re.findall(r'^PROBE_JSON:(.*)$', result.get('output') or '', re.M)
        if result.get('returncode') != 0 or len(matches) != 1:
            return dict(status='invalid', reason='probe did not complete')
        return dict(json.loads(matches[0]), profile=profile)
    except Exception as exc:
        return dict(status='invalid', reason=str(exc))
