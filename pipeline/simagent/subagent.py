from __future__ import annotations

import contextlib
import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Callable, Optional

try:  # the tool-call models raise this when a response carries zero tool calls
    from minisweagent.exceptions import FormatError
except Exception:  # pragma: no cover - allows importing/testing without mini installed
    class FormatError(Exception):
        messages: list = []

from simagent.phase_hook import extract_command, extract_observation, extract_thought


# Tool-free sub-agent calls bypass mini-swe-agent's normal model wrapper. Bound them here so a
# provider that sends response headers and then stalls while streaming the body cannot pin an
# instance forever. Existing model kwargs win, and every knob can be disabled with 0.
SUBAGENT_QUERY_TIMEOUT = max(0.0, float(os.getenv("SUBAGENT_QUERY_TIMEOUT", "300") or 0))
SUBAGENT_MAX_COMPLETION_TOKENS = max(
    0, int(os.getenv("SUBAGENT_MAX_COMPLETION_TOKENS", "8192") or 0))
SUBAGENT_RESPONSE_CHAR_CAP = max(
    0, int(os.getenv("SUBAGENT_RESPONSE_CHAR_CAP", "60000") or 0))


# Tool-free sub-agent calls bypass mini-swe-agent's normal model wrapper.  Bound them here so a
# provider that sends response headers and then stalls while streaming the body cannot pin an
# instance forever.  Existing model kwargs win, and every knob can be disabled with 0.
SUBAGENT_QUERY_TIMEOUT = max(0.0, float(os.getenv("SUBAGENT_QUERY_TIMEOUT", "300") or 0))
SUBAGENT_MAX_COMPLETION_TOKENS = max(
    0, int(os.getenv("SUBAGENT_MAX_COMPLETION_TOKENS", "8192") or 0))
SUBAGENT_RESPONSE_CHAR_CAP = max(
    0, int(os.getenv("SUBAGENT_RESPONSE_CHAR_CAP", "60000") or 0))


TRACE_CALL_CHAIN_SCRIPT = r'''
import ast, os, json, sys
from collections import defaultdict

# --- Phase 1: Build index by parsing all source files under $REPO_PATH ---
func_defs = defaultdict(list)
func_callees = {}
func_callers = defaultdict(set)
func_siblings = defaultdict(set)
file_def_ranges = defaultdict(list)   # rel_path -> [(lineno, end_lineno, name, cls), ...]
class_defs = defaultdict(list)        # class name -> [(rel_path, lineno, end_lineno)]
data_defs = defaultdict(list)         # module/class-level NAME = ... -> [(rel_path, lineno, end_lineno)]

# --- dynamic-dispatch modelling (CHAIN_DYNAMIC=0 disables all three rules) --------------------
# A literal ast.Call is not the only way control flows into a function: an operator dispatches to
# a dunder (a * b -> __mul__), calling a class runs its (possibly inherited) __new__/__init__, and
# a function stored in an import-time registry dict is invoked through a loop variable (sympy's
# Basic._constructor_postprocessor_mapping -> matexpr's _postprocessor closure -- the producer
# frame the purely-literal chain provably missed on sympy-17630). Each is modelled as a bounded
# over-approximation; the existing same-file/same-package ranking and the analyst arbitrate.
_DYN = os.environ.get('CHAIN_DYNAMIC', '1').lower() not in ('0', 'false', 'no')
_OP_DUNDERS = {'Add': '__add__', 'Sub': '__sub__', 'Mult': '__mul__', 'MatMult': '__matmul__',
               'Div': '__truediv__', 'FloorDiv': '__floordiv__', 'Mod': '__mod__', 'Pow': '__pow__'}
class_bases = {}                      # class name -> [base-class names]
func_inner = defaultdict(set)         # outer def -> nested def names (closures a factory returns)
func_symbols = {}                     # def site key -> every Name/Attribute symbol in its body
registry_stores = {}                  # registry name -> {'keys': set, 'funcs': set, 'locs': set}

class CallVisitor(ast.NodeVisitor):
    """Extract function/method call names from a function body."""
    def __init__(self):
        self.calls = set()
        self.symbols = set()
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.calls.add(node.func.attr)
        for arg in node.args:
            if isinstance(arg, ast.Name):
                self.calls.add(arg.id)
        for kw in node.keywords:
            if isinstance(kw.value, ast.Name):
                self.calls.add(kw.value.id)
        self.generic_visit(node)
    def visit_BinOp(self, node):
        # a * b dispatches to some type's __mul__ -- record the dunder as a callee unless both
        # operands are literals. resolve_sites' locality ranking scopes the fan-out.
        if _DYN:
            d = _OP_DUNDERS.get(type(node.op).__name__)
            if d and not (isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant)):
                self.calls.add(d)
        self.generic_visit(node)
    def visit_Name(self, node):
        self.symbols.add(node.id)
    def visit_Attribute(self, node):
        self.symbols.add(node.attr)
        self.generic_visit(node)

def _harvest_registries(tree, rel_path):
    """Record import-time ``<obj>[Key] = <value containing functions>`` registry stores.

    Module/class level only (registration happens at import time) -- function bodies are NOT
    descended into, so ordinary ``d[k] = v`` code contributes nothing. The subscript KEY (usually
    a class) is kept: objects of a subclass flowing through arithmetic/constructors is what makes
    the registry fire, so the key gates where the bridge is advertised.
    """
    stack = [tree]
    while stack:
        parent = stack.pop()
        for node in ast.iter_child_nodes(parent):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(node, ast.ClassDef):
                stack.append(node)
                continue
            if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Subscript):
                tgt = node.targets[0]
                base = tgt.value
                reg = base.attr if isinstance(base, ast.Attribute) else (base.id if isinstance(base, ast.Name) else '')
                if not reg or len(reg) < 5:
                    continue
                sl = tgt.slice
                if sl.__class__.__name__ == 'Index':   # py<3.9 wraps the subscript key
                    sl = sl.value
                key = sl.id if isinstance(sl, ast.Name) else (sl.attr if isinstance(sl, ast.Attribute) else '')
                stored, arg_names = set(), set()
                for sub in ast.walk(node.value):
                    if isinstance(sub, ast.Call):
                        f = sub.func
                        fn = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else '')
                        if fn:
                            stored.add(fn)
                        # a factory's ARGUMENTS (get_postprocessor(Mul)) are not stored functions
                        for a in ast.walk(sub):
                            if isinstance(a, ast.Name) and a is not f:
                                arg_names.add(a.id)
                    elif isinstance(sub, ast.Name):
                        stored.add(sub.id)
                stored -= arg_names
                if stored:
                    r = registry_stores.setdefault(reg, {'keys': set(), 'funcs': set(), 'locs': set()})
                    r['keys'].add(key)
                    r['funcs'].update(stored)
                    r['locs'].add(rel_path + ':' + str(node.lineno))
            else:
                stack.append(node)

def _iter_with_class(node, cls):
    """Yield (child_node, enclosing_class_name) so we know each def's class scope."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            yield child, cls
            yield from _iter_with_class(child, child.name)
        else:
            yield child, cls
            yield from _iter_with_class(child, cls)

def _def_end_line(src_lines, lineno):
    """Python < 3.8 fallback (``ast`` has no end_lineno there): scan to the last line indented
    deeper than the ``def`` line. Without this, every def range collapses to a single line, the
    frame sources degrade to bare signatures, and the file:line self-heal can never match."""
    if lineno > len(src_lines):
        return lineno
    i0 = lineno - 1
    # py<3.8 also points a DECORATED def's lineno at its first decorator -- take the indent (and
    # start the body scan) from the real ``def`` line, else the scan stops at the def itself.
    for j in range(i0, min(i0 + 50, len(src_lines))):
        s = src_lines[j].lstrip()
        if s.startswith('def ') or s.startswith('async def '):
            i0 = j
            break
    head = src_lines[i0]
    indent = len(head) - len(head.lstrip())
    end = i0 + 1
    for i in range(i0 + 1, len(src_lines)):
        line = src_lines[i]
        stripped = line.strip()
        if not stripped:
            continue
        if len(line) - len(line.lstrip()) <= indent and not stripped.startswith((')', ']', '}')):
            break
        end = i + 1
    return end

def index_file(filepath, rel_path):
    try:
        with open(filepath, 'r', errors='replace') as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except:
        return
    src_lines = source.splitlines()
    if _DYN:
        _harvest_registries(tree, rel_path)
    # Module/class-level data symbols (NAME = ... / NAME: T = ...). A registry dict or constant
    # is a real, editable symbol; without this index the tracer reports it as "NO definition"
    # (measured on qutebrowser _WEBENGINE_SETTINGS, which the localizer then discarded).
    # FunctionDef bodies are NOT descended into -- function-local variables are not symbols.
    _dstack = [tree]
    while _dstack:
        _dparent = _dstack.pop()
        for _dnode in ast.iter_child_nodes(_dparent):
            if isinstance(_dnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(_dnode, ast.ClassDef):
                _dstack.append(_dnode)
                continue
            _dtargets = []
            if isinstance(_dnode, ast.Assign):
                _dtargets = [t for t in _dnode.targets if isinstance(t, ast.Name)]
            elif isinstance(_dnode, ast.AnnAssign) and isinstance(_dnode.target, ast.Name):
                _dtargets = [_dnode.target]
            for _dt in _dtargets:
                data_defs[_dt.id].append(
                    (rel_path, _dnode.lineno, getattr(_dnode, 'end_lineno', None) or _dnode.lineno))
    for node, cls in _iter_with_class(tree, ''):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fname = node.name
            # Record the def WITH its file and enclosing class so callees can be
            # resolved file-/class-locally later (cuts cross-file same-name noise).
            func_defs[fname].append((rel_path, node.lineno, cls or ''))
            # Also record the def's line RANGE so a reported file:line can be mapped back
            # to its real enclosing function when the model named a non-existent one.
            _end = getattr(node, 'end_lineno', None) or _def_end_line(src_lines, node.lineno)
            file_def_ranges[rel_path].append((node.lineno, _end, fname, cls or ''))
            cv = CallVisitor()
            cv.visit(node)
            # Keep a same-name callee (e.g. a module fn `resolve` calling `obj.resolve()`): site-based
            # traversal resolves it to the OTHER def and uses site cycle-detection for real recursion.
            callees = sorted(cv.calls)
            key = (fname, rel_path, cls or '')
            func_callees[key] = sorted(set(func_callees.get(key, [])) | set(callees))
            for callee in callees:
                if callee != fname:                  # don't register a function as its own caller
                    func_callers[callee].add((fname, rel_path, cls or ''))
            if _DYN:
                func_symbols[key] = cv.symbols
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub is not node:
                        func_inner[fname].add(sub.name)
        elif isinstance(node, ast.ClassDef):
            class_name = node.name
            class_defs[class_name].append(
                (rel_path, node.lineno, getattr(node, 'end_lineno', None) or node.lineno))
            if _DYN and class_name not in class_bases:
                class_bases[class_name] = [b.id if isinstance(b, ast.Name) else getattr(b, 'attr', '')
                                           for b in node.bases]
            for item in node.body:
                if isinstance(item, ast.Assign) and isinstance(item.value, ast.Call):
                    co_refs = []
                    for arg in item.value.args:
                        if isinstance(arg, ast.Name):
                            func_callers[arg.id].add((class_name, rel_path, class_name))
                            co_refs.append(arg.id)
                    for kw in item.value.keywords:
                        if isinstance(kw.value, ast.Name):
                            func_callers[kw.value.id].add((class_name, rel_path, class_name))
                            co_refs.append(kw.value.id)
                    if len(co_refs) > 1:
                        for fn in co_refs:
                            func_siblings[fn].update(c for c in co_refs if c != fn)

_REPO_ROOT = os.environ.get('REPO_PATH', '/testbed')
for root, dirs, files in os.walk(_REPO_ROOT):
    dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', '.tox', 'node_modules', '.eggs')
               and 'test' not in d.lower()]
    for fname in files:
        if fname.endswith(('.py', '.java', '.js', '.ts', '.go', '.rs')) and 'test' not in fname.lower():
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, _REPO_ROOT)
            index_file(full, rel)

# --- dynamic-dispatch post-passes: constructor edges + registry-reader edges -------------------
ctor_edges = defaultdict(set)   # def site -> {(ctor_name, file, class)} for classes it CALLS
if _DYN:
    _ctor_cache = {}
    def _ctor_site(cname, _depth=0):
        """The __new__/__init__ def a call to class ``cname`` actually runs (base-chain walk)."""
        if cname in _ctor_cache:
            return _ctor_cache[cname]
        if _depth > 6 or cname not in class_bases:
            return None
        site = None
        for ctor in ('__new__', '__init__'):
            for f, ln, k in func_defs.get(ctor, []):
                if k == cname:
                    site = (ctor, f, k)
                    break
            if site:
                break
        if site is None:
            for b in class_bases.get(cname, []):
                site = _ctor_site(b, _depth + 1)
                if site:
                    break
        _ctor_cache[cname] = site
        return site

    # Calling a class (``MatMul(...)``) is a call to its constructor: bridge the name the
    # CallVisitor already recorded (which resolves to no function def) to the real ctor SITE.
    for _key in list(func_callees):
        for _c in func_callees[_key]:
            if _c in class_bases:
                _s = _ctor_site(_c)
                if _s and _s != _key:
                    ctor_edges[_key].add(_s)
                    func_callers[_s[0]].add(_key)

    def _registry_funcs(reg):
        """Known function defs stored in registry ``reg``, with factory closures expanded."""
        out = set()
        for fn in registry_stores.get(reg, {}).get('funcs', ()):
            if fn in func_defs:
                out.add(fn)
            for inner in func_inner.get(fn, ()):
                if inner in func_defs:
                    out.add(inner)
        return out

    # A function that reads a registry and calls through a variable gets call edges to every
    # function stored in that registry (capped) -- e.g. Basic._exec_constructor_postprocessors
    # -> get_postprocessor/_postprocessor. Factories also get an edge to their returned closure.
    for _key, _syms in func_symbols.items():
        for _reg in registry_stores:
            if _reg not in _syms:
                continue
            for _fn in sorted(_registry_funcs(_reg))[:6]:
                if _fn == _key[0]:
                    continue
                func_callees[_key] = sorted(set(func_callees.get(_key, [])) | {_fn})
                func_callers[_fn].add(_key)
                for _inner in sorted(func_inner.get(_fn, ()))[:3]:
                    if _inner in func_defs and func_defs.get(_fn):
                        _f, _ln, _k = func_defs[_fn][0]
                        _fkey = (_fn, _f, _k)
                        func_callees[_fkey] = sorted(set(func_callees.get(_fkey, [])) | {_inner})
                        func_callers[_inner].add(_fkey)

# --- Phase 2: Enumerate explicit call PATHS up to DEPTH levels in each direction ---
# For target A we want concrete chains:
#   caller paths (who reaches A):  B -> C -> D -> A      (DEPTH hops UP, presented root-first)
#   callee paths  (what A invokes): A -> E -> F -> G     (DEPTH hops DOWN)
# so the agent can simulate execution along the whole chain, not just read a neighbor list.
DEPTH = int(os.environ.get('CHAIN_DEPTH', '3'))
MAX_PATHS = int(os.environ.get('CHAIN_MAX_PATHS', '24'))      # cap paths per direction
MAX_BRANCH = int(os.environ.get('CHAIN_MAX_BRANCH', '6'))     # cap fan-out per node

# Issue-aware branch ranking: when a node has more callees/callers than MAX_BRANCH, expand the
# ones the ISSUE text mentions (by function name, else by defining file) and same-file neighbours
# FIRST instead of cutting alphabetically -- an alphabetical cut silently dropped the gold callee
# `luqum_parser` (openlibrary-9bdfd29). Every cut is recorded and rendered (BRANCH-CAP
# TRUNCATIONS) so downstream steps can SEE what was not expanded and re-trace it.
_ISSUE_TOKENS = {t for t in os.environ.get('CHAIN_ISSUE_TOKENS', '').lower().split(',')
                 if len(t) >= 4}
branch_truncations = []   # [{'at': name, 'kind': 'callees'|'callers', 'dropped': [names]}]

def _branch_pick(site_at, nxt, kind, ctx_file=''):
    if len(nxt) <= MAX_BRANCH:
        return nxt
    def rank(s):
        name, f, _cls = s
        ln, lf = name.lower(), (f or '').lower()
        sc = 0.0
        for t in _ISSUE_TOKENS:
            if t in ln:
                sc -= 2.0
            elif t in lf:
                sc -= 1.0
        if ctx_file and f == ctx_file:
            sc -= 0.5
        return sc
    ranked = sorted(nxt, key=rank)   # stable: original (alphabetical) order within a score tier
    kept, dropped = ranked[:MAX_BRANCH], ranked[MAX_BRANCH:]
    branch_truncations.append({'at': site_at[0], 'kind': kind,
                               'dropped': [d[0] for d in dropped]})
    return kept

def _pkg(path):
    """The directory (package) a file lives in, used to route an overloaded name to the nearest one."""
    return path.rsplit('/', 1)[0] if '/' in path else ''

# A "site" is a concrete definition: (name, file, class). Resolving a callee NAME to candidate
# sites -- preferring same class, then same file, then same package -- keeps the chain scoped: an
# ambiguous name defined in many unrelated files (append/get/arg/...) resolves to nothing local and
# is DROPPED instead of fanning into noise. Untyped method dispatch (obj.foo()) can hit more than
# one same-name def (e.g. URLPattern.resolve vs URLResolver.resolve), so this returns a RANKED LIST
# and the DFS branches into them -- one branch may reach the real producer the others miss.
_MAX_SITES = 3  # cap candidate defs returned for one ambiguous callee name
def resolve_sites(name, ctx_file, ctx_cls, avoid=None):
    defs = func_defs.get(name, [])
    if not defs:
        return []
    cands = [(name, f, k) for f, ln, k in defs]
    # Method-dispatch: a call like obj.foo() inside foo() resolves by name to foo's OWN site; when
    # other defs of the same name exist, drop self so the path follows the dispatch to the other def
    # (e.g. a module-level `resolve` calling URLResolver.resolve) instead of dead-ending on self.
    if avoid is not None and len(cands) > 1:
        cands = [c for c in cands if c != avoid] or cands
    if ctx_cls:                                      # 1) same file AND same class -> exact, single
        exact = [c for c in cands if c[1] == ctx_file and c[2] == ctx_cls]
        if exact:
            return exact[:1]
    same_file = [c for c in cands if c[1] == ctx_file]
    if same_file:                                    # 2) same file (any class)
        return same_file[:_MAX_SITES]
    same_pkg = [c for c in cands if _pkg(c[1]) == _pkg(ctx_file)]
    if same_pkg:                                     # 3) same package/dir (may be several -> branch)
        return same_pkg[:_MAX_SITES]
    if len({c[1] for c in cands}) == 1:              # 4) unambiguous: one def file
        return cands[:1]
    return []                                        # 5) ambiguous cross-file -> drop as noise

def site_callees(name, file, cls):
    return func_callees.get((name, file, cls), []) or func_callees.get((name, file, ''), [])

callee_paths = []
def dfs_callees(site, trail, visited):
    if len(callee_paths) >= MAX_PATHS:
        return
    name, file, cls = site
    nxt, seen = [], set()
    for c in site_callees(name, file, cls):
        for r in resolve_sites(c, file, cls, avoid=site)[:2]:   # up to 2 dispatch targets per callee
            if r in visited or r in seen:            # SITE-based cycle check (distinct same-name defs OK)
                continue
            seen.add(r)
            nxt.append(r)
    for r in sorted(ctor_edges.get(site, ()))[:2]:   # class-call -> its (inherited) constructor
        if r not in visited and r not in seen:
            seen.add(r)
            nxt.append(r)
    if not nxt or len(trail) - 1 >= DEPTH:
        callee_paths.append(list(trail))             # already root-first: A -> E -> F -> G
        return
    for r in _branch_pick(site, nxt, 'callees', ctx_file=file):
        dfs_callees(r, trail + [r], visited | {r})   # trail carries SITES, not bare names

# Callers already carry their own def site (caller_name, caller_file, caller_class), so the
# caller side is concrete; we de-dup by SITE and cap fan-out.
def site_callers(name):
    out, seen = [], set()
    for cn, cf, ck in sorted(func_callers.get(name, set())):
        site = (cn, cf, ck)
        if cn in func_defs and site not in seen:
            seen.add(site)
            out.append(site)
    return out

caller_paths = []
def dfs_callers(site, trail, visited):
    if len(caller_paths) >= MAX_PATHS:
        return
    nxt = [s for s in site_callers(site[0]) if s not in visited]
    if not nxt or len(trail) - 1 >= DEPTH:
        caller_paths.append(list(reversed(trail)))   # reverse to root-first: B -> C -> D -> A
        return
    for s in _branch_pick(site, nxt, 'callers'):
        dfs_callers(s, trail + [s], visited | {s})   # trail carries SITES, not bare names

func_name = sys.argv[1]
hint_file = sys.argv[2] if len(sys.argv) > 2 else ''
hint_line = 0
if len(sys.argv) > 3:
    try:
        hint_line = int(sys.argv[3])
    except ValueError:
        hint_line = 0

def _norm_path(p):
    return p.lstrip('/').replace('\\', '/')

def _enclosing_def(hfile, hline):
    """Map a reported file:line back to the name of the function whose body encloses it.

    Prefer the INNERMOST def whose [lineno, end_lineno] range contains the line; if the
    reported line sits between defs (e.g. it pointed at a comment just above the def), fall
    back to the nearest preceding def in the same file. File matching is suffix-based since
    the model may report an absolute or partial path.
    """
    hfile = _norm_path(hfile)
    cands = [rp for rp in file_def_ranges if rp == hfile or rp.endswith('/' + hfile)]
    if not cands and hfile:
        base = hfile.split('/')[-1]
        cands = [rp for rp in file_def_ranges if rp.split('/')[-1] == base]
    best = None        # innermost containing def: (lineno, name)
    prec = None        # nearest preceding def: (lineno, name)
    for rp in cands:
        for lo, hi, name, _cls in file_def_ranges[rp]:
            if lo <= hline <= hi and (best is None or lo > best[0]):
                best = (lo, name)
            if lo <= hline and (prec is None or lo > prec[0]):
                prec = (lo, name)
    if best:
        return best[1]
    return prec[1] if prec else None

# SELF-HEAL: the model may name a function that does not exist in the repo (it inferred the
# file+line correctly but guessed the enclosing function name). When that happens and we have a
# file+line, recover the REAL enclosing function so the call chain is traced on a real def.
healed_from = None
if func_name not in func_defs and hint_file and hint_line:
    _enc = _enclosing_def(hint_file, hint_line)
    if _enc and _enc != func_name:
        healed_from = func_name
        func_name = _enc

_target_sites = resolve_sites(func_name, hint_file, '')
target_site = _target_sites[0] if _target_sites else (func_name, hint_file, '')
dfs_callees(target_site, [target_site], {target_site})
dfs_callers(target_site, [target_site], {target_site})

# Drop a bare single-node "path" (no neighbors found) from each side.
callee_site_paths = [p for p in callee_paths if len(p) > 1]
caller_site_paths = [p for p in caller_paths if len(p) > 1]

def _site_lineno(site):
    n, f, k = site
    for fl, ln, kk in func_defs.get(n, []):
        if fl == f and kk == k:
            return ln
    return 0

# FUNCTION LOCATIONS from the sites the DFS actually TRAVERSED. A repo-wide re-resolution of the
# bare name (the old behaviour, capped at 3 defs) can list same-name defs that are NOT on the
# chain and silently drop the real caller (e.g. Django's many ``process_response`` defs hiding
# the one middleware that calls the target).
involved_sites = set([target_site])
for p in callee_site_paths + caller_site_paths:
    involved_sites.update(p)
locations = {}
for site in sorted(involved_sites):
    n, f, k = site
    if not f:
        continue
    ln = _site_lineno(site)
    loc = (f + ':' + str(ln)) if ln else f
    locations.setdefault(n, [])
    if loc not in locations[n]:
        locations[n].append(loc)

# DYNAMIC-DISPATCH BRIDGES: registry-stored functions that RUN when objects of a chain frame's
# class flow through constructors/arithmetic yet appear as a literal call NOWHERE (sympy's
# constructor postprocessors). Advertised whenever a chain frame's class is (a subclass of) the
# registry's key class -- depth caps cannot hide these because they bypass the path DFS entirely.
dynamic_bridges = []
if _DYN and registry_stores:
    def _base_closure(cname):
        out, stack = set(), [cname]
        while stack:
            c = stack.pop()
            for b in class_bases.get(c, []):
                if b and b not in out:
                    out.add(b)
                    stack.append(b)
        return out
    _frame_classes = sorted({s[2] for s in involved_sites if s[2]})
    # builtin key classes make every subclass "match" -- only project classes gate a bridge
    _GENERIC_KEYS = {'dict', 'list', 'tuple', 'set', 'frozenset', 'str', 'int', 'float',
                     'complex', 'bool', 'bytes', 'object', 'type', 'Exception'}
    for _reg in sorted(registry_stores):
        if len(dynamic_bridges) >= 3:
            break
        _fns = sorted(_registry_funcs(_reg))[:6]
        if not _fns:
            continue
        _why = ''
        for _kc in sorted(registry_stores[_reg]['keys']):
            if not _kc or _kc in _GENERIC_KEYS:
                continue
            for _fc in _frame_classes:
                if _kc == _fc or _kc in _base_closure(_fc):
                    _why = _fc + ' is a ' + _kc
                    break
            if _why:
                break
        if not _why:
            continue
        _entries = []
        for _fn in _fns:
            _d = func_defs.get(_fn, [])
            _loc = (_d[0][0] + ':' + str(_d[0][1])) if _d else ''
            _entries.append({'name': _fn, 'loc': _loc})
            if _loc:
                locations.setdefault(_fn, [])
                if _loc not in locations[_fn]:
                    locations[_fn].append(_loc)
        dynamic_bridges.append({'registry': _reg, 'why': _why,
                                'stored_at': sorted(registry_stores[_reg]['locs'])[:2],
                                'funcs': _entries})

# The JSON (and the rendered chain) keep NAME paths -- downstream parsers rely on that shape.
callee_paths = [[s[0] for s in p] for p in callee_site_paths]
caller_paths = [[s[0] for s in p] for p in caller_site_paths]

# --- Phase 3: attach the SOURCE BODY of the key call-chain frames -----------------------------
# The analyst reasons with no file access, so it otherwise GUESSES what each function does from its
# name (e.g. wrongly assuming `_populate` builds the per-request kwargs). Giving it the real bodies
# of the frames on the chain lets it verify which function actually constructs the wrong value.
MAX_FRAMES = int(os.environ.get('CHAIN_SRC_FRAMES', '14'))
MAX_FN_LINES = int(os.environ.get('CHAIN_SRC_LINES', '40'))
MAX_FN_CHARS = int(os.environ.get('CHAIN_SRC_CHARS', '1400'))
MAX_SRC_TOTAL = int(os.environ.get('CHAIN_SRC_TOTAL', '12000'))
# Ubiquitous helpers/builtins whose bodies are noise (and often live in unrelated files).
_SRC_SKIP = {'dict', 'list', 'set', 'tuple', 'append', 'extend', 'get', 'str', 'int', 'len',
             'update', 'isinstance', 'escape', 'keep_lazy', 'mark_safe', 'format'}
_tgt_file = target_site[1] if target_site else (hint_file or '')
_tgt_pkg = _pkg(_tgt_file)
_src_lines = {}
def _read_src(rel):
    if rel not in _src_lines:
        try:
            with open(os.path.join(_REPO_ROOT, rel), errors='replace') as fh:
                _src_lines[rel] = fh.readlines()
        except Exception:
            _src_lines[rel] = []
    return _src_lines[rel]

def _pick_def(name):
    """Choose the most relevant def of `name`: same file as the target, then same package, else first."""
    defs = func_defs.get(name, [])
    if not defs:
        return None
    for f, ln, k in defs:
        if f == _tgt_file:
            return (f, ln, k)
    for f, ln, k in defs:
        if _pkg(f) == _tgt_pkg and _tgt_pkg:
            return (f, ln, k)
    return defs[0]

# Order frames by execution proximity to the target: target first, then dynamic-bridge frames
# (producer candidates the paths cannot show -- their source must survive the frame budget),
# then callee-path nodes (the producer side), then caller-path nodes.
_ordered_sites, _seen = [target_site], set()
for b in dynamic_bridges:
    for e in b['funcs']:
        for f, ln, k in func_defs.get(e['name'], [])[:1]:
            s = (e['name'], f, k)
            if s not in _ordered_sites:
                _ordered_sites.append(s)
for p in callee_site_paths + caller_site_paths:
    for s in p:
        if s not in _ordered_sites:
            _ordered_sites.append(s)
frame_sources, _src_total = [], 0
for site in _ordered_sites:
    name = site[0]
    if name in _SRC_SKIP or site in _seen or len(frame_sources) >= MAX_FRAMES or _src_total >= MAX_SRC_TOTAL:
        continue
    # Exact def of THIS traversed site, so an overloaded name shows the caller that is actually
    # on the chain (not a same-name def elsewhere); heuristic pick only for unresolved sites.
    pick = None
    for fl, ln0, kk in func_defs.get(name, []):
        if fl == site[1] and kk == site[2]:
            pick = (fl, ln0, kk)
            break
    if pick is None:
        pick = _pick_def(name)
    if not pick:
        continue
    f, ln, k = pick
    end = ln
    for lo, hi, nm, cls in file_def_ranges.get(f, []):
        if lo == ln and nm == name:
            end = hi
            break
    lines = _read_src(f)
    if not lines:
        continue
    body = ''.join(lines[ln - 1:min(end, ln - 1 + MAX_FN_LINES)]).rstrip()[:MAX_FN_CHARS]
    if not body.strip():
        continue
    _seen.add(site)
    _src_total += len(body)
    qual = (k + '.' if k else '') + name
    frame_sources.append({'name': name, 'loc': f'{f}:{ln}', 'qual': qual, 'code': body})

# --- non-function symbol fallback ------------------------------------------------------------
# 'defined' below only reflects FUNCTION defs. Before a symbol is reported absent, check the
# class and module-level-data indexes: 'absent' now means verified absent from all three, while
# a class or data symbol gets its definition source and the functions referencing it (the
# pseudo-callers the call graph cannot show).
symbol_kind = ('function' if func_name in func_defs else
               'class' if func_name in class_defs else
               'data' if func_name in data_defs else 'absent')
symbol_defs, symbol_source, referencing = [], '', []
if symbol_kind in ('class', 'data'):
    _sdefs = (class_defs if symbol_kind == 'class' else data_defs)[func_name]
    symbol_defs = [f'{f}:{ln}' for f, ln, _e in _sdefs[:3]]
    _f0, _ln0, _end0 = _sdefs[0]
    _slines = _read_src(_f0)
    symbol_source = ''.join(_slines[_ln0 - 1:min(_end0, _ln0 - 1 + 40)]).rstrip()[:2000]
    # Referencing functions: recorded callers (calling a class ``Name(...)`` registers a call to
    # the name) plus any def whose body mentions the symbol (CallVisitor's symbol harvest).
    _refs = []
    for fn, rel, k in sorted(func_callers.get(func_name, ())):
        _refs.append(((k + '.' if k else '') + fn, rel))
    for (fn, rel, k), syms in sorted(func_symbols.items()):
        if func_name in syms:
            _refs.append(((k + '.' if k else '') + fn, rel))
    _seen_r = set()
    for q, rel in _refs:
        if (q, rel) in _seen_r:
            continue
        _seen_r.add((q, rel))
        referencing.append(f'{q} ({rel})')
        if len(referencing) >= 12:
            break

print(json.dumps({
    'target': func_name,
    'healed_from': healed_from,
    'defined': func_name in func_defs,
    'symbol_kind': symbol_kind,
    'symbol_defs': symbol_defs,
    'symbol_source': symbol_source,
    'referencing_functions': referencing,
    'target_defs': [f"{fl}:{ln}" for fl, ln, _k in func_defs.get(func_name, [])],
    'depth': DEPTH,
    'callee_paths': callee_paths,
    'caller_paths': caller_paths,
    # SITE-qualified paths -- the same routes with each hop as [name, file, class] instead of a
    # bare name. The name-only lists above collapse every same-named method in the repo into ONE
    # node (GalaxyCLI.__init__ == GalaxyAPI.__init__), which is harmless for the stock renderer
    # but manufactures false edges for any consumer that builds a GRAPH out of them.
    'callee_site_paths': [[list(s) for s in p] for p in callee_site_paths],
    'caller_site_paths': [[list(s) for s in p] for p in caller_site_paths],
    'locations': locations,
    'frame_sources': frame_sources,
    'dynamic_bridges': dynamic_bridges,
    'branch_truncations': branch_truncations[:12],
}))
'''


# =============================================================================
# Language profile (SWE-bench Pro multi-language support)
# =============================================================================
# The pipeline rebinds this per instance from ``instance["repo_language"]``. Path-matching
# regexes across the pipeline are built from ``src_ext_alt()`` and RECOMPILED on every
# ``set_lang`` through ``on_lang_change`` hooks, so a Python instance matches exactly the
# extensions it always did; BEHAVIOR differences (test runner, def-grep grammar, tracer
# script, smoke check) branch on ``is_go()`` / ``is_js()`` / ``is_python()``.
LANG = (os.environ.get("PIPELINE_LANG") or "python").strip().lower()

_LANG_ALIASES = {
    "py": "python", "python": "python", "python3": "python",
    "go": "go", "golang": "go",
    # SWE-bench Pro labels TypeScript repos both "js" (element-web, webclients) and "ts"
    # (tutanota); they share one toolchain (node + jest/mocha), hence one profile.
    "js": "js", "javascript": "js", "ts": "js", "typescript": "js", "node": "js",
}
_SRC_EXTS = {
    "python": (".py",),
    "go": (".go",),
    "js": (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"),
}
_LANG_HOOKS: "list" = []


def _normalize_lang(lang: str) -> str:
    return _LANG_ALIASES.get((lang or "python").strip().lower(), "python")


def set_lang(lang: str) -> None:
    global LANG
    LANG = _normalize_lang(lang)
    for fn in _LANG_HOOKS:
        fn()


def on_lang_change(fn):
    """Register (and immediately run) a hook that rebuilds language-derived module state --
    the ``.(?:py|...)`` path regexes -- so they always reflect the active profile."""
    _LANG_HOOKS.append(fn)
    fn()
    return fn


def is_go() -> bool:
    return LANG == "go"


def is_js() -> bool:
    return LANG == "js"


def is_python() -> bool:
    return LANG == "python"


def src_exts() -> "tuple[str, ...]":
    return _SRC_EXTS.get(LANG, (".py",))


def src_ext_alt() -> str:
    """Regex alternation of the active language's source extensions (``py`` / ``go`` /
    ``js|jsx|ts|tsx|mjs|cjs``) for building dotted-extension path patterns."""
    return "|".join(e[1:] for e in src_exts())


def is_src_path(path: str) -> bool:
    """Does ``path`` carry one of the active language's source extensions?"""
    return (path or "").endswith(src_exts())


def src_includes() -> str:
    """Complete ``--include=`` flag string for grep over this language's sources."""
    return " ".join(f"--include='*{e}'" for e in src_exts())


def src_ls_globs() -> str:
    """Quoted pathspec globs for ``git ls-files`` over this language's sources."""
    return " ".join(f"'*{e}'" for e in src_exts())


def def_pattern(name: str) -> str:
    """grep -E pattern matching a definition of ``name`` (function/method; +type/class)."""
    esc = re.escape(name)
    if is_go():
        # plain func, method with receiver, or type declaration
        return r"^func(\s*\([^)]*\))?\s+" + esc + r"\(|^type\s+" + esc + r"\b"
    if is_js():
        # function declaration; const/let arrow or function expression; class/object method
        # shorthand (incl. TS modifiers); `name: function`/`name: () =>` object property;
        # `Obj.name = function` assignment (CommonJS module style); class/interface/type/enum.
        return (
            r"^\s*(export\s+)?(default\s+)?(async\s+)?function\s*\*?\s*" + esc + r"\s*\("
            r"|^\s*(export\s+)?(const|let|var)\s+" + esc + r"\s*(:.*)?=\s*(async\s*)?(function\b|\(|[A-Za-z_$][A-Za-z0-9_$]*\s*=>)"
            r"|^\s+((public|private|protected|readonly|static)\s+)*" + esc + r"\s*=\s*(async\s*)?\("
            r"|^\s*((public|private|protected|static|async|readonly|override|get|set)\s+)*" + esc + r"\s*(<[^>]*>)?\s*\([^)]*\)\s*(:[^{;=]*)?\{"
            r"|^\s*" + esc + r"\s*:\s*(async\s+)?(function\b|\([^)]*\)\s*=>)"
            r"|^\s*([A-Za-z_$][A-Za-z0-9_$]*\.)+" + esc + r"\s*=\s*(async\s+)?(function\b|\([^)]*\)\s*=>)"
            r"|^\s*(export\s+)?(default\s+)?(abstract\s+)?(class|interface|type|enum)\s+" + esc + r"\b"
        )
    return r"^\s*(async\s+)?def\s+" + esc + r"\(|^\s*class\s+" + esc + r"\b"


_JS_TEST_DIRS = {"test", "tests", "__tests__", "__snapshots__", "__mocks__", "cypress", "e2e"}
_JS_TEST_FILE_RE = re.compile(r"(?:[.-]test|\.spec)\.[cm]?[jt]sx?$|\.snap$")


def is_test_path(path: str) -> bool:
    """Language-aware test-file test (repo-relative path or basename)."""
    base = (path or "").rsplit("/", 1)[-1]
    if base.endswith("_test.go"):
        return True
    if base.startswith("test_") or base.endswith("_test.py"):
        return True
    if is_js():
        # jest/mocha conventions: ``X.test.ts``, ``X-test.tsx``, ``X.spec.js``, snapshot
        # files, and anything under a test/tests/__tests__/__mocks__/cypress directory.
        if _JS_TEST_FILE_RE.search(base):
            return True
        parts = (path or "").split("/")[:-1]
        if any(d in _JS_TEST_DIRS for d in parts):
            return True
    return "/tests/" in f"/{path}" or (path or "").startswith("tests/")


# Go call-chain tracer: same argv/env interface and same output JSON contract as
# TRACE_CALL_CHAIN_SCRIPT, but indexes Go sources with a regex grammar (func defs with
# optional receivers; body = def line .. first column-0 '}'). No dynamic-dispatch modelling
# (Go has no operator dunders / import-time registry dicts); interface-method dispatch is
# approximated by name-level matching, which is exactly what the name-keyed graph does.
GO_TRACE_CALL_CHAIN_SCRIPT = r'''
import json, os, re, sys

REPO = os.environ.get('REPO_PATH', '/testbed')
DEPTH = int(os.environ.get('CHAIN_DEPTH', '3'))
MAX_PATHS = int(os.environ.get('CHAIN_MAX_PATHS', '24'))
MAX_BRANCH = int(os.environ.get('CHAIN_MAX_BRANCH', '6'))
ISSUE_TOKENS = set(t for t in os.environ.get('CHAIN_ISSUE_TOKENS', '').lower().split(',') if t)
MAX_FRAMES = int(os.environ.get('CHAIN_SRC_FRAMES', '14'))
MAX_FN_LINES = int(os.environ.get('CHAIN_SRC_LINES', '40'))
MAX_FN_CHARS = int(os.environ.get('CHAIN_SRC_CHARS', '1400'))
MAX_SRC_TOTAL = int(os.environ.get('CHAIN_SRC_TOTAL', '12000'))

func_name = sys.argv[1] if len(sys.argv) > 1 else ''
hint_file = sys.argv[2] if len(sys.argv) > 2 else ''
hint_line = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 0

FUNC_RE = re.compile(r'^func\s+(?:\(\s*\w+\s+\*?([\w\.]+)\s*\)\s+)?([A-Za-z_]\w*)\s*\(')
CALL_RE = re.compile(r'\b([A-Za-z_]\w*)\s*\(')
NOISE = {'if', 'for', 'switch', 'select', 'func', 'return', 'go', 'defer', 'range', 'chan',
         'map', 'make', 'new', 'len', 'cap', 'append', 'copy', 'delete', 'close', 'panic',
         'recover', 'print', 'println', 'string', 'byte', 'rune', 'int', 'int8', 'int16',
         'int32', 'int64', 'uint', 'uint8', 'uint16', 'uint32', 'uint64', 'float32',
         'float64', 'bool', 'error', 'interface', 'struct', 'complex64', 'complex128',
         'uintptr', 'any', 'min', 'max', 'clear', 'Sprintf', 'Errorf', 'Printf', 'Fprintf',
         'Error', 'String', 'Wrap', 'Wrapf', 'New', 'Background', 'Context', 'WithField'}

func_defs = {}         # name -> [(rel, line, recv)]
file_def_ranges = {}   # rel -> [(lo, hi, name, recv)]
file_lines = {}

for root, dirs, files in os.walk(REPO):
    dirs[:] = [d for d in dirs if d not in ('.git', 'vendor', 'testdata', 'node_modules',
                                            '_fixtures', 'fixtures')]
    for fn in files:
        if not fn.endswith('.go') or fn.endswith('_test.go'):
            continue
        full = os.path.join(root, fn)
        rel = os.path.relpath(full, REPO)
        try:
            with open(full, errors='replace') as fh:
                lines = fh.readlines()
        except Exception:
            continue
        defs = []
        for i, ln in enumerate(lines, 1):
            m = FUNC_RE.match(ln)
            if m:
                defs.append((i, m.group(2), m.group(1) or ''))
        if not defs:
            continue
        file_lines[rel] = lines
        ranges = []
        for j, (lo, name, recv) in enumerate(defs):
            nxt = defs[j + 1][0] - 1 if j + 1 < len(defs) else len(lines)
            hi = nxt
            for k in range(lo, nxt):
                if lines[k].startswith('}'):
                    hi = k + 1
                    break
            ranges.append((lo, hi, name, recv))
            func_defs.setdefault(name, []).append((rel, lo, recv))
        file_def_ranges[rel] = ranges

callees_of = {}
callers_of = {}
for rel, ranges in file_def_ranges.items():
    lines = file_lines[rel]
    for lo, hi, name, recv in ranges:
        body = ''.join(lines[lo:hi])
        for t in CALL_RE.findall(body):
            if t in NOISE or t == name or t not in func_defs:
                continue
            lst = callees_of.setdefault(name, [])
            if t not in lst:
                lst.append(t)
            cl = callers_of.setdefault(t, [])
            if name not in cl:
                cl.append(name)

healed_from = None
if func_name not in func_defs and hint_file:
    rel = hint_file.lstrip('/')
    ranges = file_def_ranges.get(rel) or []
    if ranges and hint_line:
        for lo, hi, name, recv in ranges:
            if lo <= hint_line <= hi:
                healed_from, func_name = func_name, name
                break
    if healed_from is None and ranges and len(func_defs.get(func_name, [])) == 0:
        # fall back to the closest def above the hint line, else the file's first def
        best = None
        for lo, hi, name, recv in ranges:
            if hint_line and lo <= hint_line:
                best = name
        if best:
            healed_from, func_name = func_name, best

branch_truncations = []


def rank(names, at, kind):
    kept = sorted(names, key=lambda n: (0 if n.lower() in ISSUE_TOKENS else 1, n))[:MAX_BRANCH]
    if len(names) > len(kept):
        dropped = [n for n in names if n not in kept]
        branch_truncations.append({'at': at, 'kind': kind, 'dropped': dropped[:10]})
    return kept


def walk(start, graph, kind):
    paths = []

    def dfs(node, path):
        if len(paths) >= MAX_PATHS:
            return
        nxt = [n for n in graph.get(node, []) if n not in path]
        kept = rank(nxt, node, kind) if nxt else []
        if not kept or len(path) >= DEPTH + 1:
            if len(path) > 1:
                paths.append(path)
            return
        for n in kept:
            dfs(n, path + [n])

    dfs(start, [start])
    return paths

callee_paths = walk(func_name, callees_of, 'callees')
caller_paths = [list(reversed(p)) for p in walk(func_name, callers_of, 'callers')]

names_on = set([func_name])
for p in callee_paths + caller_paths:
    names_on.update(p)
locations = {}
for n in sorted(names_on):
    locs = [rel + ':' + str(lo) for rel, lo, recv in func_defs.get(n, [])[:2]]
    if locs:
        locations[n] = locs

ordered = [func_name]
for p in callee_paths + caller_paths:
    for n in p:
        if n not in ordered:
            ordered.append(n)
frame_sources, total = [], 0
for n in ordered:
    if len(frame_sources) >= MAX_FRAMES or total >= MAX_SRC_TOTAL:
        break
    defs = func_defs.get(n, [])
    if not defs:
        continue
    rel, lo, recv = defs[0]
    hi = lo
    for l0, h0, nm, rc in file_def_ranges.get(rel, []):
        if l0 == lo and nm == n:
            hi = h0
            break
    lines = file_lines.get(rel, [])
    body = ''.join(lines[lo - 1:min(hi, lo - 1 + MAX_FN_LINES)]).rstrip()[:MAX_FN_CHARS]
    if not body.strip():
        continue
    total += len(body)
    qual = ((recv + '.') if recv else '') + n
    frame_sources.append({'name': n, 'loc': rel + ':' + str(lo), 'qual': qual, 'code': body})

print(json.dumps({
    'target': func_name,
    'healed_from': healed_from,
    'defined': func_name in func_defs,
    'target_defs': [rel + ':' + str(lo) for rel, lo, recv in func_defs.get(func_name, [])],
    'depth': DEPTH,
    'callee_paths': callee_paths,
    'caller_paths': caller_paths,
    'locations': locations,
    'frame_sources': frame_sources,
    'dynamic_bridges': [],
    'branch_truncations': branch_truncations[:12],
}))
'''


# JS/TS call-chain tracer: same argv/env interface and JSON output contract as the Python and Go
# scripts. Regex grammar over .js/.jsx/.ts/.tsx/.mjs/.cjs (function declarations, const/let arrow and
# function expressions incl. annotations containing `=>`, class-property arrows, class/object method
# shorthand, `Obj.name = function` assignments, class/enum/interface nodes); bodies are brace-matched
# by a string/comment-aware scanner; JSX `<Component>` usage counts as a call edge. Written in the JS
# port (2026-09-01) with the 2026-09-02 audit fixes; lost from this file and restored 2026-09-16 --
# without it every JS/TS trace ran the Python tracer and graph localization built 0 nodes (J7).
JS_TRACE_CALL_CHAIN_SCRIPT = r'''
import json, os, re, sys

REPO = os.environ.get('REPO_PATH', '/testbed')
DEPTH = int(os.environ.get('CHAIN_DEPTH', '3'))
MAX_PATHS = int(os.environ.get('CHAIN_MAX_PATHS', '24'))
MAX_BRANCH = int(os.environ.get('CHAIN_MAX_BRANCH', '6'))
ISSUE_TOKENS = set(t for t in os.environ.get('CHAIN_ISSUE_TOKENS', '').lower().split(',') if t)
MAX_FRAMES = int(os.environ.get('CHAIN_SRC_FRAMES', '14'))
MAX_FN_LINES = int(os.environ.get('CHAIN_SRC_LINES', '40'))
MAX_FN_CHARS = int(os.environ.get('CHAIN_SRC_CHARS', '1400'))
MAX_SRC_TOTAL = int(os.environ.get('CHAIN_SRC_TOTAL', '12000'))
MAX_FILE_LINES = int(os.environ.get('CHAIN_MAX_FILE_LINES', '12000'))

func_name = sys.argv[1] if len(sys.argv) > 1 else ''
hint_file = sys.argv[2] if len(sys.argv) > 2 else ''
hint_line = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 0

ID = r'[A-Za-z_$][\w$]*'
DEF_RES = [
    # function declaration
    re.compile(r'^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(' + ID + r')\s*\('),
    # const name[: annotation] = [async] function / (..) => / x =>  -- the annotation is
    # matched lazily (.*?) because React types routinely contain `=>` and `=`
    # (`React.FC<{onClick: () => void}>` broke the old [^=]* form: 2026-09-02 audit).
    re.compile(r'^\s*(?:export\s+)?(?:const|let|var)\s+(' + ID + r')\s*(?::.*?)?=\s*(?:async\s*)?'
               r'(?:function\b|\(|' + ID + r'\s*=>)'),
    # class-property arrow/function member: `private onLoggedOut = async (): Promise<void> => {`
    # (element-web's dominant handler style; previously NOT indexed at all)
    re.compile(r'^\s+(?:(?:public|private|protected|readonly|static)\s+)*(' + ID + r')\s*'
               r'=\s*(?:async\s*)?\([^)]*\)\s*(?::.*?)?=>'),
    # Obj.name = [async] function / (..) =>   (CommonJS / prototype style)
    re.compile(r'^\s*(?:' + ID + r'\.)+(' + ID + r')\s*=\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*=>)'),
    # name: [async] function / (..) =>   (object property)
    re.compile(r'^\s*(' + ID + r')\s*:\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*=>)'),
    # class / object method shorthand (TS modifiers allowed)
    re.compile(r'^\s*(?:(?:public|private|protected|static|async|readonly|override|get|set)\s+)*'
               r'(' + ID + r')\s*(?:<[^>]*>)?\s*\([^()]*(?:\([^()]*\)[^()]*)*\)\s*(?::\s*[^{;=]+?)?\s*\{'),
]
# class / enum / interface: indexed as TARGETABLE nodes too (a queried component-class or enum
# name used to come back "NO definition"; 2026-09-02 audit found 76 such existing symbols).
CLASS_RE = re.compile(r'^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?(?:class|enum|interface)\s+(' + ID + r')\b')
# KEYWORDS: the only names never indexable as DEFINITIONS. NOISE (below) additionally filters
# CALL matching -- applying it to defs made every React `render()` unfindable (10 unresolved
# runs queried `render` -> "NO definition").
# #1 components defined through a wrapper call. TOP-LEVEL only: an indented `const x = withY(...)` is almost
# always a data helper (measured on webclients: `with*` consts were overwhelmingly helper results), and `with*`
# counts only when its argument is a component or function expression.
WRAPPED_DEF_RE = re.compile(
    r'^(?:export\s+)?(?:const|let|var)\s+(' + ID + r')\s*(?::.*?)?=\s*'
    r'(?:(?:React\.)?(?:memo|forwardRef|lazy)|observer)\s*(?:<.*?>)?\s*\('
    r'|^(?:export\s+)?(?:const|let|var)\s+(' + ID + r')\s*(?::.*?)?=\s*with[A-Z][\w$]*\s*\('
    r'(?=\s*(?:function\b|async\b|\(|[A-Z][\w$]*\s*[,)]))')
INNER_FN_RE = re.compile(r'(?<![\w$.])function\s*\*?\s*(' + ID + r')\s*\(')
# anonymous default export: indexed under the file's stem (the name callers import it as)
ANON_DEFAULT_RE = re.compile(
    r'^\s*export\s+default\s+(?:async\s+)?(?:function\s*\*?\s*\(|\(|' + ID + r'\s*=>|class\s*(?:extends\b|\{)'
    r'|(?:React\.)?(?:memo|forwardRef)\s*\()')
# #2 method / class-property-arrow signatures whose parameter list spans lines (Prettier wraps long TS
# signatures). Only accepted after the matching `)` is found -- see multiline_def.
ML_METHOD_RE = re.compile(r'^\s*(?:(?:public|private|protected|static|async|readonly|override|get|set)\s+)*'
                          r'(' + ID + r')\s*(?:<[^>]*>)?\s*\(')
ML_PROP_ARROW_RE = re.compile(r'^\s+(?:(?:public|private|protected|readonly|static)\s+)*(' + ID + r')\s*'
                              r'(?::.*?)?=\s*(?:async\s*)?(?:<[^>]*>)?\s*\(')
# #5 dispatch: emit/fire of a key, and handler registration for a key (string-literal or identifier key)
KEY = r'(?:[\'"`]([^\'"`\n]+)[\'"`]|([A-Za-z_$][\w$]*(?:\.[\w$]+)*))'
EMIT_RE = re.compile(r'\.(?:emit|fire|trigger|publish)\(\s*' + KEY + r'\s*[,)]')
LISTEN_RE = re.compile(r'\.(?:on|once|addListener|prependListener|addEventListener|subscribe)\(\s*' + KEY +
                       r'\s*,\s*(?:(?:this|self)\.)?([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)?')
HOOK_REG_RE = re.compile(r'hook\s*:\s*[\'"`]([^\'"`\n]+)[\'"`]\s*,\s*method\s*:\s*(?:(?:this|self)\.)?'
                         r'([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)?')
MAX_BRIDGE_FANOUT = int(os.environ.get('CHAIN_MAX_BRIDGE_FANOUT', '6'))
KEYWORDS = {'if', 'for', 'while', 'switch', 'catch', 'function', 'return', 'typeof', 'new',
            'await', 'async', 'do', 'else', 'try', 'throw', 'yield', 'delete', 'void', 'in',
            'of', 'instanceof', 'with', 'constructor', 'super'}
CALL_RE = re.compile(r'(?<![\w$.])(?:this\.|self\.)?(' + ID + r')\s*\(|\.(' + ID + r')\s*\(')
# JSX component usage is a call edge too: `<CurrentDeviceSection device={..}/>` (uppercase only)
JSX_RE = re.compile(r'<([A-Z][\w$]*)[\s/>]')
NOISE = {'if', 'for', 'while', 'switch', 'catch', 'function', 'return', 'typeof', 'new',
         'await', 'async', 'require', 'import', 'export', 'super', 'constructor', 'do', 'else',
         'try', 'throw', 'yield', 'delete', 'void', 'in', 'of', 'instanceof', 'with',
         'describe', 'it', 'test', 'expect', 'beforeEach', 'afterEach', 'beforeAll', 'afterAll',
         'log', 'error', 'warn', 'info', 'debug', 'push', 'pop', 'shift', 'map', 'filter',
         'forEach', 'reduce', 'find', 'some', 'every', 'join', 'split', 'slice', 'splice',
         'concat', 'indexOf', 'includes', 'keys', 'values', 'entries', 'assign', 'freeze',
         'then', 'finally', 'resolve', 'reject', 'all', 'race', 'toString', 'trim',
         'replace', 'match', 'exec', 'parse', 'stringify', 'bind', 'call', 'apply',
         'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval', 'String', 'Number',
         'Boolean', 'Array', 'Object', 'Promise', 'Error', 'Date', 'Map', 'Set', 'Symbol',
         'JSON', 'Math', 'parseInt', 'parseFloat', 'isNaN', 'encodeURIComponent',
         'decodeURIComponent', 'useState', 'useEffect', 'useCallback', 'useMemo', 'useRef',
         'useContext', 'render', 'createElement', 'cloneElement', 'get', 'set', 'has', 'add',
         'clear', 'size', 'length'}
SKIP_DIRS = {'.git', 'node_modules', 'dist', 'build', 'coverage', '.next', '.cache', 'vendor',
             '__snapshots__', '__mocks__', '__tests__', 'test', 'tests', 'cypress', 'e2e',
             'fixtures', '_fixtures'}
TEST_FILE_RE = re.compile(r'(?:[.-]test|\.spec)\.[cm]?[jt]sx?$')
EXTS = ('.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs')


def body_end(lines, lo):
    """1-based line where the block opened on/after line ``lo`` closes (string/comment aware).
    Returns ``lo`` for brace-less one-liners (`const f = x => x + 1`)."""
    depth, pdepth, opened, state = 0, 0, False, 'code'
    for i in range(lo - 1, min(len(lines), lo - 1 + 4000)):
        ln = lines[i]
        j = 0
        while j < len(ln):
            c = ln[j]
            nxt = ln[j + 1] if j + 1 < len(ln) else ''
            if state == 'code':
                if c == '/' and nxt == '/':
                    break
                if c == '/' and nxt == '*':
                    state = 'block'
                    j += 2
                    continue
                if c in ('"', "'"):
                    state = c
                elif c == '`':
                    state = '`'
                elif c == '{':
                    depth += 1
                    opened = True
                elif c == '}':
                    depth -= 1
                elif c == '(':
                    pdepth += 1
                elif c == ')':
                    pdepth -= 1
            elif state == 'block':
                if c == '*' and nxt == '/':
                    state = 'code'
                    j += 2
                    continue
            elif state in ('"', "'"):
                if c == '\\':
                    j += 2
                    continue
                if c == state:
                    state = 'code'
            elif state == '`':
                if c == '\\':
                    j += 2
                    continue
                if c == '`':
                    state = 'code'
            j += 1
        if state in ('"', "'"):
            state = 'code'  # unterminated quote on this line: recover
        # depth is judged at END of line, not at the first closing brace: destructured params and inline
        # object types open AND close braces before the body's own `{` on the same line
        # (`({ a, b }: Props) => {`), and an early return there ended every such React component at
        # its signature line, dropping all of its callee edges (2026-09-16).
        if opened and depth <= 0:
            return i + 1
        # the brace-less one-liner rules apply only OUTSIDE parentheses: a parameter list spanning lines
        # (`loadRooms(\n force: boolean,\n ...\n): Promise<void> {`) ends its lines with `,` and has no `{`
        # for several lines, and both rules used to end the body inside the signature (2026-09-16)
        if not opened and pdepth <= 0 and i > lo - 1 and ln.rstrip().endswith((';', ',')):
            return i + 1
        if not opened and pdepth <= 0 and i - (lo - 1) >= 3:
            return lo
        if not opened and i - (lo - 1) >= 40:    # an unbalanced `(` (e.g. in a regex literal) must not run away
            return lo
    return lo if not opened else min(len(lines), lo - 1 + 4000)


def multiline_def(lines, i, m, arrow):
    """#2: a signature whose `(` opens on line i (1-based) and closes on a LATER line is a definition only when the
    matching `)` is followed by `[: ReturnType] {` (method) or `[: ReturnType] =>` (property arrow). A multi-line
    CALL such as `dispatch(` ... `);` or `return (` ... `);` fails that test."""
    row, col, depth, state = i - 1, m.end() - 1, 0, 'code'
    while row < len(lines) and row < i - 1 + 25:
        s = lines[row]
        while col < len(s):
            c = s[col]
            if state == 'code':
                if c in ('"', "'", '`'):
                    state = c
                elif c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        if row == i - 1:
                            return False      # closes on the same line: the single-line patterns own it
                        rest = s[col + 1:]
                        rx = r'\s*(?::.*?)?=>' if arrow else r'\s*(?::\s*[^;=]*?)?\s*\{'
                        return re.match(rx, rest) is not None
            elif c == '\\':
                col += 1
            elif c == state:
                state = 'code'
            col += 1
        if state in ('"', "'"):
            state = 'code'
        row += 1
        col = 0
    return False


func_defs = {}         # name -> [(rel, line, recv)]
file_def_ranges = {}   # rel -> [(lo, hi, name, recv)]
file_lines = {}

for root, dirs, files in os.walk(REPO):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]
    for fn in files:
        if not fn.endswith(EXTS) or fn.endswith('.d.ts') or fn.endswith('.min.js'):
            continue
        if TEST_FILE_RE.search(fn):
            continue
        full = os.path.join(root, fn)
        rel = os.path.relpath(full, REPO)
        try:
            with open(full, errors='replace') as fh:
                lines = fh.readlines()
        except Exception:
            continue
        if len(lines) > MAX_FILE_LINES:
            continue
        defs = []
        class_stack = []   # [(name, close_line)]
        class_decl_lines = set()
        anon_default_lines = []
        for i, ln in enumerate(lines, 1):
            while class_stack and i > class_stack[-1][1]:
                class_stack.pop()
            cm = CLASS_RE.match(ln)
            if cm:
                class_stack.append((cm.group(1), body_end(lines, i)))
                defs.append((i, cm.group(1), ''))   # the class/enum itself is a queryable node
                class_decl_lines.add(i)             # ...but contributes NO body/edges (below)
                continue
            names = []
            wm = WRAPPED_DEF_RE.match(ln)
            if wm:                                           # #1 memo/forwardRef/observer/withX(...)
                names.append(wm.group(1) or wm.group(2))
                im = INNER_FN_RE.search(ln, wm.end())        # withX(function Inner() {...}): Inner too
                if im:
                    names.append(im.group(1))
            else:
                for rx in DEF_RES:
                    m = rx.match(ln)
                    if m:
                        names.append(m.group(1))
                        break
            if not names:                                    # #2 multi-line signatures
                m = ML_PROP_ARROW_RE.match(ln)
                if m and m.group(1) not in KEYWORDS and multiline_def(lines, i, m, True):
                    names.append(m.group(1))
                else:
                    m = ML_METHOD_RE.match(ln)
                    if m and m.group(1) not in KEYWORDS and multiline_def(lines, i, m, False):
                        names.append(m.group(1))
            if not names and ANON_DEFAULT_RE.match(ln):
                anon_default_lines.append(i)
            recv = class_stack[-1][0] if class_stack else ''
            for name in dict.fromkeys(names):
                if name and name not in KEYWORDS:
                    defs.append((i, name, recv))
        if anon_default_lines:                               # #1 anonymous default export -> file stem
            stem = fn.rsplit('.', 1)[0]
            if stem in ('index', 'main'):
                stem = os.path.basename(root)
            if re.fullmatch(ID, stem) and stem not in {n for _l, n, _r in defs}:
                defs.append((anon_default_lines[0], stem, ''))
                defs.sort(key=lambda d: d[0])
        if not defs:
            continue
        file_lines[rel] = lines
        ranges = []
        for j, (lo, name, recv) in enumerate(defs):
            # a class/enum node is TARGETABLE but bodiless: spanning the whole class would
            # register every inner call as the class's own edge (mega-node).
            hi = lo if lo in class_decl_lines else body_end(lines, lo)
            if hi < lo:
                hi = lo
            ranges.append((lo, hi, name, recv))
            func_defs.setdefault(name, []).append((rel, lo, recv))
        file_def_ranges[rel] = ranges

callees_of = {}
callers_of = {}
for rel, ranges in file_def_ranges.items():
    lines = file_lines[rel]
    for lo, hi, name, recv in ranges:
        # the definition's OWN line is part of the body: a one-line function (`function f(p) { return g(p); }`)
        # has all of its calls there, and they were never seen (2026-09-16). The def's own name on that line is
        # still excluded below (`t == name`).
        body = ''.join(lines[lo - 1:hi])
        for hit in CALL_RE.findall(body) + JSX_RE.findall(body):
            t = (hit[0] or hit[1]) if isinstance(hit, tuple) else hit
            if not t or t in NOISE or t == name or t not in func_defs:
                continue
            lst = callees_of.setdefault(name, [])
            if t not in lst:
                lst.append(t)
            cl = callers_of.setdefault(t, [])
            if name not in cl:
                cl.append(name)

# #5 dispatch edges: a function that EMITS/FIRES a key reaches the handlers REGISTERED for that key although no
# literal call links them: EventEmitter .emit/.on, NodeBB plugins.hooks.fire + hooks.register({hook, method}), and
# NodeBB socket.io (client socket.emit('user.follow') -> server SocketUser.follow in src/socket.io/user.js).
# A key with more than MAX_BRIDGE_FANOUT handlers is too generic to bridge.
import bisect
emitters, listeners = {}, {}          # key -> [emitting def]; key -> [(handler, via)]


def _handler(chain, registrant):
    parts = [p for p in (chain or '').split('.') if p]
    if parts and parts[-1] in ('bind', 'call', 'apply'):
        parts = parts[:-1]
    h = parts[-1] if parts else ''
    return h if h and h not in KEYWORDS and h not in ('async', 'function') else registrant


def _add(table, key, ent):
    lst = table.setdefault(key, [])
    if ent not in lst:
        lst.append(ent)


# Whole-file scan, each site attributed to its INNERMOST enclosing definition: a registration inside an
# unindexed `constructor` still registers its named handler, and a file-wide wrapper such as NodeBB's
# `module.exports = function (Posts) {` does not claim every emit and registration in the file.
for rel, ranges in file_def_ranges.items():
    flines = file_lines[rel]
    text = ''.join(flines)
    starts = [0]
    for ln in flines:
        starts.append(starts[-1] + len(ln))

    def enclosing(pos, ranges=ranges, starts=starts):
        line = bisect.bisect_right(starts, pos)
        best = None
        for lo, hi, name, recv in ranges:
            if lo <= line <= hi and (best is None or hi - lo < best[1] - best[0]):
                best = (lo, hi, name)
        return best[2] if best else None

    for m in EMIT_RE.finditer(text):
        em = enclosing(m.start())
        if em:
            _add(emitters, m.group(1) or m.group(2), em)
    for m in LISTEN_RE.finditer(text):
        h = _handler(m.group(3), None) or enclosing(m.start())
        if h:
            _add(listeners, m.group(1) or m.group(2), (h, 'event'))
    for m in HOOK_REG_RE.finditer(text):
        h = _handler(m.group(2), None) or enclosing(m.start())
        if h:
            _add(listeners, m.group(1), (h, 'hook'))

SOCKET_ROOT = 'src/socket.io'
if os.path.isdir(os.path.join(REPO, SOCKET_ROOT)):
    for key in emitters:
        parts = key.split('.')
        if len(parts) >= 2 and re.fullmatch(ID, parts[-1]):
            prefix = SOCKET_ROOT + '/' + parts[0]
            if any(rel.startswith(prefix) for rel, _lo, _rc in func_defs.get(parts[-1], [])):
                ent = (parts[-1], 'socket.io')
                lst = listeners.setdefault(key, [])
                if ent not in lst:
                    lst.append(ent)

event_edges = []                      # (key, emitter, handler, via)
for key, ems in emitters.items():
    hs = listeners.get(key) or []
    if not hs or len(hs) > MAX_BRIDGE_FANOUT or len(ems) > 3 * MAX_BRIDGE_FANOUT:
        continue
    for em in ems:
        for h, via in hs:
            if h == em or h not in func_defs:
                continue
            lst = callees_of.setdefault(em, [])
            if h not in lst:
                lst.append(h)
            cl = callers_of.setdefault(h, [])
            if em not in cl:
                cl.append(em)
            event_edges.append((key, em, h, via))

healed_from = None
if func_name not in func_defs and hint_file:
    rel = hint_file.lstrip('/')
    ranges = file_def_ranges.get(rel) or []
    if ranges and hint_line:
        for lo, hi, name, recv in ranges:
            if lo <= hint_line <= hi:
                healed_from, func_name = func_name, name
                break
    if healed_from is None and ranges and len(func_defs.get(func_name, [])) == 0:
        best = None
        for lo, hi, name, recv in ranges:
            if hint_line and lo <= hint_line:
                best = name
        if best:
            healed_from, func_name = func_name, best

branch_truncations = []


def rank(names, at, kind):
    kept = sorted(names, key=lambda n: (0 if n.lower() in ISSUE_TOKENS else 1, n))[:MAX_BRANCH]
    if len(names) > len(kept):
        dropped = [n for n in names if n not in kept]
        branch_truncations.append({'at': at, 'kind': kind, 'dropped': dropped[:10]})
    return kept


def walk(start, graph, kind):
    paths = []

    def dfs(node, path):
        if len(paths) >= MAX_PATHS:
            return
        nxt = [n for n in graph.get(node, []) if n not in path]
        kept = rank(nxt, node, kind) if nxt else []
        if not kept or len(path) >= DEPTH + 1:
            if len(path) > 1:
                paths.append(path)
            return
        for n in kept:
            dfs(n, path + [n])

    dfs(start, [start])
    return paths

callee_paths = walk(func_name, callees_of, 'callees')
caller_paths = [list(reversed(p)) for p in walk(func_name, callers_of, 'callers')]

names_on = set([func_name])
for p in callee_paths + caller_paths:
    names_on.update(p)
locations = {}
for n in sorted(names_on):
    locs = [rel + ':' + str(lo) for rel, lo, recv in func_defs.get(n, [])[:2]]
    if locs:
        locations[n] = locs

ordered = [func_name]
for p in callee_paths + caller_paths:
    for n in p:
        if n not in ordered:
            ordered.append(n)
frame_sources, total = [], 0
for n in ordered:
    if len(frame_sources) >= MAX_FRAMES or total >= MAX_SRC_TOTAL:
        break
    defs = func_defs.get(n, [])
    if not defs:
        continue
    rel, lo, recv = defs[0]
    hi = lo
    for l0, h0, nm, rc in file_def_ranges.get(rel, []):
        if l0 == lo and nm == n:
            hi = h0
            break
    lines = file_lines.get(rel, [])
    body = ''.join(lines[lo - 1:min(hi, lo - 1 + MAX_FN_LINES)]).rstrip()[:MAX_FN_CHARS]
    if not body.strip():
        continue
    total += len(body)
    qual = ((recv + '.') if recv else '') + n
    frame_sources.append({'name': n, 'loc': rel + ':' + str(lo), 'qual': qual, 'code': body})

def _loc(n):
    d = func_defs.get(n) or []
    return (d[0][0] + ':' + str(d[0][1])) if d else ''


def _site(n):
    d = func_defs.get(n) or []
    return [n, d[0][0], d[0][2]] if d else [n, '', '']


# report the bridges that touch this chain, those involving the traced function itself first: with name-level
# matching a generic on-path name (init, get, end) otherwise fills the 8 slots with unrelated edges
dynamic_bridges = []
for key, em, h, via in sorted(event_edges, key=lambda e: (0 if func_name in (e[1], e[2]) else 1)):
    if (em in names_on or h in names_on) and len(dynamic_bridges) < 8:
        rec = {'kind': 'event', 'key': key, 'via': via, 'from': em, 'to': h,
               'from_loc': _loc(em), 'to_loc': _loc(h)}
        if rec not in dynamic_bridges:
            dynamic_bridges.append(rec)

print(json.dumps({
    'target': func_name,
    'healed_from': healed_from,
    'defined': func_name in func_defs,
    'target_defs': [rel + ':' + str(lo) for rel, lo, recv in func_defs.get(func_name, [])],
    'depth': DEPTH,
    'callee_paths': callee_paths,
    'caller_paths': caller_paths,
    'callee_site_paths': [[_site(n) for n in p] for p in callee_paths],
    'caller_site_paths': [[_site(n) for n in p] for p in caller_paths],
    'locations': locations,
    'frame_sources': frame_sources,
    'dynamic_bridges': dynamic_bridges,
    'branch_truncations': branch_truncations[:12],
}))
'''


_MAX_RENDER_PATHS = 20  # cap caller/callee paths shown in the rendered chain


def _format_chain(data: dict, function_name: str) -> str:
    """Render the path-based call chain into a transcript the sub-agent can simulate over.

    Shows the caller paths that REACH the target (``B -> C -> D -> A``) and the callee paths the
    target INVOKES (``A -> E -> F -> G``), plus a file:line map so the agent knows where each
    function lives.
    """
    import json as _json

    if isinstance(data, str):
        data = _json.loads(data)
    target = data.get("target", function_name)
    depth = data.get("depth", 3)
    caller_paths = data.get("caller_paths", []) or []
    callee_paths = data.get("callee_paths", []) or []
    locations = data.get("locations", {}) or {}

    out = [f"=== CALL CHAIN for {target} (callers <={depth} levels  ->  {target}  ->  callees <={depth} levels) ==="]
    healed_from = data.get("healed_from")
    if healed_from:
        out.append(
            f"NOTE: the reported name '{healed_from}' has no definition in the repo; resolved to "
            f"the actual enclosing function '{target}' at the reported file:line."
        )
    defs = data.get("target_defs") or []
    if defs:
        out.append(f"target defined at: {', '.join(defs[:3])}")
    if not data.get("defined", True):
        kind = data.get("symbol_kind") or ""
        sdefs = ", ".join((data.get("symbol_defs") or [])[:3])
        refs = data.get("referencing_functions") or []
        if kind == "data":
            out.append(
                f"NOTE: '{target}' is not a function -- it is a module-level DATA structure "
                f"(dict/constant/registry) defined at {sdefs}. It EXISTS in the repo; do NOT "
                f"treat it as a symbol that must be added. Call paths cannot route through a "
                f"data symbol, so reason from its definition and the functions that read or "
                f"write it (listed below)."
            )
        elif kind == "class":
            out.append(
                f"NOTE: '{target}' is a CLASS defined at {sdefs}; it has no function node of "
                f"its own on the call graph. It EXISTS in the repo; reason from its definition "
                f"and the functions that reference or instantiate it (listed below)."
            )
        else:
            out.append(
                f"NOTE: '{target}' has NO definition in the repo -- it may be a method/function that must "
                f"be ADDED to fix the bug, so its call paths are sparse. Reason about where it SHOULD be "
                f"invoked from (its would-be callers) and what it must call."
            )
            if kind == "absent":
                out.append(
                    "(verified by repo scan: no function, class, or module-level assignment "
                    "with this name exists -- unless the fix must CREATE it, the name is likely "
                    "invented and should not be trusted as an edit site.)"
                )
        if refs:
            out.append("")
            out.append(f"REFERENCING FUNCTIONS (pseudo-callers -- these read/write/instantiate {target}):")
            for r in refs[:12]:
                out.append("  " + r)
        src = (data.get("symbol_source") or "").rstrip()
        if src:
            out.append("")
            out.append(f"DEFINITION SOURCE of {target} ({sdefs}):")
            out.append(src)

    out.append("")
    out.append(f"CALLER PATHS (execution reaches {target} via):")
    if caller_paths:
        for p in caller_paths[:_MAX_RENDER_PATHS]:
            out.append("  " + " -> ".join(p))
        if len(caller_paths) > _MAX_RENDER_PATHS:
            out.append(f"  ... (+{len(caller_paths) - _MAX_RENDER_PATHS} more caller paths)")
    else:
        out.append("  (none found)")

    out.append("")
    out.append(f"CALLEE PATHS ({target} invokes, in order):")
    if callee_paths:
        for p in callee_paths[:_MAX_RENDER_PATHS]:
            out.append("  " + " -> ".join(p))
        if len(callee_paths) > _MAX_RENDER_PATHS:
            out.append(f"  ... (+{len(callee_paths) - _MAX_RENDER_PATHS} more callee paths)")
    else:
        out.append("  (none found)")

    truncs = data.get("branch_truncations") or []
    if truncs:
        out.append("")
        out.append(
            "BRANCH-CAP TRUNCATIONS (these call edges EXIST but were NOT expanded above because "
            "the node's fan-out exceeded the cap -- if one looks relevant to the issue, it may be "
            "the missing producer; trace it explicitly):"
        )
        for t in truncs[:8]:
            dropped = ", ".join(t.get("dropped", [])[:10])
            out.append(f"  {t.get('at', '?')} also has unexpanded {t.get('kind', '?')}: {dropped}")

    bridges = data.get("dynamic_bridges") or []
    # JS/TS event/hook/socket edges carry kind="event"; Python registry bridges are rendered exactly as before.
    event_bridges = [b for b in bridges if b.get("kind") == "event"]
    bridges = [b for b in bridges if b.get("kind") != "event"]
    if event_bridges:
        out.append("")
        out.append(
            "EVENT-DISPATCH EDGES (no literal call links these functions: the first EMITS/FIRES the key and the "
            "second is REGISTERED to handle it, so it runs as a consequence. These edges are already part of "
            "the paths above):"
        )
        for b in event_bridges:
            out.append(
                f"  {b.get('from', '?')} ({b.get('from_loc') or '?'}) --[{b.get('via', 'event')} "
                f"'{b.get('key', '?')}']--> {b.get('to', '?')} ({b.get('to_loc') or '?'})"
            )
    if bridges:
        out.append("")
        out.append(
            "DYNAMIC-DISPATCH BRIDGES (these functions appear as a literal call on NO path above, "
            "but they DO run: Python dispatches to them through a registry when objects of the "
            "classes below are constructed or combined with operators. Treat them as PRODUCER "
            "candidates -- the wrong value may be FIRST CONSTRUCTED inside one of them):"
        )
        for b in bridges:
            fns = ", ".join(
                (f.get("name", "?") + (f" ({f['loc']})" if f.get("loc") else ""))
                for f in b.get("funcs", [])
            )
            out.append(
                f"  registry '{b.get('registry')}' (registered at "
                f"{', '.join(b.get('stored_at', []) or ['?'])}; active here because "
                f"{b.get('why', '?')}) dispatches to: {fns}"
            )

    # Machine-readable, SITE-qualified duplicate of the two path sections above. Off by default:
    # it is not for the model to read, it exists so a graph-building consumer can tell
    # ``a/cli.py::GalaxyCLI.__init__`` from ``a/api.py::GalaxyAPI.__init__`` instead of merging
    # them into one node. localize_graph sets CHAIN_QUALIFIED_PATHS=1 and strips the section
    # after parsing, so the stock pipeline's rendered chain is unchanged.
    if os.environ.get("CHAIN_QUALIFIED_PATHS") == "1":
        qual_lines = []
        for kind, key in (("CALLER", "caller_site_paths"), ("CALLEE", "callee_site_paths")):
            for p in (data.get(key) or [])[:_MAX_RENDER_PATHS]:
                hops = []
                for s in p:
                    name, sfile, scls = (list(s) + ["", ""])[:3]
                    hops.append(f"{sfile}::{scls + '.' if scls else ''}{name}")
                if len(hops) >= 2:
                    qual_lines.append(f"  {kind}: " + " -> ".join(hops))
        if qual_lines:
            out.append("")
            out.append("QUALIFIED SITE PATHS (machine-readable; node = <file>::<Class.name>):")
            out.extend(qual_lines)

    if locations:
        out.append("")
        out.append("FUNCTION LOCATIONS:")
        for name in sorted(locations):
            locs = locations[name]
            if locs:
                out.append(f"  {name}: {', '.join(locs)}")

    # Real source of the chain frames -- so the analyst VERIFIES what each function does (which one
    # actually constructs the wrong value) instead of guessing the role from the name.
    sources = data.get("frame_sources") or []
    if sources:
        out.append("")
        out.append("SOURCE OF KEY CALL-CHAIN FRAMES (read these to see what each frame ACTUALLY "
                   "does -- do NOT guess a function's behaviour from its name):")
        for s in sources:
            out.append(f"\n--- {s.get('qual', s.get('name', '?'))}  ({s.get('loc', '?')}) ---")
            out.append(s.get("code", "").rstrip())
    return "\n".join(out)


def make_trace_call_chain_runner(env, *, repo_path: str = "/testbed", timeout: int = 120, depth: int = 3):


    def trace(function_name: str, file_path: str = "", line: int = 0) -> str:
        if env is None:
            return "[trace_call_chain unavailable: no execution environment]"
        hint = (file_path or "").strip()
        if hint:
            rp = repo_path.rstrip("/") + "/"
            if hint.startswith(rp):
                hint = hint[len(rp):]
            hint = hint.lstrip("/")
        argv = shlex.quote(function_name)
        if hint:
            # A line hint lets the script self-heal a non-existent function name to its real
            # enclosing def; it is positional, so only pass it when we also have a file.
            argv += " " + shlex.quote(hint)
            if line and int(line) > 0:
                argv += " " + shlex.quote(str(int(line)))
        # Forward any CHAIN_* tuning vars the host set, so the in-container script can be told to
        # keep longer frame bodies / wider chains than its defaults.
        forwarded = ""
        for k in ("CHAIN_SRC_LINES", "CHAIN_SRC_CHARS", "CHAIN_SRC_TOTAL", "CHAIN_SRC_FRAMES",
                  "CHAIN_MAX_PATHS", "CHAIN_MAX_BRANCH", "CHAIN_ISSUE_TOKENS"):
            v = os.environ.get(k)
            if v:
                forwarded += f"{k}={shlex.quote(v)} "
        script = (GO_TRACE_CALL_CHAIN_SCRIPT if is_go() else
                  JS_TRACE_CALL_CHAIN_SCRIPT if is_js() else TRACE_CALL_CHAIN_SCRIPT)
        command = (
            f"REPO_PATH={shlex.quote(repo_path)} CHAIN_DEPTH={int(depth)} {forwarded}python3 - {argv} "
            f"<<'__MSWEA_TRACE_EOF__'\n{script}\n__MSWEA_TRACE_EOF__\n"
        )
        try:
            result = env.execute({"command": command}, timeout=timeout)
        except Exception as e:  # Submitted/timeouts/etc -- never let the tool kill the run
            return f"[trace_call_chain failed: {type(e).__name__}: {e}]"
        stdout = (result.get("output") or "").strip()
        if result.get("returncode", 0) != 0 or not stdout:
            return f"[trace_call_chain produced no usable output: {stdout[:300] or 'empty'}]"
        # The script prints exactly one JSON line; tolerate banner noise by taking the last line.
        line = stdout.splitlines()[-1].strip()
        start = line.find("{")
        try:
            import json as _json

            data = _json.loads(line[start:] if start != -1 else line)
        except Exception:
            return f"[trace_call_chain output not JSON: {stdout[:300]}]"
        return _format_chain(data, function_name)

    return trace


#: clip window for the problem statement in every prompt. SWE-bench Pro statements embed the
#: graded Requirements + interface spec in their TAIL (up to ~11k chars); head-clipping below
#: that silently drops the spec the hidden tests check, so the window must cover the whole text.
PS_CLIP = int(os.getenv("PS_CLIP", "12000"))


def _clip(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + "\n...[truncated]..."


def _clip_tail(text: str, n: int) -> str:
    """Keep the LAST ``n`` chars (most recent context wins for an in-progress investigation)."""
    text = text or ""
    return text if len(text) <= n else "...[earlier history truncated]...\n" + text[-n:]


#: max chars kept per single observation block, and for the whole transcript.
_HISTORY_OBS_CLIP = 1200
_HISTORY_TOTAL_CLIP = 14000


def _format_history(messages: list, *, skip_first_user: bool = True) -> str:

    lines: list[str] = []
    step = 0
    seen_first_user = False
    for m in messages or []:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            if skip_first_user and not seen_first_user:
                seen_first_user = True
                continue
            seen_first_user = True
            content = m.get("content")
            content = content if isinstance(content, str) else extract_observation([m])
            if content and content.strip():
                lines.append(f"[note] {_clip(content.strip(), _HISTORY_OBS_CLIP)}")
        elif role == "assistant":
            step += 1
            thought = extract_thought(m).strip()
            if thought.lower() in ("", "null"):
                thought = "(no thought text; tool call only)"
            command = extract_command(m).strip()
            block = [f"[agent step {step}] THOUGHT: {_clip(thought, _HISTORY_OBS_CLIP)}"]
            if command:
                block.append(f"  COMMAND: {_clip(command, 600)}")
            lines.append("\n".join(block))
        elif role == "tool":
            obs = extract_observation([m]).strip()
            if obs:
                lines.append(f"  OBSERVATION:\n{_clip(obs, _HISTORY_OBS_CLIP)}")
    transcript = "\n".join(lines).strip()
    return _clip_tail(transcript, _HISTORY_TOTAL_CLIP) if transcript else "(no prior history)"


_LOCATE_PROMPT = """\
You are the BUG-LOCALIZATION step of an automated reproduction sub-agent for a software issue.
Read the issue AND the main agent's investigation so far, then name the SINGLE most likely
function (or method) where the bug lives, so we can extract its call chain next. Prefer a real
source function named or strongly implied by the issue or surfaced by the agent's exploration
(a traceback frame, an API the issue calls, a function the agent just grepped/read).

IMPORTANT -- prefer the PRODUCER, not the symptom surface: if the issue describes a WRONG / EXTRA /
MISSING VALUE in some output (a stray ``None``, a bad field, a "None" string, an incorrect
URL/number), name the function that CONSTRUCTS / PARSES that value (the data origin -- often a
resolve / parse / build / ``groupdict`` / ``__init__`` step), NOT the outer API the issue calls
where the wrong value is merely rendered or reported. The wrapper the user invokes is usually the
symptom; the producer upstream is usually the bug.

=== ISSUE (problem statement) ===
{problem_statement}

=== WHAT THE MAIN AGENT HAS ALREADY DONE (its prior commands + observations, up to now) ===
{history}

A WARNING on three systematic traps, then the format:
  - The issue author's SUGGESTED FIX SITE (an "ISTM we could fix X", an inline diff) and the
    DEEPEST TRACEBACK FRAME are both HYPOTHESES, not ground truth -- reporters name a plausible
    site, and a traceback names where the crash SURFACED, which is often machinery consuming a
    value that some other function built.
  - MULTI-FAULT ISSUES: when the issue enumerates SEVERAL distinct broken behaviors (a
    Requirements list, several bullet points -- e.g. "aliases don't map" AND "binding is not
    greedy" AND "codes aren't normalized"), each behavior can live in a DIFFERENT function. The
    ALT slots below are then NOT optional second guesses: name one candidate per REMAINING
    distinct behavior (the function that implements THAT behavior), so every described fault gets
    a call chain. Answering "none" while the issue lists multiple behaviors is WRONG. Only a
    genuinely single-fault issue may use the ALT slots for competing hypotheses instead: prefer
    (a) an UPSTREAM producer in a DIFFERENT file whose output the primary consumes, and (b) when
    the issue shows a traceback or proposes a fix site, one candidate OUTSIDE those frames.
  - Name functions that EXIST in the repository's SOURCE now (surfaced by the exploration, a
    grep, a read file). A name that only appears in a TEST file or in the issue's wished-for API
    may not exist yet -- if you must name one, put it in an ALT slot, never as FUNCTION.
  Their call chains will be extracted too, and a later simulation step decides between the
  candidates from their real source.

Respond in EXACTLY this format, nothing else:
FUNCTION: <bare function or method name, no parentheses, no module path>
FILE: <relative path to the file if you can infer it, else: unknown>
LINE: <approximate line number in that file where the bug code lives, else: unknown>
REASON: <one sentence on why this is the likely bug site>
ALT-FUNCTION-2: <function> | <relative path or unknown> | <one-phrase hypothesis or the distinct behavior it implements; or: none>
ALT-FUNCTION-3: <function> | <relative path or unknown> | <one-phrase hypothesis or the distinct behavior it implements; or: none>
ALT-FUNCTION-4: <function> | <relative path or unknown> | <one-phrase hypothesis or the distinct behavior it implements; or: none>
"""

# ``ALT-FUNCTION-2: func | path | hypothesis`` lines from the LOCATE answer (markdown-tolerant).
_ALT_LINE_RE = re.compile(r"^[\s>*\-`#]*ALT[-_ ]?FUNCTION[-_ ]?\d*\s*:\s*(.+?)\s*$",
                          re.IGNORECASE | re.MULTILINE)

# Multi-candidate LOCATE: trace the alternate candidates' call chains and show them to
# SIMULATE/PLAN as competing hypotheses (the PRODUCER CHECK arbitrates from real source).
MULTI_LOCATE = os.getenv("MULTI_LOCATE", "1") in ("1", "true", "yes")
MULTI_LOCATE_MAX = int(os.getenv("MULTI_LOCATE_MAX", "3"))
# Each alternate chain is clipped before merging so the primary chain stays dominant.
ALT_CHAIN_CLIP = int(os.getenv("ALT_CHAIN_CLIP", "5000"))

# Alternate-candidate SIMULATION: the stock SIMULATE step is contractually anchored to the
# PRIMARY candidate ("pick the caller path that reaches {function}"), so an alternate's chain is
# only ever read, never EXECUTED -- and a hypothesis nobody traces can never win arbitration
# (14155: '__repr__' was nominated, merged, and still went unsimulated in every run). Give each
# merged alternate its own short execution trace, ending in a parseable verdict the edit-site
# union downstream can act on.
ALT_SIM = os.getenv("ALT_SIM", "1") in ("1", "true", "yes")
ALT_SIM_MAX = int(os.getenv("ALT_SIM_MAX", "2"))


_INPUT_MATRIX_PROMPT = """\
You are the INPUT-MATRIX step of an automated bug-localization sub-agent. A primary execution
simulation has already been run on this issue using ONE simple/default concrete input. Your job:
identify up to 3 DISTINCT input CLASSES that the issue text itself calls for but that a simple
default input would NOT cover -- each will be handed to a separate follow-up simulation.

=== ISSUE (problem statement) ===
{problem_statement}

Look for input classes the text quantifies or conditions over, for example: multiple/repeated
elements in one input ("multiple linkages", "several matches"), missing/unresolvable data or
mandated error cases, alternate formats/variants/branches ("both XML and binary"), empty or
boundary forms, special encodings/scripts, size or limit extremes, ordering/concurrency. Only
claim a class the issue text actually supports -- the quote must be COPIED VERBATIM from the
issue text above, it will be checked by exact substring match.

Respond with 1-3 lines in EXACTLY this format, most important first, nothing else:
INPUT-CLASS: <UPPERCASE-SHORT-LABEL> | <one sentence: how to construct such an input> | "<verbatim quote from the issue that demands this class>"

If the issue text implies no special input classes, respond with exactly: NONE
"""

_INPUT_CLASS_LINE_RE = re.compile(
    r"^[\s>*\-`#]*INPUT-CLASS(?:-\d+)?\s*:\s*([^|\n]+)\|([^|\n]+)\|\s*(.+?)\s*$", re.M)


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\\n", " ").replace("`", "").replace('"', "")
                  ).strip().lower()


def _input_directives_llm(query_fn, problem_statement: str,
                          log=lambda m: None) -> "list[tuple[str, str, str]]":
    """LLM-extracted (label, instruction, evidence-clause) input classes.

    Grounding: an entry survives only if its quote appears verbatim (whitespace/quote
    normalized) in the issue text -- hallucinated requirements are dropped. Deliberate NONE is
    respected (returns []); parse failure / all-dropped / exception falls back to the lexical
    harvest so the matrix never silently disappears on a bad roll.
    """
    try:
        reply = query_fn(_INPUT_MATRIX_PROMPT.format(
            problem_statement=_clip(problem_statement, PS_CLIP))) or ""
    except Exception as e:
        log(f"    [subagent] (3a) input-matrix LLM call failed ({type(e).__name__}); "
            "falling back to lexical harvest")
        return _input_directives_lexical(problem_statement)
    if re.search(r"^\s*NONE\s*$", reply, re.M) and not _INPUT_CLASS_LINE_RE.search(reply):
        return []  # deliberate NONE -- respect it, no fallback
    if not _INPUT_CLASS_LINE_RE.search(reply):
        log("    [subagent] (3a) input-matrix: unparseable LLM reply; "
            "falling back to lexical harvest")
        return _input_directives_lexical(problem_statement)
    ps_norm = _norm_ws(problem_statement)
    out, dropped, seen = [], 0, set()
    for m in _INPUT_CLASS_LINE_RE.finditer(reply):
        label = re.sub(r"[^A-Z0-9/ _-]", "", m.group(1).strip().upper()).strip() or "SPEC-CLASS"
        instruction = m.group(2).strip()
        clause = m.group(3).strip().strip('"').strip()
        if label in seen or not instruction or not clause:
            continue
        if _norm_ws(clause) not in ps_norm:
            dropped += 1
            continue
        seen.add(label)
        out.append((label, instruction, _clip(clause, 220)))
        if len(out) >= 3:
            break
    if dropped:
        log(f"    [subagent] (3a) input-matrix: dropped {dropped} ungrounded class(es) "
            "(quote not verbatim in issue)")
    if not out and _INPUT_CLASS_LINE_RE.search(reply):
        log("    [subagent] (3a) input-matrix: no LLM class survived grounding; "
            "falling back to lexical harvest")
        return _input_directives_lexical(problem_statement)
    # Top up with non-overlapping lexical hits: LLM extraction is open-world but
    # sampling-variable (observed on 111347e: one roll skipped the "multiple linkages" class
    # the regex family always catches). Union keeps both strengths; overlap = either clause
    # contains the other after normalization.
    if len(out) < 3:
        chosen = [_norm_ws(c) for _l, _i, c in out]
        for lex in _input_directives_lexical(problem_statement):
            lc = _norm_ws(lex[2])
            if any(lc in c or c in lc for c in chosen):
                continue
            out.append(lex)
            chosen.append(lc)
            log(f"    [subagent] (3a) input-matrix: topped up with lexical class {lex[0]}")
            if len(out) >= 3:
                break
    return out

_INPUT_QUANTIFIERS = [
    ("MULTIPLE-INSTANCE",
     r"\b(?:multiple|several|more than one|two or more|repeated|many)\b",
     "construct it with MULTIPLE instances of the quantified element (several entries/linkages/"
     "occurrences in ONE input), not just one"),
    ("MISSING/ERROR-CASE",
     r"\b(?:missing|absent|dangling|unresolvable|not\s+found|treated as an error|"
     r"raise[sd]?\s+(?:an?\s+)?\w*(?:error|exception))\b",
     "construct it so the referenced data is MISSING/unresolvable, and trace how the code "
     "(mis)handles that case"),
    ("OTHER-VARIANT",
     r"\b(?:both|either|consistently\s+(?:for|across))\b[^.\n]{0,80}"
     r"\b(?:formats?|parsers?|variants?|entries|implementations?|modes?|branch(?:es)?)\b",
     "use the OTHER format/variant/branch than the primary simulation traced"),
    ("EMPTY/BOUNDARY",
     r"\b(?:empty|zero-length|blank|no\s+\w+\s+(?:present|given|provided|at all))\b",
     "use the empty/boundary form of the input"),
]


def _input_directives_lexical(problem_statement: str, cap: int = 3) -> "list[tuple[str, str, str]]":
    """(label, instruction, evidence-clause) input classes the issue text quantifies over.

    Purely lexical: first hit per quantifier family, evidence = the containing line, priority
    order as listed (MULTIPLE first -- the classic under-simulated class). Some Pro problem
    statements carry literal "\\n" escapes instead of newlines -- treat those as line breaks
    so the evidence clause stays one sentence, not the whole document head.
    """
    text = (problem_statement or "").replace("\\n", "\n")
    out: "list[tuple[str, str, str]]" = []
    for label, pat, instruction in _INPUT_QUANTIFIERS:
        m = re.search(pat, text, re.IGNORECASE)
        if not m:
            continue
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        line = text[line_start:line_end if line_end != -1 else len(text)]
        # Narrow to the SENTENCE containing the match (a line can hold several sentences;
        # whole-line clauses defeat the LLM-union overlap dedupe downstream).
        off = m.start() - line_start
        clause, pos = line, 0
        for part in re.split(r"(?<=[.!?])\s+", line):
            if pos <= off < pos + len(part) + 1:
                clause = part
                break
            pos += len(part) + 1
        clause = clause.strip().lstrip("-*#> ").strip()
        out.append((label, instruction, _clip(clause, 220)))
        if len(out) >= cap:
            break
    return out


def _render_input_directive(label: str, instruction: str, clause: str) -> str:
    return (
        "\nINPUT DIRECTIVE (mandatory): the primary simulation already covered the default "
        f"input shape -- do not repeat it. Your CONCRETE INPUT must be a {label} input: "
        f"{instruction}.\nThis input class comes from the issue text: \"{clause}\"\n"
    )

_ALT_SIM_PROMPT = """\
You are the ALTERNATE-CANDIDATE SIMULATION step of an automated bug-localization sub-agent. An
earlier step flagged '{alt_func}' as a COMPETING hypothesis for where this bug should be fixed
{alt_hyp}. The primary suspect ('{function}') has already been simulated separately -- do NOT
re-trace it. Your job is to give THIS candidate its own execution trace, from its real source
below, and then judge it. Do NOT write or run any code -- this is a mental trace.

=== ISSUE (problem statement) ===
{problem_statement}

=== THE ALTERNATE CANDIDATE's call chain and source ===
{alt_chain}

Trace ONE concrete input from the issue (or invent a plausible one -- never refuse) through the
candidate's chain, hop by hop, grounding every step in the frame SOURCE above. Two things to
establish: (1) what '{alt_func}' actually does to the value the issue complains about, and
(2) whether the USER-VISIBLE wrong behaviour can be fixed here -- note that a CONSUMER/renderer
can be the right fix site even when the value it receives was built elsewhere (fixing how it is
rendered), so judge fix-suitability, not only where the value is constructed.
{input_directive}
Respond in EXACTLY these labelled sections, nothing else:
CONCRETE INPUT:
<the input>

EXECUTION SIMULATION:
<hop-by-hop trace through '{alt_func}'s chain; keep it under ~30 lines; mark any wrong value's
 construction with "FIRST CONSTRUCTED in <function> at <file:line>">

VERDICT: end with EXACTLY ONE of these two lines --
MUST-CHANGE: {alt_func} | <file>:<line> | <one phrase: what change here fixes the issue>
EXONERATED: <one phrase why no change is needed here>
"""

_ALT_SECTION_HDR_RE = re.compile(r"=== ALTERNATE CANDIDATE #\d+: '([^']+)'[^\n]*===\n")


def _alt_chain_section(call_chain: str, func: str, file: str = "") -> str:
    """The ``=== ALTERNATE CANDIDATE #k: 'func' ... ===`` section body merged for ``func``.

    ``file`` disambiguates same-named candidates (``A.__init__`` and ``B.__init__`` in different
    files are two distinct candidates): a section whose header names a DIFFERENT file is skipped,
    so the second candidate's simulation cannot silently re-read the first candidate's chain.
    """
    fallback = ""
    for m in _ALT_SECTION_HDR_RE.finditer(call_chain or ""):
        if m.group(1).split(".")[-1] != func.split(".")[-1]:
            continue
        nxt = _ALT_SECTION_HDR_RE.search(call_chain, m.end())
        body = call_chain[m.end():nxt.start() if nxt else len(call_chain)]
        hdr = m.group(0)
        if file and f"[{file}]" in hdr:
            return body
        if file and re.search(r"\[[^\]\n]+\]", hdr):
            continue          # header names some OTHER file -- not this candidate
        fallback = fallback or body
    return fallback


def _alt_candidates(locate_text: str) -> "list[tuple[str, str, str]]":
    """Parse ``ALT-FUNCTION-N: func | file | hypothesis`` lines into (bare_func, file, hypothesis).

    Tolerates a missing file/hypothesis; skips ``none`` entries and template echoes.

    DEDUP is on (bare name, FILE), not on the name alone. An issue that spans several classes
    routinely names ``A.method`` and ``B.method`` in different files, and the class qualifier is
    stripped here because the tracer is addressed by (name, file, line) -- so deduping on the name
    alone silently discarded every candidate after the first. Measured on ansible-83909bf, where
    ``GalaxyLogin.__init__ | galaxy/login.py`` was the ONLY candidate naming the module the fix
    had to delete. A candidate whose file the model omitted cannot be told apart from a same-named
    one, so those still merge -- and a later line that DOES carry a file upgrades an entry the
    earlier one left file-less.
    """
    out: "list[tuple[str, str, str]]" = []
    for m in _ALT_LINE_RE.finditer(locate_text or ""):
        parts = [p.strip().strip("`") for p in m.group(1).split("|")]
        head = re.sub(r"\(.*$", "", parts[0]).strip()
        # Skip empty/none entries and the template placeholders echoed verbatim ("<function>").
        if not head or head.startswith("<") or head.lower() in ("none", "n/a", "-"):
            continue
        fm = re.search(r"[A-Za-z_][\w.]*", head)
        if not fm:
            continue
        bare = fm.group(0).split(".")[-1]
        file = ""
        if len(parts) > 1 and parts[1].lower() not in ("unknown", "n/a", "none", ""):
            pm = _PATH_TOKEN_RE.search(parts[1])
            file = pm.group(0) if pm else ""
        # Models often insert a LINE part ("func | file | 154 | hypothesis") -- take the first
        # non-numeric tail part as the hypothesis.
        hyp = next((p for p in parts[2:] if p and not p.isdigit()), "")
        dup = next((i for i, (b, f, _h) in enumerate(out)
                    if b == bare and (f == file or not f or not file)), None)
        if dup is None:
            out.append((bare, file, hyp))
        elif file and not out[dup][1]:   # the earlier line named no file; this one does
            out[dup] = (bare, file, out[dup][2] or hyp)
    return out

_SIMULATE_PROMPT = """\
You are the SIMULATION step of an automated reproduction sub-agent. You will produce ONE
concrete input and trace execution across the suspected bug's call chain with reasoning. Do NOT write
or run any code -- this is a mental trace.

=== ISSUE (problem statement) ===
{problem_statement}

=== WHAT THE MAIN AGENT HAS ALREADY DONE (its prior commands + observations, up to now) ===
{history}

=== SUSPECTED BUG SITE ===
function: {function} | file: {file}
reason: {reason}

=== CALL CHAIN around the bug site (caller/callee paths AND the real SOURCE of each frame) ===
{call_chain}

The call chain above includes the actual SOURCE BODY of each frame. Read each body and trace over
the REAL code -- never guess what a function does from its name. As you trace, state for each
function what its source actually does to the value (e.g. "match.groupdict() builds {{...: None}}";
"_populate only fills the reverse lookup dict and never touches these kwargs").

Do BOTH (grounding the input/trace in any actual code the agent has already read above):
  1. CONCRETE INPUT: give ONE concrete, valid input that should trigger the bug. Use the exact
     example the issue / PR gives if it gives one; OTHERWISE YOU MUST INVENT a concrete, plausible one
     from the frame SOURCE above and your knowledge of the code -- e.g. a concrete URL pattern with an
     optional named group plus a matching URL, or concrete argument values that hit the buggy branch.
     A terse issue with NO example is NORMAL and expected -- construct the example yourself; that is
     the whole point of this step. Do NOT refuse: answering "no concrete input", "cannot construct",
     "not possible", "the issue does not provide an example", or "Unable to determine" is a FAILURE of
     this step, not a valid answer. Never write "some input".
  2. EXECUTION SIMULATION ALONG THE WHOLE CALL CHAIN: pick the caller path that the concrete input
     actually takes to reach {function} (e.g. B -> C -> D -> {function}), then continue DOWN the
     callee paths it triggers ({function} -> E -> F -> G). Walk the chain end to end as if you were
     the interpreter: for EACH function on the path state, in order, what it receives, what it
     evaluates to, and how the value flows to the next hop -- until you reach the buggy line and
     then on to the chain's result. Name each function/line as you go.
     IMPORTANT -- track the WRONG VALUE to its birthplace: the moment a value first becomes wrong
     (a bad/extra/missing entry, a None that should be absent, an incorrect number/string), call it
     out explicitly as "<value> is FIRST CONSTRUCTED here, in <function> at <file:line>". A later
     function that merely forwards, consumes, or renders that value into visible output is NOT where
     it goes wrong -- the producer that constructed it is. Flag the producer even if it sits on a
     different branch of the chain (e.g. the resolve/parse branch) than where the symptom surfaces.

Respond with the clearly labelled sections below:
CONCRETE INPUT:
<the input>

EXECUTION SIMULATION:
<step-by-step trace following caller path -> {function} -> callee path, naming each function/line;
 mark where each wrong value is FIRST CONSTRUCTED vs where it is later used/rendered>

PRODUCER CHECK:
<adversarial self-check -- the SUSPECTED BUG SITE above is a HYPOTHESIS from an earlier step, and
 it is often the frame where the symptom SURFACES (a traceback line, the API the issue names, a
 site the issue's author guessed), not where the wrong value is BORN. Re-examine your trace: is
 the value/structure ALREADY wrong when it ENTERS {function}? Is the wrong object constructed by
 a helper, a different branch, or generic machinery in ANOTHER FILE whose result {function} merely
 consumes or crashes on? If the call chain contains ALTERNATE CANDIDATE sections, those are
 competing hypotheses an earlier step flagged -- weigh each against {function} using its real
 source shown there. Check the callers'/callees' frame SOURCE before answering; disagreeing
 with the suspected site is a VALID and useful outcome, not a contradiction.
 End with EXACTLY one line in this format (full function name, real file path):
 PRODUCER: <function> | <file>:<line> | <the wrong value it first constructs>
 -- name {function} itself ONLY if your trace shows the wrong value is genuinely first
 constructed inside it.>
"""

# Re-query when SIMULATE refuses to invent a concrete input. A terse issue is normal; the model must
# construct the example from the call-chain SOURCE rather than declaring the simulation impossible.
_SIMULATE_RETRY = """\
Your previous answer REFUSED to produce a concrete input -- that is NOT allowed and is a failure of
this step. A terse issue with no example is normal: you MUST INVENT a plausible concrete
input YOURSELF from the call-chain SOURCE below and your knowledge of the code, then hand-trace
execution over it. Do NOT say the issue lacks an example, that you cannot construct one, or that the
simulation is impossible.

=== SUSPECTED BUG SITE ===
function: {function} | file: {file}

=== CALL CHAIN (caller/callee paths AND the real SOURCE of each frame) ===
{call_chain}

Respond with exactly the labelled sections below, nothing else:
CONCRETE INPUT:
<a concrete INVENTED input with real values -- e.g. a URL pattern containing an optional named group
 plus a matching URL, or concrete argument values that hit the buggy branch. Never "some input",
 never a refusal>

EXECUTION SIMULATION:
<step-by-step trace of that input through the call chain, naming each function/line, marking where
 the wrong value is FIRST CONSTRUCTED (the producer) vs where it is later used/rendered>

PRODUCER CHECK:
<adversarial self-check: is the value ALREADY wrong when it ENTERS {function}, i.e. is the true
 producer a different function (a helper, another branch, another file) whose result {function}
 merely consumes or crashes on? End with EXACTLY one line:
 PRODUCER: <function> | <file>:<line> | <the wrong value it first constructs>
 -- name {function} itself ONLY if the wrong value is genuinely first constructed inside it.>
"""

_PREDICT_PROMPT = """\
You are the PREDICT-AND-COMPARE step of an automated reproduction sub-agent. Using the
simulation below, predict what the BUGGY code actually produces and compare it to what the
issue expects.

=== ISSUE (problem statement) ===
{problem_statement}

=== CONCRETE INPUT + SIMULATION (from the previous step) ===
{simulation}

Respond in EXACTLY these labelled sections:
PREDICTED ACTUAL OUTPUT: <the exact output/exception/wrong value the buggy code produces on that input>
EXPECTED OUTPUT: <the correct output the issue says it should produce>
ROOT CAUSE FRAME: <the function where the wrong value is FIRST CONSTRUCTED / INTRODUCED. Trace the
  bad value BACKWARD to its origin. For a DATA defect -- a wrong / extra / missing value (e.g. a
  dict, kwargs, list, or field that holds a bad or None entry) -- the root cause is the function
  that CREATES that value, NOT a later function that forwards, consumes, or renders it into visible
  output. Worked example: if URL resolution produces a kwargs dict holding x=None for an absent
  optional group and a later reverse/format step turns that None into the string "None" in the
  output, the root cause is the PRODUCER that built the x=None entry (the resolve/match step), NOT
  the formatter that rendered it. Name that producing function -- even if it is on a DIFFERENT branch
  than where the symptom surfaces, and even if it was not expanded in the chain above (use your
  knowledge of the code to name it and its file:line). Only blame the consumer/renderer if the value
  it received was already CORRECT and it corrupts it itself. EARLIEST WINS: if the simulation marks
  the value as "constructed" at more than one frame, choose the EARLIEST in execution order -- a
  function that merely RECEIVES an already-wrong value did not construct it. VERIFY against the frame
  SOURCE in the simulation: confirm the function you name literally builds the value (don't trust a
  role you assumed from its name). Format: function | file:line | one phrase on what wrong value it
  constructs>
VERDICT: REPRODUCED or NOT_REPRODUCED  (REPRODUCED iff the predicted actual output is the SAME
  failure the issue reports -- same exception type+message or same wrong value)
"""


_PREDICT_ROOTCAUSE_RETRY = """\
Your previous answer did not isolate a ROOT CAUSE FRAME, but this bug originates in a SPECIFIC
function -- not the outer function where the symptom merely surfaces. Trace the wrong value BACK to
where it is first CONSTRUCTED, then RE-EMIT the full answer in the exact labelled format.

=== CALL CHAIN (caller paths INTO the site  ->  the site  ->  callee paths OUT of it) ===
{call_chain}

=== YOUR PREVIOUS ANSWER ===
{prediction}

Respond in EXACTLY these labelled sections, nothing else:
PREDICTED ACTUAL OUTPUT: <the exact wrong value/exception the buggy code produces>
EXPECTED OUTPUT: <the correct output the issue says it should produce>
ROOT CAUSE FRAME: <function | file:line | what wrong value it constructs -- the function that FIRST
  CONSTRUCTS the bad value (for a wrong/extra/missing/None value, the PRODUCER that creates it), NOT
  a later function that forwards, consumes, or renders it into output. Name the producer even if it
  is on a different branch than the symptom or was not expanded in the chain above; use your code
  knowledge to give its file:line. Only blame a consumer if the value it received was already CORRECT.>
VERDICT: REPRODUCED or NOT_REPRODUCED
"""

_SCRIPT_PROMPT = """\
You are the FINAL step of an automated reproduction sub-agent. Based on everything below,
(1) list the COMPLETE set of source files that must be edited to FIX this bug -- the root-cause file
AND any caller / sibling / base file that must change too -- and (2) write a concrete,
self-contained reproduction script that PRINTS the actual result so the failure can be OBSERVED.

=== ISSUE (problem statement) ===
{problem_statement}

=== WHAT THE MAIN AGENT HAS ALREADY DONE (its prior commands + observations, up to now) ===
{history}

=== SUSPECTED BUG SITE ===
function: {function} | file: {file}

=== CALL CHAIN ===
{call_chain}

=== SIMULATION ===
{simulation}

=== PREDICTION vs EXPECTED (note the ROOT CAUSE FRAME -- the deepest function where the value first goes wrong) ===
{prediction}

FILES_TO_EDIT -- find the COMPLETE set of edit sites, not just one. Reason from the call chain and
the frame SOURCE shown above:
  1. PRIMARY: the ROOT CAUSE FRAME above (the function that constructs the wrong value). Do NOT pick
     the surface wrapper where the symptom merely surfaces if the defect originates deeper.
  2. ALSO include every OTHER frame that must change for a COMPLETE fix:
       - a CALLER that must be updated to handle the root's corrected value/return (e.g. a base-class
         caller that consumes what the producer returns and mishandles the empty/None/edge case),
       - a SIBLING implementation of the same method in another module with the SAME defect (e.g. a
         method overridden per backend/subclass),
       - a shared BASE / default implementation the fix must also cover.
     Hidden tests often target the BASE/abstract class directly, so if the call chain or source shows
     a base-class or caller frame with the same bug pattern, it almost certainly needs editing TOO.
  Verify each file against the frame SOURCE -- only list a file whose code must actually change, but
  ERR TOWARD COMPLETENESS for a fix that spans a producer + its caller or several backends.

Respond in EXACTLY these labelled sections:
FILES_TO_EDIT:
- <relative/path/one.py>   (the root-cause file)
- <relative/path/two.py>   (any caller / sibling / base that must ALSO change -- list ALL of them, one per line)

REPRODUCTION_SCRIPT:
```bash
<a single bash command or heredoc that creates and runs the reproduction, printing the actual
result; it must reproduce the issue's OWN example and exit/print so the failure is visible>
```
"""


# Models answer in markdown -- "**FUNCTION:** `foo`", "## FILES_TO_EDIT", etc. -- so all the
# parsers strip markdown emphasis/heading/backtick noise before matching the labels.
_MD_NOISE_RE = re.compile(r"[*`#>]+")


# The sub-agent reasons in plain text, but agentic/tool-call models (e.g. minimax-m2.5) tend to
# IGNORE the requested label format and instead continue investigating -- emitting their native
# tool-call syntax as literal text content (``THOUGHT: ... [TOOL_CALL]{"command": "sed ..."}``).
# That text carries no parseable answer, so every step's labels come back empty and the reminder
# degrades to "SUSPECTED BUG SITE: unknown". The system prompt below removes the agentic frame
# (it tells the model it has no tools), and ``_looks_like_toolcall`` detects the failure so the
# query layer can retry once with ``_TOOLCALL_CORRECTION`` before giving up.
_SUBAGENT_SYSTEM = (
    "You are an offline analysis assistant with NO tools. You cannot run commands, open files, "
    "or call any tool. Do not emit tool calls, shell commands, JSON command objects, or "
    "[TOOL_CALL] blocks. Using ONLY the information the user gives you, answer with the EXACT "
    "labelled plain-text format the user asks for -- nothing else."
)
_TOOLCALL_CORRECTION = (
    "That response tried to call a tool or run a command, but you have NO tools and cannot run "
    "anything. Do not investigate further. Using ONLY the information already provided above, "
    "answer NOW in the exact labelled plain-text format requested -- no commands, no JSON, no "
    "[TOOL_CALL] blocks."
)
# Matches the agentic tool-call forms the model emits as text INSTEAD of answering -- always a
# non-answer for every step (none of the labelled formats use these), so flagged unconditionally:
#   - an explicit ``[TOOL_CALL]`` / ``[/TOOL_CALL]`` marker,
#   - a JSON command object (``{"command":``, ``{command:``, ``{'command':``),
#   - a ``<function>`` / ``</function>`` tool-call tag (some models' native wrapper),
#   - any ``<tool_*>`` / ``[tool_*]`` wrapper, INCLUDING a namespace prefix -- minimax leaks several
#     native spellings as text (``<tool_call>``, ``<tool_code>``, ``<minimax:tool_call>``,
#     ``[tool_call]``, ``[/tool_code]`` ...), so match the whole ``[ns:]tool_<word>`` family,
#   - an ``<invoke ...>`` tag (the inner element of minimax's tool-call block),
#   - a bare ``COMMAND:`` line (the mini-swe-agent shell convention leaking as text).
# Kept anchored / structured so legitimate analysis prose that merely mentions "command" does not
# trigger a spurious retry.
_TOOLCALL_RE = re.compile(
    r"\[/?[\w:]*tool_\w+\]|\{\s*[\"']?command[\"']?\s*:|</?function\b|</?[\w:]*tool_\w+\b|"
    r"</?invoke\b|^\s*command\s*:",
    re.IGNORECASE | re.MULTILINE,
)
# A markdown shell fence (```bash / ```sh / ```shell / ```console). This is the OTHER way an
# agentic model "calls a tool" as text -- it emits a shell command to investigate instead of
# answering. It is only a non-answer for the reasoning steps; the SCRIPT step legitimately returns
# a ```bash reproduction block, so callers there pass allow_commands=True.
_SHELL_FENCE_RE = re.compile(r"```+\s*(?:bash|sh|shell|console|zsh)\b", re.IGNORECASE)


def _looks_like_toolcall(text: str, allow_commands: bool = False) -> bool:
    """True if ``text`` is the model investigating (tool-call/shell) rather than answering.

    ``allow_commands=True`` exempts a markdown shell fence (for the SCRIPT step, whose answer
    legitimately contains a ```bash reproduction block); the explicit [TOOL_CALL]/JSON-command
    forms are always treated as non-answers.
    """
    text = text or ""
    if _TOOLCALL_RE.search(text):
        return True
    if not allow_commands and _SHELL_FENCE_RE.search(text):
        return True
    return False


# The rendered call chain opens with "=== CALL CHAIN for <name> (...". When the tracer self-heals
# a non-existent function name to its real enclosing def, <name> is the HEALED name, so parsing it
# back lets the reproducer correct ``bug_function`` for the reminder/simulation downstream.
_CHAIN_TARGET_RE = re.compile(r"^=== CALL CHAIN for (\S+) ", re.MULTILINE)


def _chain_target(chain_text: str) -> str:
    m = _CHAIN_TARGET_RE.search(chain_text or "")
    return m.group(1) if m else ""


_LABEL_LINE_RE = re.compile(r"^\s*[A-Z][A-Z _]{2,}\s*:")  # e.g. "VERDICT:", "EXPECTED OUTPUT:"

# "kwargs {'year': None} is FIRST CONSTRUCTED here, in match at resolvers.py:156" -- the SIMULATE
# step is told to mark the producer explicitly. Harvest that frame when PREDICT's ROOT CAUSE FRAME
# is missing or (worse) names the symptom wrapper instead of the producer.
_FIRST_CONSTRUCTED_RE = None   # built by _compile_frame_regexes (per active language)
_FRAME_FILE_RE = None


@on_lang_change
def _compile_frame_regexes():
    """Frame-file regexes. Python and Go keep the historical ``py|go`` alternation unchanged; JS/TS
    add their extensions, which the old hard-coded form could never match (a JS frame's file was
    silently dropped)."""
    global _FIRST_CONSTRUCTED_RE, _FRAME_FILE_RE
    # longest extensions first: alternation takes the first branch that matches, so `ts|tsx` would read
    # `GroupModal.tsx:22` as `GroupModal.ts` and lose the line number
    js_ext = "|".join(sorted(src_ext_alt().split("|"), key=len, reverse=True)) if is_js() else ""
    ext = "py|go" + ("|" + js_ext if js_ext else "")
    _FIRST_CONSTRUCTED_RE = re.compile(
        r"FIRST\s+CONSTRUCTED\b[^\n]*?\bin\s+`?([A-Za-z_][\w.]*)`?\s*(?:\(\))?"
        r"[^\n]*?(?:\bat\s+|,\s*|\bin\s+)?([\w./-]+\.(?:" + ext + r"))?\s*:?\s*(\d+)?",
        re.IGNORECASE,
    )
    _FRAME_FILE_RE = re.compile(r"([\w./-]+\.(?:" + ext + r"))\s*:?\s*(\d+)?")

# "PRODUCER: <function> | <file>:<line> | <wrong value>" -- the SIMULATE step's adversarial
# PRODUCER CHECK verdict. This is the trace's FINAL answer on where the wrong value is born (it
# may overrule the suspected site LOCATE anchored on), so it outranks the inline
# FIRST CONSTRUCTED markers scattered through the trace body.
_PRODUCER_LINE_RE = re.compile(r"^\s*`?PRODUCER`?\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _parse_frame(text: str) -> "tuple[str, str, int]":
    """Parse a ``func | file:line | phrase`` frame into (BARE_func, file, line).

    Returns the BARE function name (last dotted component, so ``RegexPattern.match`` -> ``match``)
    because the tracer resolves functions by their ``def <name>``; file/line when present, else
    ``("", "", 0)``.
    """
    rc = (text or "").strip()
    if not rc:
        return "", "", 0
    parts = [p.strip() for p in rc.split("|")]
    func = re.sub(r"\(.*$", "", parts[0]).strip().rstrip(":") if parts else ""
    bare = func.split(".")[-1] if func else ""
    file, line = "", 0
    if len(parts) > 1:
        m = _FRAME_FILE_RE.search(parts[1])
        if m:
            file, line = m.group(1), (int(m.group(2)) if m.group(2) else 0)
    return bare, file, line


def _producer_frame(root_cause: str, simulation: str) -> "tuple[str, str, int]":
    """The function where the wrong value is FIRST CONSTRUCTED (the fix's true root cause).

    Prefer PREDICT's ROOT CAUSE FRAME; then the ``PRODUCER: <func> | <file>:<line>`` verdict of
    the SIMULATE step's adversarial PRODUCER CHECK (its final answer, which may overrule the
    suspected site); then the inline ``FIRST CONSTRUCTED ... in <func> at <file:line>`` markers.
    Returns (bare_func, file, line) or ("","",0).
    """
    bare, file, line = _parse_frame(root_cause)
    if bare:
        return bare, file, line
    pm = _PRODUCER_LINE_RE.search(simulation or "")
    if pm:
        bare, file, line = _parse_frame(pm.group(1))
        # Guard against the template placeholder echoed verbatim ("<function>"): only accept a
        # real bare identifier.
        if bare and re.fullmatch(r"[A-Za-z_]\w*", bare):
            return bare, file, line
    m = _FIRST_CONSTRUCTED_RE.search(simulation or "")
    if m:
        return m.group(1).split(".")[-1], (m.group(2) or ""), (int(m.group(3)) if m.group(3) else 0)
    return "", "", 0


def _extract_root_cause(prediction: str) -> str:
    """Pull the ROOT CAUSE FRAME value, whether inline or on the line(s) after the label.

    Models often emit ``ROOT CAUSE FRAME:`` on its own line (or wrap the value in a code fence),
    which the same-line ``_line_field`` parser misses -- silently dropping a frame that WAS
    produced. This tolerates both shapes: take the inline value if present, else the first
    non-empty, non-fence line(s) after the label, stopping at the next ``LABEL:`` (e.g. VERDICT).
    """
    inline = _line_field("ROOT CAUSE FRAME", prediction)
    # _line_field's optional colon can capture a bare ":" off a label-only line; require real text.
    if inline and re.search(r"[A-Za-z0-9]", inline):
        return inline
    lines = _strip_md(prediction).splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*ROOT\s+CAUSE\s+FRAME\s*:", line, re.IGNORECASE):
            for nxt in lines[i + 1:]:
                if not nxt.strip():
                    continue
                if _LABEL_LINE_RE.match(nxt):  # hit the next labelled section -> no value
                    return ""
                return nxt.strip()
            return ""
    return ""


# The SIMULATE step sometimes REFUSES to invent a concrete input when the issue gives no example
# ("execution simulation is not possible", "cannot construct", "the issue does not provide an
# example") -- which collapses PREDICT to "Unable to determine". These phrasings never occur in a
# real hand-trace, so they are a reliable refusal signal that triggers a re-query.
_REFUSAL_RE = re.compile(
    r"execution simulation is not possible"
    r"|cannot (?:construct|determine|identify|provide|trace|simulate)"
    r"|can'?t (?:construct|determine|identify|trace|simulate)"
    r"|unable to (?:construct|determine|identify|trace|simulate)"
    r"|no concrete (?:input|example)"
    r"|not possible to (?:construct|simulate|trace)"
    r"|does not (?:provide|give|include|contain) (?:a |an |any )?(?:specific |concrete )?"
    r"(?:example|input|test case|url pattern)",
    re.IGNORECASE,
)


def _simulation_refused(text: str) -> bool:
    """True if the SIMULATE step gave up instead of inventing a concrete input + trace."""
    t = text or ""
    return not t.strip() or bool(_REFUSAL_RE.search(t))


def _chain_has_multiple_frames(chain_text: str) -> bool:
    """True if the rendered chain shows at least one A -> B path (so a deeper root frame exists).

    A single-frame chain (just the target, no caller/callee paths) has no deeper frame to point
    at, so there is no point re-querying PREDICT for a distinct ROOT CAUSE FRAME.
    """
    return " -> " in (chain_text or "")


def _strip_md(text: str) -> str:
    return _MD_NOISE_RE.sub("", text or "")


def _line_field(label: str, text: str) -> str:
    """Return the value after ``LABEL:`` on the first matching (markdown-stripped) line."""
    pat = re.compile(rf"^\s*{label}\s*:?\s*(.+?)\s*$", re.IGNORECASE)
    for line in _strip_md(text).splitlines():
        m = pat.match(line)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return ""


_BARE_NAME_RE = re.compile(r"[A-Za-z_][\w.]*")
# A path-looking token (has a slash or a file extension); used to harvest files-to-edit.
_PATH_TOKEN_RE = re.compile(r"[\w./\\-]*[\w](?:/[\w./\\-]+|\.[A-Za-z]{1,4})")
_FILE_PATH_RE = re.compile(r"(?:[\w.-]+/)+[\w.-]+\.[A-Za-z][A-Za-z0-9]{0,4}\b")
# The reproduction-SCRIPT filename the model invents (reproduce_bug.py, repro.py, ...) is not a fix
# target, so it is dropped from FILES_TO_EDIT.
_REPRO_NAME_RE = re.compile(r"(?:^|/)(?:repro|reproduce|bug_?repro|test_repro)", re.IGNORECASE)
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_]*\s*\n(.*?)```", re.DOTALL)
# Split on the REPRODUCTION_SCRIPT *label* -- anchored at line start so the prose phrase
# "reproduction script" inside a THOUGHT does NOT trigger an early split. Tolerates a mistyped
# label (e.g. "REPRODUCTION_CRIPT" / "REPRODUCTION SCRIPT").
_SECTION_SPLIT_RE = re.compile(r"^\s*REPRODUCTION[_ ]?S?CRIPT", re.IGNORECASE | re.MULTILINE)


@dataclass
class SubAgentFindings:
    """The structured result of the reproduction sub-agent's five-step pipeline."""

    bug_function: str = ""
    bug_file: str = ""
    bug_line: int = 0
    locate_reason: str = ""
    call_chain: str = ""
    simulation: str = ""
    prediction: str = ""
    root_cause: str = ""  # deepest frame where the value first goes wrong (PREDICT step), the EDIT target
    files_to_edit: list = field(default_factory=list)
    reproduction_script: str = ""
    ok: bool = False  # True once the pipeline produced something usable
    # Raw model text for each LLM step, kept verbatim so nothing is lost if a label
    # parser misses (the reminder still surfaces the model's actual work).
    raw_locate: str = ""
    raw_plan: str = ""
    # Multi-candidate LOCATE: (bare_func, file, hypothesis) alternates whose chains were merged.
    alt_candidates: list = field(default_factory=list)
    # Quantifier-driven input matrix: {label, clause} input classes assigned to the ALT slots.
    input_directives: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bug_function": self.bug_function,
            "bug_file": self.bug_file,
            "bug_line": self.bug_line,
            "locate_reason": self.locate_reason,
            "call_chain": self.call_chain,
            "simulation": self.simulation,
            "prediction": self.prediction,
            "root_cause": self.root_cause,
            "files_to_edit": self.files_to_edit,
            "reproduction_script": self.reproduction_script,
            "ok": self.ok,
            "raw_locate": self.raw_locate,
            "raw_plan": self.raw_plan,
            "alt_candidates": self.alt_candidates,
            "input_directives": self.input_directives,
        }

    def to_reminder(self) -> str:
        """Render the findings into the reminder injected into the MAIN agent's history."""
        files = "\n".join(f"    - {f}" for f in self.files_to_edit) or "    (none identified)"
        script = self.reproduction_script.strip()
        if not script:
            # Parser missed the fenced block -- surface the raw plan text so the agent
            # still benefits from the sub-agent's proposed files + script.
            script = self.raw_plan.strip() or "(the sub-agent did not produce a script; write a minimal one yourself)"
        # The PREDICT step names the deepest frame where the value first goes wrong; that, not the
        # surface function the symptom surfaces in, is the place to EDIT. Fall back to the located
        # site when the model did not isolate a distinct root frame.
        root_cause = self.root_cause.strip() or (
            f"{self.bug_function or 'unknown'} (file: {self.bug_file or 'unknown'})"
        )
        return SUBAGENT_REMINDER_TEMPLATE.format(
            function=self.bug_function or "unknown",
            file=self.bug_file or "unknown",
            reason=_clip(self.locate_reason, 400),
            call_chain=_clip(self.call_chain, 2500),
            simulation=_clip(self.simulation, 2500),
            prediction=_clip(self.prediction, 1500),
            root_cause=_clip(root_cause, 300),
            files_to_edit=files,
            reproduction_script=_clip(script, 2500),
        )

    def to_patch_preamble(self) -> str:
        """A root-cause anchor block prepended to the patch-phase reminder.

        Carries the reproduction sub-agent's localisation forward into the FIX step so the edit
        targets the same ROOT CAUSE frame (the deepest function where the value first goes wrong),
        not the surface function where the symptom merely surfaces -- the django-11477 failure mode
        where the agent patched the wrapper instead of the resolver.
        """
        root_cause = self.root_cause.strip() or (
            f"{self.bug_function or 'unknown'} (file: {self.bug_file or 'unknown'})"
        )
        files = ", ".join(self.files_to_edit) or "(see the analysis from the reproduce step)"
        return SUBAGENT_PATCH_PREAMBLE_TEMPLATE.format(
            root_cause=_clip(root_cause, 300),
            function=self.bug_function or "unknown",
            file=self.bug_file or "unknown",
            files_to_edit=_clip(files, 300),
        )


SUBAGENT_REMINDER_TEMPLATE = """\
[REASONING INTERVENTION -- pre-reproduction analysis from a reproduction sub-agent]

A reproduction sub-agent has already analysed this bug for you. Its findings:

  SYMPTOM SURFACES AT: {function}  (file: {file})   <- where the failure is OBSERVED
    why: {reason}

  ROOT CAUSE -- EDIT HERE: {root_cause}   <- the deepest frame where the value FIRST goes wrong
    Fixing the ROOT resolves EVERY caller path in the chain below; patching the surface function
    where the symptom merely surfaces fixes only the one path you reproduce. Do NOT patch the
    surface wrapper if the defect originates deeper -- unless the surface IS itself the root.

  CALL CHAIN around the bug site (callers / callees / siblings):
{call_chain}

  CONCRETE INPUT + EXECUTION SIMULATION across that call chain:
{simulation}

  PREDICTED ACTUAL vs EXPECTED (incl. ROOT CAUSE FRAME):
{prediction}

  FILES THAT LIKELY NEED EDITING to fix the bug (the root-cause file AND any caller/sibling/base
  that must ALSO change -- edit ALL of them):
{files_to_edit}

  PROPOSED REPRODUCTION SCRIPT:
{reproduction_script}

>>> WHAT TO DO IN YOUR NEXT STEP (do BOTH, in this order, in your very next response):

  STEP A -- REASON FIRST (in your THOUGHT): restate the concrete input, the predicted ACTUAL
    (buggy) result and the EXPECTED result from the analysis above, and the exact symptom the
    ISSUE reports (precise error type+message, or the exact wrong value).

  STEP B -- THEN ACT: in the SAME response, issue ONE bash command that RUNS a reproduction
    script (you may adapt the proposed script above) which PRINTS the actual result, so you can
    OBSERVE the failure (red). After it runs, compare the OBSERVED output against the issue's
    symptom and conclude REPRODUCED or NOT-REPRODUCED. If NOT reproduced (a different error, a
    setup/import error, or a clean pass), you are mis-localized -- fix the reproduction to match
    the issue's own example BEFORE attempting any fix.

  STEP C -- WHEN YOU FIX (after reproducing): make the edit at the ROOT CAUSE frame above, not the
    surface function, and confirm your fix resolves the symptom for ALL caller paths in the call
    chain -- not just the single path you reproduced.

Do not spend this step on more exploration -- your next command must be the reproduction script.
"""


# Prepended to the patch-phase reminder so the FIX is anchored on the reproduction sub-agent's
# root-cause localisation (carried over from the L_reproduce step), rather than re-derived.
SUBAGENT_PATCH_PREAMBLE_TEMPLATE = """\
[REASONING INTERVENTION -- pre-fix/patch reasoning, anchored on the reproduction sub-agent's analysis]

The reproduction sub-agent already localised this bug. Anchor your fix on its analysis:

  ROOT CAUSE -- EDIT HERE: {root_cause}   <- the deepest frame where the value FIRST goes wrong
  SYMPTOM SURFACES AT: {function}  (file: {file})   <- where the failure is merely observed
  FILES THAT LIKELY NEED EDITING (edit ALL of them): {files_to_edit}

  >>> Make the primary edit at the ROOT CAUSE frame above, NOT the surface function where the symptom
  surfaces. Fixing the root resolves EVERY caller path; patching a surface wrapper fixes only the one
  path you reproduced. If MORE THAN ONE file is listed above, this fix spans multiple sites (a
  producer plus a caller, or the same defect in a base class / sibling backend) -- edit EVERY listed
  file, not just the root, or the hidden tests that target the other site will still fail. If you are
  convinced the true root is a DIFFERENT function, state the concrete reason BEFORE editing.

"""


class SubAgentReproducer:
    """Runs the five-step reproduction pipeline out of band from the main agent.

    Dependency-injected for testability:
      * ``query_fn(prompt) -> str`` -- a single-shot LLM call (wraps ``agent.model``).
      * ``trace_fn(function_name, file_path="") -> str`` -- the call-chain extractor
        (from :func:`make_trace_call_chain_runner`).
    None of these touch the main agent's message history.
    """

    def __init__(
        self,
        *,
        query_fn: Callable[[str], str],
        trace_fn: Callable[..., str],
        problem_statement: str,
        history: str = "",
        log: Optional[Callable[[str], None]] = None,
    ):
        self.query_fn = query_fn
        self.trace_fn = trace_fn
        self.problem_statement = problem_statement or ""
        self.history = history or "(no prior history)"
        self._log = log or (lambda _msg: None)

    # -- step 1 -----------------------------------------------------------------------
    def locate(self) -> "tuple[str, str, int, str, str]":
        text = self.query_fn(
            _LOCATE_PROMPT.format(
                problem_statement=_clip(self.problem_statement, PS_CLIP),
                history=_clip_tail(self.history, 10000),
            )
        ) or ""
        # Markdown-tolerant: pull the value off the FUNCTION:/FILE:/LINE:/REASON: lines, then take
        # the bare (last-segment) identifier for the function name.
        fn_raw = _line_field("FUNCTION", text)
        m = _BARE_NAME_RE.search(fn_raw)
        function = m.group(0).split(".")[-1] if m else ""
        file = _line_field("FILE", text)
        if file.lower() in ("unknown", "n/a", "none", ""):
            file = ""
        else:
            fm = _PATH_TOKEN_RE.search(file)
            file = fm.group(0) if fm else file
        # Line: prefer the explicit LINE: field, else scrape "line NNNN" / "file:NNNN" from the
        # full answer (the model usually mentions it in the reason, e.g. "around line 1242").
        line = 0
        lm = re.search(r"\d{1,7}", _line_field("LINE", text))
        if lm:
            line = int(lm.group(0))
        if not line:
            lm = re.search(r"lines?\s*[:#]?\s*(\d{1,7})|:(\d{1,7})\b", text, re.IGNORECASE)
            if lm:
                line = int(lm.group(1) or lm.group(2))
        reason = _line_field("REASON", text)
        return function, file, line, reason, text

    # -- step 2 -----------------------------------------------------------------------
    def trace(self, function: str, file: str, line: int = 0) -> str:
        if not function:
            return "[no bug function identified; call chain unavailable]"
        return self.trace_fn(function, file, line)

    # -- step 3 -----------------------------------------------------------------------
    def simulate(self, function: str, file: str, reason: str, call_chain: str) -> str:
        text = self.query_fn(
            _SIMULATE_PROMPT.format(
                problem_statement=_clip(self.problem_statement, PS_CLIP),
                history=_clip_tail(self.history, 9000),
                function=function or "unknown",
                file=file or "unknown",
                reason=reason or "(none)",
                # Generous budget: the chain now carries the real source bodies of its frames, which
                # the simulation must reason over rather than guessing each function's behaviour.
                call_chain=_clip(call_chain, 16000),
            )
        ) or ""
        # If the model refused to invent a concrete input (the terse-issue failure mode), re-query
        # once with a hard instruction to construct one from the call-chain source.
        if _simulation_refused(text):
            self._log("    [subagent] (3*) simulation refused a concrete input; re-querying with a hard instruction")
            retry = self.query_fn(
                _SIMULATE_RETRY.format(
                    function=function or "unknown",
                    file=file or "unknown",
                    call_chain=_clip(call_chain, 16000),
                )
            ) or ""
            if retry and not _simulation_refused(retry):
                return retry
        return text

    # -- step 4 -----------------------------------------------------------------------
    def predict(self, simulation: str, call_chain: str = "") -> str:
        text = self.query_fn(
            _PREDICT_PROMPT.format(
                problem_statement=_clip(self.problem_statement, PS_CLIP),
                simulation=_clip(simulation, 4000),
            )
        ) or ""
        # Guard: if the model skipped the ROOT CAUSE FRAME and the chain spans multiple frames,
        # re-query once demanding it (the deeper frame is the edit target). Keep the retry only if
        # it actually produced a frame; otherwise fall back to the original answer.
        if not _extract_root_cause(text) and _chain_has_multiple_frames(call_chain):
            self._log("    [subagent] (4*) no root-cause frame; re-querying PREDICT with the call chain")
            retry = self.query_fn(
                _PREDICT_ROOTCAUSE_RETRY.format(
                    call_chain=_clip(call_chain, 6000),
                    prediction=_clip(text, 2000),
                )
            ) or ""
            if _extract_root_cause(retry):
                return retry
        return text

    # -- step 5 -----------------------------------------------------------------------
    def plan_and_script(self, function: str, file: str, call_chain: str, simulation: str, prediction: str) -> "tuple[list, str, str]":
        text = self.query_fn(
            _SCRIPT_PROMPT.format(
                problem_statement=_clip(self.problem_statement, PS_CLIP),
                history=_clip_tail(self.history, 7000),
                function=function or "unknown",
                file=file or "unknown",
                call_chain=_clip(call_chain, 5000),
                simulation=_clip(simulation, 2500),
                prediction=_clip(prediction, 1500),
            )
        ) or ""
        # Split into the FILES section (before REPRODUCTION_SCRIPT) and the script section.
        parts = _SECTION_SPLIT_RE.split(text, maxsplit=1)
        files_section, script_section = (parts[0], parts[1]) if len(parts) == 2 else (text, "")
        files = self._parse_files(files_section)
        # Script: prefer a fenced code block; else fall back to the prose under the label.
        fence = _FENCE_RE.search(script_section) or _FENCE_RE.search(text)
        if fence:
            script = fence.group(1).strip()
        else:
            script = _strip_md(script_section).strip()
        return files, script, text

    @staticmethod
    def _parse_files(files_section: str) -> list:
        """Harvest real source paths (``dir/.../name.ext``) from the FILES_TO_EDIT section.

        Uses the strict :data:`_FILE_PATH_RE` so a verbose, multi-site answer (root file PLUS a
        caller/base, each annotated with prose) yields just the genuine file paths -- not the dotted
        module names / truncated identifiers the prose contains. Takes the FIRST path per line so a
        bullet like ``- a/b.py (also edit the caller in c/d.py)`` resolves to the bullet's file.
        """
        files: list = []
        for line in _strip_md(files_section).splitlines():
            line = line.strip()
            if not line or line.lower().lstrip("-*0123456789. ").startswith("files_to_edit"):
                continue
            m = _FILE_PATH_RE.search(line)
            if not m:
                continue
            cand = m.group(0).strip().lstrip("/")
            for pre in ("testbed/", "a/", "b/"):  # container abs-path / diff prefixes
                if cand.startswith(pre):
                    cand = cand[len(pre):]
            if not cand or cand.lower() in ("none", "n/a") or _REPRO_NAME_RE.search(cand):
                continue  # drop the reproduction-script file -- it is not a fix target
            if cand not in files:
                files.append(cand)
        return files

    # -- orchestration ----------------------------------------------------------------
    def run(self) -> SubAgentFindings:
        findings = SubAgentFindings()
        try:
            (
                findings.bug_function,
                findings.bug_file,
                findings.bug_line,
                findings.locate_reason,
                findings.raw_locate,
            ) = self.locate()
            self._log(
                f"    [subagent] (1) bug site: {findings.bug_function or '?'} "
                f"(file: {findings.bug_file or '?'}:{findings.bug_line or '?'})"
            )
            findings.call_chain = self.trace(findings.bug_function, findings.bug_file, findings.bug_line)
            # If the tracer self-healed a non-existent function name to its real enclosing def,
            # adopt that name so the reminder + simulation downstream use the function that exists.
            healed = _chain_target(findings.call_chain)
            if healed and healed != findings.bug_function:
                self._log(
                    f"    [subagent] (2a) healed bug function '{findings.bug_function}' -> "
                    f"'{healed}' via {findings.bug_file}:{findings.bug_line}"
                )
                findings.bug_function = healed
            # (2a*) RE-ANCHOR when the primary has no definition in the repo (typically a symbol
            # that only appears in a TEST file or a wished-for API): its chain is empty and would
            # poison SIMULATE/PLAN. Adopt the first ALT candidate that IS defined instead; the
            # undefined name is kept as a note (it may be a function the fix must ADD).
            if MULTI_LOCATE and "has NO definition in the repo" in findings.call_chain[:700]:
                primary_bare = (findings.bug_function or "").split(".")[-1]
                for afunc, afile, ahyp in _alt_candidates(findings.raw_locate)[:MULTI_LOCATE_MAX]:
                    if not afunc or afunc == primary_bare:
                        continue
                    achain = self.trace(afunc, afile, 0)
                    if (achain and "call chain unavailable" not in achain[:80]
                            and "has NO definition in the repo" not in achain[:700]):
                        self._log(
                            f"    [subagent] (2a*) primary anchor '{findings.bug_function}' has "
                            f"no definition in the repo; re-anchoring to ALT '{afunc}'"
                        )
                        findings.call_chain = achain + (
                            f"\n\nNOTE: the originally suspected site '{findings.bug_function}' "
                            "has NO definition in the repo (it may be a test-only symbol or a "
                            "function the fix must ADD); the chain above anchors on the strongest "
                            "EXISTING candidate instead."
                        )
                        findings.bug_function, findings.bug_file, findings.bug_line = afunc, afile or "", 0
                        break
            self._log(f"    [subagent] (2) traced call chain ({len(findings.call_chain)} chars)")
            # (2b) MULTI-CANDIDATE LOCATE: trace the alternate candidates and merge their chains
            # (clipped) so SIMULATE's PRODUCER CHECK arbitrates between hypotheses from real
            # source instead of only ever seeing the primary anchor. Purely additive.
            findings.alt_candidates = _alt_candidates(findings.raw_locate)[:MULTI_LOCATE_MAX]
            primary_bare = (findings.bug_function or "").split(".")[-1]
            for k, (afunc, afile, ahyp) in enumerate(findings.alt_candidates, start=2):
                if afunc == primary_bare:
                    continue
                achain = self.trace(afunc, afile, 0)
                if not achain or "call chain unavailable" in achain[:80]:
                    continue
                findings.call_chain += (
                    f"\n\n=== ALTERNATE CANDIDATE #{k}: '{afunc}'"
                    # The file is part of the candidate's identity: two classes' same-named
                    # methods are two candidates, and _alt_chain_section keys on this tag.
                    f"{f' [{afile}]' if afile else ''}"
                    f"{f' ({ahyp})' if ahyp else ''} -- a COMPETING hypothesis for the true "
                    f"bug site; its call chain/source ===\n" + _clip(achain, ALT_CHAIN_CLIP)
                )
                self._log(f"    [subagent] (2b) merged alternate candidate '{afunc}"
                          f"{f' ({afile})' if afile else ''}' "
                          f"({ahyp or 'no hypothesis given'}) into the call chain")
            findings.simulation = self.simulate(
                findings.bug_function, findings.bug_file, findings.locate_reason, findings.call_chain
            )
            self._log(f"    [subagent] (3) generated concrete input + execution simulation ({len(findings.simulation)} chars)")
            # (3a) ALTERNATE-CANDIDATE SIMULATIONS: give each merged alternate its own short
            # execution trace + MUST-CHANGE/EXONERATED verdict (the stock simulation above only
            # ever executes the PRIMARY candidate's path). Appended to the simulation text so
            # PREDICT/PLAN and the downstream frame-union see the verdicts.
            if ALT_SIM and findings.alt_candidates:
                # Quantifier-driven input matrix: each ALT slot gets a DISTINCT spec-quantified
                # input class (multiple-instance, missing/error, other-variant, ...) so the
                # sims stop re-sampling the primary's default input shape.
                directives = _input_directives_llm(
                    self.query_fn, self.problem_statement, log=self._log)
                findings.input_directives = [
                    {"label": l, "clause": c} for l, _i, c in directives]
                if directives:
                    self._log("    [subagent] (3a) input-matrix directives: "
                              f"{[l for l, _i, _c in directives]}")
                slot = 0
                for afunc, afile, ahyp in findings.alt_candidates[:ALT_SIM_MAX]:
                    section = _alt_chain_section(findings.call_chain, afunc, afile)
                    if not section.strip():
                        continue  # candidate was never merged (trace failed) -> nothing to simulate
                    directive = ""
                    if slot < len(directives):
                        label, instruction, clause = directives[slot]
                        directive = _render_input_directive(label, instruction, clause)
                    slot += 1
                    alt_sim = self.query_fn(
                        _ALT_SIM_PROMPT.format(
                            alt_func=f"{afunc} ({afile})" if afile else afunc,
                            alt_hyp=f"({ahyp})" if ahyp else "(no hypothesis given)",
                            function=findings.bug_function or "unknown",
                            problem_statement=_clip(self.problem_statement, PS_CLIP),
                            alt_chain=_clip(section, 9000),
                            input_directive=directive,
                        )
                    ) or ""
                    if not alt_sim.strip() or _simulation_refused(alt_sim):
                        self._log(f"    [subagent] (3a) alternate simulation for '{afunc}' "
                                  "refused/empty; skipped")
                        continue
                    findings.simulation += (
                        f"\n\n=== SIMULATION on ALTERNATE CANDIDATE '{afunc}"
                        f"{f' [{afile}]' if afile else ''}' "
                        "(independent trace of the competing hypothesis) ===\n"
                        + _clip(alt_sim, 2500)
                    )
                    verdict = "MUST-CHANGE" if re.search(
                        r"^[\s>*\-`#]*(?:[A-Za-z][\w ]{0,12}:\s*)?MUST[-_ ]?CHANGE\s*:",
                        alt_sim, re.IGNORECASE | re.MULTILINE
                    ) else "exonerated/unclear"
                    self._log(f"    [subagent] (3a) simulated alternate candidate '{afunc}': "
                              f"{verdict}")
            findings.prediction = self.predict(findings.simulation, findings.call_chain)
            # Elevate the deepest "ROOT CAUSE FRAME" the PREDICT step isolated; this is the EDIT
            # target the reminder advertises (distinct from the surface symptom site) and what the
            # plan step is told to point FILES_TO_EDIT at.
            findings.root_cause = _extract_root_cause(findings.prediction)
            self._log(
                f"    [subagent] (4) predicted actual vs expected ({len(findings.prediction)} chars); "
                f"root cause: {findings.root_cause or '(not isolated)'}"
            )
            # (4b) RE-ANCHOR ON THE PRODUCER. When LOCATE anchored on the symptom wrapper (e.g. the
            # reverse/format side), the traced call chain does NOT contain the function that FIRST
            # CONSTRUCTS the wrong value (e.g. the resolve/parse side), so PLAN can't see its source
            # and mislocates the fix. If PREDICT/SIMULATE named a producer frame in a DIFFERENT
            # function, trace it and MERGE its source into the chain PLAN sees. Purely additive.
            prod_func, prod_file, prod_line = _producer_frame(findings.root_cause, findings.simulation)
            sym = (findings.bug_function or "").split(".")[-1]
            if prod_func and prod_func != sym:
                rc_chain = self.trace(prod_func, prod_file or findings.bug_file, prod_line)
                if rc_chain and "call chain unavailable" not in rc_chain[:80] and rc_chain not in findings.call_chain:
                    findings.call_chain += (
                        "\n\n=== ADDITIONAL CALL CHAIN around the ROOT-CAUSE / PRODUCER frame "
                        f"'{prod_func}' (where the wrong value is FIRST CONSTRUCTED) ===\n{rc_chain}"
                    )
                    self._log(
                        f"    [subagent] (4b) merged producer frame '{prod_func}' source into the "
                        f"call chain for PLAN (symptom was '{sym}')"
                    )
            findings.files_to_edit, findings.reproduction_script, findings.raw_plan = self.plan_and_script(
                findings.bug_function, findings.bug_file, findings.call_chain,
                findings.simulation, findings.prediction,
            )
            self._log(
                f"    [subagent] (5) files to edit: {findings.files_to_edit or '[]'}; "
                f"reproduction script: {'yes' if findings.reproduction_script else 'no'}"
            )
            # Require a localized bug FUNCTION: without it the call chain is unavailable and the
            # reminder would advertise "SUSPECTED BUG SITE: unknown", which is worse than the plain
            # precheck fallback. (A non-compliant model that only ever emits tool-call text fails
            # here even though simulation/raw_plan are non-empty strings of that same junk.)
            findings.ok = bool(findings.bug_function) and bool(
                findings.simulation or findings.reproduction_script or findings.raw_plan
            )
        except Exception as e:  # never let the sub-agent abort the main run
            self._log(f"    [subagent] pipeline error ({type(e).__name__}: {e}); falling back to plain reminder")
            findings.ok = False
        return findings


def _content_to_text(content) -> str:
    """Flatten a model message ``content`` (string or list-of-parts) into plain text."""
    if isinstance(content, list):
        return " ".join((x.get("text", "") or "") for x in content if isinstance(x, dict))
    return content or ""


def _text_from_format_error(err: "FormatError") -> str:
    """Recover the assistant text from the response persisted on a no-tool-call FormatError.

    ``LitellmModel.query`` stashes ``response.model_dump()`` onto the error's reprompt message
    before raising (its "persist the response on FormatError" contract). The sub-agent asks for
    plain text, so a tool-call model legitimately returns zero tool calls and raises here -- we
    pull the text out instead of treating it as a failure.
    """
    try:
        response = (err.messages[0].get("extra") or {}).get("response")
    except (AttributeError, IndexError, TypeError):
        return ""
    if not isinstance(response, dict):
        return ""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    parts = [_content_to_text(message.get("content"))]
    for key in ("reasoning_content", "reasoning"):
        val = message.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return "\n".join(p for p in parts if p)


@contextlib.contextmanager
def _force_no_tool_call(model):
    """Best-effort: set ``tool_choice="none"`` so the sub-agent's reasoning turns issue no tool.

    Some providers honour it (returning text directly); others ignore it (we still recover the
    text from the FormatError). No-op for models without a ``config.model_kwargs`` dict.
    """
    cfg = getattr(model, "config", None)
    mk = getattr(cfg, "model_kwargs", None)
    if not isinstance(mk, dict):
        yield
        return
    sentinel = object()
    prev = mk.get("tool_choice", sentinel)
    mk["tool_choice"] = "none"
    try:
        yield
    finally:
        if prev is sentinel:
            mk.pop("tool_choice", None)
        else:
            mk["tool_choice"] = prev


def _usage_from_response(litellm, resp, model_name: str = "") -> dict:
    """Extract ``{prompt_tokens, completion_tokens, total_tokens, cost}`` from a litellm response.

    ``resp`` may be a litellm ``ModelResponse`` (attrs) or a ``model_dump`` dict. Cost is computed
    with ``litellm.completion_cost`` (best-effort; 0.0 if the model is unregistered / it errors).
    """
    def _g(o, k):
        return o.get(k) if isinstance(o, dict) else getattr(o, k, None)

    u = _g(resp, "usage")
    pt = int(_g(u, "prompt_tokens") or 0) if u is not None else 0
    ct = int(_g(u, "completion_tokens") or 0) if u is not None else 0
    tt = int((_g(u, "total_tokens") if u is not None else None) or (pt + ct))
    cost = float((_g(u, "cost") if u is not None else None) or 0.0)
    if litellm is not None:
        try:
            cost = float(litellm.completion_cost(completion_response=resp, model=model_name or None) or 0.0)
        except Exception:
            pass
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt, "cost": cost}






def _bounded_response(text: str) -> str:
    """Cap persisted/prompted text while retaining both labels at the head and conclusions at tail."""
    text = text or ""
    cap = SUBAGENT_RESPONSE_CHAR_CAP
    if not cap or len(text) <= cap:
        return text
    half = max(1, (cap - 120) // 2)
    omitted = len(text) - (2 * half)
    return (text[:half] + f"\n\n... [SUBAGENT RESPONSE TRUNCATED: {omitted} chars] ...\n\n"
            + text[-half:])


def _is_timeout_error(exc: BaseException) -> bool:
    """Recognize provider/http timeout wrappers without importing optional HTTP clients."""
    seen = set()
    cur: "BaseException | None" = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        name = type(cur).__name__.lower()
        msg = str(cur).lower()
        if "timeout" in name or "timed out" in msg or "read timeout" in msg:
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return False


def _query_via_model(model, prompt: str, on_usage: Optional[Callable[[dict], None]] = None) -> str:
    """Fallback ``prompt -> text`` using the agent's own ``model.query``.

    Used when a direct litellm call is unavailable (e.g. test fakes, custom model classes).
    The sub-agent's prompts ask for plain text, which makes tool-call models raise
    :class:`FormatError` (zero tool calls); we nudge ``tool_choice="none"`` and, if the model
    still raises, recover the text from the response it persists on the error. Reports token/cost
    usage via ``on_usage`` when the model surfaces it (``extra.response.usage`` / ``extra.cost``).
    """
    fmt = getattr(model, "format_message", None)

    def _mk(role: str, content: str) -> dict:
        return fmt(role=role, content=content) if callable(fmt) else {"role": role, "content": content}

    def _one(messages: list) -> str:
        with _force_no_tool_call(model):
            try:
                resp = model.query(messages)
            except FormatError as e:
                if on_usage is not None:
                    try:
                        response = (e.messages[0].get("extra") or {}).get("response")
                        if isinstance(response, dict):
                            ev = _usage_from_response(None, response)
                            if isinstance(response.get("usage"), dict):
                                on_usage(ev)
                    except Exception:
                        pass
                return _text_from_format_error(e)
        if isinstance(resp, dict):
            if on_usage is not None:
                extra = resp.get("extra") or {}
                ev = _usage_from_response(None, extra.get("response") or {})
                ev["cost"] = float(extra.get("cost") or ev.get("cost") or 0.0)
                on_usage(ev)
            return _bounded_response(_content_to_text(resp.get("content")))
        return _bounded_response(str(resp or ""))

    # The SCRIPT step's answer legitimately contains a ```bash block; every other step must not.
    allow_commands = "REPRODUCTION_SCRIPT" in prompt
    messages = [_mk("system", _SUBAGENT_SYSTEM), _mk("user", prompt)]
    text = _one(messages)
    if not _looks_like_toolcall(text, allow_commands=allow_commands):
        return text
    # The model emitted a tool call / shell command instead of answering -- retry ONCE.
    messages.append(_mk("assistant", text))
    messages.append(_mk("user", _TOOLCALL_CORRECTION))
    return _one(messages) or text


class QueryReply(str):
    """Text-compatible reply retaining completion status for structured consumers."""
    def __new__(cls, text, finish_reason=None, truncated=False):
        obj = super().__new__(cls, text)
        obj.finish_reason = finish_reason
        obj.truncated = truncated
        return obj


def _litellm_query_compliant(litellm, model_name, prompt, base_kwargs, on_usage,
                             before_query=None) -> str:
    """One sub-agent step via ``litellm.completion`` (no tools), retried once if non-compliant.

    Sends a system prompt that strips the agentic frame plus the step's user prompt; if the model
    answers with its tool-call text instead of the requested labels, appends the offending turn and
    :data:`_TOOLCALL_CORRECTION` and re-queries once. Usage is reported per underlying call.
    """
    messages = [
        {"role": "system", "content": _SUBAGENT_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    def _one() -> str:
        if before_query is not None:
            before_query()
        resp = litellm.completion(model=model_name, messages=messages, **base_kwargs)
        finish = getattr(resp.choices[0], "finish_reason", None)
        if on_usage is not None:
            on_usage(dict(_usage_from_response(litellm, resp, model_name), finish_reason=finish))
        message = resp.choices[0].message
        text = _content_to_text(getattr(message, "content", None))
        # Internal reasoning is not the requested structured answer.
        bounded = _bounded_response(text or "")
        return QueryReply(bounded, finish, bounded != (text or ""))

    allow_commands = "REPRODUCTION_SCRIPT" in prompt  # only the SCRIPT step may answer with ```bash
    text = _one()
    if not _looks_like_toolcall(text, allow_commands=allow_commands):
        return text
    messages.append({"role": "assistant", "content": text})
    messages.append({"role": "user", "content": _TOOLCALL_CORRECTION})
    return _one() or text


def _make_agent_query_fn(agent, on_usage: Optional[Callable[[dict], None]] = None) -> Optional[Callable[[str], str]]:
    """Wrap the agent's model into a ``prompt -> text`` callable for the sub-agent's steps.

    The sub-agent reasons in plain text, but the main agent's model is configured as a
    *tool-call* model (mini always passes a bash tool), so asking it to "just answer" makes it
    either emit a bash tool call instead of answering or return empty content. So the PRIMARY
    path calls ``litellm.completion`` directly with NO tools, reusing the agent's
    ``model_name`` + ``model_kwargs`` (minus tool-only kwargs). This guarantees a clean text
    answer. If litellm or the model name is unavailable (test fakes, custom models), it falls
    back to :func:`_query_via_model`.

    ``on_usage`` (if given) is called once per model call with a token/cost dict so the caller can
    account the sub-agent's spend (the direct litellm path does NOT accrue to ``agent.cost``, so
    this is how the sub-agent's tokens/cost get reflected in the trajectory).
    """
    model = getattr(agent, "model", None)
    if model is None:
        return None
    cfg = getattr(model, "config", None)
    model_name = getattr(cfg, "model_name", None)

    try:
        import litellm  # noqa: F401
    except Exception:
        litellm = None

    if litellm is not None and model_name:
        base_kwargs = dict(getattr(cfg, "model_kwargs", {}) or {})
        for k in ("tools", "tool_choice", "parallel_tool_calls"):
            base_kwargs.pop(k, None)
        # Drop None-valued kwargs (e.g. ``temperature: null`` in the minimax config) which some
        # providers reject; keep reasoning/thinking effort settings intact.
        base_kwargs = {k: v for k, v in base_kwargs.items() if v is not None}
        base_kwargs.setdefault("drop_params", True)
        if SUBAGENT_QUERY_TIMEOUT:
            base_kwargs.setdefault("timeout", SUBAGENT_QUERY_TIMEOUT)
        if SUBAGENT_MAX_COMPLETION_TOKENS:
            base_kwargs.setdefault("max_tokens", SUBAGENT_MAX_COMPLETION_TOKENS)

        def record_usage(event):
            charge = getattr(model, "record_direct_usage", None)
            if callable(charge):
                charge(event)
            if on_usage is not None:
                on_usage(event)

        def query(prompt: str, *, max_tokens=None) -> str:
            kwargs = dict(base_kwargs)
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            try:
                return _litellm_query_compliant(litellm, model_name, prompt, kwargs, record_usage,
                                                 getattr(model, "check_budget", None))
            except Exception as e:
                # Retrying the same stalled provider through model.query defeats the timeout and
                # can double the maximum wall time.  Let the caller record a failed/incomplete
                # unit; retain the fallback for compatibility/provider-parameter failures.
                if _is_timeout_error(e):
                    raise
                # Provider hiccup / unsupported kwarg -> fall back to the agent's own model.
                if hasattr(model, "query"):
                    return _query_via_model(model, prompt, on_usage=on_usage)
                return ""

        query.supports_output_limit = True
        return query

    if hasattr(model, "query"):
        return lambda prompt: _query_via_model(model, prompt, on_usage=on_usage)
    return None


def _repo_path_for(agent, override: Optional[str]) -> str:
    if override:
        return override
    cfg = getattr(getattr(agent, "env", None), "config", None)
    cwd = getattr(cfg, "cwd", "") or ""
    return cwd if cwd and cwd != "/" else "/testbed"


# =============================================================================
# The hook
# =============================================================================
def sum_usage(events: list) -> dict:
    """Aggregate token/cost usage over a list of sub-agent usage events."""
    agg = {"calls": len(events), "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}
    for e in events:
        agg["prompt_tokens"] += int(e.get("prompt_tokens") or 0)
        agg["completion_tokens"] += int(e.get("completion_tokens") or 0)
        agg["total_tokens"] += int(e.get("total_tokens") or 0)
        agg["cost"] += float(e.get("cost") or 0.0)
    return agg


