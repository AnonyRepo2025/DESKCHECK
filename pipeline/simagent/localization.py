from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path



from simagent import subagent as _sub
from simagent.subagent import _strip_md

# Language-derived path regexes, rebuilt on every _sub.set_lang (Python instances match
# exactly ``.py``; Go ``.go``; JS the node source extensions).
_FRAME_PAIR_LOOSE_RE = _SRC_HIT_LINE_RE = _SRC_HIT_RE = _SYM_HIT_RE = None


@_sub.on_lang_change
def _compile_lang_regexes():
    global _FRAME_PAIR_LOOSE_RE, _SRC_HIT_LINE_RE, _SRC_HIT_RE, _SYM_HIT_RE
    ext = _sub.src_ext_alt()
    _FRAME_PAIR_LOOSE_RE = re.compile(
        r"([A-Za-z_][\w.]*)\s*(?:\([^)]*\))?\s*\|\s*((?:[\w.-]+/)*[\w-][\w.-]*\.(?:" + ext + r"))")
    _SRC_HIT_LINE_RE = re.compile(r"^(\S+?\.(?:" + ext + r")):(\d+):")
    _SRC_HIT_RE = re.compile(r"^(\S+?\.(?:" + ext + r")):\d+:")
    _SYM_HIT_RE = re.compile(r"\s*([A-Za-z_]\w*):\s*(\S+\.(?:" + ext + r")):\d+")


MODEL = os.getenv("MODEL", "anthropic/claude-sonnet-4-5-20250929")


os.environ.setdefault("CHAIN_SRC_LINES", "200")    # was 40
os.environ.setdefault("CHAIN_SRC_CHARS", "8000")   # was 1400
os.environ.setdefault("CHAIN_SRC_TOTAL", "40000")  # was 12000 (cumulative across frames)
os.environ.setdefault("CHAIN_SRC_FRAMES", "18")    # was 14

PLAN_SEARCH = os.getenv("PLAN_SEARCH", "1") in ("1", "true", "yes")


# SIBLING-EXPANSION (Tier-1): one extra model pass that asks WHERE ELSE the same defect
# manifests -- same-name methods in other backends/factories (grounded by a deterministic
# ``def <symbol>(`` sweep) and PARALLEL PATHS to the same user-visible output (a readonly
# renderer vs the form widget, sync vs async). PLAN_SEARCH greps for the defect PATTERN, which
# misses siblings that express the defect differently (django-13512: the readonly path calls
# ``field.get_prep_value``, not ``json.dumps``, so no pattern grep ever surfaced it). Every
# proposed site is verified to exist via grep before it is added.
SIBLING_EXPAND = os.getenv("SIBLING_EXPAND", "1") in ("1", "true", "yes")


import minisweagent as _minisweagent  # noqa: E402
SWEBENCH_YAML = Path(_minisweagent.__file__).parent / "config/benchmarks/swebench.yaml"
OUT_DIR = Path(os.getenv("OUT_DIR", "runs/localization"))


_HUNK_CTX_RE = re.compile(r"^@@.*@@\s*(.*)$")
# Python def/class OR Go func (with optional receiver) / type -- one capture group each way.
_DEF_RE = re.compile(r"\b(?:def|class)\s+(\w+)|\bfunc\s+(?:\([^)]*\)\s*)?(\w+)\s*\(|\btype\s+(\w+)\b")


def _def_group(m) -> str:
    """First non-empty capture of _DEF_RE/_BODY_DEF_RE (py def/class, go func, go type)."""
    return next((g for g in m.groups() if g), "") if m else ""


def _strip_ab(path: str) -> str:
    """Normalise a path token (diff ``a/``/``b/`` prefix, ``/testbed`` root, leading slash)."""
    p = (path or "").strip().strip('"').replace("\\", "/")
    p = p.lstrip("/")
    for pre in ("testbed/", "a/", "b/", "./"):
        while p.startswith(pre):
            p = p[len(pre):]
    return p


_BODY_DEF_RE = re.compile(
    r"\s*(?:async\s+)?(?:def|class)\s+(\w+)|func\s+(?:\([^)]*\)\s*)?(\w+)\s*\(|type\s+(\w+)\b")
_TOPLEVEL_ASSIGN_RE = re.compile(r"(\w+)\s*=")  # column-0 module-level assignment target


def parse_gold_patch(patch: str) -> "tuple[list[str], list[str]]":
    """Return (gold_files, gold_functions) from a unified-diff gold patch.

    Files come from ``+++ b/<path>`` (the post-image; skips ``/dev/null`` deletions) and
    ``diff --git`` headers (covers renames).

    Functions: each changed (``+``/``-``) line is attributed to the nearest PRECEDING
    ``def``/``class`` line that exists in the ORIGINAL file (a context or ``-`` line inside the
    hunk body), falling back to a column-0 module-level assignment target (e.g. a module regex the
    patch edits in place), then to the ``@@ ... @@ <context>`` hunk-header def. The old
    header-only heuristic named the def the hunk STARTS inside -- often the function ABOVE the
    edited one (run-3: it scored ``catch_all_view``/``delete_cookie``/``standard_duration_re``
    predictions as misses). Purely-ADDED defs never become anchors: a brand-new function's name is
    unpredictable pre-patch, so lines inside it attribute to the surrounding original def (the
    insertion site).
    """
    files: list[str] = []
    funcs: list[str] = []

    def _add_func(name: str) -> None:
        if name and name not in funcs:
            funcs.append(name)

    lines = (patch or "").splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                for tok in parts[2:4]:
                    f = _strip_ab(tok)
                    if f and f != "dev/null" and f not in files:
                        files.append(f)
        elif line.startswith("+++ ") or line.startswith("--- "):
            p = line[4:].strip()
            if p and p != "/dev/null":
                f = _strip_ab(p)
                if f and f not in files:
                    files.append(f)
        elif line.startswith("@@"):
            header_func = ""
            m = _HUNK_CTX_RE.match(line)
            if m and m.group(1).strip():
                dm = _DEF_RE.search(m.group(1))
                if dm:
                    header_func = _def_group(dm)
            last_def = ""     # nearest preceding def/class present in the ORIGINAL file
            last_assign = ""  # nearest preceding column-0 assignment target (original lines)
            j = i + 1
            while j < n and not lines[j].startswith(("@@", "diff --git", "--- ", "+++ ")):
                b = lines[j]
                tag, body = (b[0], b[1:]) if b[:1] in ("+", "-", " ") else ("", b)
                if tag in (" ", "-"):  # exists in the original file -> can anchor
                    dm = _BODY_DEF_RE.match(body)
                    if dm and _def_group(dm):
                        last_def = _def_group(dm)
                    elif _TOPLEVEL_ASSIGN_RE.match(body):  # anchored at column 0 only
                        last_assign = _TOPLEVEL_ASSIGN_RE.match(body).group(1)
                stripped = body.strip()
                # Import lines are module plumbing, not a function edit -- attributing them would
                # inject the hunk-header context (often a docstring line git picked up, e.g.
                # ``class Child(Model):`` for django-16256's asgiref import) into the gold set.
                if tag in ("+", "-") and stripped and not re.match(r"(?:from|import)\s", stripped):
                    _add_func(last_def or last_assign or header_func)
                j += 1
            i = j
            continue
        i += 1
    return files, funcs


def _file_match(pred: str, gold: str) -> bool:
    """True if predicted and gold paths refer to the same file (segment-aligned suffix match)."""
    pn, gn = _strip_ab(pred), _strip_ab(gold)
    if not pn or not gn:
        return False
    if pn == gn:
        return True
    return pn.endswith("/" + gn) or gn.endswith("/" + pn)


def _any_match(pred: str, golds: "list[str]") -> bool:
    return any(_file_match(pred, g) for g in golds)


def _harvest_paths(text: str) -> "list[str]":
    """Pull file-path-looking tokens (``dir/.../name.ext``) out of a free-text field."""
    out: list[str] = []
    for m in re.finditer(r"(?:[\w.-]+/)+[\w.-]+\.[A-Za-z][A-Za-z0-9]{0,4}", text or ""):
        f = _strip_ab(m.group(0))
        if f and f not in out:
            out.append(f)
    return out


def _harvest_funcs(text: str) -> "list[str]":
    """First bareword token(s) of a ROOT CAUSE line like ``foo | path:line | ...``."""
    head = (text or "").split("|", 1)[0]
    return [m.group(0).split(".")[-1] for m in re.finditer(r"[A-Za-z_]\w+", head)][:1]


def _split_site(entry: str) -> "tuple[str, str]":
    """Split a ``<file> :: <symbol>`` edit-site entry into (file, bare-symbol).

    Tolerates ``::``, ``|`` or `` - `` separators; returns ("", "") parts that are absent. The
    symbol is reduced to its last dotted segment for ``Class.method`` so it matches a gold ``def``
    name, but the class name is also recoverable by the caller when needed.
    """
    parts = re.split(r"\s*(?:::|\||\s-\s)\s*", entry or "", maxsplit=1)
    fpaths = _harvest_paths(parts[0])
    f = fpaths[0] if fpaths else _strip_ab(parts[0])
    sym = ""
    if len(parts) > 1:
        sm = re.search(r"[A-Za-z_][\w.]*", parts[1])
        sym = sm.group(0) if sm else ""
    return f, sym


def _func_match(pred: str, gold_funcs: "list[str]") -> bool:
    """True if any dotted segment of a predicted symbol (``Class.method``) is a gold def/class."""
    return any(seg and seg in gold_funcs for seg in (pred or "").split("."))


def score_localization(findings, gold_files: "list[str]", gold_funcs: "list[str]") -> dict:
    """Compare the sub-agent's localization fields to the gold patch and compute metrics.

    Understands BOTH shapes of ``files_to_edit``: plain ``path/file.py`` (stock mode) and the
    localization-only ``path/file.py :: function`` edit sites (the EMIT_SCRIPT=0 default), from
    which it also scores FUNCTION-level localization against the gold patch's hunk functions.
    """
    bug_file = findings.bug_file or ""
    files_to_edit = list(findings.files_to_edit or [])
    # root_cause is free text "func | file:line | phrase"; harvest its file + function.
    rc_files = _harvest_paths(findings.root_cause)
    rc_funcs = _harvest_funcs(findings.root_cause)

    # Each edit site may carry a function: "<file> :: <symbol>". Split into files + symbols.
    site_pairs = [_split_site(s) for s in files_to_edit]
    fte = []
    fte_funcs = []
    for f, sym in site_pairs:
        f = _strip_ab(f)
        if f and f not in fte:
            fte.append(f)
        if sym and sym not in fte_funcs:
            fte_funcs.append(sym)

    # The union of every file the sub-agent points at (locate site + plan list + root-cause frame).
    predicted_files: list[str] = []
    for f in ([bug_file] if bug_file else []) + fte + rc_files:
        f = _strip_ab(f)
        if f and f not in predicted_files:
            predicted_files.append(f)

    # files_to_edit precision/recall vs gold files.
    matched_preds = [p for p in fte if _any_match(p, gold_files)]
    matched_gold = [g for g in gold_files if any(_file_match(p, g) for p in fte)]
    precision = (len(matched_preds) / len(fte)) if fte else 0.0
    recall = (len(matched_gold) / len(gold_files)) if gold_files else 0.0

    # FUNCTION-level: every function the sub-agent named (locate fn + root-cause fn + edit-site fns).
    funcs_pred: list[str] = []
    for f in ([findings.bug_function] if findings.bug_function else []) + rc_funcs + fte_funcs:
        if f and f not in funcs_pred:
            funcs_pred.append(f)
    func_hit = any(_func_match(f, gold_funcs) for f in funcs_pred)
    matched_gold_funcs = [g for g in gold_funcs if any(_func_match(p, g) for p in funcs_pred)]
    func_recall = (len(matched_gold_funcs) / len(gold_funcs)) if gold_funcs else 0.0
    edit_func_preds = [s for s in fte_funcs]
    edit_func_prec = (
        sum(1 for s in edit_func_preds if _func_match(s, gold_funcs)) / len(edit_func_preds)
        if edit_func_preds else 0.0
    )

    return {
        "locate_file_hit": _any_match(bug_file, gold_files) if bug_file else False,
        "root_cause_file_hit": any(_any_match(f, gold_files) for f in rc_files),
        "any_file_hit": any(_any_match(f, gold_files) for f in predicted_files),
        "files_to_edit_precision": round(precision, 3),
        "files_to_edit_recall": round(recall, 3),
        "files_to_edit_full_recall": bool(gold_files) and recall == 1.0,
        "func_hit": func_hit,
        "func_recall": round(func_recall, 3),
        "func_full_recall": bool(gold_funcs) and func_recall == 1.0,
        "edit_site_func_precision": round(edit_func_prec, 3),
        "predicted_files": predicted_files,
        "predicted_funcs": funcs_pred,
        "files_to_edit": fte,
        "edit_sites": [f"{f} :: {s}" if s else f for f, s in site_pairs],
        "matched_gold_files": matched_gold,
        "matched_gold_funcs": matched_gold_funcs,
    }


_EDIT_LINE_RE = re.compile(r"^\s*EDIT\s*:\s*(.+?)\s*$", re.IGNORECASE)


_RULEDOUT_LINE_RE = re.compile(r"^\s*RULED[-_ ]?OUT\s*:\s*(.+?)\s*$", re.IGNORECASE)

#: Sites PLAN_LOCALIZE explicitly ruled out (plus candidates VERIFY-PRUNE dropped) for the LAST
#: instance processed. Reset per ``_localize_plan_and_script`` call; the pipeline copies it onto
#: ``findings.demoted_sites`` so ``repair_sites`` can re-inspect them as advisory candidates
#: (recall-biased routing: a demoted site is cheap for repair to inspect-and-decline, but a
#: dropped one is unrecoverable).
_DEMOTED_SITES: "list[str]" = []
#: Why each demoted site was demoted (PLAN's own rule-out reason / verify-prune) -- surfaced to
#: the repair_sites reconciliation so it can weigh the demotion instead of guessing.
_DEMOTED_REASONS: "dict[str, str]" = {}


#: Set per-instance by ``localize_one`` to a ``search(pattern, glob="") -> str`` bound to the env.
_REPO_SEARCH_TOOL = None
#: Set per-instance by ``localize_one`` so the deterministic helpers can run raw grep/awk.
_ENV = None
_REPO_PATH = "/testbed"


def make_repo_search_tool(env, repo_path: str = "/testbed", *, sink: "Optional[list]" = None,
                          timeout: int = 60, max_chars: int = 4000, max_hits: int = 60):
    """Return a READ-ONLY ``search(pattern, glob="") -> str`` that greps the repo inside ``env``.

    We construct the command ourselves (``grep -rnE`` over ``$REPO_PATH`` with test/vendor dirs
    excluded) -- the model only supplies the pattern (+ optional ``--include`` glob), so it cannot
    run arbitrary shell. Hits are capped. Each call is appended to ``sink`` so it shows up in the
    trajectory in call order alongside the model steps.
    """
    def search(pattern: str, glob: str = "") -> str:
        out = ""
        if env is not None and pattern:
            include = f"--include={shlex.quote(glob)} " if glob else f"{_sub.src_includes()} "
            excl = "--exclude-dir=.git --exclude-dir=tests --exclude-dir=test --exclude-dir=node_modules"
            if _sub.is_go():
                excl += " --exclude='*_test.go' --exclude-dir=vendor --exclude-dir=testdata"
            elif _sub.is_js():
                excl += (" --exclude='*.test.*' --exclude='*-test.*' --exclude='*.spec.*' "
                         "--exclude='*.d.ts' --exclude='*.min.js' --exclude-dir=__tests__ "
                         "--exclude-dir=__mocks__ --exclude-dir=__snapshots__ --exclude-dir=dist "
                         "--exclude-dir=build --exclude-dir=coverage --exclude-dir=vendor")
            cmd = (
                f"grep -rnE {include}{excl} -- {shlex.quote(pattern)} {shlex.quote(repo_path)} "
                f"2>/dev/null | head -n {int(max_hits)}"
            )
            try:
                result = env.execute({"command": cmd}, timeout=timeout)
                out = (result.get("output") or "")[:max_chars]
            except Exception as e:  # never let a search kill the run
                out = f"[search failed: {type(e).__name__}: {e}]"
        if sink is not None:
            sink.append({
                "idx": len(sink),
                "type": "repo_search",
                "step": "PLAN_SEARCH",
                "args": {"pattern": pattern, "glob": glob},
                "output": out,
            })
        return out

    return search


def _parse_edit_sites(text: str, reasons: "Optional[dict]" = None) -> "list[str]":
    """Harvest ``<file> :: <symbol>`` edit sites from the localization-only PLAN answer.

    Drops the reproduction-script filename if the model still slips one in, and de-dups. Returns
    the raw ``file :: symbol`` strings (stored verbatim in ``files_to_edit`` so the scorer can
    recover both the file and the function). When ``reasons`` is a dict, the model's own
    ``-- <why>`` tail of each EDIT line is recorded per site (surfaced to the repair agent).
    """
    sites: list[str] = []
    for line in _strip_md(text).splitlines():
        m = _EDIT_LINE_RE.match(line)
        if not m:
            continue
        parts = re.split(r"\s+--\s+|\s+—\s+", m.group(1), maxsplit=1)
        f, sym = _split_site(parts[0])
        if not f or _sub._REPRO_NAME_RE.search(f):
            continue
        entry = f"{f} :: {sym}" if sym else f
        if entry not in sites:
            sites.append(entry)
            if reasons is not None and len(parts) > 1 and parts[1].strip():
                reasons[entry] = "PLAN chose it: " + parts[1].strip()
    return sites


def _parse_ruled_out(text: str, reasons: "Optional[dict]" = None) -> "list[str]":
    """Harvest ``RULED-OUT: <file> :: <symbol> -- <reason>`` lines (reason kept when asked)."""
    out: "list[str]" = []
    for line in _strip_md(text).splitlines():
        m = _RULEDOUT_LINE_RE.match(line)
        if not m:
            continue
        parts = re.split(r"\s+--\s+|\s+—\s+", m.group(1), maxsplit=1)
        f, sym = _split_site(parts[0])
        if not f:
            continue
        entry = f"{f} :: {sym}" if sym else f
        if entry not in out:
            out.append(entry)
            if reasons is not None and len(parts) > 1 and parts[1].strip():
                reasons[entry] = parts[1].strip()
    return out


_NOISE_FUNCS = {"self", "print", "format", "split", "join", "len", "range", "str", "int", "list",
                "dict", "get", "set", "append", "super", "type", "isinstance", "getattr", "setattr",
                "return", "assert", "import", "value", "object", "tuple", "bool", "float"}


# FRAME-UNION helpers: frames the simulations/prediction ESTABLISHED must change. The path may be
# bare ("residue_ntheory.py") or missing -- models copy whatever the call chain showed them.
_MUST_CHANGE_RE = re.compile(r"FRAMES\s+THAT\s+MUST\s+CHANGE\s*:?(.*)", re.IGNORECASE | re.S)
# _FRAME_PAIR_LOOSE_RE: bound by _compile_lang_regexes (per-language source extensions)
# "MUST-CHANGE: <func> | <file>:<line> | <why>" -- the alternate-candidate simulation's verdict
# that the competing hypothesis is a real fix site (see _sub._ALT_SIM_PROMPT). Tolerates a
# leading label ("VERDICT: MUST-CHANGE: ...") -- the prompt's "VERDICT: end with ..." wording
# invites that concatenation and models produce it.
_MUST_CHANGE_LINE_RE = re.compile(
    r"^[\s>*\-`#]*(?:[A-Za-z][\w ]{0,12}:\s*)?MUST[-_ ]?CHANGE\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE)


def _mandatory_frames(simulation: str, prediction: str) -> "list[tuple[str, str]]":
    """(func, file) frames PLAN is not allowed to silently drop.

    Sources, in order: the SIMULATION-SUMMARY's ``FRAMES THAT MUST CHANGE`` block, then the
    producer frame (PREDICT's ROOT CAUSE FRAME, the SIMULATE step's adversarial ``PRODUCER:``
    verdict, or the inline FIRST CONSTRUCTED marker -- ``_sub._producer_frame``'s priority).
    ``file`` may be a bare filename or empty; the caller resolves it against the repo.
    """
    out: "list[tuple[str, str]]" = []

    def _add(func: str, fpath: str) -> None:
        bare = (func or "").split(".")[-1]
        if not re.fullmatch(r"[A-Za-z_]\w*", bare or "") or bare in _NOISE_FUNCS:
            return
        if any(bare == f.split(".")[-1] for f, _ in out):
            return
        out.append((func, _strip_ab(re.sub(r":\d+.*$", "", fpath or ""))))

    fm = _MUST_CHANGE_RE.search(simulation or "")
    if fm:
        for func, fpath in _FRAME_PAIR_LOOSE_RE.findall(fm.group(1)):
            _add(func, fpath)
    # Alternate-candidate simulation verdicts: a traced competing hypothesis that concluded
    # "MUST-CHANGE" is a mandatory frame (PLAN may still explicitly rule it out).
    for m in _MUST_CHANGE_LINE_RE.finditer(simulation or ""):
        body = m.group(1)
        pm = _FRAME_PAIR_LOOSE_RE.search(body)
        if pm:
            _add(pm.group(1), pm.group(2))
        else:  # no file given -- take the leading identifier, resolve the file later
            fm2 = re.match(r"`?([A-Za-z_][\w.]*)`?", body)
            if fm2:
                _add(fm2.group(1), "")
    pf, pfile, _pline = _sub._producer_frame(_sub._extract_root_cause(prediction or ""), simulation or "")
    if pf:
        _add(pf, pfile)
    return out


def _resolve_frame_file(fpath: str, bare_func: str) -> str:
    """Resolve a frame's file to a repo-relative path (frames often carry only the basename).

    A path with a directory is returned as-is. A bare filename (or no file at all) is resolved by
    grepping the repo for ``def <bare_func>`` (restricted to that basename when given); falls back
    to the bare filename itself -- the scorer and repair still match it by basename suffix.
    """
    f = _strip_ab(re.sub(r":\d+.*$", "", fpath or ""))
    if f and "/" in f:
        return f
    if _REPO_SEARCH_TOOL is not None and bare_func:
        hits = _REPO_SEARCH_TOOL(_sub.def_pattern(bare_func), f or "") or ""
        for ln in hits.splitlines():
            m = _SRC_HIT_RE.match(ln.strip())
            if not m:
                continue
            relf = _strip_ab(m.group(1))
            if _sub.is_test_path(relf):
                continue
            return relf
    return f


# =============================================================================
# Trajectory capture -- record every sub-agent step (prompt + output) in call order
# =============================================================================
# Each LLM step's prompt opens with a distinctive phrase; map it to a readable step label so the
# saved trajectory reads LOCATE -> SIMULATE -> PREDICT -> PLAN_SCRIPT (plus the two retry prompts).
_STEP_MARKERS = [
    ("BUG-LOCALIZATION step", "LOCATE"),
    ("Your previous answer REFUSED", "SIMULATE_RETRY"),
    ("ALTERNATE-CANDIDATE SIMULATION step", "SIMULATE_ALT"),
    ("SIMULATION step of an automated", "SIMULATE"),
    ("did not isolate a ROOT CAUSE", "PREDICT_RETRY"),
    ("PREDICT-AND-COMPARE step", "PREDICT"),
    ("SEARCH-PLANNING step", "PLAN_SEARCH"),
    ("SIBLING-EXPANSION step", "SIBLING_EXPAND"),
    ("FINAL LOCALIZATION step", "PLAN_LOCALIZE"),
    ("VERIFY-PRUNE step", "VERIFY_PRUNE"),
    ("FINAL step of an automated", "PLAN_SCRIPT"),
]


def _step_label(prompt: str) -> str:
    for marker, label in _STEP_MARKERS:
        if marker in (prompt or ""):
            return label
    return "QUERY"


def _make_recorders(query_fn, trace_fn, sink: "Optional[list]" = None):
    """Wrap the sub-agent's query/trace callables so each call is appended to a trajectory list.

    Returns ``(wrapped_query, wrapped_trace, trajectory)`` -- the trajectory is an ordered list of
    ``{idx, type, step, prompt|args, output}`` events, one per model call and per call-graph trace,
    captured exactly as the pipeline issues them. Pass ``sink`` to append into an EXISTING list
    (so other instrumented tools -- e.g. the PLAN-step repo search -- interleave in call order).
    """
    trajectory: list[dict] = sink if sink is not None else []

    def wrapped_query(prompt: str) -> str:
        out = query_fn(prompt) or ""
        trajectory.append({
            "idx": len(trajectory),
            "type": "model",
            "step": _step_label(prompt),
            "prompt": prompt,
            "output": out,
        })
        return out

    def wrapped_trace(function_name: str, file_path: str = "", line: int = 0) -> str:
        out = trace_fn(function_name, file_path, line)
        trajectory.append({
            "idx": len(trajectory),
            "type": "trace_call_chain",
            "step": "TRACE",
            "args": {"function": function_name, "file": file_path, "line": line},
            "output": out,
        })
        return out

    return wrapped_query, wrapped_trace, trajectory


# =============================================================================
# Aggregation
# =============================================================================
def summarize(records: "list[dict]") -> dict:
    scored = [r for r in records if "metrics" in r]
    n = len(scored)

    def _rate(key: str) -> float:
        return round(sum(1 for r in scored if r["metrics"].get(key)) / n, 3) if n else 0.0

    def _mean(key: str) -> float:
        return round(sum(r["metrics"].get(key, 0.0) for r in scored) / n, 3) if n else 0.0

    return {
        "n_instances": len(records),
        "n_scored": n,
        "n_no_trigger": sum(1 for r in records if "note" in r and "metrics" not in r),
        "n_errored": sum(1 for r in records if "error" in r),
        "locate_file_hit_rate": _rate("locate_file_hit"),
        "any_file_hit_rate": _rate("any_file_hit"),
        "root_cause_file_hit_rate": _rate("root_cause_file_hit"),
        "files_to_edit_mean_precision": _mean("files_to_edit_precision"),
        "files_to_edit_mean_recall": _mean("files_to_edit_recall"),
        "files_to_edit_full_recall_rate": _rate("files_to_edit_full_recall"),
        "func_hit_rate": _rate("func_hit"),
        "func_mean_recall": _mean("func_recall"),
        "func_full_recall_rate": _rate("func_full_recall"),
        "edit_site_func_mean_precision": _mean("edit_site_func_precision"),
    }


def _load_instances(ids: "list[str]", want_all: bool, limit: int) -> "list[dict]":
    local = os.getenv("INSTANCES_JSONL", "")
    if local and Path(local).exists():
        by_id: dict[str, dict] = {}
        with open(local) as f:
            for line in f:
                line = line.strip()
                if line:
                    inst = json.loads(line)
                    by_id[inst["instance_id"]] = inst
    else:
        from datasets import load_dataset

        by_id = {i["instance_id"]: i for i in load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")}

    if want_all:
        chosen = list(by_id.values())
        if limit:
            chosen = chosen[:limit]
        return chosen
    out = []
    for iid in ids:
        if iid in by_id:
            out.append(by_id[iid])
        else:
            print(f"!! unknown instance {iid}", flush=True)
    return out


