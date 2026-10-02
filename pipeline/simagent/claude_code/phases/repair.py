"""Call 2 -- REPAIR: fix -> simulate the patch by pure code reasoning -> refine on confirmed findings.

R1 (full tools) produces the initial patch. R2/R3 (no tools) are the stock sim-loop lenses and
adversarial verifier (`_run_reasoning_review`), run INSIDE the repair conversation via a
tools-off query callable; the harness grounds every citation and anchors every finding in the
spec. R4 (full tools) runs only when an anchored, confirmed finding survives, and its edits pass
the stock mutation gate (import smoke, covering tests, executed-regression veto).
Both patches are saved: 4_initial_patch.diff and 4_refined_patch.diff.
"""
from __future__ import annotations

import json
import os
import re
import shlex

from jinja2 import Template, Undefined

from simagent import pipeline as sp
from simagent import plainrepair as pr

from .. import flow, prompts
from . import localize

_sub = sp._sub
REPAIR_TURNS = int(os.getenv("REPAIR_STEPS", "200"))
REFINE_TURNS = int(os.getenv("SIMLOOP_REFINE_STEPS", "40"))


# --- J1 (JS/TS port, 2026-09-20): deterministic 4.1 units for JavaScript/TypeScript ----------------
# ``pr._extract_units`` keeps only ``.py`` diff files, so every JS patch produced ZERO units and the
# whole review + refine step was silently skipped (the same failure Go had before _extract_units_go).
_JS_DECL_RE = re.compile(
    r"^(?P<indent>\s*)(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?(?:"
    r"(?:async\s+)?function\s*\*?\s*(?P<fn>[A-Za-z_$][\w$]*)\s*\("
    r"|(?:const|let|var)\s+(?P<var>[A-Za-z_$][\w$]*)\s*(?::[^=]*)?=\s*(?:async\s*)?(?:function\b|\(|<|[A-Za-z_$][\w$]*\s*=>)"
    r"|(?:class|interface|enum)\s+(?P<cls>[A-Za-z_$][\w$]*)\b"
    r"|(?:(?:public|private|protected|static|readonly|override|async|get|set)\s+)*(?P<meth>[A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*\([^;]*\)\s*(?::[^{;=]*)?\{"
    r"|(?P<prop>[A-Za-z_$][\w$]*)\s*:\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*=>)"
    r")")
_JS_STRIP_RE = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`(?:\\.|[^`\\])*`|//[^\n]*|/\*.*?\*/", re.S)


def _js_block_end(lines: "list[str]", start: int, limit: int) -> int:
    """Last line of the declaration beginning at ``start`` (0-based), by brace/paren depth over
    code with strings and comments blanked out. A declaration with no brace (``const f = x => y;``)
    ends at its first line, or at the line closing the expression it opened."""
    depth = 0
    seen = False
    for i in range(start, min(limit, len(lines))):
        code = _JS_STRIP_RE.sub("", lines[i])
        for ch in code:
            if ch in "{([":
                depth += 1
                seen = True
            elif ch in "})]":
                depth -= 1
        if seen and depth <= 0:
            return i
    return min(limit, len(lines)) - 1


def _js_units_from_source(path: str, src: str, changed: "set[int]") -> "list[dict]":
    """Declarations (function / arrow const / class / method / object property) enclosing the
    changed POST-patch lines, with their complete source. Nested declarations are kept as their
    own units -- a method inside a class is the unit the fix usually lives in -- and a unit is
    dropped when an inner unit already covers every changed line it holds."""
    lines = (src or "").splitlines()
    starts = []
    for i, l in enumerate(lines):
        m = _JS_DECL_RE.match(l)
        if m:
            name = m.group("fn") or m.group("var") or m.group("cls") or m.group("meth") or m.group("prop")
            if name and name not in ("if", "for", "while", "switch", "catch", "return", "function"):
                starts.append((i, name, len(m.group("indent"))))
    rows = []
    for k, (s, name, _ind) in enumerate(starts):
        e = _js_block_end(lines, s, len(lines))
        hits = sum(1 for ln in changed if s + 1 <= ln <= e + 1)
        if not hits:
            continue
        rows.append({"file": path, "symbol": name, "code": "\n".join(lines[s:e + 1])[:12000],
                     "hits": hits, "line": s + 1, "start": s, "end": e})
    # prefer the innermost unit: drop a row whose changed lines are all covered by a nested row
    inner = []
    for r in rows:
        if any(o is not r and o["start"] >= r["start"] and o["end"] <= r["end"] and o["hits"] >= r["hits"]
               and (o["end"] - o["start"]) < (r["end"] - r["start"]) for o in rows):
            continue
        inner.append(r)
    return inner or rows


def _extract_units_js(env, repo_path: str, patch_text: str, cap: int) -> "list[dict]":
    changed = {f: v for f, v in pr._changed_lines(patch_text).items()
               if _sub.is_src_path(f) and not _sub.is_test_path(f) and "/node_modules/" not in f}
    rows = []
    for f, v in changed.items():
        try:
            src = env.execute({"command": f"cd {repo_path} && cat {shlex.quote(f)}"}, timeout=30).get("output", "") or ""
        except Exception:
            continue
        rows += _js_units_from_source(f, src, set(v))
    rows.sort(key=lambda r: (-r["hits"], r["file"], r["line"]))
    return [{"id": f"U{i}", "file": r["file"], "symbol": r["symbol"], "code": r["code"]}
            for i, r in enumerate(rows[:cap], 1)]

def _merged_lens_prompt() -> str:
    """Both review lenses in one prompt with one copy of the materials (stock task blocks)."""
    rules = pr._REVIEW_COMMON_RULES
    t_task = pr._REVIEW_TRACE_PROMPT.split("# TASK", 1)[1]
    c_task = pr._REVIEW_CONTRACT_PROMPT.split("# TASK", 1)[1]
    for r in (rules,):
        t_task = t_task.replace(r, "")
        c_task = c_task.replace(r, "")
    return ("{context}\n=== SPECIFICATION ===\n{problem_statement}\n\n=== REQUIREMENTS ===\n{requirements}\n\n"
            "=== INTERFACE ===\n{interface}\n\n{units}\n\n=== SUPPORTING POST-PATCH SOURCE (callees / neighbours) ===\n{pack}\n\n"
            "=== DEFINITIONS OF WHAT THE UNITS CALL (read from the repository) ===\n{callees}\n\n"
            "=== EXISTING CALLERS OF THE CHANGED SYMBOLS ===\n{callers}\n\n=== EXISTING TESTS COVERING THE CHANGED MODULES ===\n{tests}\n\n"
            "=== THE PATCH (what changed; `+` lines are new) ===\n{diff}\n\n=== FORWARDING FACTS (parsed from the patched source) ===\n{facts}\n\n"
            "Apply BOTH lenses below in ONE answer; number findings continuously across lenses.\n\n# TASK" + t_task.rstrip()
            + "\n\n# TASK" + c_task.rstrip() + "\n\n" + rules)


def _render(t: str, **v) -> str:
    return Template(t, undefined=Undefined).render(**v)


MODCACHE_TURNS = int(os.getenv("MODCACHE_FIX_STEPS", "40"))

MODCACHE_PROMPT = """\
[HARNESS FACT -- your change does not ship]
`go mod verify` reports that you edited Go MODULE CACHE source, outside the repository:

{detail}

Only the repository at {repo_path} is collected (`git diff`); a dependency edited under the module
cache is invisible to the graders, so the code that compiles for you now will NOT compile for them
(vuls-a76302c1 died exactly this way: `nvd.Cvss40 undefined`).

Do now, in order:
  1. Restore the dependency to its published state: `go clean -modcache` is too slow -- instead run
     `go mod verify` again after restoring each reported module from the module cache download zip,
     or simply undo your edits in those files.
  2. Make the change SHIPPABLE inside the repository. Either upgrade the dependency so the symbol you
     need exists (`go get <module>@<version>` so go.mod / go.sum record it -- both files ship), or
     implement what you need in repository code instead of the dependency.
  3. Re-run `go build ./...` (or the packages you touched) and confirm `go mod verify` says "all
     modules verified". Stop when the build is clean with an unmodified module cache.
"""


def _guard_module_cache(ctx: flow.Ctx, s: flow.Session) -> None:
    """Go: a fix made in the module cache compiles in the container and ships as nothing
    (baseline-only forensics 2026-09-19, vuls-a76302c1). Detect it and give the session one
    bounded round to move the change into the repository."""
    if not _sub.is_go():
        return
    try:
        out = (ctx.env.execute({"command": f"cd {ctx.repo_path} && go mod verify 2>&1 | tail -5"},
                               timeout=300).get("output") or "").strip()
    except Exception as e:  # noqa: BLE001
        ctx.record.setdefault("repair_modcache", {})["error"] = f"{type(e).__name__}: {e}"
        return
    ok = "all modules verified" in out
    ctx.record.setdefault("repair_modcache", {})["verify"] = out[:300]
    if ok or not out:
        return
    ctx.log(f"[repair] go mod verify FAILED (module cache edited): {out[:160]!r} -- one fix round")
    r = s.step("modcache_fix", MODCACHE_PROMPT.format(detail=out[:800], repo_path=ctx.repo_path)
               + flow.RUNTIME_NOTE.format(repo_path=ctx.repo_path),
               tools=flow.FULL_TOOLS, max_turns=MODCACHE_TURNS)
    try:
        out2 = (ctx.env.execute({"command": f"cd {ctx.repo_path} && go mod verify 2>&1 | tail -3"},
                                timeout=300).get("output") or "").strip()
    except Exception:
        out2 = "(re-check failed)"
    ctx.record["repair_modcache"].update(fix_exit=r.exit_status, turns=r.n_calls, verify_after=out2[:300],
                                         resolved="all modules verified" in out2)
    ctx.log(f"[repair] after modcache fix: {out2[:120]!r}")


def run(ctx: flow.Ctx) -> dict:
    # Anchoring: a reasoned finding whose EXPECTED/BASIS the spec never states cannot trigger the refine.
    # Default = the configuration of the reported runs: on for Python, off for Go/JS/TS (CCFLOW_ANCHOR overrides).
    pr.SIMLOOP_REASON_ANCHOR = os.getenv("CCFLOW_ANCHOR", "1" if _sub.is_python() else "0") in ("1", "true", "yes")
    env, rp, inst = ctx.env, ctx.repo_path, ctx.instance
    ps = _sub._clip(ctx.problem_statement, _sub.PS_CLIP)
    s = flow.Session(ctx, "repair", system_prompt=prompts.BASELINE_SYSTEM)
    # R1 fix
    before = pr._untracked(env, rp)
    if getattr(ctx.findings, "repro", None):
        from . import repro_localize
        ref = repro_localize.reference_block(ctx.findings)
    else:
        ref = localize.reference_block(ctx.findings) if ctx.findings else ""
    task = inst["problem_statement"] + ("\n\n" + ref if ref else "")
    r1 = s.step("fix", prompts.baseline_prompt(task, rp), tools=flow.FULL_TOOLS, max_turns=REPAIR_TURNS)
    pr._guard_scratch(env, rp, before)
    _guard_module_cache(ctx, s)
    initial = sp._extract_patch(env, rp)
    (ctx.inst_out / "4_initial_patch.diff").write_text(initial)
    ctx.patches["initial"] = initial
    meta: dict = {"initial_chars": len(initial), "fix_exit": r1.exit_status, "fix_turns": r1.n_calls}
    if ctx.baseline:
        (ctx.inst_out / "4_refined_patch.diff").write_text(initial)
        ctx.patches["refined"] = initial
        meta.update(baseline=True, refined_chars=len(initial), changed=False)
        s.save(ctx.inst_out / "2_repair.traj.json", meta=meta)
        ctx.record["repair"] = {**meta, "steps": s.steps, "cost": s.cost, "n_calls": s.n_calls}
        return meta
    if not initial.strip():
        ctx.log("[repair] empty initial patch -- skipping simulation")
        s.save(ctx.inst_out / "2_repair.traj.json", meta=meta)
        ctx.record["repair"] = {**meta, "steps": s.steps, "cost": s.cost}
        return meta
    # deterministic: units, callers, tests, pack, callee definitions
    # Go: declarations, not Python ast (the .py-only extractor returned no units on every Go patch,
    # which silently skipped the whole review and refine)
    if _sub.is_go():
        units = pr._extract_units_go(env, rp, initial, pr.SIMLOOP_MAX_UNITS)
    elif _sub.is_js():
        units = _extract_units_js(env, rp, initial, pr.SIMLOOP_MAX_UNITS)
    else:
        units = pr._extract_units(env, rp, initial, cap=pr.SIMLOOP_MAX_UNITS)
    callers, tests = pr._review_context(env, rp, units, initial) if units else ("", "")
    try:
        pack = sp._audit_evidence_pack(env, rp, initial, [], ctx.findings, cap=10000 if True else 14000)
    except Exception as e:  # noqa: BLE001
        pack = f"(pack unavailable: {type(e).__name__})"
    try:
        callees = pr._callee_definitions(env, rp, units) if units else ""
    except Exception as e:  # noqa: BLE001
        callees = f"(callee definitions unavailable: {type(e).__name__})"
    meta["units"] = [f"{u['file']} :: {u['symbol']}" for u in units]
    if not units:
        ctx.log("[repair] no edited units extracted -- skipping simulation")
        s.save(ctx.inst_out / "2_repair.traj.json", meta=meta)
        ctx.record["repair"] = {**meta, "steps": s.steps, "cost": s.cost}
        return meta
    # R2/R3: lenses + grounding + verifier, tools off, same conversation
    meter = pr._UsageMeter()
    inst2 = dict(inst)
    for fld in ("requirements", "interface"):
        inst2[fld] = sp._unquote_spec_field(inst2.get(fld) or "")
    ctx_txt = ("=== YOUR OWN WORK PRODUCING THE INITIAL PATCH ===\n(this conversation above -- you wrote the "
               "patch under review; be adversarial with yourself)")
    qf = s.qf("simulate")
    merged = _merged_lens_prompt()
    orig_t, orig_c = pr._REVIEW_TRACE_PROMPT, pr._REVIEW_CONTRACT_PROMPT
    pr._REVIEW_TRACE_PROMPT, pr._REVIEW_CONTRACT_PROMPT = merged, "__SKIP_LENS__"
    base_qf = qf

    def qf(prompt, **kw):
        if prompt.startswith("__SKIP_LENS__"):
            return "NO FINDINGS"
        return base_qf(prompt, **kw)
    qf.supports_output_limit = False
    # The lenses run INSIDE the repair conversation, so the model legitimately quotes any file it
    # read there; the harness corpus (units + pack + callers + tests) is narrower. Ground against
    # both, plus the full diff and the spec text (b1_two_arms: true findings on 1be7de78 /
    # 0fc6d110 / bf98f031 were dropped as ungrounded).
    read_txt = "\n".join(str(m.get("content") or "") for m in s.messages if m.get("role") == "tool")[-400000:]
    spec_txt = pr._spec_corpus(inst2, ps)
    orig_ground = pr._ground_finding

    def _ground(f, code_corpus, basis_corpus):
        return orig_ground(f, code_corpus + "\n" + initial + "\n" + read_txt, basis_corpus + "\n" + spec_txt + "\n" + read_txt)
    pr._ground_finding = _ground
    try:
        rows, steps, vsteps, log = pr._run_reasoning_review(
            qf, meter, ctx=ctx_txt, ps=ps, instance=inst2, units=units, pack=pack,
            callers=callers, tests=tests, diff=initial, callees=callees)
    finally:
        pr._ground_finding = orig_ground
        pr._REVIEW_TRACE_PROMPT, pr._REVIEW_CONTRACT_PROMPT = orig_t, orig_c
    meta["lean"] = {k: True for k in ("MERGE_LENSES", "TRIM")}
    anchored = [r for r in rows if r.get("anchored", True)]
    unanchored = [r for r in rows if not r.get("anchored", True)]
    meta["review"] = {"candidates": log.get("n_candidates"), "grounded": log.get("n_grounded"),
                      "confirmed": log.get("n_confirmed"), "anchored": len(anchored), "unanchored": len(unanchored),
                      "dropped": log.get("dropped"), "refuted": log.get("refuted")}
    (ctx.inst_out / "4.2_repair_review.json").write_text(json.dumps(
        {"rows": rows, "log": log, "steps": [(a, b, c[:2000], d) for a, b, c, d in steps],
         "verify": [(a, b, c[:2000], d) for a, b, c, d in vsteps]}, indent=1, default=str))
    ctx.log(f"[repair] reasoning review: {log.get('n_candidates')} candidates, {log.get('n_grounded')} grounded, "
            f"{log.get('n_confirmed')} confirmed, {len(anchored)} anchored")
    refined = initial
    if anchored:
        snap = sp._tree_snapshot(env, rp)
        smoke_before = sp._import_smoke(env, rp, initial)[0]
        mism = pr._render_mismatches(rows)
        prompt = _render(pr._REFINE_INSTANCE, task=inst["problem_statement"], repair_context=ctx_txt,
                         diff_so_far=_sub._clip(initial, 9000), mismatches=mism, test_evidence="(none)",
                         phase_body=pr._REFINE_BODY)
        prompt = flow.strip_submit_protocol(prompt) + flow.RUNTIME_NOTE.format(repo_path=rp)
        r4 = s.step("refine", prompt, tools=flow.FULL_TOOLS, max_turns=REFINE_TURNS)
        exit_status = {"Completed": "Submitted"}.get(r4.exit_status, r4.exit_status)
        after, gate, _m, _s = sp._gate_phase_mutation(
            env, rp, inst2, ctx.findings, "repair_refine", exit_status, initial, 0, smoke_before,
            snap_before=snap, check_tests_on_conflict=True, revert_on_test_regression=True,
            intended_symbols=[r.get("symbol") for r in anchored if r.get("symbol") and r.get("symbol") != "?"],
            log=ctx.log)
        refined = after
        meta["refine"] = {"exit": r4.exit_status, "turns": r4.n_calls, "gate": gate}
    else:
        ctx.log("[repair] no anchored confirmed finding -- initial patch kept")
    (ctx.inst_out / "4_refined_patch.diff").write_text(refined)
    ctx.patches["refined"] = refined
    meta["refined_chars"] = len(refined)
    meta["changed"] = refined != initial
    s.save(ctx.inst_out / "2_repair.traj.json", meta=meta)
    ctx.record["repair"] = {**meta, "steps": s.steps, "cost": s.cost, "n_calls": s.n_calls}
    return meta
