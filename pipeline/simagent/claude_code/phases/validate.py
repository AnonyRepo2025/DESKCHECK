"""Call 4 -- VALIDATE: reason input->expected (corner cases) -> write tests -> run -> fix source.

V1 (no tools): the pre-validation table, scored by the stock compliance checker + structural
check, re-asked with the missing criterion. V2 (full tools): tests hard-coding the table against
the real entry points, executed; Stop hook refuses to end until an asserting run happened.
Harness: require-improvement mutation gate on source edits, then the regression guard over the
covering tests; V3 (full tools) fixes the source if the guard finds regressions, one round.
"""
from __future__ import annotations

import os
import shlex

from jinja2 import Template, Undefined

from simagent import interventions as _rie
from simagent import pipeline as sp

from .. import flow, runner, validate_cc
from ..env import WORK_DIR_IN

_sub = sp._sub
class _SkipRegressionFix(Exception):
    """Raised to skip the regression-fix round (all broken symbols are named by the specification)."""


VALIDATE_TURNS = int(os.getenv("VALIDATE_STEPS", "50"))
FIX_TURNS = int(os.getenv("REGRESSION_FIX_STEPS", "40"))
RETRIES = int(os.getenv("VALIDATE_SIM_RETRIES", "3"))
LITERAL_GUARD = os.getenv("CCFLOW_VALIDATE_LITERAL_GUARD", "1") in ("1", "true", "yes")
LITERAL_FIX_TURNS = int(os.getenv("VALIDATE_LITERAL_FIX_STEPS", "30"))

V3_PROMPT = """\
REGRESSION GUARD: after your validation the repository's EXISTING covering tests show these
regressions (they passed on the un-patched code and fail now):

{regressions}

Test command output (head/tail):
{tail}

Diagnose and FIX THE SOURCE so these tests pass again without undoing the intended change. Never
edit the existing tests. If a test encodes the pre-fix behaviour that the specification replaces,
say so explicitly with the test id and leave it. Stop when done.
"""


V2_BASELINE_REMINDER = """\
[VALIDATION (baseline arm)]
The real grading tests are HIDDEN and are NOT in this repo. Write new tests for the fix above that
drive the REAL code through its actual entry points (import & call the real function / method /
view, or a real request via the test client) and run them. Never re-implement or mock the code path
you are validating.
"""

V2_BASELINE_HOWTO = """\
## WHAT TO DO (VALIDATE)
1. Write tests for the issue's behaviour(s) and run them against the current (already-patched) code.
2. If a real-code test FAILS or RAISES, the FIX is at fault: diagnose and EDIT THE SOURCE so the
   real-code test passes; never weaken, delete or replace the test, and never edit the repo's
   existing tests. NEVER run `git checkout`, `git reset`, `git stash` or `git restore` on repository
   files -- the fix must still be applied when you stop.
3. Stop as soon as your tests pass on the patched code. Do not run the repository's full test suite.
"""


# G3 (Go batch 1): validation that sets up MORE than the specification / the package's own tests hides
# nil-dependency panics and precondition paths (navidrome-23bebe4e: every request carried credentials the
# spec says are absent; vuls-2c84be80: the receiver was built with a logger the hidden test leaves nil).
GO_VALIDATE_PRECONDITIONS = """
GO VALIDATION PRECONDITIONS (binding for this phase):
  - Build the receiver / router / struct under test EXACTLY the way this package's existing *_test.go files
    do (grep them for the constructor call or struct literal, e.g. `&redhatBase{}` or `New(nil, nil, ...)`).
    Do NOT add loggers, clients, stores, config or other dependencies those tests leave nil or zero: the
    hidden tests use the same minimal construction, so a nil field your patch dereferences panics there.
  - When the specification says something works WITHOUT a precondition (without authentication /
    credentials / a parameter / configuration), exercise it with that precondition ABSENT: send no auth
    params, headers or tokens beyond what the specification itself names.
  - When the specification says warnings or errors are appended / collected / recorded, assert on the
    collection the code exposes; logging a message does not satisfy it.
"""


def _render(t: str, **v) -> str:
    return Template(t, undefined=Undefined).render(**v)


def _seeds(ctx: flow.Ctx, patch: str) -> str:
    out = []
    ci = getattr(ctx.findings, "concrete_inputs", None) or []
    if ci:
        out.append("=== CONCRETE INPUTS TRACED DURING LOCALIZATION (reuse as probe seeds; verify against the "
                   "patched source; assert the FIXED behaviour) ===\n" + "\n".join(f"  - {x[:300]}" for x in ci))
    try:
        ex = sp._spec_examples_regex(ctx.instance)
        if ex:
            out.append("=== SPEC-STATED EXAMPLES (run the patched function on each input; a mismatch is a bug to FIX) ===\n"
                       + "\n".join(f"  - input {i!r}  =>  MUST produce  {e!r}" for i, e in ex))
    except Exception:
        pass
    try:
        nb = sp._new_branch_directives(patch)
        if nb:
            out.append("=== NEW-BRANCH PROBES (construct an input that REACHES each branch the fix introduces and RUN it) ===\n"
                       + "\n".join(f"  - {x}" for x in nb))
    except Exception:
        pass
    return "\n\n".join(out)


SPEC_FIX = os.getenv("CCFLOW_SPEC_FIX", "1") in ("1", "true", "yes")
SPEC_FIX_TURNS = int(os.getenv("CCFLOW_SPEC_FIX_TURNS", "40"))


def _spec_fix_round(ctx: flow.Ctx, s: flow.Session, cur: str, streams: list, meta: dict) -> str:
    """Fix 1: re-run the session's spec tests frozen at their first assertion failure; if they still fail on
    the final tree, one bounded SOURCE-only fix round with those tests read-only. Kept only when frozen
    failures strictly shrink with no new ones, the covering tests gain no failures, and the smoke check holds."""
    env, rp = ctx.env, ctx.repo_path
    rec: dict = {}
    meta["spec_fix"] = rec
    try:
        fz = validate_cc.frozen_failing_tests(env, streams, rp)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"freeze: {type(e).__name__}: {e}"
        return cur
    rec["frozen"] = sorted(fz["files"])
    rec["unreliable"] = fz["unreliable"][:8]
    if not fz["files"]:
        return cur
    go = _sub.is_go()
    targets = sorted({("./" + r.rsplit("/", 1)[0]) if (go and "/" in r) else ("./." if go else r) for r in fz["files"]})
    live = {rel: validate_cc.read_repo_file(env, rp, rel) for rel in fz["files"]}   # False = did not exist

    def put_frozen():
        for rel, c in fz["files"].items():
            runner.put_file(env, f"{rp}/{rel}", c)

    def put_live():
        for rel, c in live.items():
            if c is False:   # the session had deleted it: remove the frozen copy again
                env.execute({"command": f"cd {rp} && rm -f {shlex.quote(rel)}"}, timeout=30)
            elif c is not None:
                runner.put_file(env, f"{rp}/{rel}", c)

    try:
        put_frozen()
        failed, out = sp._run_test_files(env, rp, targets)
        rec["frozen_fail_before"] = sorted(failed)[:12] if failed is not None else None
        if not failed:
            rec["outcome"] = "frozen tests pass on the final tree" if failed is not None else "frozen run unusable"
            return cur
        ctx.log(f"[validate] spec-fix: {len(failed)} frozen spec test(s) still fail -> source-only fix round")
        tf = sp._find_covering_tests(env, rp, cur)
        cov_before, _ = sp._run_test_files(env, rp, tf) if tf else (set(), "")
        put_frozen()
        snap = sp._tree_snapshot(env, rp)
        guard = validate_cc.install_spec_fix_guard(env, f"{WORK_DIR_IN}/hooks_specfix", rp, sorted(fz["files"]), runner.put_file)
        tail = sp._head_tail(out or "", 4000, "OUTPUT MIDDLE ELIDED") if hasattr(sp, "_head_tail") else (out or "")[-4000:]
        r = s.step("spec_fix", validate_cc.SPEC_FIX_PROMPT.format(
                       failed="\n".join(f"  - {t}" for t in sorted(failed)[:15]), tail=tail)
                   + flow.RUNTIME_NOTE.format(repo_path=rp),
                   tools=flow.FULL_TOOLS, max_turns=SPEC_FIX_TURNS, extra_args=guard)
        rec["fix"] = {"exit": r.exit_status, "turns": r.n_calls}
        put_frozen()   # belt and braces: the guard blocks edits, but a frozen file must be what is judged
        failed2, _ = sp._run_test_files(env, rp, targets)
        cov_after, _ = sp._run_test_files(env, rp, tf) if tf else (set(), "")
        new_patch = sp._extract_patch(env, rp)
        smoke = sp._import_smoke(env, rp, new_patch)[0]
        rec["frozen_fail_after"] = sorted(failed2)[:12] if failed2 is not None else None
        improved = (failed2 is not None and len(set(failed2)) < len(set(failed)) and not (set(failed2) - set(failed)))
        cov_ok = cov_after is not None and cov_before is not None and not (set(cov_after) - set(cov_before))
        rec.update(improved=improved, covering_ok=cov_ok, smoke=smoke, changed=new_patch.strip() != cur.strip())
        if improved and cov_ok and smoke is not False and new_patch.strip() != cur.strip():
            rec["outcome"] = "kept"
            ctx.log(f"[validate] spec-fix kept: frozen failures {len(failed)} -> {len(failed2)}")
            cur = new_patch
        else:
            sp._tree_restore(env, rp, snap)
            rec["outcome"] = "reverted"
            ctx.log(f"[validate] spec-fix reverted (improved={improved} covering_ok={cov_ok} smoke={smoke})")
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        try:
            put_live()
        except Exception:
            pass
    return cur


def run(ctx: flow.Ctx) -> dict:
    env, rp = ctx.env, ctx.repo_path
    inst = ctx.instance
    patch = ctx.patches.get("audited") or ctx.patches.get("refined") or ctx.patches.get("initial") or ""
    meta: dict = {}
    if not patch.strip():
        ctx.log("[validate] empty patch -- skipped")
        ctx.record["validate"] = {"skipped": True}
        return meta
    s = flow.Session(ctx, "validate", system_prompt="You are a meticulous software engineer validating a fix.")
    fix_ref = ("=== FIX APPLIED BY THE REPAIR CALL (validate it) ===\nFiles/edits (diff so far):\n"
               + _sub._clip(patch, 3000) + "\n\n" + _seeds(ctx, patch))
    # V1 reasoning table (no tools), checker-driven retries inside the same conversation
    if ctx.baseline:
        reason = {"table": "", "attempts": [], "complied": None, "baseline": True}
    else:
        reason = validate_cc.reason_step(
        lambda prompt, label: s.step(label, prompt, tools=flow.NO_TOOLS, max_turns=2),
        task=inst["problem_statement"], fix_reference=fix_ref, max_retries=RETRIES,
        checker=_rie.default_newtest_compliance_checker, log=ctx.log)
    (ctx.inst_out / "4a_validate_reasoning.json").write_text(__import__("json").dumps(reason, indent=1))
    meta["reason"] = {"complied": reason["complied"], "attempts": len(reason["attempts"]),
                      "table_chars": len(reason["table"])}
    if reason["table"]:
        fix_ref += validate_cc.ACT_BLOCK.format(table=reason["table"][:12000])
    # V2 write + run tests (full tools, Stop guard)
    hook_dir = f"{WORK_DIR_IN}/hooks_validate"
    extra = validate_cc.install_hooks(env, "validate", hook_dir, runner.put_file)
    reminder = V2_BASELINE_REMINDER if ctx.baseline else validate_cc.CC_VALIDATE_REMINDER
    if _sub.is_go() and not ctx.baseline:
        reminder += GO_VALIDATE_PRECONDITIONS
    prompt = _render(flow.lang_text(sp.VALIDATE_INSTANCE), task=inst["problem_statement"], fix_reference=fix_ref,
                     phase_reminder=reminder,
                     phase_howto=V2_BASELINE_HOWTO if ctx.baseline else flow.lang_text(sp.VALIDATE_HOWTO))
    prompt = flow.strip_submit_protocol(prompt) + flow.RUNTIME_NOTE.format(repo_path=rp)
    snap = sp._tree_snapshot(env, rp)
    smoke_before = sp._import_smoke(env, rp, patch)[0]
    r2 = s.step("tests", prompt, tools=flow.FULL_TOOLS, max_turns=VALIDATE_TURNS, extra_args=extra)
    spec_streams = [getattr(r2, "stream_path", "")]
    hooks = validate_cc.hook_outcome(env, hook_dir)
    meta["guard"] = hooks
    exit_status = {"Completed": "Submitted"}.get(r2.exit_status, r2.exit_status)
    # literal preservation: a table literal rewritten in the session's own test/probe -> one bounded
    # follow-up that treats the source as at fault (b2 qutebrowser-305e7c96, both rolls)
    if reason.get("table") and not ctx.baseline and LITERAL_GUARD:
        lits = validate_cc.table_literals(reason["table"])
        hits = validate_cc.literal_rewrites(getattr(r2, "stream_path", ""), lits)
        meta["literal_guard"] = {"literals": lits[:20], "rewrites": hits}
        if hits:
            ctx.log(f"[validate] literal guard: {len(hits)} table literal rewrite(s) in session tests -> follow-up")
            txt = "\n".join(f"  - {h['literal']!r} removed via {h['via']}" + (f" in {h['file']}" if h['file'] else "")
                            + f": {h['snippet']!r}" for h in hits[:6])
            r2b = s.step("literal_fix", validate_cc.LITERAL_FOLLOWUP.format(hits=txt) + flow.RUNTIME_NOTE.format(repo_path=rp),
                         tools=flow.FULL_TOOLS, max_turns=LITERAL_FIX_TURNS, extra_args=extra)
            spec_streams.append(getattr(r2b, "stream_path", ""))
            meta["literal_guard"]["fix"] = {"exit": r2b.exit_status, "turns": r2b.n_calls}
            hooks = validate_cc.hook_outcome(env, hook_dir)
            meta["guard"] = hooks
            exit_status = {"Completed": "Submitted"}.get(r2b.exit_status, r2b.exit_status)
            weak = validate_cc.weak_literal_assertions(env, hits)
            passed = validate_cc.passing_run_after_last_edit(getattr(r2b, "stream_path", ""))
            meta["literal_guard"]["verify"] = {"weak": weak[:6], "passing_run_after_edit": passed}
            if weak or not passed:
                ctx.log(f"[validate] literal guard follow-up unverified (weak={len(weak)}, passing_run={passed}) -> one more round")
                wtxt = ("\n".join(f"  - {w['file']}: {w['line']}" for w in weak[:6])
                        or "  - no passing test run followed your last edit")
                r2c = s.step("literal_verify", validate_cc.LITERAL_VERIFY_FOLLOWUP.format(weak=wtxt) + flow.RUNTIME_NOTE.format(repo_path=rp),
                             tools=flow.FULL_TOOLS, max_turns=LITERAL_FIX_TURNS, extra_args=extra)
                spec_streams.append(getattr(r2c, "stream_path", ""))
                meta["literal_guard"]["verify"]["round2"] = {
                    "exit": r2c.exit_status, "turns": r2c.n_calls,
                    "weak_after": validate_cc.weak_literal_assertions(env, hits)[:6],
                    "passing_run_after_edit": validate_cc.passing_run_after_last_edit(getattr(r2c, "stream_path", ""))}
                hooks = validate_cc.hook_outcome(env, hook_dir)
                meta["guard"] = hooks
                exit_status = {"Completed": "Submitted"}.get(r2c.exit_status, r2c.exit_status)
    after, gate, _m, _s = sp._gate_phase_mutation(
        env, rp, inst, ctx.findings, "validate", exit_status, patch, 0, smoke_before, snap_before=snap,
        check_tests_on_conflict=True, revert_on_test_regression=True, require_improvement=True, log=ctx.log)
    meta["gate"] = gate
    cur = after
    # regression guard over the covering tests
    reg: dict = {}
    try:
        tf = sp._find_covering_tests(env, rp, cur)
        reg["test_files"] = tf
        if tf:
            failed_base, _ = sp._regression_baseline(env, rp, cur, tf)
            failed_after, out_after = sp._run_test_files(env, rp, tf)
            if failed_base is not None and failed_after is not None:
                regs = sorted(set(failed_after) - set(failed_base))
                reg["regressions"] = regs
                if regs:
                    ctx.log(f"[validate] regression guard: {len(regs)} regression(s) -> fix round")
                    spec_tests = validate_cc.session_test_targets(getattr(r2, "stream_path", ""), rp, _sub.is_go())
                    spec_pass_before: set = set()
                    if spec_tests:
                        _f, _ = sp._run_test_files(env, rp, spec_tests)
                        if _f is not None:   # targets that ran clean on the patched tree, pre-fix
                            spec_pass_before = set(spec_tests) - set(_f)
                        ctx.log(f"[validate] spec tests {spec_tests} passing before fix: {sorted(spec_pass_before)}")
                    snap2 = sp._tree_snapshot(env, rp)
                    builds = [t for t in regs if str(t).startswith("BUILD:")]
                    spec_txt_all = (inst.get("problem_statement", "") + "\n" + str(inst.get("requirements") or "")
                                    + "\n" + str(inst.get("interface") or "")).replace("\\n", "\n")
                    if _sub.is_go() and builds:
                        # every broken symbol named by the specification == a sanctioned shape change: the hidden
                        # tests use the NEW shape, so "repairing" the break can only move away from them
                        _errs, _syms = validate_cc.go_compile_errors(out_after or "", spec_txt_all)
                        if _syms and all(_syms.values()):
                            reg["go_build"] = {"errors": _errs[:10], "symbols": _syms, "skipped": "all symbols spec-named"}
                            ctx.log(f"[validate] BUILD regression on spec-named symbols {list(_syms)} -- fix round skipped")
                            meta["fix"] = {"skipped": "all broken symbols are spec-named"}
                            raise _SkipRegressionFix
                    if _sub.is_go() and builds:
                        spec_txt = (inst.get("problem_statement", "") + "\n" + str(inst.get("requirements") or "")
                                    + "\n" + str(inst.get("interface") or "")).replace("\\n", "\n")
                        errs, syms = validate_cc.go_compile_errors(out_after or "", spec_txt)
                        reg["go_build"] = {"errors": errs[:10], "symbols": syms}
                        guard_args = validate_cc.install_test_guard(env, f"{WORK_DIR_IN}/hooks_fix", rp, runner.put_file)
                        r3 = s.step("fix", validate_cc.GO_BUILD_FIX_PROMPT.format(
                                        regressions="\n".join(f"  - {t}" for t in regs[:20]),
                                        errors="\n".join(f"  {e}" for e in errs) or "  (not parsed -- see the output below)\n"
                                               + (out_after or "")[-3000:],
                                        symbols="\n".join(f"  - {k}: {'NAMED in the specification' if v else 'NOT named in the specification'}"
                                                          for k, v in syms.items()) or "  (none parsed)")
                                    + flow.RUNTIME_NOTE.format(repo_path=rp),
                                    tools=flow.FULL_TOOLS, max_turns=FIX_TURNS, extra_args=guard_args)
                    else:
                        # JS/TS: edits to pre-existing test and snapshot files are stripped from the shipped
                        # diff exactly as in Go, so the fixer gets the same PreToolUse guard (J1, JS port).
                        fix_args = (validate_cc.install_test_guard(env, f"{WORK_DIR_IN}/hooks_fix", rp, runner.put_file)
                                    if _sub.is_js() else ())
                        r3 = s.step("fix", V3_PROMPT.format(regressions="\n".join(f"  - {t}" for t in regs[:20]),
                                                             tail=sp._head_tail(out_after or "", 5000, "OUTPUT MIDDLE ELIDED")
                                                             if hasattr(sp, "_head_tail") else (out_after or "")[-5000:])
                                    + flow.RUNTIME_NOTE.format(repo_path=rp),
                                    tools=flow.FULL_TOOLS, max_turns=FIX_TURNS, extra_args=fix_args)
                    # tests are not the fixer's to change: restore the covering test files before
                    # re-running (b1 f0341c0b: the fixer "fixed" the regression by editing the test)
                    try:
                        chg = env.execute({"command": f"cd {rp} && git diff --name-only -- " + " ".join(__import__('shlex').quote(t) for t in tf)}, timeout=60).get("output") or ""
                        if _sub.is_go():
                            # Go covering "tests" are PACKAGE DIRECTORIES: checking out the directory also reverted the
                            # fixer's SOURCE repair (rerun vuls-edb324c3 restored the helper signatures in scan/base.go and
                            # the restore wiped it; flipt-756f00f7 lost config.go the same way). Restore only test files.
                            edited = [x for x in chg.split() if x.endswith("_test.go")]
                            if edited:
                                env.execute({"command": f"cd {rp} && git checkout -- " + " ".join(__import__('shlex').quote(t) for t in edited)}, timeout=60)
                                reg["fixer_edited_tests"] = edited
                                ctx.log(f"[validate] regression fix edited covering tests {edited} -- restored before re-check")
                        elif chg.strip():
                            env.execute({"command": f"cd {rp} && git checkout -- " + " ".join(__import__('shlex').quote(t) for t in tf)}, timeout=60)
                            reg["fixer_edited_tests"] = chg.split()
                            ctx.log(f"[validate] regression fix edited covering tests {chg.split()} -- restored before re-check")
                    except Exception as e:  # noqa: BLE001
                        reg["test_restore_error"] = f"{type(e).__name__}: {e}"
                    failed_fix, _ = sp._run_test_files(env, rp, tf)
                    regs2 = sorted(set(failed_fix or []) - set(failed_base)) if failed_fix is not None else regs
                    reg["after_fix"] = regs2
                    # V1: a fix that satisfies an existing test by undoing the SPECIFIED behaviour is not an
                    # improvement, however the counts move. The session's own spec-derived tests are the check.
                    spec_lost = []
                    if spec_tests and spec_pass_before:
                        after_fail, _ = sp._run_test_files(env, rp, spec_tests)
                        if after_fail is not None:
                            spec_lost = sorted(set(spec_pass_before) & set(after_fail))
                        reg["spec_tests"] = {"targets": spec_tests, "passing_before": sorted(spec_pass_before)[:8],
                                             "lost_after_fix": spec_lost[:8]}
                    if spec_lost:
                        sp._tree_restore(env, rp, snap2)
                        reg["fix_reverted"] = True; reg["reverted_for"] = "spec behaviour lost"
                        ctx.log(f"[validate] regression fix broke the session's spec tests {spec_lost[:3]} -- reverted")
                    elif len(regs2) >= len(regs):
                        sp._tree_restore(env, rp, snap2)
                        reg["fix_reverted"] = True
                        ctx.log("[validate] regression fix did not improve -- reverted")
                    else:
                        cur = sp._extract_patch(env, rp)
                    meta["fix"] = {"exit": r3.exit_status, "turns": r3.n_calls}
            else:
                reg["inconclusive"] = True
    except _SkipRegressionFix:
        pass
    except Exception as e:  # noqa: BLE001
        reg["error"] = f"{type(e).__name__}: {e}"
    meta["regression"] = reg
    if not ctx.baseline and SPEC_FIX:
        cur = _spec_fix_round(ctx, s, cur, spec_streams, meta)
    ctx.patches["final"] = cur
    s.save(ctx.inst_out / "4_validate.traj.json", meta=meta,
           interventions=[{"intervention": "cc_prevalidate_reasoning", **a} for a in reason["attempts"]],
           intervention_triggers=[{"intervention": "cc_stop_guard", **hooks}])
    ctx.record["validate"] = {**meta, "steps": s.steps, "cost": s.cost, "n_calls": s.n_calls}
    return meta
