#!/usr/bin/env python3
"""Pipeline variant: PLAIN REPAIR -- the repair phase is a *default* agent.

Motivation / experiment
-----------------------
The stock pipeline's phase 3 is a heavily engineered repair sub-agent: a pre-fix
reasoning-intervention hook (REPAIR_REMINDER), a long HOW-TO that scripts the
hand-simulation (REPAIR_HOWTO), a localization block carrying a binding EDIT-or-REFUTE
CONTRACT plus every derived device (signature contracts, preserve contracts, free
constants, message phrases, obligation ledger, SENTINEL-PRONE TYPES, synth vectors,
BASE-CODE GUARDRAILS), the stated-examples contract, and the outcome contracts.

This variant removes ALL of that from the repair phase. Phase 3 becomes the plain
mini-swe-agent default agent: the issue, the stock swebench instance template, and the
localization sub-agent's findings attached *as a reference only* (explicitly non-binding).
No code reasoning is imposed on the repair agent.

Everything else is untouched:
    explore -> localize (full device stack)
        -> repair (PLAIN, this module's change)
        -> spec_audit -> audit_fix -> validate    (all exactly as usual)

The downstream phases still see the applied patch, and audit_fix still receives the FULL
localization reference -- only the repair phase is stripped.

Usage
-----
    python simagent/plainrepair.py <instance_id> [...] [--out DIR]
    python simagent/plainrepair.py --all --limit 20

All the usual env knobs of subagent_pipeline apply (MODEL, MODEL_CONFIG, OUT_DIR,
REPAIR_STEPS, INSTANCE_COST_CAP, SKIP_SPEC_AUDIT, SKIP_VALIDATE, ...).

Variant-specific env knobs:
    PLAIN_KEEP_REPAIR_SITES=1   keep phase 3b (repair_sites reconciliation).  Default OFF:
                                3b exists to force an EDIT-or-REFUTE decision on every
                                predicted site, which is exactly the coercion this variant
                                removes ("findings are reference only").
    PLAIN_SCRATCH_GUARD=0       disable the scratch-file guard.  Default ON: the default
                                agent's workflow tells it to "create a script to reproduce
                                the issue", and _extract_patch runs `git add -A -N`, so a
                                leftover repro script at the repo root would otherwise ship
                                inside the diff.  The guard excludes untracked files the
                                repair phase created whose names look like scratch.
    PLAIN_FINDINGS=0            run the repair agent on the bare issue with NO localization
                                reference at all (A/B against the reference-only block).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import signal
import sys
import threading
import time
from pathlib import Path


from simagent import pipeline as sp

_sub = sp._sub

VARIANT = "plainrepair"

PLAIN_KEEP_REPAIR_SITES = os.getenv("PLAIN_KEEP_REPAIR_SITES", "") in ("1", "true", "yes")
PLAIN_SCRATCH_GUARD = os.getenv("PLAIN_SCRATCH_GUARD", "1") in ("1", "true", "yes")
PLAIN_FINDINGS = os.getenv("PLAIN_FINDINGS", "1") in ("1", "true", "yes")
# Heartbeat for the repair agent loop. `_run_phase_agent` prints only "starting"/"done" and
# n_calls reaches disk only when the phase RETURNS, so a long phase is unobservable from the
# log -- on a slow provider that is indistinguishable from a hang. Print every Nth step.
# 0 disables.
PLAIN_STEP_LOG = int(os.getenv("PLAIN_STEP_LOG", "10"))
# Disable the DETERMINISTIC tier of spec_audit (sp._mech_audit / _mech_audit_go, which also
# carries _preservation_flags' `strlit` family). The behavioral pure-reasoning audit, the
# checklist, import smoke and the evidence-protected-line machinery all still run.
# Motivation (ansible-83909bf plain_r1): repair shipped a correct patch and three mechanical
# facts inverted the spec against it -- `strlit` demanded the base error string be RESTORED
# though requirement 2 commands the change (its exemption only fires when the replacement is
# quoted verbatim in the spec), and `undef:login.py` reported the requirement-1 deletion as
# "file missing or unparseable". audit_fix cannot refute a MECHANICAL finding, so both shipped.
# NOTE: import smoke is a SEPARATE tier and is not disabled here -- _import_smoke derives its
# module list from every .py in the diff including DELETED ones, so a deliberate file deletion
# still reads as SMOKE_FAIL.
PLAIN_SKIP_MECH = os.getenv("PLAIN_SKIP_MECH", "") in ("1", "true", "yes")


# --- the plain repair prompt ------------------------------------------------------------------
# Fallback only: normally the template comes verbatim from the base swebench.yaml agent config
# (i.e. literally "the default agent"), with two mechanical edits -- see _plain_instance_template.
_FALLBACK_INSTANCE = """\
<pr_description>
Consider the following PR description:
{{task}}
</pr_description>

<instructions>
You're a software engineer interacting continuously with a computer by submitting commands.
Your task is to make changes to non-test files in /testbed in order to fix the issue described
in the PR description, in a way that is general and consistent with the codebase.

Do NOT modify tests, configuration, or packaging files.

When you've completed your work, submit your changes as a git patch, in SEPARATE commands:
  1) git -C /testbed diff -- <the source files you changed> > /tmp/patch.txt
  2) echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /tmp/patch.txt
</instructions>
"""

# Mechanical hygiene, not reasoning guidance: the stock template redirects the diff to a
# relative `patch.txt`, which lands INSIDE the repo and would be picked up by the pipeline's
# `git add -A -N` patch extraction. Move it to /tmp.
_SCRATCH_NOTE = """

## Scratch files
Write any reproduction script, probe, or note you create under /tmp (e.g. /tmp/repro.py), never
inside the repository -- files left in the repo are picked up by the patch extraction.
"""

_REFERENCE_HEADER = """\
=== LOCALIZATION SUB-AGENT FINDINGS (REFERENCE ONLY -- NOT BINDING) ===
A separate localization sub-agent analysed this issue before you and reported the following.
Treat it as a hint that may save you time, nothing more: it is not verified, it may be
incomplete, and it may be wrong. You are NOT required to edit the files it names, and you are
NOT required to justify or refute any of them. Verify anything you use against the real source,
and fix the issue wherever the source actually says it lives.
"""


def _plain_instance_template(base_agent: dict, repo_path: str = "/testbed") -> str:
    """The DEFAULT agent's own instance template + a reference-only findings slot.

    ``repo_path`` is substituted for the ``/testbed`` written into _FALLBACK_INSTANCE, the
    same way every other phase template is rebased (``sp.REPAIR_HOWTO.replace(...)`` etc.).
    A config-supplied ``instance_template`` already names the right directory (``/app`` on
    SWE-bench Pro), so the substitution is a no-op on that path.
    """
    tmpl = (base_agent or {}).get("instance_template") or _FALLBACK_INSTANCE
    from simagent.evidence import clarify_scope
    tmpl = clarify_scope(tmpl)
    # Keep the agent's scratch diff out of the repository (see _SCRATCH_NOTE).
    tmpl = re.sub(r"(?<![\w./])patch\.txt", "/tmp/patch.txt", tmpl)
    tmpl = tmpl.rstrip("\n") + "\n" + _SCRATCH_NOTE
    tmpl += sp._lang_note()   # Go / JS language note; empty for Python
    return (tmpl + "\n{{ localization_reference }}\n").replace("/testbed", repo_path)


def _plain_localization_reference(findings) -> str:
    """Flat, advisory rendering of the localization findings.

    Deliberately NOT sp._localization_reference: no EDIT-or-REFUTE contract, no evidence-tier
    ranking rules, no directive, no signature/preserve contracts, no free constants, no message
    phrases, no obligation ledger, no SENTINEL-PRONE TYPES, no BASE-CODE GUARDRAILS. Just what
    was found and why it was flagged.

    ONE exception: RULE-COVERAGE synth vectors ARE included (qutebrowser-70248f25 -- repair
    self-tested 13 cases across 4 rounds and never once constructed an order-violated or
    duplicated-component input for a stated `XhYmZs`-style format, because nothing suggested
    that category to it). A synth vector is a concrete adversarial INPUT STRING, not a
    contract about what the code must do with it -- it costs the plain agent nothing to try
    and stays consistent with "reference only, non-binding": repair is free to run it and
    conclude its own fix already handles it correctly.
    """
    if findings is None or not PLAIN_FINDINGS:
        return ""
    reasons = getattr(findings, "site_reasons", {}) or {}
    lines = []
    for f in getattr(findings, "files_to_edit", None) or []:
        why = (reasons.get(f) or "").strip()
        lines.append(f"  - {f}" + (f"  ({_sub._clip(why, 160)})" if why else ""))
    files = "\n".join(lines) or "  (none identified)"
    root_cause = (getattr(findings, "root_cause", "") or "").strip() or "(not determined)"
    return (
        _REFERENCE_HEADER
        + f"\nSUSPECTED ROOT CAUSE: {root_cause}\n"
        + f"SYMPTOM SURFACES AT: {getattr(findings, 'bug_function', None) or 'unknown'}  "
          f"(file: {getattr(findings, 'bug_file', None) or 'unknown'})\n"
        + "CANDIDATE FILES / SITES:\n"
        + files
        + sp._render_synth_vectors(getattr(findings, "synth_vectors", []) or [])
        + "\n=== end of reference ===\n"
    )


# --- contract-block stripping (fallback path) --------------------------------------------------
# run_pipeline appends the STATED EXAMPLES CONTRACT and OUTCOME CONTRACTS blocks to the repair
# task before handing it over. This variant repairs from the bare issue, so they are removed.
# Normally we substitute the problem statement captured from run_pipeline; this regex is the
# belt-and-braces path if that capture ever misses.
_CONTRACT_BLOCK_RE = re.compile(
    r"\n\n=== (?:STATED EXAMPLES CONTRACT|OUTCOME CONTRACTS)\b.*\Z", re.S)


def _strip_contract_blocks(task: str) -> str:
    return _CONTRACT_BLOCK_RE.sub("", task or "")


# --- scratch-file guard ------------------------------------------------------------------------
# Anchored word + an explicit separator (NOT \b, which does not fire before '_'): matches
# reproduce_bug.py / test-case.go / debug.py, but leaves testbed.py, checkout.py, verifier.py,
# patches.py and other legitimate source names alone.
_SCRATCH_NAME_RE = re.compile(
    r"^(?:reproduce|reproducer|repro|test|tests|debug|scratch|tmp|temp|check|verify|demo|bug|"
    r"poc|minimal|issue|fix|patch|out|output|foo|bar|script)(?:[-_.0-9]|$)", re.I)
_SCRATCH_EXT = (".txt", ".log", ".patch", ".diff", ".out", ".rej", ".orig", ".bak")


# JS probes the agents write despite the /tmp instruction land at the repo root as
# `_probe.js` / `probe.mjs` / `repro.ts` (measured: nodebb-0f788b8e shipped `_probe.js`).
# JS-only so Python/Go scratch detection is unchanged.
_JS_SCRATCH_NAME_RE = re.compile(
    r"^[_.]?(?:probe|probes|repro\w*|reproduce\w*|scratch\w*|tmp\w*|temp\w*|check\w*|debug\w*|"
    r"verify\w*|demo\w*|poc\w*|test\w*)\.[cm]?[jt]sx?$", re.I)


def _looks_like_scratch(path: str) -> bool:
    base = os.path.basename(path)
    if _sub.is_js() and _JS_SCRATCH_NAME_RE.match(base) and "/" not in path.strip("./"):
        return True   # root-level probe file (a real JS source never lives at the repo root)
    return bool(_SCRATCH_NAME_RE.match(base)) or base.lower().endswith(_SCRATCH_EXT)


def _untracked(env, repo_path: str) -> "set[str]":
    try:
        out = env.execute(
            {"command": f"cd {repo_path} && git ls-files --others --exclude-standard"},
            timeout=30).get("output", "") or ""
    except Exception:
        return set()
    return {l.strip() for l in out.splitlines() if l.strip()}


def _guard_scratch(env, repo_path: str, before: "set[str]") -> None:
    """Exclude scratch files the repair phase created from every later patch extraction."""
    new = sorted(f for f in (_untracked(env, repo_path) - before) if _looks_like_scratch(f))
    if not new:
        return
    dirt = list(getattr(env, "_pipeline_start_dirt", None) or [])
    env._pipeline_start_dirt = dirt + [f for f in new if f not in dirt]
    print(f"[phase:repair] scratch guard: excluding agent-created files from the diff: {new}",
          flush=True)


# --- monkeypatched seams -----------------------------------------------------------------------
_STATE: dict = {"problem_statement": "", "findings": None, "instance": None, "inst_out": None}

_orig_run_phase_agent = sp._run_phase_agent
_orig_localization_reference = sp._localization_reference
_orig_run_pipeline = sp.run_pipeline
_orig_combined_trajectory = sp._combined_trajectory


def _capturing_localization_reference(findings):
    """Full rendering (unchanged, for repair_sites / audit_fix) + capture for the repair swap.

    The call site evaluates this argument immediately before calling _run_phase_agent, so the
    captured object is always the one belonging to the phase about to run.
    """
    _STATE["findings"] = findings
    return _orig_localization_reference(findings)


def _install_step_log(step_limit: int, label: str):
    """Return an ``on_agent`` hook that prints a heartbeat every PLAIN_STEP_LOG steps.

    Wraps ``agent.query`` the same way `sp._install_submit_guard` does -- the wrapper is the
    only place inside the loop that runs once per step. Reports elapsed wall time and cost so a
    slow provider is distinguishable from a stalled run.
    """
    def _hook(agent):
        orig_query = agent.query
        t0 = time.time()

        def logged_query():
            n = agent.n_calls
            if n and n % PLAIN_STEP_LOG == 0:
                print(f"[phase:{label}] step {n}/{step_limit}  "
                      f"{time.time() - t0:.0f}s  ${agent.cost:.4f}", flush=True)
            return orig_query()

        agent.query = logged_query
        return None
    return _hook


def _chain_step_log(existing, step_limit: int, label: str):
    """Add the heartbeat WITHOUT displacing a phase's own on_agent hook.

    validate (newtest intervention) and classic repair (patch intervention) pass real hooks
    whose return value is stored as ``hook`` and read downstream for intervention_log, so the
    heartbeat has to compose with them and return THEIR value, not its own.
    """
    logger = _install_step_log(step_limit, label)

    def _hook(agent):
        h = existing(agent) if existing is not None else None
        logger(agent)
        return h
    return _hook


def _plain_run_phase_agent(name, model, env, base_agent, **kw):
    """Swap the repair phase's whole prompt for the default agent's; every other phase passes
    through untouched (apart from the step heartbeat, which is observability only)."""
    if name != "repair":
        if PLAIN_STEP_LOG > 0 and kw.get("step_limit", 0) > PLAIN_STEP_LOG:
            kw["on_agent"] = _chain_step_log(kw.get("on_agent"),
                                             kw.get("step_limit", 0), name)
        return _orig_run_phase_agent(name, model, env, base_agent, **kw)

    repo_path = _sub._repo_path_for(sp.SimpleNamespace(env=env), None)
    kw["instance"] = _plain_instance_template(base_agent, repo_path)
    kw["task"] = _STATE["problem_statement"] or _strip_contract_blocks(kw.get("task", ""))
    kw["localization_reference"] = _plain_localization_reference(_STATE["findings"])
    kw["on_agent"] = None          # no pre-fix reasoning intervention
    for slot in ("phase_reminder", "phase_howto", "phase_body", "explore_transcript"):
        kw.pop(slot, None)         # not referenced by the default template

    kw["on_agent"] = _install_step_log(kw.get("step_limit", 0), name)

    before = _untracked(env, repo_path) if PLAIN_SCRATCH_GUARD else set()
    print(f"[phase:repair] variant={VARIANT} -- DEFAULT agent "
          f"(findings={'reference-only' if PLAIN_FINDINGS else 'withheld'}, "
          f"no intervention, no contracts, no guardrails)", flush=True)
    try:
        repair = _orig_run_phase_agent(name, model, env, base_agent, **kw)
    finally:
        try:
            _guard_scratch(env, repo_path, before)
        except Exception as e:
            print(f"[phase:repair] scratch guard FAILED ({type(e).__name__}: {e})",
                  flush=True)

    # Post-repair simulate-and-refine (4.1 extract -> 4.2 cases -> 4.3 simulate -> 4.4 refine).
    # Runs inside phase 3 so every downstream phase still sees one applied patch and one repair
    # record; the loop's own steps are persisted as separate 4.x artifacts.
    if PLAIN_SIMLOOP and _STATE.get("inst_out") is not None:
        try:
            repair["simloop"] = _STATE["simloop_meta"] = _run_simloop(
                model, env, base_agent, repo_path, _STATE["inst_out"], repair,
                system=kw.get("system", ""))
        except Exception as e:
            import traceback
            print(f"[simloop] FAILED ({type(e).__name__}: {e})", flush=True)
            traceback.print_exc()
            repair["simloop"] = _STATE["simloop_meta"] = {"error": f"{type(e).__name__}: {e}"}
    return repair


# =================================================================================================
# POST-REPAIR SIMULATE-AND-REFINE LOOP  (PLAIN_SIMLOOP)
#
# Runs INSIDE phase 3, after the plain repair agent has produced and applied its patch:
#   4.1 EXTRACT  -- one tool-free query: which code units did the initial patch edit, and what
#                   is each unit's POST-PATCH source?  Grounded on a deterministic source pack
#                   so the emitted snippets are real code rather than diff-reconstructions.
#   4.2 CASES    -- per unit, one tool-free query: derive concrete INPUT / EXPECTED pairs FROM
#                   THE SPEC (each must quote the requirement sentence it rests on, so the
#                   oracle cannot be read off the very code under test).
#   4.3 SIMULATE -- per unit, one tool-free query: hand-execute the unit on each INPUT, emit the
#                   concrete ACTUAL value and a MATCH / MISMATCH verdict.
#   4.4 REFINE   -- if anything MISMATCHed, one bounded agent loop carrying those mismatches.
#
# All four steps are seeded with the repair agent's own transcript (_repair_context), so they
# continue the initial patch generation rather than re-deriving it cold.
#
# The initial and refined patches are written to SEPARATE files (4_initial_patch.diff /
# 4_refined_patch.diff). A refine round that breaks import smoke is reverted wholesale.
# =================================================================================================

PLAIN_SIMLOOP = os.getenv("PLAIN_SIMLOOP", "1") in ("1", "true", "yes")
SIMLOOP_ROUNDS = int(os.getenv("SIMLOOP_ROUNDS", "1"))
SIMLOOP_REFINE_STEPS = int(os.getenv("SIMLOOP_REFINE_STEPS", "40"))
# 5 was BINDING on every large patch and silently under-covered them: ansible-d72025be edited 20
# units, the loop inspected 5, reported clean, and the instance scored 0/5 f2p in every setting
# (measured again 2026-08-12 under LOCALIZE_MODE=graph). 4.2/4.3 cost one model call per unit, so
# this raises per-instance simloop cost roughly 4x on big patches and is a no-op on small ones.
SIMLOOP_MAX_UNITS = int(os.getenv("SIMLOOP_MAX_UNITS", "20"))
SIMLOOP_CTX_CLIP = int(os.getenv("SIMLOOP_CTX_CLIP", "9000"))
# Units per 4.2/4.3 call. 4.2 and 4.3 used to issue ONE call per unit, and every call re-sent an
# IDENTICAL header (the repair-transcript context, the problem statement, the requirements and
# the interface) with only the unit body varying. Measured on ansible-811093f0 (20 units): 40
# calls, 20 of them >3 KB averaging 22,062 chars, of which the first **18,284 chars were
# byte-identical across every call** -- 366 KB of the 444 KB sent was redundant context, and
# provider latency scales with input size. Batching amortises that header over `SIMLOOP_BATCH`
# units. 1 restores the old one-call-per-unit behaviour.
SIMLOOP_BATCH = max(1, int(os.getenv("SIMLOOP_BATCH", "5")))
# Absolute wall-clock ceiling around each tool-free query. This is deliberately a little above
# SUBAGENT_QUERY_TIMEOUT (the HTTP attempt timeout): providers/LiteLLM may retry attempts, which
# otherwise turned a nominal 300-second timeout into an unbounded multi-attempt wait.
SIMLOOP_QUERY_WALL_TIMEOUT = max(
    0.0, float(os.getenv("SIMLOOP_QUERY_WALL_TIMEOUT", "330") or 0))

SIMLOOP_EXEC = os.getenv("SIMLOOP_EXEC", "1") in ("1", "true", "yes")
SIMLOOP_SIM_MIN = int(os.getenv("SIMLOOP_SIM_MIN", "2"))          # simulated-only trigger floor
SIMLOOP_EXEC_TIMEOUT = int(os.getenv("SIMLOOP_EXEC_TIMEOUT", "45"))  # seconds per harness
# Units the cases call left without a single case (9759e0ca: 4 of 5 units in a batch, the
# whole install path untested) get one more, smaller cases call for just those units.
SIMLOOP_CASES_REASK = os.getenv("SIMLOOP_CASES_REASK", "1") in ("1", "true", "yes")
# After a kept refine, every case that MATCHED before the refine is executed again; a refine
# that flips a previously matching executed case is reverted (the executed analogue of the
# gate's "no new violations").
SIMLOOP_EXEC_REGRESSION = os.getenv("SIMLOOP_EXEC_REGRESSION", "1") in ("1", "true", "yes")
# An executed mismatch triggers refine on its own only when its EXPECTED is ANCHORED in the
# specification: a quoted literal, number or exception name of EXPECTED appears verbatim in
# the issue/requirements/interface. An EXPECTED the spec never states is the case-writer's
# inference from the same reading the repair used -- harness-graded over 3 passes such
# mismatches broke 4 correct patches (d72025be, fcfa069a x2, e1e50298's sibling) and fixed
# none. Unanchored mismatches are still shown to refine (marked) when something else triggers.
SIMLOOP_ANCHORED_TRIGGER = os.getenv("SIMLOOP_ANCHORED_TRIGGER", "1") in ("1", "true", "yes")
# Same policy for REASONED findings (SIMLOOP_MODE=reason): a confirmed finding whose EXPECTED the
# spec never states (its BASIS is an existing test or caller) marks the row unanchored, so it is
# shown to refine but cannot trigger one on its own. Measured (ccpipe m3_A2, a26c325b): the
# contract lens cited a pre-change test pinning `fallback_mock.call_count == 17`; the spec adds a
# parameter that necessarily adds an 18th fallback call; refine "fixed" the count and the hidden
# (updated) test failed -- a harness-verified-correct patch broken by a stale-test basis.
# Default OFF to keep the stock arm reproducible; ccpipe sets it on.
SIMLOOP_REASON_ANCHOR = os.getenv("SIMLOOP_REASON_ANCHOR", "") in ("1", "true", "yes")
# G8 (Go defect assessment 2026-09-16): on Go a confirmed REASONED finding triggers refine only
# when it passes the checks named here ("touched", "anchored"; comma-separated, "" = off).
# touched: a TRACE quote is a line the patch ADDED -- a finding traced only through base lines
#   describes behaviour the change did not produce (vuls-fe8d252c: "error if the server name is
#   absent" read as unconditional; refine dropped `toLocalFile &&` from an untouched check).
# anchored: EXPECTED's distinguishing content is stated verbatim in the specification, a BASIS
#   sentence alone is not enough (teleport-bb69574e: an invented `{{literal.foo}}` input made refine
#   add a field to a struct the hidden tests compare whole; teleport-288c5519: "should continue to
#   be derived from the hostname" turned into a rewrite of subject.Organization).
# All 3 Go baseline-only losses of that class repeated in 3/3 attempts. Failing rows are still
# shown to refine, marked, when another row triggers it.
SIMLOOP_GO_REASON_TRIGGER = tuple(x for x in os.getenv("SIMLOOP_GO_REASON_TRIGGER", "touched,anchored")
                                  .replace(" ", "").split(",") if x)

_UNIT_RE = re.compile(
    r"===\s*UNIT\s+id=(\S+)\s+file=(\S+)\s+symbol=(\S+)\s*===\n(.*?)\n===\s*END UNIT\s*===",
    re.S)
_HARNESS_RE = re.compile(r"HARNESS:\s*```(?:python|py)?[ \t]*\n(.*?)\n```", re.S | re.I)
_CASE_HDR_RE = re.compile(r"===\s*CASE\s+id=(\S+)\s*===[ \t]*\n", re.M)
_CASE_END_RE = re.compile(r"\n?===\s*END CASE\s*===")
_VERDICT_RE = re.compile(
    r"\[([^\]\s]+)\]\s*VERDICT:\s*(MATCH|MISMATCH|UNVERIFIABLE)\b", re.I)






class _QueryWallTimeout(BaseException):
    """BaseException so provider ``except Exception`` retry loops cannot swallow our deadline."""


@contextlib.contextmanager
def _query_wall_deadline(seconds: float):
    """Enforce one absolute deadline around all provider attempts on the main pipeline thread."""
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def expire(_signum, _frame):
        raise _QueryWallTimeout(f"tool-free query exceeded {seconds:g}s wall-clock deadline")

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _normalize_case_id(case_id: str) -> str:
    """Normalize harmless model decoration such as ``[<U5#1>]`` to the stored ``U5#1``."""
    out = (case_id or "").strip().strip("`")
    while len(out) >= 2 and ((out[0], out[-1]) in (("<", ">"), ("'", "'"), ('"', '"'))):
        out = out[1:-1].strip()
    return out


def _write_json_checkpoint(path: Path, payload: dict) -> None:
    """Atomically persist a trajectory so an interrupt cannot erase completed unit work."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    tmp.replace(path)


def _repair_context(repair: dict) -> str:
    """The initial-patch generation transcript, as shared context for steps 4.1-4.4."""
    parts = []
    for m in (repair.get("messages") or []):
        role = m.get("role")
        if role not in ("assistant", "tool"):
            continue
        c = m.get("content")
        if c is None and role == "assistant":
            for t in (m.get("tool_calls") or []):
                try:
                    c = "$ " + json.loads(t["function"]["arguments"]).get("command", "")
                except Exception:
                    c = None
        if isinstance(c, str) and c.strip():
            parts.append(f"[{role}] {c.strip()}")
    body = "\n".join(parts) or "(no transcript captured)"
    return ("=== YOUR OWN WORK PRODUCING THE INITIAL PATCH (reasoning + commands + outputs) ===\n"
            "This is the context in which the patch below was written. Continue from it.\n"
            + _sub._clip_tail(body, SIMLOOP_CTX_CLIP) + "\n")


_EXTRACT_PROMPT = """\
{context}
=== THE INITIAL PATCH YOU JUST PRODUCED ===
{manifest}

{diff}

=== POST-PATCH SOURCE (deterministic reads from the patched tree) ===
{pack}

# TASK
List every CODE UNIT (function, method, or class body) that the patch above ADDED or MODIFIED.
Skip pure documentation, comment-only, and import-only edits. At most {cap} units, most
central to the fix first.

For each unit emit EXACTLY this block, and nothing else between the markers:

=== UNIT id=U1 file=<repo-relative path> symbol=<qualified name, e.g. ClassName.method> ===
<the unit's COMPLETE POST-PATCH source, copied verbatim from the source section above>
=== END UNIT ===

Rules:
- The source you copy must be the PATCHED source as it now exists, not the diff and not the
  pre-patch version. If the post-patch source section does not show a unit, omit that unit
  rather than reconstructing it from memory.
- Number the ids U1, U2, ... in order.
"""


_SIMULATE_PROMPT = """\
{context}
=== UNIT UNDER TEST: {uid}  ({file} :: {symbol}) ===
{code}

=== SUPPORTING POST-PATCH SOURCE (callees / neighbours, for resolving what the unit calls) ===
{pack}

=== CASES TO EXECUTE ===
{cases}

# TASK
Hand-execute the unit above on EACH case. Do not run anything; trace the code in your reasoning.

For each case, in order:
1. Walk the unit line by line on that INPUT, writing down the CONCRETE VALUE each relevant
   expression takes. When the unit calls something shown in the supporting source, trace into
   it; when it calls something not shown, state the assumption you are making about its result.
2. State the ACTUAL final value the patched code produces.
3. Compare ACTUAL against EXPECTED **literally** -- byte for byte for strings, element for
   element for sequences, exact type for exceptions. "Equivalent" is not MATCH.

Emit one block per case:

[<case-id>] VERDICT: MATCH|MISMATCH|UNVERIFIABLE
TRACE: <the concrete value walk, briefly>
ACTUAL: <the exact value the patched code produces>
WHY: <for MISMATCH: name the exact line whose value diverges and state the value it produces
      versus the value the case requires; for UNVERIFIABLE: exactly what you could not resolve>

DIAGNOSE ONLY -- DO NOT PRESCRIBE A FIX. In WHY, do not propose an edit, do not write
replacement code, and do not say "to fix, add ...". You are reasoning without executing
anything, so you cannot know that a suggested edit is safe for the other callers of this unit
or for the rest of the patch. State only what diverges and where; deciding what to change is
the next step's job, made with the tools you do not have.

Be adversarial with yourself: you wrote this patch, so you are predisposed to read it
charitably. A case you cannot resolve is UNVERIFIABLE, never MATCH.
"""


# --- batched 4.2 / 4.3 -------------------------------------------------------------------------
# Same instructions as the single-unit prompts; the only change is that N units share one header.
# The per-unit answer is keyed by the CASE id (`<uid>#<n>`), which is how each case and verdict is
# attributed back to its unit -- see _uid_of.
_CASES_PROMPT_BATCH = """\
{context}
=== SPECIFICATION (the ground truth; the code is NOT the oracle) ===
{problem_statement}

REQUIREMENTS:
{requirements}

INTERFACE:
{interface}

{units}

# TASK
Derive up to 4 concrete test CASES **for EACH unit above**, from the SPECIFICATION ONLY.

The single most important rule: the EXPECTED value must come from what the specification says
must happen, NOT from reading what the code above does. If you find yourself computing EXPECTED
by tracing the code, you have written a tautology and the case is worthless -- delete it and
pick a behaviour the spec states explicitly.

Cover, where the spec supports it: the behaviour the issue asks for; any edge/boundary the spec
names; and at least one case that must be UNCHANGED by this fix (a regression guard).

=== MANDATORY CASES (derived from the signature changes in this patch) ===
{required}

Any mandatory case listed above MUST be emitted with the exact id given. It is not optional and
does not count against the four-case budget.

Work through the units IN THE ORDER GIVEN. The case id MUST begin with that unit's id followed
by `#` and a number -- e.g. for unit U3 the ids are `U3#1`, `U3#2`, ... A case whose id does not
name one of the units above is discarded.

Emit EXACTLY these blocks and nothing else between the markers:

=== CASE id=<UNIT_ID>#<n> ===
INPUT: <a concrete call with literal argument values, plus any state the call depends on>
EXPECTED: <the exact expected return value / raised exception / observable effect, literally>
EXPECTED_REPR: <the exact Python repr() the harness below must print for EXPECTED, e.g. 'done'
or {{'key': 1}} or True -- or `RAISES: <ExceptionClassName>` when the spec requires an exception>
BASIS: <one sentence quoted or closely paraphrased from the specification that fixes EXPECTED>
HARNESS:
```python
<a SELF-CONTAINED script, run from the repository root with the package importable, that
imports the unit through the repository's real import path, builds INPUT (stubbing external
services/hardware with the simplest fake that lets the unit run), and ends with EXACTLY ONE line
print("ACTUAL=" + repr(<the value EXPECTED talks about>)). No pytest, no network, no writes
outside /tmp. If the behaviour genuinely cannot be exercised in-process, write HARNESS: none>
```
=== END CASE ===

The HARNESS is executed for real; its printed ACTUAL decides MATCH/MISMATCH, so make it faithful
to INPUT and keep it minimal. A harness that fails to import or run does not count against the
code -- it is simply unverifiable.

If the specification genuinely does not constrain a given unit's observable behaviour, emit no
blocks for that unit and say `NO SPEC-GROUNDED CASES: <unit id>` with one sentence of
justification. Units are independent -- one unit having no cases does not excuse the others.
"""

_SIMULATE_PROMPT_BATCH = """\
{context}
{units}

=== SUPPORTING POST-PATCH SOURCE (callees / neighbours, for resolving what the units call) ===
{pack}

=== CASES TO EXECUTE (the id prefix names the unit each case belongs to) ===
{cases}

# TASK
Hand-execute EVERY case above against the post-patch source of ITS OWN unit and give one
verdict per case, in the case's own id:

[<case id>] VERDICT: MATCH | MISMATCH | UNVERIFIABLE
ACTUAL: <what the code above actually produces for that INPUT, traced step by step>
WHY:    <for MISMATCH/UNVERIFIABLE: where the divergence arises. One or two sentences.>

DIAGNOSE ONLY -- DO NOT PRESCRIBE A FIX. In WHY, do not propose an edit, do not write
replacement code, and do not say "to fix, add ...". You are reasoning without executing
anything, so you cannot know that a suggested edit is safe for the other callers of this unit
or for the rest of the patch. State only what diverges and where; deciding what to change is
the next step's job, made with the tools you do not have.

Be adversarial with yourself: you wrote this patch, so you are predisposed to read it
charitably. A case you cannot resolve is UNVERIFIABLE, never MATCH.
"""

_REVIEW_COMMON_RULES = """\
Rules for every FINDING:
- The complete specification takes precedence over pre-change tests when it explicitly changes
  their behavior. Do not infer a production requirement from a test-only interface declaration.
- Trace values through their consumers to the public observation (including serialization,
  display and missing-item behavior), not only through the producer's intermediate tuple/object.
- Check required declarations and application resources as well as edited functions. A fallback
  for a missing registration is not evidence that the requested feature is registered.
- Keep the answer concise: at most six strongest findings, with only the decisive trace lines.
- INPUT is concrete (a call with literal arguments, or a precise scenario).
- EXPECTED is what the specification, an existing test, or an existing caller REQUIRES -- not
  what you would prefer. BASIS quotes the sentence or line it comes from VERBATIM.
- TRACE quotes the exact lines of the post-patch source you followed, one per line, each
  starting with "> ". Quote real lines; paraphrase is not evidence. Quoted lines that do not
  exist in the text above are discarded mechanically.
- ACTUAL is what the quoted lines produce for INPUT, step by step.
- Report ONLY divergences. Do not prescribe a fix. If you find none, write "NO FINDINGS".
- Be adversarial with yourself: this patch is yours, you are predisposed to read it charitably.

FINDING format (repeat per finding):
=== FINDING id=F<n> ===
UNIT: <unit id>
KIND: BEHAVIOUR | CONTRACT
INPUT: <concrete input or call>
EXPECTED: <required outcome>
BASIS: <verbatim quote from the specification, an existing test, or a caller>
TRACE:
> <quoted source line>
> <quoted source line>
ACTUAL: <traced outcome>
=== END FINDING ===
"""

_REVIEW_TRACE_PROMPT = """\
{context}
=== SPECIFICATION ===
{problem_statement}

=== REQUIREMENTS ===
{requirements}

=== INTERFACE ===
{interface}

{units}

=== SUPPORTING POST-PATCH SOURCE (callees / neighbours) ===
{pack}

=== DEFINITIONS OF WHAT THE UNITS CALL (read from the repository; trace what a callee does
with the values it is given -- a constructor or helper may transform them) ===
{callees}

=== EXISTING CALLERS AND REPOSITORY TESTS ===
{callers}
{tests}

# TASK (lens 1: requirement trace)
For EACH requirement and interface clause above that the units implement, choose one or two
concrete inputs that exercise it -- including the boundary the clause implies (absent value,
empty collection, disabled flag, second call, mismatched case) -- and hand-trace the post-patch
code path for that input, line by line. Compare the traced result with what the clause states.
Report every divergence as a FINDING of KIND BEHAVIOUR.

""" + _REVIEW_COMMON_RULES

_REVIEW_CONTRACT_PROMPT = """\
{context}
=== SPECIFICATION ===
{problem_statement}

=== REQUIREMENTS ===
{requirements}

=== INTERFACE ===
{interface}

{units}

=== EXISTING CALLERS OF THE CHANGED SYMBOLS (pre-existing code that invokes them) ===
{callers}

=== EXISTING TESTS COVERING THE CHANGED MODULES (as they exist in the repository) ===
{tests}

=== THE PATCH (what changed; `+` lines are new) ===
{diff}

=== DEFINITIONS OF WHAT THE UNITS CALL (read from the repository) ===
{callees}

=== FORWARDING FACTS (computed by parsing the patched source -- nothing was executed) ===
For each function that gained a parameter: the call in its body that receives the most of the
function's other parameters IS its sibling path, and whether the new parameter is on it.
These facts override any contrary reading of the code; a fact ending in "IS NOT passed" is a
CONTRACT finding unless the specification names a different call for that value.
{facts}

# TASK (lens 2: contract review)
Work in two parts. Part 1 is MANDATORY even when you expect no findings: a "NO FINDINGS"
answer without the table is not accepted.

PART 1 -- CONTRACT TABLE. One row per NEW or CHANGED parameter of every function the patch
touches, and one row per function whose return value the patch changes:
  FUNCTION | NEW/CHANGED ITEM | SIBLING PATH: the call inside the function that receives the
  MOST of the function's other parameters -- count them and name the call (quote the line);
  a constructor or helper that receives only the new item is NOT the sibling path |
  NEW ITEM PASSED ON THAT SAME CALL, as a keyword argument? YES/NO (quote the line) |
  WHO PINS IT: the existing test that asserts THAT call's argument list, or the caller (quote)
  A requirement sentence that says where the value is used or stored does not change the
  sibling path: the siblings' call is where tests assert the argument list.
For return values: FUNCTION | BRANCH | KIND RETURNED (entity, dict, list, None, ...) | KIND THE
CALLERS/TESTS CONSUME (quote).
PART 2 -- FINDINGS. Every row whose answer is NO, and every function whose branches return
different kinds or a kind the callers/tests do not consume, is a FINDING of KIND CONTRACT
(BASIS = the pinning quote from the table). Only after the table may you write "NO FINDINGS".

Background for the table:
For EACH function or method the patch adds or changes, compare its CONTRACT against three
sources of truth: (a) the interface/requirements text (parameter names, defaults, where a value
must be forwarded to, the type of what is returned); (b) how EXISTING TESTS invoke it and what
they assert about its calls and results (a test that asserts the exact keyword arguments of a
forwarded call, or `.attr` access on the result, fixes the call shape and the return type);
(c) how EXISTING CALLERS use it. The existing tests predate the change: use them for the SHAPE
of a contract (which callee receives the forwarded keyword arguments, what attribute or type
the result must have, which arguments are positional), never to forbid the new parameter or
behaviour the specification adds -- a test that asserts an exact keyword list will be updated
to include the new keyword on the SAME call, so the finding to make is "the new keyword is not
forwarded to that call", not "a new keyword was added". Check in particular: a new parameter is
forwarded along the SAME path as its sibling parameters (into the same call the tests assert); every branch of a function returns the SAME kind of value
(an entity vs a dict, a list vs None) and the kind the callers and tests consume; nothing the
tests pass positionally changed position; defaults match the text. Report every deviation as a
FINDING of KIND CONTRACT, with BASIS quoting the test line, caller line or clause.

""" + _REVIEW_COMMON_RULES

_REVIEW_VERIFY_PROMPT = """\
{context}
=== AUTHORITATIVE SPECIFICATION ===
{problem_statement}
=== REQUIREMENTS ===
{requirements}
=== INTERFACE ===
{interface}
The specification overrides conflicting pre-change test expectations. Test-only interface
declarations are not requirements to ship test classes as production implementation.
{units}

=== SUPPORTING POST-PATCH SOURCE (callees / neighbours) ===
{pack}

=== EXISTING CALLERS ===
{callers}

=== EXISTING TESTS ===
{tests}

=== DEFINITIONS OF WHAT THE UNITS CALL (read from the repository) ===
{callees}

=== FORWARDING FACTS (computed by parsing the patched source -- nothing was executed) ===
{facts}

=== CANDIDATE FINDINGS (from an independent first review) ===
{findings}

# TASK (verifier)
You did NOT write these findings. For EACH one, re-derive it from scratch: take its INPUT,
trace the post-patch source yourself, and decide whether the divergence is REAL. A finding is
REFUTED when the code actually produces EXPECTED, when EXPECTED is not required by the quoted
BASIS, or when BASIS misreads the specification/test/caller. It is CONFIRMED only when your own
trace reaches the same ACTUAL and the BASIS really requires EXPECTED.

Two rules of judgement for CONTRACT findings:
- Existing tests predate the change. A test that asserts a call's exact argument list fixes
  WHICH call carries the function's parameters; it will be updated to include the new
  parameter on that same call. So "the existing test does not list the new parameter" never
  refutes a finding -- it is the reason the finding exists. Refute only if the new parameter IS
  on that call, or the specification names a different call.
- Behavioural equivalence is not contract equivalence. Routing a value through a constructor,
  an attribute, a fallback or a helper instead of the call the siblings use produces the same
  result today and still fails the asserted call shape; that is CONFIRMED, not REFUTED.

Answer one block per finding, in the finding's own id:
[F<n>] VERDICT: CONFIRMED | REFUTED
TRACE:
> <quoted source line>
> <quoted source line>
WHY: <one or two sentences>
"""

_FINDING_RE = re.compile(r"===\s*FINDING\s+id=(F\d+)\s*===\s*\n(.*?)===\s*END FINDING\s*===", re.S)
_VERIFY_RE = re.compile(r"\[([A-Z]?F\d+)\]\s*VERDICT:\s*(CONFIRMED|REFUTED)(.*?)(?=\n\[[A-Z]?F\d+\]\s*VERDICT:|\Z)", re.S)


# "NO FINDINGS" as the model actually writes it: as a markdown heading (`# NO FINDINGS`), in
# bold, or followed by a dash/colon clause ("NO FINDINGS -- all requirements pass"). The bare
# form was the only one accepted, so 39 of the 45 incomplete trace steps in ccpipe batches 2-3
# were a correctly-reviewed instance spending a second call to say the same thing again.
_NO_FINDINGS_RE = re.compile(
    r"^[ \t]{0,3}(?:[#>*_]{1,6}[ \t]*)?NO[ \t]+FINDINGS[ \t]*(?:[*_#]{1,4})?"
    r"[ \t]*(?:[:\u2014\u2013-][^\n]*)?[.!]?[ \t]*$", re.M)

_PROSE_LABEL_RE = re.compile(
    r"[ \t]{0,3}(?:[-+#>*_]{1,4}[ \t]*)?(INPUT|TRACE|EXPECTED|ACTUAL|BASIS|UNIT|KIND)\b[^:\n]{0,24}?:")
# emphasis only: a backtick is never stripped, because anchoring reads backticked spans as
# literals and an unpaired one would let a bare identifier anchor a finding (622a493a).
_PROSE_DECOR_RE = re.compile(r"(?:[*_]{1,3})$")


def _prose_normalize(text: str) -> str:
    """Strip markdown decoration from field labels so `**TRACE:**` / `- **EXPECTED:** x` /
    `**INPUT 1a: valid state**` read as the plain `LABEL: value` the block format uses."""
    out = []
    for ln in (text or "").splitlines():
        m = _PROSE_LABEL_RE.match(ln)
        if m:
            rest = _PROSE_DECOR_RE.sub("", ln[m.end():].strip()).strip()
            rest = re.sub(r"^(?:[*_]{1,3})", "", rest).strip()
            ln = f"{m.group(1)}: {rest}".rstrip()
        out.append(ln)
    return "\n".join(out)


def _prose_pass(actual: str) -> bool:
    """True when a prose segment's ACTUAL reports the case passing -- a tick, or a leading
    pass word. The review prompt asks for passing cases to be omitted; when the model narrates
    them anyway they are not candidate findings."""
    a = (actual or "").strip()
    if not a or any(ch in a for ch in "\u2713\u2714\u2611"):
        return True
    return bool(re.match(r"^[*_`\s]*(OK|PASS(?:ES|ED)?|CORRECT|MATCHES|AS EXPECTED|REQUIREMENT MET|"
                         r"NO MISMATCH|SAME|UNCHANGED|YES)\b", a, re.I))


def _prose_review_summary(text: str) -> "tuple[list[dict], int]":
    """Salvage findings from a free-form review that never used the FINDING block format.

    Returns (findings, n_segments). A segment is one `INPUT:`-headed stretch; it becomes a
    candidate only when it cites at least one TRACE quote AND states EXPECTED and an ACTUAL
    that does not read as a pass. Grounding and the verifier still decide -- this only makes
    the finding visible to them. n_segments tells the caller the model did review case by case
    (all passing) rather than answering in an unusable shape."""
    lines = _prose_normalize(text).splitlines()
    idx = [i for i, ln in enumerate(lines) if ln.startswith("INPUT:")]
    out = []
    for j, i in enumerate(idx):
        body = "\n".join(lines[i:idx[j + 1] if j + 1 < len(idx) else len(lines)])
        quotes = [l[1:].strip() for l in body.splitlines() if l.startswith(">")]
        if not quotes:   # TRACE written inline: the backticked spans are the citation
            quotes = [q for q in re.findall(r"`([^`\n]{12,})`", _field(body, "TRACE"))]
        expected, actual = _field(body, "EXPECTED"), _field(body, "ACTUAL")
        if not (quotes and expected and actual) or _prose_pass(actual):
            continue
        out.append({"id": "P?", "uid": _field(body, "UNIT"), "kind": _field(body, "KIND") or "BEHAVIOUR",
                    "input": _field(body, "INPUT"), "expected": expected,
                    "basis": _field(body, "BASIS"), "actual": actual,
                    "quotes": [q for q in quotes if q], "from_prose": True})
    return out, len(idx)


def _parse_findings(text: str) -> "list[dict]":
    out = []
    for fid, raw in _FINDING_RE.findall(text or ""):
        # `**TRACE:**` / `- TRACE:` decorated the label on a correctly-blocked finding, the quote
        # regex below (bare `TRACE:`) found nothing, and the finding died ungrounded -- js_batch2
        # element-web-b007ea81, where the dropped finding named the defect exactly. _field is
        # already decoration-tolerant; normalize the body so the TRACE quotes are too.
        body = _prose_normalize(raw)
        tr = re.search(r"^TRACE:\s*\n((?:>.*\n?)+)", body, re.M)
        quotes = [l[1:].strip() for l in (tr.group(1).splitlines() if tr else []) if l.startswith(">")]
        out.append({"id": fid, "uid": _field(body, "UNIT"), "kind": _field(body, "KIND") or "BEHAVIOUR",
                    "input": _field(body, "INPUT"), "expected": _field(body, "EXPECTED"),
                    "basis": _field(body, "BASIS"), "actual": _field(body, "ACTUAL"),
                    "quotes": [q for q in quotes if q]})
    # Fall back to the prose salvage only when the reply used no blocks at all and did not
    # declare NO FINDINGS: an explicit disclaimer is respected, never overridden.
    if out or not (text or "").strip() or _NO_FINDINGS_RE.search(text):
        return out
    return _prose_review_summary(text)[0]


def _findings_from_table(reply: str, units: "list[dict]") -> "list[dict]":
    """Contract-table rows answered NO that the reviewer did not turn into FINDING blocks
    (a26c325b roll 8: the table said "NO -- the sibling Request.open call ... does not receive
    use_netrc" with the pinning test quoted, then no Part 2). Each such row becomes a
    candidate finding; grounding and the verifier still decide."""
    out = []
    if not reply or "|" not in reply:
        return out
    by_sym = {}
    for u in units:
        sym = (u.get("symbol") or "")
        by_sym[sym] = u["id"]; by_sym[sym.split(".")[-1]] = u["id"]
    for line in reply.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        # the YES/NO verdict is the 4th column; "No sibling call" / "N/A" cells elsewhere are not it
        ans = cells[3] if len(cells) > 3 else ""
        if not re.match(r"^\**\s*NO\b(?!\s+(sibling|forwarding|call|such))", ans, re.I) or re.match(r"^N/?A\b", ans, re.I):
            continue
        func = cells[0]
        uid = next((by_sym[k] for k in by_sym if k and re.search(r"\b" + re.escape(k) + r"\b", func)), "")
        quotes = re.findall(r"`([^`]{12,})`", line)
        pin = cells[4] if len(cells) > 4 else cells[-1]
        out.append({"id": "T?", "uid": uid, "kind": "CONTRACT",
                    "input": f"{func}: {cells[1]}",
                    "expected": f"{cells[1]} is passed on the same call as its sibling parameters ({cells[2][:120]})",
                    "basis": pin, "actual": ans, "quotes": quotes,
                    "from_table": True})
    return out


def _ground_finding(f: dict, code_corpus: str, basis_corpus: str) -> "tuple[bool, str]":
    """A finding is grounded when >= 1 quoted TRACE line exists in the reviewed source text
    (whitespace-normalised, >= 12 chars) and its BASIS quote exists in the spec / tests /
    callers text. This checks citations, it does not run anything."""
    from simagent.evidence import quote_text
    cc = sp._ws_squash(code_corpus or ""); bc = sp._ws_squash(basis_corpus or "")
    good_q = [q for q in f.get("quotes", []) if len(quote_text(q)) >= 12 and quote_text(q) in cc]
    if not good_q:
        return False, "no quoted TRACE line exists in the reviewed source"
    if f.get("from_facts"):
        return True, ""   # the basis is a mechanical fact about the source, not a quote
    basis = f.get("basis", "")
    lits = [next(g for g in m.groups() if g) for m in _QUOTED_RE.finditer(basis)]
    cands = lits + [basis]
    if not any(len(c.strip()) >= 12 and sp._ws_squash(c.strip().strip("`'\"")) in bc for c in cands):
        # tolerate a close paraphrase: 6 consecutive words of the basis in the corpus
        words = basis.split()
        if not any(" ".join(words[i:i + 6]).lower() in bc.lower() for i in range(max(0, len(words) - 5))):
            return False, "BASIS is not a quote of the specification, an existing test or a caller"
    return True, ""


# ---- reasoning-review context helpers: read-only facts about the patched source -------------

import builtins as _builtins
_PY_BUILTIN_NAMES = set(dir(_builtins))


def _forwarding_facts_from_sources(pre_src: str, post_src: str, path: str) -> "list[dict]":
    """For every function/method whose signature GAINED parameters between pre and post source:
    the call inside its body that receives the most of the function's parameters (the sibling
    path), and whether each new parameter is passed on that same call. Pure ast reading."""
    import ast as _ast
    out = []
    try:
        pre_t = _ast.parse(pre_src or ""); post_t = _ast.parse(post_src or "")
    except SyntaxError:
        return out
    def funcs(tree):
        res = {}
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                a = node.args
                params = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
                if a.vararg: params.append(a.vararg.arg)
                if a.kwarg: params.append(a.kwarg.arg)
                res.setdefault(node.name, []).append((node, params))
        return res
    pre_f, post_f = funcs(pre_t), funcs(post_t)
    post_lines = (post_src or "").splitlines()
    for name, defs in post_f.items():
        for node, params in defs:
            pre_params = set()
            for _n, pp in pre_f.get(name, []):
                pre_params |= set(pp)
            new = [q for q in params if q not in pre_params and q not in ("self", "cls")]
            if not new or not pre_f.get(name):
                continue
            pset = set(params) - {"self", "cls"}
            best = None
            for c in _ast.walk(node):
                if not isinstance(c, _ast.Call):
                    continue
                passed = set()
                for kw in c.keywords:
                    if kw.arg and isinstance(kw.value, _ast.Name) and kw.value.id in pset:
                        passed.add(kw.value.id)
                for a in c.args:
                    if isinstance(a, _ast.Name) and a.id in pset:
                        passed.add(a.id)
                if len(passed) >= 2 and (best is None or len(passed) > len(best[1])):
                    best = (c, passed)
            if best is None:
                continue
            call, passed = best
            callee = _ast.unparse(call.func) if hasattr(_ast, "unparse") else "<call>"
            line = post_lines[call.lineno - 1].strip() if 0 < call.lineno <= len(post_lines) else ""
            for q in new:
                out.append({"path": path, "function": name, "new_param": q, "callee": callee,
                            "line_no": call.lineno, "line": line, "n_siblings": len(passed),
                            "forwarded": q in passed})
    return out


def _forwarding_facts(env, repo_path: str, patch: str) -> "tuple[str, list[dict]]":
    """Rendered FORWARDING FACTS for the contract lens + the raw fact rows."""
    facts = []
    for f in sp._DIFF_FILES_RE.findall(patch or "")[:6]:
        if not f.endswith(".py") or "/test" in f or f.startswith("test"):
            continue
        try:
            post = env.execute({"command": f"cd {repo_path} && cat {shlex.quote(f)}"}, timeout=30).get("output", "") or ""
            pre = env.execute({"command": f"cd {repo_path} && git show HEAD:{shlex.quote(f)} 2>/dev/null"}, timeout=30).get("output", "") or ""
        except Exception:
            continue
        facts += _forwarding_facts_from_sources(pre, post, f)
    if not facts:
        return "(no function gained a parameter, or no call in its body receives 2+ of its parameters)", []
    lines = []
    for x in facts:
        lines.append(f"- {x['path']} :: {x['function']}() gained parameter `{x['new_param']}`. Its sibling path is the call "
                     f"`{x['callee']}(...)` at line {x['line_no']} (receives {x['n_siblings']} of the function's parameters):\n"
                     f"    > {x['line'][:160]}\n"
                     f"  `{x['new_param']}` {'IS' if x['forwarded'] else 'IS NOT'} passed on that call.")
    return "\n".join(lines), facts


_GO_NON_CALLEES = frozenset((
    "if for switch select func return go defer make new len cap append copy delete close panic "
    "recover print println complex real imag min max clear string int int8 int16 int32 int64 "
    "uint uint8 uint16 uint32 uint64 uintptr byte rune float32 float64 bool error any").split())


def _go_call_names(code: str, defined=frozenset()) -> "list[str]":
    """Function/method names a Go unit calls (``f(`` / ``x.f(``), minus keywords and builtins."""
    names = []
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", code or ""):
        n = m.group(1)
        if n in _GO_NON_CALLEES or n in defined or len(n) < 3 or n in names:
            continue
        names.append(n)
    return names


def _callee_definitions_go(env, repo_path: str, units: "list[dict]", cap: int = 9000) -> str:
    """Go twin of _callee_definitions (defect G2): the Python version greps `def|class` in *.py
    only, so the callee context was empty in 58 of 59 Go batch1 reviews."""
    defined = {(u.get("symbol") or "").split(".")[-1] for u in units}
    names = []
    for u in units:
        for n in _go_call_names(u.get("code") or "", defined):
            if n not in names:
                names.append(n)
    parts, used = [], 0
    for n in names[:14]:
        try:
            out = env.execute({"command":
                f"cd {repo_path} && grep -rn -m1 -E '^(func (\\([^)]*\\) )?{n}\\b|type {n}\\b)' "
                f"--include='*.go' . --exclude-dir=.git --exclude-dir=vendor 2>/dev/null "
                f"| grep -v '_test\\.go:' | head -1"}, timeout=30).get("output", "") or ""
            m = re.match(r"\./(\S+?):(\d+):", out.strip())
            if not m:
                continue
            f, ln = m.group(1), int(m.group(2))
            body = env.execute({"command": f"cd {repo_path} && sed -n '{ln},{ln + 44}p' {shlex.quote(f)}"},
                               timeout=30).get("output", "") or ""
            chunk = f"--- {f}:{ln} ({n}) ---\n{body.rstrip()}\n"
            if used + len(chunk) > cap:
                break
            parts.append(chunk); used += len(chunk)
        except Exception:
            continue
    return "\n".join(parts) or "(no callee definitions found)"


def _callee_definitions(env, repo_path: str, units: "list[dict]", cap: int = 9000) -> str:
    """Definitions (first ~45 lines) of the functions/classes the units CALL, read from the
    repository -- so the reviewer can trace what a callee does with the values it receives
    (305e7c96: ListCategory turns an explicit None column into ''; its source was not in view)."""
    if _sub.is_go():
        return _callee_definitions_go(env, repo_path, units, cap)
    import ast as _ast
    names, defined = [], set()
    for u in units:
        code = u.get("code") or ""
        try:
            t = _ast.parse(code)
        except SyntaxError:
            try:
                t = _ast.parse("if True:\n" + "\n".join("    " + l for l in code.splitlines()))
            except SyntaxError:
                t = None
        if t is None:
            for m in re.finditer(r"\b([A-Za-z_]\w+)\(", code):
                names.append(m.group(1))
            continue
        for node in _ast.walk(t):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                defined.add(node.name)
            if isinstance(node, _ast.Call):
                fn = node.func
                nm = fn.id if isinstance(fn, _ast.Name) else (fn.attr if isinstance(fn, _ast.Attribute) else None)
                if nm:
                    names.append(nm)
    seen, order = set(), []
    for n in names:
        if n in seen or n in defined or n in _PY_BUILTIN_NAMES or len(n) < 3:
            continue
        seen.add(n); order.append(n)
    parts, used = [], 0
    for n in order[:14]:
        try:
            out = env.execute({"command":
                f"cd {repo_path} && grep -rn -m1 -E '^\\s*(def|class)\\s+{re.escape(n)}\\b' --include='*.py' . "
                f"--exclude-dir=.git --exclude-dir=tests --exclude-dir=test 2>/dev/null | head -1"}, timeout=30).get("output", "") or ""
            m = re.match(r"\./(\S+?):(\d+):", out.strip())
            if not m:
                continue
            f, ln = m.group(1), int(m.group(2))
            body = env.execute({"command": f"cd {repo_path} && sed -n '{ln},{ln + 44}p' {shlex.quote(f)}"}, timeout=30).get("output", "") or ""
            chunk = f"--- {f}:{ln} ({n}) ---\n{body.rstrip()}\n"
            if used + len(chunk) > cap:
                break
            parts.append(chunk); used += len(chunk)
        except Exception:
            continue
    return "\n".join(parts) or "(no callee definitions found)"


# In-container excerpting of a covering test file by TEST FUNCTION (reads text only): each
# `def test_*` / `class Test*` block is scored by the rarity weights of the changed names it
# mentions (+0.3 for call/attribute assertions) and the best blocks are emitted whole until the
# cap. argv: <file> <json weights> <cap chars>.
_TEST_EXCERPT_PY = r'''
import re, sys, json
w = json.loads(sys.argv[2]); cap = int(sys.argv[3])
src = open(sys.argv[1], errors="ignore").read().splitlines()
idx = [i for i, l in enumerate(src)
       if re.match(r"\s*(async\s+)?def\s+test|\s*class\s+Test|^(func|it\(|test\(|describe\()", l)]
idx.append(len(src)); blocks = []
for a, b in zip(idx, idx[1:]):
    body = "\n".join(src[a:b])
    sc = sum(v for n, v in w.items() if re.search(r"\b" + re.escape(n) + r"\b", body))
    if sc > 0:
        if re.search(r"assert_called|assert_.*with|assert .*\.\w+\s*(==|is)", body):
            sc += 0.3
        blocks.append((sc, a, body))
blocks.sort(key=lambda t: (-t[0], t[1])); out = []; used = 0
# Parametrization tables and fixtures often contain the contract, not the assertion body.
# Include referenced module-level definitions, bounded independently from the test bodies.
try:
    import ast
    tree = ast.parse("\n".join(src))
    selected = "\n".join(body for _, _, body in blocks[:4])
    refs = set(re.findall(r"\b[A-Za-z_]\w*\b", selected))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.AsyncFunctionDef)):
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = {node.name} if not node.name.startswith("test") else set()
            if refs & names:
                a = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
                body = "\n".join(src[a - 1:node.end_lineno])
                chunk = "# dependency lines %d-%d\n%s\n" % (a, node.end_lineno, body)
                remaining = cap // 2 - used
                if remaining > 120:
                    if len(chunk) > remaining:
                        half = (remaining - 40) // 2
                        chunk = chunk[:half] + "\n# DEPENDENCY MIDDLE ELIDED\n" + chunk[-half:]
                    out.append(chunk); used += len(chunk)
except (SyntaxError, ValueError):
    pass
for sc, a, body in blocks:
    chunk = "# lines %d-%d (score %.2f)\n%s\n" % (a + 1, a + body.count("\n") + 1, sc, body)
    if used + len(chunk) > cap:
        chunk = chunk[:max(0, cap - used)]
    out.append(chunk); used += len(chunk)
    if used >= cap:
        break
print("".join(out))
'''


def _resource_context(env, repo_path, specification, units):
    """Read application declarations referenced by the contract or edited code, not gold data."""
    text = specification + "\n" + "\n".join(u.get("code", "") for u in units)
    keys = sorted(set(re.findall(r"[`'\"]([A-Za-z_]\w*(?:[.-]\w+)+)[`'\"]", text)))[:30]
    if not keys:
        return ""
    script = r'''
import json, pathlib, subprocess, sys
keys = json.loads(sys.argv[1]); used = 0
paths = subprocess.check_output(['git', 'ls-files', '-z'], text=True).split('\0')
for name in paths:
    p = pathlib.Path(name)
    if p.suffix not in ('.yaml', '.yml', '.json', '.toml', '.ini') or any(x in p.parts for x in ('tests', 'test', 'node_modules')):
        continue
    if not p.is_file() or p.stat().st_size > 1000000:
        continue
    lines = p.read_text(errors='replace').splitlines()
    for i, line in enumerate(lines):
        if any(key in line for key in keys):
            chunk = '--- %s:%d ---\n%s\n' % (name, i + 1, '\n'.join(lines[max(0, i-2):i+10]))
            if used + len(chunk) > 6000:
                sys.exit(0)
            print(chunk); used += len(chunk)
'''
    try:
        out = env.execute({"command": f"cd {shlex.quote(repo_path)} && python3 - {shlex.quote(json.dumps(keys))} <<'PYRESOURCE'\n{script}\nPYRESOURCE"}, timeout=60)
        return (out.get("output") or "")[:6000]
    except Exception as e:
        return f"(resource lookup unavailable: {type(e).__name__})"


def _review_context(env, repo_path: str, units: "list[dict]", patch: str) -> "tuple[str, str]":
    """(callers_text, tests_text): pre-existing code that invokes the changed symbols, and
    symbol-focused excerpts of the repository's covering test files. Reading text only."""
    names = []
    for u in units:
        # every dotted component counts: `Request.open` contributes both the class name
        # (`Request`, rare -> test_Request.py) and the method name (`open`, everywhere -> ~0)
        for part in (u.get("symbol") or "").split("."):
            if len(part) >= 4 and part not in names and not part.startswith("__"):
                names.append(part)
    # Deleted/renamed entrypoints identify the predecessor tests of new implementations.
    for part in sp._patch_removed_symbols(patch):
        if part not in names:
            names.append(part)
    names = names[:18]
    callers = tests = ""
    if names:
        pat = "|".join(re.escape(n) for n in names)
        try:
            out = env.execute({"command":
                f"cd {repo_path} && grep -rn --include='*.py' --include='*.go' --include='*.js' --include='*.ts' "
                f"-B2 -A3 -E '\\b({pat})\\(' . --exclude-dir=.git --exclude-dir=node_modules "
                f"| grep -vE '^\\S+[-:][0-9]+[-:]\\s*(def|func|function)\\s' | head -160"}, timeout=60)
            callers = _sub._clip((out.get("output", "") or "").strip(), 6000)
        except Exception as e:
            callers = f"(callers lookup unavailable: {type(e).__name__})"
        try:
            # Test files are chosen by CONTENT, not by name convention: the files that mention
            # the changed symbols most often, then the whole test functions around each mention
            # (a26c325b: the name convention picked test_gzip.py; the kwargs contract lived in
            # test_urls.py::test_open_url, which never entered the review).
            # Rarity-weighted ranking: a test file scores 1/(files mentioning that name) per changed
            # symbol it mentions, so `open`/`main` (everywhere) count for little and `fetch_url`
            # (three files) for much; files under a directory named after a changed module get a
            # bonus. (a26c325b roll 2: raw counts ranked test files full of `open(` above
            # test_Request.py, the one asserting the kwargs contract.)
            inc = ("--include='test_*.py' --include='*_test.py' --include='*.test.js' "
                   "--include='*.spec.ts' --include='*_test.go'")
            per_name = {}
            for n in names:
                out = env.execute({"command":
                    f"cd {repo_path} && grep -rlE '\\b{re.escape(n)}\\b' {inc} . --exclude-dir=.git "
                    f"--exclude-dir=node_modules 2>/dev/null | head -60"}, timeout=60)
                per_name[n] = [l.strip().lstrip("./") for l in (out.get("output", "") or "").splitlines() if l.strip()]
            _generic = {"lib", "src", "test", "tests", "units", "unit", "module_utils", "modules",
                        "plugins", "utils", "common", "core", "base", "main", "internal", "pkg"}
            stems = set()
            for f in sp._DIFF_FILES_RE.findall(patch or ""):
                parts = f.split("/")
                stem = parts[-1].rsplit(".", 1)[0]
                for tok in (stem, parts[-2] if len(parts) > 1 else ""):
                    if len(tok) >= 4 and tok not in _generic and tok != "__init__":
                        stems.add(tok)
            score = {}
            for n, files in per_name.items():
                for f in files:
                    score[f] = score.get(f, 0.0) + 1.0 / len(files)
            for f in list(score):
                if any(re.search(r"(^|[/_.])" + re.escape(st) + r"([/_.]|$)", f) for st in stems):
                    score[f] += 0.5
            tf = [f for f, _ in sorted(score.items(), key=lambda kv: -kv[1])][:4]
            if len(tf) < 4:
                # NEW symbols appear in no test yet: fall back to the tests of their SIBLINGS --
                # the other top-level defs of the changed files -- which show the data shapes
                # the module's tests expect (305e7c96: test_models.py tests tab_focus's sibling
                # completion models and shows rows read back as None).
                sib = []
                for f in sp._DIFF_FILES_RE.findall(patch or "")[:4]:
                    if f.endswith(".py"):
                        out = env.execute({"command":
                            f"cd {repo_path} && grep -oE '^(def|class) [A-Za-z_]\\w+' {shlex.quote(f)} | awk '{{print $2}}' | head -40"}, timeout=30)
                        sib += [l.strip() for l in (out.get("output", "") or "").splitlines() if len(l.strip()) >= 4 and l.strip() not in names]
                    elif _sub.is_go() and f.endswith(".go") and not f.endswith("_test.go"):
                        # G2: Go siblings -- the file's top-level funcs/methods/types
                        out = env.execute({"command":
                            f"cd {repo_path} && grep -oE '^(func (\\([^)]*\\) )?|type )[A-Za-z_]\\w+' {shlex.quote(f)} "
                            f"| sed -E 's/^(func (\\([^)]*\\) )?|type )//' | head -40"}, timeout=30)
                        sib += [l.strip() for l in (out.get("output", "") or "").splitlines() if len(l.strip()) >= 4 and l.strip() not in names]
                sib = list(dict.fromkeys(sib))[:12]
                sscore = {}
                for n in sib:
                    out = env.execute({"command":
                        f"cd {repo_path} && grep -rlE '\\b{re.escape(n)}\\b' {inc} . --exclude-dir=.git --exclude-dir=node_modules 2>/dev/null | head -40"}, timeout=60)
                    files = [l.strip().lstrip("./") for l in (out.get("output", "") or "").splitlines() if l.strip()]
                    for f in files:
                        sscore[f] = sscore.get(f, 0.0) + 1.0 / len(files)
                tf = list(dict.fromkeys(tf + [f for f, _ in sorted(sscore.items(), key=lambda kv: -kv[1])]))[:4]
                if tf:
                    per_name.update({n: [] for n in sib})   # excerpt weights below use `weights`
                    weights_extra = {n: 1.0 / max(1, sum(1 for f in sscore)) for n in sib}
                else:
                    weights_extra = {}
            else:
                weights_extra = {}
            if not tf:
                tf = sp._find_covering_tests(env, repo_path, patch, cap=3)
            if _sub.is_go():
                # G2: Go covering tests are PACKAGE DIRECTORIES; the excerpt reader needs files
                # (a directory read as nothing: test context < 100 chars in 27 of 59 Go reviews).
                files_tf = []
                for t in tf:
                    if t.endswith(".go"):
                        files_tf.append(t)
                        continue
                    o = env.execute({"command": f"cd {repo_path} && ls {shlex.quote(t.rstrip('/') or '.')}/*_test.go 2>/dev/null | head -6"}, timeout=30)
                    files_tf += [l.strip()[2:] if l.strip().startswith("./") else l.strip()
                                 for l in (o.get("output", "") or "").splitlines()
                                 if l.strip().endswith("_test.go")]
                tf = list(dict.fromkeys(files_tf))[:4]
            # Excerpt by TEST FUNCTION, most specific first: each `def test_*` block is scored by
            # the rarity weights of the changed names it mentions (+ a bonus for call/attribute
            # assertions), and the best blocks are emitted whole until the per-file cap. A plain
            # line-window grep spends the cap on the first mentions of a common name and never
            # reaches the one function that asserts the contract (a26c325b roll 5:
            # test_Request.py was in context, test_open_url's assert_called_once_with was not).
            weights = {n: 1.0 / len(files) for n, files in per_name.items() if files}
            weights.update(weights_extra)
            wjson = json.dumps(weights)
            parts = []
            for t in tf[:4]:
                out = env.execute({"command":
                    f"cd {repo_path} && python3 - {shlex.quote(t)} {shlex.quote(wjson)} 4200 <<'PYX'\n{_TEST_EXCERPT_PY}PYX"}, timeout=60)
                txt = (out.get("output", "") or "").strip()
                if not txt or "Traceback" in txt[:200]:
                    out = env.execute({"command":
                        f"cd {repo_path} && grep -n -B6 -A22 -E '\\b({pat})\\b' {shlex.quote(t)} | head -200"}, timeout=60)
                    txt = _sub._clip((out.get("output", "") or "").strip(), 4200)
                if txt:
                    parts.append(f"--- {t} ---\n{txt}")
            tests = _sub._clip("\n\n".join(parts), 17000)
        except Exception as e:
            tests = f"(covering tests unavailable: {type(e).__name__})"
    return callers or "(no pre-existing callers found)", tests or "(no covering tests found)"


def _review_reply_complete(reply, finding_ids=None):
    if (not reply or getattr(reply, "finish_reason", None) in ("length", "tool_calls")
            or getattr(reply, "truncated", False)):
        return False
    if finding_ids is not None:
        return set(finding_ids) <= {fid for fid, _, _ in _VERIFY_RE.findall(reply)}
    if _NO_FINDINGS_RE.search(reply):
        return "=== FINDING" not in reply
    n_open = reply.count("=== FINDING id=")
    if n_open:
        return len(_parse_findings(reply)) == n_open == reply.count("=== END FINDING ===")
    # No blocks: a free-form review is usable when the salvage finds a mismatch, and is a
    # complete "nothing to report" when it walked >= 2 cases and every one of them passed.
    prose, n_segments = _prose_review_summary(reply)
    return bool(prose) or n_segments >= 2


def _ask_review(qf, meter, prompt, label, steps, finding_ids=None):
    """One bounded retry; incompleteness remains distinct from a negative verdict."""
    for attempt in range(2):
        retry_prompt = prompt if not attempt else prompt + (
            "\nYour previous answer was incomplete or did not match the required format. "
            "Respond concisely with complete finding/verdict blocks or NO FINDINGS; "
            "omit discussion of passing cases. Finish every block.\n")
        # G5 (2026-09-12): the FIRST attempt gets twice the default budget. At the 8192 default
        # the reasoning consumed the whole cap on ~16% of review calls (finish=length, empty
        # text), and every one of those needed a retry anyway -- one wasted call each. The
        # retry keeps a strictly larger budget than the first attempt.
        reply = _ask(qf, retry_prompt, label, meter,
                     max_tokens=((4 if attempt else 2) * _sub.SUBAGENT_MAX_COMPLETION_TOKENS
                                 or (32768 if attempt else 16384)))
        steps.append((label, f"structured review attempt {attempt + 1}", retry_prompt, reply))
        if _review_reply_complete(reply, finding_ids):
            return reply, True
    return reply, False


def _reason_trigger_failures(f: dict, diff: str, spec_corpus: str, checks) -> "list[str]":
    """Which SIMLOOP_GO_REASON_TRIGGER checks a confirmed reasoned finding fails (see there)."""
    from simagent.evidence import quote_text
    failed = []
    if "touched" in checks:
        added = sp._ws_squash("\n".join(l[1:] for l in (diff or "").splitlines()
                                        if l.startswith("+") and not l.startswith("+++")))
        if not any(len(quote_text(q)) >= 12 and quote_text(q) in added for q in f.get("quotes", [])):
            failed.append("touched")
    if "anchored" in checks and not _anchored(
            re.sub(r"`[A-Za-z_][\w.]*(?:\(\))?`", " ", f.get("expected") or ""), spec_corpus):
        failed.append("anchored")
    return failed


def _run_reasoning_review(qf, meter, *, ctx: str, ps: str, instance: dict, units: "list[dict]",
                          pack: str, callers: str, tests: str, diff: str = "", facts: str = "",
                          fact_rows: "list[dict] | None" = None, callees: str = "") -> "tuple[list[dict], list[tuple], list[tuple], dict]":
    """Two review lenses -> mechanical grounding -> adversarial verifier. Returns the confirmed
    findings as mismatch rows (source 'reasoned'), the review steps, the verify steps and a log."""
    fmt = dict(context=ctx, problem_statement=ps,
               requirements=instance.get("requirements") or "(none given)",
               interface=instance.get("interface") or "(none given)",
               units=_render_units(units), pack=pack or "(no source pack)",
               callers=callers, tests=tests, diff=_sub._clip(diff or "(no diff)", 9000),
               facts=facts or "(none)", callees=callees or "(none)")
    steps, cands, incomplete = [], [], []
    for label, prompt in (("trace", _REVIEW_TRACE_PROMPT), ("contract", _REVIEW_CONTRACT_PROMPT)):
        p = prompt.format(**fmt)
        r, complete = _ask_review(qf, meter, p, f"review:{label}", steps)
        if not complete:
            incomplete.append(label)
        parsed = _parse_findings(r)
        # The contract table is harvested from EVERY lens reply, not only the one labelled
        # "contract": under CCFLOW_MERGE_LENSES both lenses answer inside the trace reply, so
        # the label-gated harvest never ran (ccpipe batches 1-3: 139 NO-verdict rows across 70
        # of 120 instances were discarded, and for 39 instances they were the only candidates).
        have = {(f.get("uid"), f.get("input", "")[:40]) for f in parsed + cands}
        parsed += [f for f in _findings_from_table(r, units) if (f["uid"], f["input"][:40]) not in have]
        for f in parsed:
            f["lens"] = label
            f["id"] = f"F{len(cands) + 1}"   # unique across both lenses; verifiers echo ids verbatim
            if not f.get("uid"):   # prose/table findings name a symbol, not a unit id
                hay = f"{f.get('input', '')} {f.get('expected', '')} {f.get('actual', '')}"
                f["uid"] = next((u["id"] for u in units if (u.get("symbol") or "").split(".")[-1]
                                 and re.search(r"\b" + re.escape((u["symbol"]).split(".")[-1]) + r"\b", hay)), "")
            cands.append(f)
    # Mechanical forwarding facts that say "IS NOT passed" are candidate findings in their own
    # right (a26c325b: the lens called the constructor the sibling path in 9 of 10 rolls).
    by_sym0 = {}
    for u in units:
        by_sym0[(u.get("symbol") or "").split(".")[-1]] = u["id"]
    for x in (fact_rows or []):
        if x.get("forwarded"):
            continue
        uid = by_sym0.get(x["function"], "")
        if any(f.get("from_facts") and f.get("input", "").startswith(x["function"]) and x["new_param"] in f.get("input", "") for f in cands):
            continue
        cands.append({"id": f"F{len(cands) + 1}", "uid": uid, "kind": "CONTRACT", "lens": "contract", "from_facts": True,
                      "input": f"{x['function']}(...) with `{x['new_param']}` set",
                      "expected": f"`{x['new_param']}` is passed on the sibling call `{x['callee']}(...)` at line {x['line_no']}, like the {x['n_siblings']} sibling parameters",
                      "basis": f"FORWARDING FACT: {x['function']}() forwards {x['n_siblings']} of its parameters on `{x['callee']}(...)`; `{x['new_param']}` IS NOT among them",
                      "actual": f"`{x['new_param']}` is not passed on `{x['callee']}(...)`",
                      "quotes": [x["line"]] if x.get("line") else []})
    code_corpus = "\n".join(u.get("code", "") for u in units) + "\n" + (pack or "") + "\n" + callers + "\n" + tests + "\n" + (callees or "") + "\n" + (facts or "")
    basis_corpus = _spec_corpus(instance, ps) + "\n" + tests + "\n" + callers
    by_uid = {u["id"]: u for u in units}
    grounded, dropped = [], []
    for f in cands:
        ok, why = _ground_finding(f, code_corpus, basis_corpus)
        (grounded if ok else dropped).append((f, why))
    log = {"n_candidates": len(cands), "n_grounded": len(grounded),
           "incomplete": incomplete,
           "dropped": [{"id": f["id"], "why": w} for f, w in dropped]}
    vsteps, confirmed = [], []
    if grounded:
        rendered = "\n\n".join(
            f"=== FINDING id={f['id']} ===\nUNIT: {f['uid']}\nKIND: {f['kind']}\nINPUT: {f['input']}\n"
            f"EXPECTED: {f['expected']}\nBASIS: {f['basis']}\nTRACE:\n" + "\n".join("> " + q for q in f['quotes'])
            + f"\nACTUAL: {f['actual']}\n=== END FINDING ===" for f, _ in grounded)
        pv = _REVIEW_VERIFY_PROMPT.format(findings=rendered, **fmt)
        rv, complete = _ask_review(qf, meter, pv, "review:verify", vsteps,
                                   finding_ids=[f["id"] for f, _ in grounded])
        if not complete:
            incomplete.append("verify")
        verdicts = {}
        for fid, v, body in _VERIFY_RE.findall(rv):
            quotes = [l.strip()[1:].strip() for l in body.splitlines() if l.strip().startswith(">")]
            verdicts[fid] = (v, quotes)
        for f, _ in grounded:
            v, quotes = verdicts.get(f["id"], ("(none)", []))
            from simagent.evidence import quote_text
            vq_ok = any(len(quote_text(q)) >= 12 and quote_text(q) in sp._ws_squash(code_corpus) for q in quotes)
            if v == "CONFIRMED" and vq_ok:
                confirmed.append(f)
            else:
                key = "refuted" if v == "REFUTED" else "unverified"
                log.setdefault(key, []).append({"id": f["id"], "verdict": v, "verifier_quotes_grounded": vq_ok})
    log["n_confirmed"] = len(confirmed)
    log["complete"] = not incomplete
    rows = []
    go_checks = SIMLOOP_GO_REASON_TRIGGER if _sub.is_go() else ()
    spec_corpus = _spec_corpus(instance, ps) if (SIMLOOP_REASON_ANCHOR or go_checks) else ""
    for f in confirmed:
        u = by_uid.get(f["uid"], {})
        anchored = True
        go_fail = _reason_trigger_failures(f, diff, spec_corpus, go_checks) if go_checks else []
        if go_fail:
            log.setdefault("go_not_trigger", []).append({"id": f["id"], "failed": go_fail})
        if SIMLOOP_REASON_ANCHOR:
            # EXPECTED anchored in the spec, or the BASIS quote itself taken from the spec (not
            # from a pre-change test/caller): either grounds the finding in the change's authority.
            # Backticked bare identifiers (`run_commands`, `module`) are NOT literals: a spec that
            # merely names a function anchors nothing about how it must be called (luna_batch1_A4
            # 622a493a: a contract-lens finding built on an existing test's mock convention passed
            # anchoring on `run_commands` alone; the refine turned a 9/9 patch into 4/9).
            _ident_span = re.compile(r"`[A-Za-z_][\w.]*(?:\(\))?`")
            _b = " ".join((f.get("basis") or "").split()).strip("\"'` ")
            _segs = [x.strip() for x in re.split(r"\.\.\.|\u2026", _b) if len(x.strip()) >= 12]
            basis_in_spec = (len(_b) >= 12 and _b in spec_corpus) or (bool(_segs) and all(x in spec_corpus for x in _segs))
            anchored = _anchored(_ident_span.sub(" ", f["expected"]), spec_corpus) or basis_in_spec
            if not anchored:
                log.setdefault("unanchored", []).append({"id": f["id"], "basis": (f.get("basis") or "")[:200]})
        why_go = ""
        if go_fail:
            anchored = False
            why_go = (" -- NOT A TRIGGER: "
                      + ("its trace runs only through lines the patch did not change, so it describes "
                         "pre-existing behaviour" if "touched" in go_fail else "")
                      + ("; " if len(go_fail) == 2 else "")
                      + ("the specification never states this EXPECTED verbatim" if "anchored" in go_fail else "")
                      + ". Act on it only if the specification itself explicitly requires the change")
        rows.append({"id": f["id"], "uid": f["uid"], "file": u.get("file", "?"), "symbol": u.get("symbol", "?"),
                     "input": f["input"], "expected": f["expected"], "actual": f["actual"],
                     "source": "reasoned", "anchored": anchored, "kind": f["kind"], "basis": f["basis"],
                     "why": f"found by reading the code against the {'specification' if f['lens']=='trace' else 'existing tests, callers and interface'} "
                            f"and independently re-derived by a second reviewer; BASIS: {f['basis'][:200]}"
                            + (why_go if go_fail else "" if anchored else
                               " -- UNANCHORED: the specification never states this EXPECTED; its BASIS is a "
                               "pre-change test or caller, which the specified change may legitimately alter. "
                               "Act on it only if the specification itself requires it")})
    return rows, steps, vsteps, log


_REFINE_INSTANCE = (
    "<pr_description>\n{{task}}\n</pr_description>\n\n"
    "{{ repair_context }}\n\n"
    "=== THE PATCH YOU APPLIED (already on disk -- extend it, do not start over) ===\n"
    "{{ diff_so_far }}\n\n"
    "=== MISMATCHES TO FIX ===\n"
    "Each block below is a case derived from the specification and checked against your own\n"
    "patched code. EXPECTED is what the spec requires; ACTUAL is what your code produces.\n"
    "SOURCE tells you HOW the check was done:\n"
    "  EXECUTED  -- the case's harness was run in this container; ACTUAL is the real printed\n"
    "               value. You cannot dispute ACTUAL; you may only dispute EXPECTED (the spec\n"
    "               reading).\n"
    "  TEST      -- one of the repository's own tests PASSES at the base commit and FAILS on\n"
    "               your patched tree; ACTUAL is its failing assertion. Either your patch\n"
    "               regressed it, or the test encodes the PRE-FIX behaviour the specification\n"
    "               replaces (then say so and leave it). The specification decides.\n"
    "  SIMULATED -- a hand-trace without execution; ACTUAL may be wrong. Verify it by running\n"
    "               the code before acting.\n"
    "  REASONED  -- found by READING your patched code against the specification, the\n"
    "               repository's existing tests and the existing callers, and independently\n"
    "               re-derived by a second reviewer. ACTUAL is a traced result; BASIS names the\n"
    "               sentence or line that requires EXPECTED. Confirm the trace by reading (or\n"
    "               running) the code, then fix the cause.\n\n"
    "{{ mismatches }}\n\n"
    "=== COVERING-TEST EVIDENCE (executed on your patched tree) ===\n"
    "These are the repository's PRE-FIX tests. A failure can mean two things: your code is\n"
    "wrong, or the test's stored expectation is stale because the issue changes that behaviour.\n"
    "Decide per failure by comparing the assertion's expected value, your ACTUAL, and what the\n"
    "ISSUE literally states -- if your ACTUAL differs from the issue's stated value (a missing or\n"
    "extra character, bracket, case, wording), your code is wrong; if the test expects the OLD\n"
    "behaviour the issue replaces, leave the code alone. Never edit tests.\n"
    "{{ test_evidence }}\n\n"
    "{{ phase_body }}\n"
)

_REFINE_BODY = """\
# YOUR JOB
Make the patched source produce EXPECTED for every mismatch above, then submit.

FIRST, FOR EACH MISMATCH, RECONCILE IT AGAINST WHAT YOU ALREADY ESTABLISHED (mandatory).
Your own work producing the patch is reproduced above. Before changing anything, check whether
it already settles this mismatch -- did you read that code, run those tests, or consciously
decide the behaviour the case now disputes? Two cases in particular:
  - You ran a test covering this behaviour and it PASSED: a SIMULATED case claiming otherwise
    is wrong -- executed evidence outranks a hand-simulation; reject it and say so. An EXECUTED
    or TEST case, however, IS executed evidence: its ACTUAL stands, so reconcile the two runs
    (different input? different code path?) instead of rejecting it.
  - You saw a test FAIL here and reasoned that the failure was correct (an old test asserting
    pre-fix behaviour): re-examine that judgement honestly now, and state whether you still
    stand by it. Do not silently reverse it, and do not silently repeat it.
State the reconciliation in one line per mismatch BEFORE any edit.

Rules:
- Fix the cause, not the symptom: change the logic that produces the wrong value.
- A mismatch may be wrong about the spec. If a case's EXPECTED contradicts an explicit
  requirement sentence, do NOT contort the code to satisfy it -- say so plainly and leave that
  case alone. State which requirement wins and why.
- A mismatch may also rest on a FALSE PREMISE -- an INPUT no real caller produces, or an
  EXPECTED that is merely plausible rather than something the specification states. Check the
  case's own BASIS: if the requirement it cites does not actually demand the expected outcome,
  reject the case instead of coding to it. Inventing behaviour to satisfy an invented
  expectation is worse than leaving the patch alone.
- Change nothing the mismatches do not require. Do not undo the rest of your fix, and do not
  add behaviour the specification does not state.
- Do NOT modify tests, configuration, or packaging files.

Submit in TWO SEPARATE commands:
  1) git -C /testbed diff -- <the source files you changed> > /tmp/refined.txt
  2) echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /tmp/refined.txt
"""


class _UsageMeter:
    """Per-step token/cost accounting for the tool-free sim-loop queries.

    Instance charging belongs to the shared model's direct-usage callback. This meter only
    records phase attribution, so a fallback through model.query is not charged twice.
    """

    def __init__(self) -> None:
        self.steps: "list[dict]" = []
        self._label = "?"

    def begin(self, label: str) -> None:
        self._label = label

    def add(self, ev: dict) -> None:
        try:
            cost = float((ev or {}).get("cost") or 0.0)
        except (TypeError, ValueError):
            cost = 0.0
        self.steps.append({
            "step": self._label, "cost": cost,
            "prompt_tokens": int((ev or {}).get("prompt_tokens") or 0),
            "completion_tokens": int((ev or {}).get("completion_tokens") or 0),
            "total_tokens": int((ev or {}).get("total_tokens") or 0),
            "finish_reason": (ev or {}).get("finish_reason")})

    def totals(self, prefix: str = "") -> dict:
        rows = [s for s in self.steps if s["step"].startswith(prefix)]
        return {"n_calls": len(rows),
                "cost": round(sum(s["cost"] for s in rows), 6),
                "prompt_tokens": sum(s["prompt_tokens"] for s in rows),
                "completion_tokens": sum(s["completion_tokens"] for s in rows),
                "total_tokens": sum(s["total_tokens"] for s in rows),
                "steps": rows}


def _qf(model, meter: "_UsageMeter | None" = None):
    """Tool-free `prompt -> text` callable over the shared per-instance model."""
    return _sub._make_agent_query_fn(sp.SimpleNamespace(model=model),
                                     on_usage=(meter.add if meter is not None else None))


def _ask(qf, prompt: str, label: str, meter: "_UsageMeter | None" = None, *, max_tokens=None) -> str:
    if sp.BUDGET.exhausted:
        print(f"[simloop:{label}] SKIPPED -- instance budget exhausted "
              f"(${sp.BUDGET.spent:.4f}/${sp.BUDGET.cap:.4f})", flush=True)
        return ""
    if meter is not None:
        meter.begin(label)
    try:
        with _query_wall_deadline(SIMLOOP_QUERY_WALL_TIMEOUT):
            kwargs = {"max_tokens": max_tokens} if max_tokens and getattr(qf, "supports_output_limit", False) else {}
            reply = qf(prompt, **kwargs)
            return reply if reply is not None else ""
    except _QueryWallTimeout as e:
        print(f"[simloop:{label}] query TIMED OUT ({e})", flush=True)
        return ""
    except Exception as e:
        print(f"[simloop:{label}] query FAILED ({type(e).__name__}: {e})", flush=True)
        return ""


_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _changed_lines(patch_text: str) -> "dict[str, set]":
    """repo-relative path -> POST-patch line numbers the diff added or deleted at.

    Deleted lines have no post-image line of their own, so they are attributed to the position
    they were removed from -- that still lands inside the enclosing definition, which is what
    the caller needs.
    """
    out: "dict[str, set]" = {}
    cur, ln = None, 0
    for raw in (patch_text or "").splitlines():
        m = re.match(r"^diff --git a/(\S+) b/(\S+)", raw)
        if m:
            cur = m.group(2)
            continue
        if cur is None:
            continue
        h = _HUNK_RE.match(raw)
        if h:
            ln = int(h.group(1))
            continue
        if raw.startswith("+++") or raw.startswith("---"):
            continue
        if raw.startswith("+"):
            out.setdefault(cur, set()).add(ln)
            ln += 1
        elif raw.startswith("-"):
            out.setdefault(cur, set()).add(ln)
        elif raw.startswith(" "):
            ln += 1
    return out


# Runs INSIDE the container: map changed line numbers to their enclosing def/class via ast and
# print each unit's complete post-patch source. Deterministic -- no model involved.
_EXTRACT_SCRIPT = r'''
import ast, json, sys
spec = json.load(open(sys.argv[1]))
out = []
for path, lines in spec.items():
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
        tree = ast.parse(src)
    except Exception:
        continue
    srclines = src.splitlines()
    units = []
    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                s = min([ch.lineno] + [d.lineno for d in getattr(ch, "decorator_list", [])])
                e = getattr(ch, "end_lineno", ch.lineno)
                name = prefix + ch.name
                isfunc = not isinstance(ch, ast.ClassDef)
                # trivial declaration: a class whose body is only a docstring/pass/ellipsis and
                # defines no methods (e.g. `class FooError(Exception): """doc"""`). Nothing to
                # simulate; measured wasting 4 of 8 cases on qutebrowser-6dd402c0.
                trivial = (not isfunc
                           and not any(isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))
                                       for b in ch.body)
                           and all(isinstance(b, (ast.Pass, ast.Expr)) for b in ch.body))
                units.append((s, e, name, isfunc, trivial))
                walk(ch, name + ".")
    walk(tree, "")
    seen = set()
    for ln in sorted(set(lines)):
        cands = [u for u in units if u[0] <= ln <= u[1]]
        if not cands:
            continue
        s, e, name, isfunc, trivial = max(cands, key=lambda x: x[0])
        if (path, name) in seen:
            continue
        seen.add((path, name))
        out.append({"file": path, "symbol": name, "lineno": s, "end_lineno": e,
                    "trivial": trivial, "n_lines": e - s + 1,
                    "code": "\n".join(srclines[s - 1:e])[:8000]})
print("EXTRACT_JSON:" + json.dumps(out))
'''


_GO_DECL_RE = re.compile(r"^(func|type|var|const)\b")
_GO_FUNC_NAME_RE = re.compile(
    r"^func\s*(?:\(\s*(?:\w+\s+)?\*?\s*([A-Za-z_]\w*)(?:\[[^\]]*\])?\s*\)\s*)?([A-Za-z_]\w*)")


def _go_decl_end(lines: "list[str]", s: int, limit: int) -> int:
    """Last line (0-based) of the top-level Go declaration starting at line ``s``."""
    head = lines[s].rstrip()
    if head.startswith("func"):
        if "{" in head and head.endswith("}") and head.count("{") == head.count("}"):
            return s                                   # one-line func
        closer = ("}",)
    elif head.endswith("("):
        closer = (")",)                                # grouped type/var/const block
    elif head.count("{") > head.count("}"):
        closer = ("}",)                                # struct / interface / composite literal
    else:
        return s                                       # single-line declaration
    for j in range(s + 1, limit):
        if lines[j].startswith(closer):
            return j
    return max(s, limit - 1)


def _go_units_from_source(path: str, src: str, changed: "set[int]") -> "list[dict]":
    """Top-level Go declarations (func/method, type, var, const) enclosing the changed POST-patch
    lines (1-based), with their complete source. gofmt puts every top-level declaration and its
    closing `}` / `)` at column 0, so the block boundaries are exact."""
    lines = (src or "").splitlines()
    starts = [i for i, l in enumerate(lines) if _GO_DECL_RE.match(l)]
    out = []
    for k, s in enumerate(starts):
        limit = starts[k + 1] if k + 1 < len(starts) else len(lines)
        e = _go_decl_end(lines, s, limit)
        hits = sum(1 for ln in changed if s + 1 <= ln <= e + 1)
        if not hits:
            continue
        head = lines[s]
        if head.startswith("func"):
            m = _GO_FUNC_NAME_RE.match(head)
            sym = (f"{m.group(1)}.{m.group(2)}" if m and m.group(1)
                   else (m.group(2) if m else f"func@{s + 1}"))
        else:
            m = re.match(r"^(?:type|var|const)\s+([A-Za-z_]\w*)", head)
            nm = m.group(1) if m else ""
            if not nm and head.rstrip().endswith("(") and s + 1 < len(lines):
                m2 = re.match(r"^\s+([A-Za-z_]\w*)", lines[s + 1])
                nm = m2.group(1) if m2 else ""
            sym = nm or f"{head.split()[0]}@{s + 1}"
        out.append({"file": path, "symbol": sym, "code": "\n".join(lines[s:e + 1])[:12000],
                    "hits": hits, "line": s + 1})
    return out


def _extract_units_go(env, repo_path: str, patch_text: str, cap: int) -> "list[dict]":
    """Deterministic 4.1 for Go (defect G2, Go batch1 2026-09-12). The model-extraction path
    returned 0 units in 26 of 85 runs: its prompt must copy each unit from the audit evidence
    pack, which rarely carries the edited bodies, and it correctly forbids reconstructing them
    (flipt-0b119520 reproduced: 0 chars without the source, 2 units with it)."""
    changed = {f: v for f, v in _changed_lines(patch_text).items()
               if f.endswith(".go") and not f.endswith("_test.go") and not f.startswith("vendor/")}
    rows = []
    for f, v in changed.items():
        try:
            src = env.execute({"command": f"cd {repo_path} && cat {shlex.quote(f)}"},
                              timeout=30).get("output", "") or ""
        except Exception:
            continue
        rows += _go_units_from_source(f, src, v)
    # Biggest edits first, as for Python: the unit with the most changed lines holds the fix.
    rows.sort(key=lambda r: (-r["hits"], r["file"], r["line"]))
    return [{"id": f"U{i}", "file": r["file"], "symbol": r["symbol"], "code": r["code"]}
            for i, r in enumerate(rows[:cap], 1)]


def _extract_units(env, repo_path: str, patch_text: str, cap: int) -> "list[dict]":
    """Deterministic 4.1: diff line numbers -> enclosing def/class -> complete post-patch source.

    Replaces an LLM call that was unreliable for the worst-case input: on ansible-6cc97447 a
    7-file / 14-function patch produced ZERO units because the (checklist-shaped, 14 KB-capped)
    audit evidence pack never contained any unit's complete body, and the prompt correctly
    forbade reconstructing from memory. The whole loop then silently skipped the instance.
    """
    import base64
    changed = {f: v for f, v in _changed_lines(patch_text).items()
               if f.endswith(".py") and "test" not in f.lower()}
    if not changed:
        return []
    spec = {f"{repo_path}/{f}": sorted(v) for f, v in changed.items()}
    b64s = base64.b64encode(_EXTRACT_SCRIPT.encode()).decode()
    b64c = base64.b64encode(json.dumps(spec).encode()).decode()
    cmd = (f"printf %s {b64c} | base64 -d > /tmp/_units.json && "
           f"printf %s {b64s} | base64 -d > /tmp/_units.py && "
           "(python3 /tmp/_units.py /tmp/_units.json 2>/dev/null || "
           "python /tmp/_units.py /tmp/_units.json)")
    try:
        out = env.execute({"command": cmd}, timeout=120).get("output", "") or ""
    except Exception as e:
        print(f"[simloop:4.1] container extract FAILED ({type(e).__name__}: {e})", flush=True)
        return []
    m = re.search(r"EXTRACT_JSON:(\[.*\])", out)
    if not m:
        print("[simloop:4.1] no EXTRACT_JSON in output -- extraction produced nothing",
              flush=True)
        return []
    try:
        rows = json.loads(m.group(1))
    except Exception:
        return []
    n_all = len(rows)
    kept = [r for r in rows if not r.get("trivial")]
    n_triv = n_all - len(kept)
    # Biggest edits first: the unit with the most changed lines is the one the fix lives in.
    weight = {f"{repo_path}/{f}": len(v) for f, v in changed.items()}
    kept.sort(key=lambda r: (-weight.get(r["file"], 0), r["lineno"]))
    units = []
    for i, r in enumerate(kept[:cap], 1):
        units.append({"id": f"U{i}",
                      "file": r["file"].replace(f"{repo_path}/", "", 1),
                      "symbol": r["symbol"], "code": r["code"]})
    if n_triv:
        print(f"[simloop:4.1] skipped {n_triv} trivial declaration unit(s)", flush=True)
    if len(kept) > cap:
        print(f"[simloop:4.1] {len(kept)} units found, capped to {cap}", flush=True)
    return units


def _parse_cases(text: str) -> "list[dict]":
    """Parse CASE blocks, tolerating a MISSING `=== END CASE ===` terminator.

    Measured on qutebrowser-6dd402c0: a reply carried 4 CASE headers but only 1 END marker, so
    a terminator-anchored regex kept ONE case and silently dropped three -- the simulator then
    returned verdicts for ids it had never been shown. Anchor on the HEADERS instead and end
    each body at the next header, the END marker, or end-of-text.
    """
    hdrs = list(_CASE_HDR_RE.finditer(text or ""))
    out = []
    for i, h in enumerate(hdrs):
        stop = hdrs[i + 1].start() if i + 1 < len(hdrs) else len(text)
        body = text[h.end():stop]
        e = _CASE_END_RE.search(body)
        if e:
            body = body[:e.start()]
        body = body.strip()
        if body:
            out.append({"id": _normalize_case_id(h.group(1)), "body": body})
    return out


def _flat_messages(steps: "list[tuple]") -> list:
    """[(label, desc, prompt, reply)] -> ONE flat mini-swe-agent message list.

    4.2 and 4.3 issue one query PER UNIT, but every other trajectory file in the pipeline
    carries its turns in a top-level ``messages`` array; nesting them under ``units[]`` makes
    the file unreadable to anything that speaks the mini-swe-agent trajectory format. Flatten
    them and keep the per-unit boundary as a phase-marker banner, exactly the way
    _combined_trajectory joins phases.
    """
    msgs: list = []
    for label, desc, prompt, reply in steps:
        msgs.append(sp._banner(label, desc))
        msgs.append({"role": "user", "content": prompt})
        msgs.append({"role": "assistant", "content": reply})
    return msgs


def _uid_of(case_id: str) -> str:
    """Owning unit of a case id: case ids are minted as ``<uid>#<n>`` (e.g. ``U5#1`` -> ``U5``).

    With batched 4.2/4.3 calls the unit is no longer implied by the loop variable, so the id
    IS the attribution channel; a case whose prefix names no unit in the batch is dropped loudly
    rather than silently mis-attributed.
    """
    return (case_id or "").split("#", 1)[0].strip()


_DEF_ANY_RE = re.compile(r"^([-+])\s*(?:async\s+)?def\s+(\w+)\s*\((.*)$")


def _param_slots(sig: str) -> "list[tuple[str, bool]]":
    """[(name, has_default)] from a parameter list, ignoring nesting inside defaults."""
    out, depth, cur, done = [], 0, "", False

    def flush(chunk):
        m = re.match(r"\s*\*{0,2}(\w+)", chunk)
        if m and m.group(1) not in ("self", "cls"):
            out.append((m.group(1), "=" in chunk))

    for ch in sig:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:          # end of the parameter list: flush the LAST param first
                done = True
                break
            depth -= 1
        if ch == "," and depth == 0:
            flush(cur)
            cur = ""
        else:
            cur += ch
    flush(cur)                      # trailing param, whether we hit ")" or ran out of line
    return out


def _signature_deltas(patch_text: str) -> "dict[str, dict]":
    """bare-func -> {'old': [names], 'gained_optional': [names]} from the diff's def lines.

    Motivation (ansible-6cc97447): `__new__(cls, value)` became
    `__new__(cls, object='', encoding=None, errors=None)`. The sim-loop generated four cases and
    supplied `encoding=` in EVERY bytes case, so it never tested the DEFAULT path -- which is the
    only one the graded test pins (`str(b'Hello')` is the repr, not a decode). The parameters a
    signature GAINS are derivable from the diff, so the case that omits them can be REQUIRED
    rather than hoped for.
    """
    old, new = {}, {}
    for ln in (patch_text or "").splitlines():
        m = _DEF_ANY_RE.match(ln)
        if not m:
            continue
        side, name, rest = m.groups()
        (old if side == "-" else new).setdefault(name, set()).update(
            n for n, _d in _param_slots(rest))
        if side == "+":
            new.setdefault(name + "\x00opt", set()).update(
                n for n, d in _param_slots(rest) if d)
    out = {}
    for name, params in new.items():
        if name.endswith("\x00opt") or name not in old:
            continue
        gained = sorted((new.get(name + "\x00opt") or set()) - old[name])
        if gained:
            out[name] = {"old": sorted(old[name]), "gained_optional": gained}
    return out


def _required_case_block(batch: "list[dict]", patch_text: str) -> str:
    """Per-unit MANDATORY case slots derived from the signature delta (see _signature_deltas)."""
    deltas = _signature_deltas(patch_text)
    lines = []
    for u in batch:
        bare = (u.get("symbol") or "").split(".")[-1]
        d = deltas.get(bare)
        if not d:
            continue
        lines.append(
            f"- {u['id']} ({u['symbol']}): this unit's signature GAINED the optional "
            f"parameter(s) {', '.join('`%s`' % x for x in d['gained_optional'])}; before this "
            f"patch it took ({', '.join(d['old']) or 'no arguments'}). Emit a case with id "
            f"`{u['id']}#G` that calls it WITHOUT any of the gained parameters -- the DEFAULT "
            f"path -- and derive EXPECTED from the behaviour that existed BEFORE the patch, not "
            f"from the new code. New optional parameters must not change the default path.")
    if not lines:
        return "(none)"
    return ("\n".join(lines))


def _render_units(batch: "list[dict]") -> str:
    """The UNIT UNDER TEST blocks for one batch, in the order the model must answer them."""
    return "\n\n".join(
        f"=== UNIT UNDER TEST: {u['id']}  ({u['file']} :: {u['symbol']}) ===\n{u['code']}"
        for u in batch)


def _parse_harness(body: str) -> str:
    m = _HARNESS_RE.search(body or "")
    if not m:
        return ""
    code = m.group(1).strip()
    return "" if code.lower() in ("none", "# none", "pass") else code


_EXEC_RUNNER = r'''
import json, sys, subprocess, os, ast
cases = json.load(open(sys.argv[1]))
repo = sys.argv[2]; tmo = int(sys.argv[3])
env = dict(os.environ)
env["PYTHONPATH"] = repo + os.pathsep + os.path.join(repo, "lib") + os.pathsep + os.path.join(repo, "src") + os.pathsep + env.get("PYTHONPATH", "")
env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
for c in cases:
    path = "/tmp/_simcase_%s.py" % c["id"].replace("#", "_").replace("/", "_")
    open(path, "w").write(c["harness"])
    res = {"id": c["id"]}
    try:
        r = subprocess.run([sys.executable, path], cwd=repo, env=env, capture_output=True, text=True, timeout=tmo)
        out, err = r.stdout, r.stderr
        line = [l for l in out.splitlines() if l.startswith("ACTUAL=")]
        res["actual"] = line[-1][len("ACTUAL="):] if line else None
        res["rc"] = r.returncode
        res["stderr_tail"] = err[-800:]
    except subprocess.TimeoutExpired:
        res["actual"] = None; res["rc"] = -9; res["stderr_tail"] = "TIMEOUT after %ss" % tmo
    except Exception as e:
        res["actual"] = None; res["rc"] = -1; res["stderr_tail"] = "%s: %s" % (type(e).__name__, e)
    print("SIMEXEC " + json.dumps(res))
'''


def _norm_value(text: str):
    """Best-effort Python value from a repr-ish string; falls back to the squashed string."""
    t = (text or "").strip()
    try:
        import ast as _ast
        return ("val", _ast.literal_eval(t))
    except Exception:
        return ("str", " ".join(t.split()).strip("'\""))


def _judge_exec(expected_repr: str, res: dict) -> "tuple[str, str]":
    """(verdict, actual_text) from one executed harness result."""
    exp = _clean_repr(expected_repr)
    err = res.get("stderr_tail") or ""
    actual = res.get("actual")
    if actual is not None:
        actual = _clean_repr(actual)
    if exp.upper().startswith("RAISES"):
        exc = exp.split(":", 1)[-1].strip().split()[0] if ":" in exp else ""
        if res.get("rc") not in (0, None) and (not exc or exc in err):
            return "MATCH", f"raised ({exc or 'an exception'}): {err.strip().splitlines()[-1][:160] if err.strip() else ''}"
        if actual is not None:
            # Harnesses frequently catch the exception themselves and print it as the ACTUAL
            # ("ACTUAL=RAISES: ValueError" / "ACTUAL='raised ValueError: ...'"): an exception
            # reported either way is the expected outcome, not a returned value.
            a_up = actual.upper()
            if a_up.startswith(("RAISES", "'RAISES", "\"RAISES", "RAISED", "'RAISED", "\"RAISED")) \
                    or (exc and exc in actual and re.search(r"rais|except|error", actual, re.I)):
                return "MATCH", actual
            return "MISMATCH", f"returned {actual} (no exception raised)"
        return "UNVERIFIABLE", f"harness failed: {err.strip().splitlines()[-1][:200] if err.strip() else 'no output'}"
    if actual is None:
        return "UNVERIFIABLE", f"harness produced no ACTUAL line: {err.strip().splitlines()[-1][:200] if err.strip() else 'rc=%s' % res.get('rc')}"
    if not exp:
        return "UNVERIFIABLE", actual
    a, e = _norm_value(actual), _norm_value(exp)
    if a == e or (a[0] == "val" and e[0] == "val" and a[1] == e[1]):
        return "MATCH", actual
    try:
        if _ws_norm(a[1]) == _ws_norm(e[1]):
            return "MATCH", actual
    except Exception:
        pass
    if " ".join(str(a[1]).split()) == " ".join(str(e[1]).split()):
        return "MATCH", actual
    return "MISMATCH", actual


def _execute_cases(env, repo_path: str, cases: "list[dict]", timeout: int = SIMLOOP_EXEC_TIMEOUT
                   ) -> "dict[str, dict]":
    """Run every case that carries a HARNESS in the container; {case id: {verdict, actual}}.
    Never raises; a case whose harness cannot run is UNVERIFIABLE, never MISMATCH."""
    import base64
    runnable = [{"id": c["id"], "harness": c["harness"]} for c in cases if c.get("harness")]
    if not runnable:
        return {}
    b64s = base64.b64encode(_EXEC_RUNNER.encode()).decode()
    b64c = base64.b64encode(json.dumps(runnable).encode()).decode()
    cmd = (f"printf %s {b64c} | base64 -d > /tmp/_simcases.json && "
           f"printf %s {b64s} | base64 -d > /tmp/_simrun.py && "
           f"cd {repo_path} && (python3 /tmp/_simrun.py /tmp/_simcases.json {repo_path} {timeout} "
           f"2>/dev/null || python /tmp/_simrun.py /tmp/_simcases.json {repo_path} {timeout})")
    try:
        out = env.execute({"command": cmd},
                          timeout=min(600, 30 + timeout * len(runnable))).get("output", "") or ""
    except Exception as e:
        print(f"[simloop:4.3] container execution FAILED ({type(e).__name__}: {e})", flush=True)
        return {}
    by_id = {c["id"]: c for c in cases}
    results = {}
    for m in re.finditer(r"^SIMEXEC (\{.*\})$", out, re.M):
        try:
            res = json.loads(m.group(1))
        except Exception:
            continue
        c = by_id.get(res.get("id"))
        if not c:
            continue
        verdict, actual = _judge_exec(c.get("expected_repr") or c.get("expected") or "", res)
        results[c["id"]] = {"verdict": verdict, "actual": actual, "rc": res.get("rc")}
    return results


_QUOTED_RE = re.compile(r"[\"\u201c\u201d]([^\"\u201c\u201d\n]{3,60})[\"\u201c\u201d]|'([^'\n]{3,60})'|`([^`\n]{3,60})`")


def _ws_norm(v):
    """Whitespace-insensitive view of a value: runs of blanks inside strings collapse to one
    space, recursively through containers. Specification prose is not a reliable witness for
    exact spacing (5e88cd99: the spec writes the CLI line with single spaces, the repo's
    convention -- and the hidden tests -- use two; four executed "mismatches" differed only there)."""
    if isinstance(v, str):
        return " ".join(v.split())
    if isinstance(v, (list, tuple)):
        return type(v)(_ws_norm(x) for x in v)
    if isinstance(v, dict):
        return {(_ws_norm(k) if isinstance(k, str) else k): _ws_norm(x) for k, x in v.items()}
    return v


def _spec_corpus(instance: dict, problem_statement: str) -> str:
    return " ".join(((problem_statement or "") + " " + str(instance.get("requirements") or "")
                     + " " + str(instance.get("interface") or "")).split())


def _anchored(expected: str, corpus: str) -> bool:
    """True when EXPECTED's distinguishing content is stated verbatim in the specification."""
    exp = (expected or "").strip()
    if not exp or not corpus:
        return False
    if exp.upper().startswith("RAISES"):
        exc = exp.split(":", 1)[-1].strip().split()[0] if ":" in exp else ""
        return bool(exc) and exc in corpus
    lits = [next(g for g in m.groups() if g) for m in _QUOTED_RE.finditer(exp)]
    if lits:
        return any(" ".join(l.split()) in corpus for l in lits if len(l.strip()) >= 3)
    nums = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w|\.\d)", exp)
    if nums:
        return any(re.search(r"(?<![\w.])" + re.escape(n) + r"(?!\w|\.\d)", corpus) for n in nums)
    core = " ".join(exp.strip("`'\"").split())
    return len(core) >= 3 and core in corpus


def _render_cases(cases: "list[dict]") -> str:
    return "\n\n".join(f"=== CASE id={c['id']} ===\n{c['body']}\n=== END CASE ===" for c in cases)


def _render_mismatches(rows: "list[dict]") -> str:
    out = []
    for r in rows:
        out.append(
            f"[{r['id']}]  unit {r['uid']} ({r['file']} :: {r['symbol']})\n"
            f"SOURCE:   {(r.get('source') or 'simulated').upper()}\n"
            f"INPUT:    {r.get('input', '(see case)')}\n"
            f"EXPECTED: {r.get('expected', '(see case)')}\n"
            f"ACTUAL:   {r.get('actual', '(not stated)')}\n"
            f"WHY:      {r.get('why', '(not stated)')}")
    return "\n\n".join(out) or "(none)"


def _field(body: str, name: str) -> str:
    m = re.search(rf"^{name}:\s*(.+?)(?=\n[A-Z][A-Z_]{{1,}}:|\Z)", body, re.S | re.M)
    return " ".join(m.group(1).split()) if m else ""


def _clean_repr(text: str) -> str:
    """Strip the decoration models put around a repr: backticks, code fences, a leading
    `EXPECTED_REPR:` echo, surrounding whitespace."""
    t = (text or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t).strip()
    t = t.strip("`").strip()
    t = re.sub(r"^(?:EXPECTED_REPR|EXPECTED)\s*:\s*", "", t).strip().strip("`").strip()
    return t


def _run_simloop(model, env, base_agent, repo_path: str, inst_out: Path, repair: dict,
                 *, system: str) -> dict:
    """Steps 4.1-4.4. Returns a meta dict; the tree is left holding whichever patch won."""
    instance = dict(_STATE["instance"] or {})
    for fld in ("requirements", "interface"):
        instance[fld] = sp._unquote_spec_field(instance.get(fld) or "")
    findings = _STATE["findings"]
    ps = _sub._clip(_STATE["problem_statement"], _sub.PS_CLIP)
    meter = _UsageMeter()
    qf = _qf(model, meter)
    meta: dict = {"rounds": [], "enabled": True}

    initial = sp._extract_patch(env, repo_path)
    (inst_out / "4_initial_patch.diff").write_text(initial or "")
    if not (initial or "").strip():
        print("[simloop] SKIPPED -- repair produced no diff", flush=True)
        meta["skipped"] = "empty initial patch"
        return meta
    if qf is None:
        print("[simloop] SKIPPED -- no tool-free query function available", flush=True)
        meta["skipped"] = "no query fn"
        return meta

    ctx = _repair_context(repair)
    pack = ""
    try:
        pack = sp._audit_evidence_pack(env, repo_path, initial, [], findings, cap=14000)
    except Exception as e:
        print(f"[simloop] evidence pack FAILED ({type(e).__name__}: {e})", flush=True)

    # --- 4.1 EXTRACT --------------------------------------------------------------------------
    # Python: deterministic (ast over the diff's post-patch line numbers). Go: no ast, so fall
    # back to the model.
    p1 = r1 = ""
    units = []
    if _sub.is_go():   # G2: deterministic declaration extraction (model path kept as fallback)
        units = _extract_units_go(env, repo_path, initial, SIMLOOP_MAX_UNITS)
        mode = "deterministic (go declarations)"
    if not _sub.is_python() and not units:   # JS (and a Go fallback): no ast -> model extraction
        p1 = _EXTRACT_PROMPT.format(
            context=ctx, manifest=sp._diff_manifest(initial),
            diff=_sub._clip(initial, 12000), pack=pack or "(no source pack)",
            cap=SIMLOOP_MAX_UNITS)
        r1 = _ask(qf, p1, "extract", meter)
        units = [{"id": u[0], "file": u[1], "symbol": u[2], "code": u[3]}
                 for u in _UNIT_RE.findall(r1)][:SIMLOOP_MAX_UNITS]
        mode = f"model ({_sub.LANG}: no ast)"
    elif _sub.is_python():
        units = _extract_units(env, repo_path, initial, SIMLOOP_MAX_UNITS)
        mode = "deterministic (ast)"
    (inst_out / "4.1_simloop_extract.traj.json").write_text(json.dumps(
        {"phase": "simloop_extract", "mode": mode, "usage": meter.totals("extract"),
         "units": [{k: v for k, v in u.items() if k != "code"} for u in units],
         "unit_source": {u["id"]: u["code"] for u in units},
         "messages": ([{"role": "user", "content": p1},
                       {"role": "assistant", "content": r1}] if r1 else [])},
        indent=1, default=str))
    print(f"[simloop:4.1] {len(units)} edited unit(s): "
          f"{[u['symbol'] for u in units] or '(none)'}", flush=True)
    if not units:
        meta["skipped"] = "no units extracted"
        return meta

    for rnd in range(1, max(1, SIMLOOP_ROUNDS) + 1):
        diff_now = sp._extract_patch(env, repo_path)
        all_cases: "list[dict]" = []
        exec_match_ids: "set[str]" = set()   # executed cases that MATCHED (regression guard)
        mismatches: "list[dict]" = []
        case_logs, sim_logs = [], []
        case_steps, sim_steps = [], []
        analysis_failures: "list[dict]" = []
        completed_units = 0
        sfx = "" if rnd == 1 else f"_r{rnd}"

        def checkpoint(stage: str, *, complete: bool = False) -> None:
            return   # the review writes its own 4.2_simloop_review file; no cases/simulate files
            progress = {"stage": stage, "complete": complete,
                        "units_completed": completed_units, "units_total": len(units),
                        "analysis_failures": analysis_failures}
            _write_json_checkpoint(
                inst_out / f"4.2_simloop_cases{sfx}.traj.json",
                {"phase": "simloop_cases", "mode": "tool-free", "round": rnd,
                 "n_calls": len(case_steps), "usage": meter.totals("cases:"),
                 "progress": progress, "units": case_logs,
                 "messages": _flat_messages(case_steps)})
            _write_json_checkpoint(
                inst_out / f"4.3_simloop_simulate{sfx}.traj.json",
                {"phase": "simloop_simulate", "mode": "tool-free", "round": rnd,
                 "n_calls": len(sim_steps), "usage": meter.totals("simulate:"),
                 "progress": progress, "units": sim_logs, "mismatches": mismatches,
                 "messages": _flat_messages(sim_steps)})

        batches = [units[i:i + SIMLOOP_BATCH] for i in range(0, len(units), SIMLOOP_BATCH)]
        batches = []   # no case generation / execution: the bug hunt is reasoning only
        resources = _resource_context(env, repo_path, _spec_corpus(instance, ps), units)
        if resources:
            pack += "\n=== APPLICATION RESOURCE DECLARATIONS ===\n" + resources
        callers_txt, tests_txt = _review_context(env, repo_path, units, diff_now)
        try:
            facts_txt, fact_rows = _forwarding_facts(env, repo_path, diff_now)
        except Exception as e:
            facts_txt, fact_rows = f"(forwarding facts unavailable: {type(e).__name__})", []
        try:
            callees_txt = _callee_definitions(env, repo_path, units)
        except Exception as e:
            callees_txt = f"(callee definitions unavailable: {type(e).__name__})"
        print(f"[simloop:4.2] review context: callers {len(callers_txt)} chars, tests {len(tests_txt)} chars, "
              f"callee defs {len(callees_txt)} chars, forwarding facts {len(fact_rows)} "
              f"({sum(1 for x in fact_rows if not x['forwarded'])} not forwarded)", flush=True)
        r_rows, r_steps, v_steps, r_log = _run_reasoning_review(
            qf, meter, ctx=ctx, ps=ps, instance=instance, units=units, pack=pack,
            callers=callers_txt, tests=tests_txt, diff=diff_now, facts=facts_txt,
            fact_rows=fact_rows, callees=callees_txt)
        mismatches += r_rows
        case_steps += r_steps
        sim_steps += v_steps
        completed_units = len(units) if r_log["complete"] else 0
        if not r_log["complete"]:
            analysis_failures.append({"stage": "review", "incomplete": r_log["incomplete"]})
        for u in units:
            case_logs.append({"unit": u["id"], "file": u["file"], "symbol": u["symbol"],
                              "n_cases": 0, "case_ids": [], "mode": "reasoning review"})
        sim_logs.append({"review": r_log})
        _write_json_checkpoint(inst_out / f"4.2_simloop_review{sfx}.traj.json",
                               {"phase": "simloop_review", "mode": "pure code reasoning",
                                "round": rnd, "usage": meter.totals("review:"),
                                "log": r_log, "callers_chars": len(callers_txt),
                                "tests_chars": len(tests_txt), "callees_chars": len(callees_txt),
                                "forwarding_facts": fact_rows,
                                "messages": _flat_messages(r_steps + v_steps)})
        print(f"[simloop:4.2/4.3] round {rnd}: REASONING REVIEW -- {r_log['n_candidates']} "
              f"candidate finding(s), {r_log['n_grounded']} grounded, "
              f"{r_log['n_confirmed']} confirmed by the verifier", flush=True)
        if r_log.get("go_not_trigger"):
            print(f"[simloop:4.2/4.3] round {rnd}: {len(r_log['go_not_trigger'])} confirmed finding(s) "
                  f"cannot trigger refine on Go: {r_log['go_not_trigger'][:4]}", flush=True)
        for bi, batch in enumerate(batches, 1):
            bids = [u["id"] for u in batch]
            blabel = "+".join(bids)
            if sp.BUDGET.exhausted:
                for u in batch:
                    analysis_failures.append({"unit": u["id"], "stage": "before_cases",
                                              "why": "instance budget exhausted"})
                print(f"[simloop:4.2/4.3] round {rnd}: stopping before batch "
                      f"{bi}/{len(batches)} ({blabel}) -- instance budget exhausted", flush=True)
                checkpoint("budget_exhausted", complete=False)
                break
            print(f"[simloop:4.2/4.3] round {rnd}: batch {bi}/{len(batches)} "
                  f"({len(batch)} unit(s): {blabel})", flush=True)
            by_uid = {u["id"]: u for u in batch}

            # --- 4.2 CASES (one call for the whole batch) --------------------------------------
            required = _required_case_block(batch, initial)
            p2 = _CASES_PROMPT_BATCH.format(
                context=ctx, problem_statement=ps,
                requirements=instance.get("requirements") or "(none given)",
                interface=instance.get("interface") or "(none given)",
                units=_render_units(batch), required=required)
            if required != "(none)":
                print(f"[simloop:4.2] batch {bi}: {required.count(chr(10)) + 1} mandatory "
                      f"default-path case(s) derived from the signature delta", flush=True)
            r2 = _ask(qf, p2, f"cases:{blabel}", meter)
            parsed = _parse_cases(r2)
            n_hdr = len(_CASE_HDR_RE.findall(r2))
            if n_hdr != len(parsed):
                print(f"[simloop:4.2] WARN batch {bi}: {n_hdr} CASE header(s) but "
                      f"{len(parsed)} parsed -- check the reply formatting", flush=True)
            # Attribute each case to its unit by the id prefix. With one call per batch the loop
            # variable no longer identifies the unit, so the id IS the attribution channel; a
            # case naming no unit in this batch has no source to simulate against -- drop it.
            cases_by_uid = {u["id"]: [] for u in batch}
            orphan = set()

            def _attribute(parsed_cases):
                for c in parsed_cases:
                    uid = _uid_of(c["id"])
                    if uid not in by_uid or uid not in cases_by_uid:
                        orphan.add(c["id"])
                        continue
                    u = by_uid[uid]
                    c.update(uid=uid, file=u["file"], symbol=u["symbol"],
                             input=_field(c["body"], "INPUT"),
                             expected=_field(c["body"], "EXPECTED"),
                             expected_repr=_field(c["body"], "EXPECTED_REPR"),
                             harness=_parse_harness(c["body"]) if SIMLOOP_EXEC else "")
                    cases_by_uid[uid].append(c)

            _attribute(parsed)
            missing = [u for u in batch if not cases_by_uid[u["id"]]]
            if missing and SIMLOOP_CASES_REASK and len(missing) < len(batch):
                mlabel = "+".join(u["id"] for u in missing)
                print(f"[simloop:4.2] batch {bi}: {len(missing)} unit(s) got no case "
                      f"({mlabel}) -- asking once more for just those", flush=True)
                p2b = _CASES_PROMPT_BATCH.format(
                    context=ctx, problem_statement=ps,
                    requirements=instance.get("requirements") or "(none given)",
                    interface=instance.get("interface") or "(none given)",
                    units=_render_units(missing), required=_required_case_block(missing, initial))
                r2b = _ask(qf, p2b, f"cases:{mlabel}", meter)
                seen = {c["id"] for cs in cases_by_uid.values() for c in cs}
                _attribute([c for c in _parse_cases(r2b) if c["id"] not in seen])
                case_steps.append((f"simloop_cases:{mlabel}",
                                   f"re-ask: derive cases for {len(missing)} unit(s) the "
                                   f"batch call left uncovered: {mlabel}", p2b, r2b))
            if orphan:
                print(f"[simloop:4.2] WARN batch {bi}: ignoring case id(s) naming no unit in "
                      f"this batch: {sorted(orphan)}", flush=True)
            batch_cases = [c for uid in bids for c in cases_by_uid[uid]]
            all_cases += batch_cases
            for u in batch:
                cs = cases_by_uid[u["id"]]
                case_logs.append({"unit": u["id"], "file": u["file"], "symbol": u["symbol"],
                                  "n_cases": len(cs), "case_ids": [c["id"] for c in cs]})
                if not cs:
                    analysis_failures.append({"unit": u["id"], "stage": "cases",
                                              "why": "no cases attributed to this unit"})
            case_steps.append((f"simloop_cases:{blabel}",
                               f"derive spec-grounded INPUT/EXPECTED cases for "
                               f"{len(batch)} unit(s): {blabel}", p2, r2))
            if not batch_cases:
                completed_units += len(batch)
                checkpoint(f"cases_failed:{blabel}", complete=False)
                continue
            # Preserve the derived cases even if the following simulation call stalls or the
            # process is interrupted.
            checkpoint(f"cases_complete:{blabel}", complete=False)

            # --- 4.3a EXECUTE (facts) ------------------------------------------------------------
            by_id = {c["id"]: c for c in batch_cases}
            exec_res = _execute_cases(env, repo_path, batch_cases) if SIMLOOP_EXEC else {}
            n_exec_ok = sum(1 for r in exec_res.values() if r["verdict"] in ("MATCH", "MISMATCH"))
            exec_match_ids |= {cid for cid, r in exec_res.items() if r["verdict"] == "MATCH"}
            n_h = sum(1 for c in batch_cases if c.get("harness"))
            print(f"[simloop:4.3] batch {bi}: {n_h}/{len(batch_cases)} case(s) carried a "
                  f"harness; executed verdicts {n_exec_ok} "
                  f"(unverifiable {len(exec_res) - n_exec_ok}, no harness "
                  f"{len(batch_cases) - len(exec_res)})", flush=True)
            for cid, r in exec_res.items():
                c = by_id[cid]
                if r["verdict"] == "MISMATCH":
                    anchored = (not SIMLOOP_ANCHORED_TRIGGER) or _anchored(
                        c.get("expected_repr") or c.get("expected", ""),
                        _spec_corpus(instance, _STATE["problem_statement"]))
                    mismatches.append({
                        "id": cid, "uid": c["uid"], "file": c["file"], "symbol": c["symbol"],
                        "input": c.get("input", ""), "expected": c.get("expected", ""),
                        "actual": r["actual"], "source": "executed", "anchored": anchored,
                        "why": "the case's harness was EXECUTED in the container; ACTUAL is the "
                               "printed value, not a trace"
                               + ("" if anchored else
                                  " -- UNANCHORED: the specification never states this EXPECTED "
                                  "value verbatim; it is the case-writer's inference and must not "
                                  "be coded to unless an explicit requirement demands it")})
            exec_verdict = {cid: r["verdict"] for cid, r in exec_res.items()
                            if r["verdict"] in ("MATCH", "MISMATCH")}
            # --- 4.3b SIMULATE the remainder (one call for the whole batch) ----------------------
            sim_cases = [c for c in batch_cases if c["id"] not in exec_verdict]
            sim_batch = [u for u in batch if any(c["uid"] == u["id"] for c in sim_cases)]
            r3 = ""
            if sim_cases:
                p3 = _SIMULATE_PROMPT_BATCH.format(
                    context=ctx, units=_render_units(sim_batch),
                    pack=_sub._clip(pack, 9000) or "(no source pack)",
                    cases=_render_cases(sim_cases))
                r3 = _ask(qf, p3, f"simulate:{blabel}", meter)
            else:
                p3 = "(all cases decided by execution -- no hand-simulation needed)"
            verdict_rows = [(_normalize_case_id(cid), verdict, cid)
                            for cid, verdict in _VERDICT_RE.findall(r3)]
            unknown = {cid for cid, _v, _raw in verdict_rows} - set(by_id)
            if unknown:
                # A verdict for a case the simulator was never given has no stored EXPECTED to
                # justify it, so it cannot be turned into refine feedback -- drop it loudly.
                print(f"[simloop:4.3] WARN batch {bi}: ignoring verdict(s) for unknown case "
                      f"id(s) {sorted(unknown)}", flush=True)
            if sim_cases and (not r3.strip() or not verdict_rows):
                for u in sim_batch:
                    analysis_failures.append(
                        {"unit": u["id"], "stage": "simulate",
                         "why": "empty, failed, budget-skipped, or unparsable response"})
            for cid, verdict, raw_cid in verdict_rows:
                if verdict.upper() != "MISMATCH" or cid not in by_id or cid in exec_verdict:
                    continue
                blk = re.search(rf"\[{re.escape(raw_cid)}\]\s*VERDICT:.*?(?=\n\[|\Z)",
                                r3, re.S)
                body = blk.group(0) if blk else ""
                c = by_id[cid]
                mismatches.append({
                    "id": cid, "uid": c["uid"], "file": c["file"], "symbol": c["symbol"],
                    "input": c.get("input", ""), "expected": c.get("expected", ""),
                    "actual": _field(body, "ACTUAL"), "why": _field(body, "WHY"),
                    "source": "simulated"})
            seen_verdict = dict(exec_verdict)
            seen_verdict.update({cid: v for cid, v, _raw in verdict_rows if cid not in exec_verdict})
            for u in sim_batch:
                rows = [{"case": c["id"], "verdict": seen_verdict.get(c["id"], "(none)"),
                         "how": "executed" if c["id"] in exec_verdict else "simulated"}
                        for c in cases_by_uid[u["id"]]]
                sim_logs.append({"unit": u["id"], "file": u["file"], "symbol": u["symbol"],
                                 "verdicts": rows})
            sim_steps.append((f"simloop_simulate:{blabel}",
                              f"execute {n_exec_ok} case(s) in the container; hand-simulate "
                              f"{len(sim_cases)} remaining case(s)", p3, r3))
            completed_units += len(batch)
            checkpoint(f"simulate_complete:{blabel}", complete=False)
            print(f"[simloop:4.2/4.3] checkpointed batch {bi}/{len(batches)} "
                  f"({len(all_cases)} cases, {len(mismatches)} mismatches)", flush=True)
        analysis_complete = completed_units == len(units) and not analysis_failures
        checkpoint("round_complete" if analysis_complete else "round_incomplete",
                   complete=analysis_complete)
        print(f"[simloop:4.2/4.3] round {rnd}: {len(all_cases)} case(s) over {len(units)} "
              f"unit(s) -> {len(mismatches)} MISMATCH(es); "
              f"analysis {'complete' if analysis_complete else 'INCOMPLETE'}", flush=True)

        # Covering-test evidence (executed) + spec-literal near-misses.
        test_failed, test_text, near_rows = [], "", []
        mismatches += near_rows
        n_exec = sum(1 for m in mismatches if m.get("source") in ("executed", "test", "reasoned")
                     and m.get("anchored", True))
        n_unanchored = sum(1 for m in mismatches if m.get("source") in ("executed", "reasoned")
                           and not m.get("anchored", True))
        n_sim = sum(1 for m in mismatches if m.get("source") == "simulated")
        if n_unanchored:
            print(f"[simloop:4.3] {n_unanchored} executed/reasoned mismatch(es) have an EXPECTED the "
                  f"specification does not state verbatim -- shown to refine but not a trigger",
                  flush=True)
        trigger = n_exec >= 1 or n_sim >= max(1, SIMLOOP_SIM_MIN)
        if trigger and not analysis_complete and n_exec == 0:
            # A partial hand-simulation is not evidence enough to rewrite a patch.
            trigger = False
            print(f"[simloop] round {rnd}: {n_sim} simulated mismatch(es) on an INCOMPLETE "
                  "analysis and no executed evidence -- not refining", flush=True)
        rec = {"round": rnd, "n_cases": len(all_cases), "n_mismatch": len(mismatches),
               "n_executed_mismatch": n_exec, "n_simulated_mismatch": n_sim,
               "analysis_complete": analysis_complete,
               "analysis_failures": analysis_failures,
               "covering_tests_failing": test_failed[:10],
               "mismatches": [{k: m.get(k) for k in ("id", "symbol", "expected", "actual", "source", "anchored")}
                              for m in mismatches]}
        if not trigger:
            meta["rounds"].append(rec)
            if not mismatches and analysis_complete:
                print(f"[simloop] round {rnd}: simulation clean -- keeping the initial patch",
                      flush=True)
            elif mismatches:
                print(f"[simloop] round {rnd}: {n_sim} simulated-only mismatch(es) below the "
                      f"corroboration floor ({SIMLOOP_SIM_MIN}) -- keeping the initial patch",
                      flush=True)
            else:
                meta["incomplete"] = analysis_failures
                print(f"[simloop] round {rnd}: analysis incomplete -- keeping the patch "
                      f"WITHOUT claiming a simulation pass", flush=True)
            break

        # --- 4.4 REFINE ------------------------------------------------------------------------
        snap = sp._tree_snapshot(env, repo_path)
        smoke_before, _ = sp._import_smoke(env, repo_path, diff_now)
        try:
            mech_before = sum(1 for m in sp._mech_audit(env, repo_path, instance, findings, diff_now)
                              if m.get("violated"))
        except Exception:
            mech_before = 0
        print(f"[simloop:4.4] round {rnd}: refining on {n_exec} executed + {n_sim} simulated "
              f"mismatch(es)", flush=True)
        phase = _orig_run_phase_agent(
            "repair_refine", model, env, base_agent,
            system=system, instance=_REFINE_INSTANCE, step_limit=SIMLOOP_REFINE_STEPS,
            task=_STATE["problem_statement"], submit_guard=True,
            repair_context=ctx,
            diff_so_far=_sub._clip(diff_now, 8000) or "(no diff captured)",
            mismatches=sp._j2safe(_render_mismatches(mismatches)),
            test_evidence=sp._j2safe(("FAILING (" + ", ".join(test_failed[:6]) + ")\n" + test_text)
                                     if test_failed else "(no failing test IDs supplied; this does "
                                     "not establish that covering tests ran or passed)"),
            phase_body=_REFINE_BODY.replace("/testbed", repo_path))
        sp._save_phase(inst_out, f"4.4_simloop_refine{sfx}", phase)

        # Acceptance through the shared mutation gate: import smoke, mechanical count, and the
        # executed veto (covering tests must not go pass->fail), with hunk bisection on conflict.
        after, gate_rec, _mech_after, _smoke_after = sp._gate_phase_mutation(
            env, repo_path, instance, findings, "repair_refine", phase["exit_status"],
            diff_now, mech_before, smoke_before, snap_before=snap,
            check_tests_on_conflict=True, revert_on_test_regression=True,
            intended_symbols=[m.get("symbol") for m in mismatches if m.get("symbol")],
            log=lambda m: print(m, flush=True),
            # G9: on Go the refine gate grants no stale-test exemption and also counts new
            # failure output under already-failing tests (see sp._gate_phase_mutation).
            strict_test_regression=_sub.is_go())
        kept = bool(gate_rec.get("kept", not gate_rec.get("changed", True)))
        changed_now = bool((after or "").strip() != (diff_now or "").strip())
        # Executed regression guard: a case that MATCHED before the refine must still match.
        if kept and changed_now and SIMLOOP_EXEC and SIMLOOP_EXEC_REGRESSION and snap:
            prev_ok = [c for c in all_cases if c.get("harness") and c["id"] in exec_match_ids]
            if prev_ok:
                again = _execute_cases(env, repo_path, prev_ok)
                flipped = sorted(cid for cid, r in again.items() if r["verdict"] == "MISMATCH")
                rec["executed_regressions"] = flipped[:8]
                if flipped:
                    print(f"[simloop:4.4] round {rnd}: {len(flipped)} previously MATCHING "
                          f"executed case(s) now MISMATCH after the refine ({flipped[:4]}) "
                          f"-- reverting the round", flush=True)
                    if sp._tree_restore(env, repo_path, snap):
                        kept = False
                        after = diff_now
                        gate_rec = dict(gate_rec, kept=False, reverted=True,
                                        why=f"executed regression: previously matching "
                                            f"case(s) {flipped[:4]} now mismatch")
                    else:
                        print(f"[simloop:4.4] WARNING: could not restore the pre-refine tree",
                              flush=True)
        rec.update(refine_kept=kept, gate=gate_rec, exit_status=phase["exit_status"],
                   n_calls=phase["n_calls"], cost=phase["cost"],
                   changed=changed_now and kept)
        # Re-execute the executed mismatches to report what the round actually fixed.
        if kept and rec["changed"] and SIMLOOP_EXEC:
            redo = [c for c in all_cases if c.get("harness") and any(
                m["id"] == c["id"] and m.get("source") == "executed" for m in mismatches)]
            if redo:
                again = _execute_cases(env, repo_path, redo)
                fixed = sum(1 for r in again.values() if r["verdict"] == "MATCH")
                rec["executed_mismatches_fixed"] = f"{fixed}/{len(redo)}"
                print(f"[simloop:4.4] round {rnd}: executed mismatches now MATCH: "
                      f"{fixed}/{len(redo)}", flush=True)
        print(f"[simloop:4.4] round {rnd} {'kept' if kept else 'REVERTED'} "
              f"(exit={phase['exit_status']}, steps={phase['n_calls']}, "
              f"diff {len(diff_now)} -> {len(after or '')} chars; {gate_rec.get('why', '')})",
              flush=True)
        meta["rounds"].append(rec)

    refined = sp._extract_patch(env, repo_path)
    (inst_out / "4_refined_patch.diff").write_text(refined or "")
    meta["usage"] = meter.totals()
    meta["initial_patch_chars"] = len(initial or "")
    meta["refined_patch_chars"] = len(refined or "")
    meta["changed"] = (refined or "").strip() != (initial or "").strip()
    t = meta["usage"]
    print(f"[simloop] done: initial {len(initial)} chars -> refined {len(refined)} chars "
          f"({'CHANGED' if meta['changed'] else 'unchanged'}); tool-free queries: "
          f"{t['n_calls']} calls, {t['total_tokens']} tokens, ${t['cost']:.4f}", flush=True)
    return meta


_SIMLOOP_DESC = {
    "simloop_extract": "sim-loop 4.1: code units the initial patch edited (post-patch source)",
    "simloop_review": "sim-loop 4.2/4.3 (reason mode): pure code-reasoning review of the initial patch -- trace lens, contract lens, adversarial verifier",
    "simloop_cases": "sim-loop 4.2: spec-grounded INPUT/EXPECTED cases per edited unit",
    "simloop_simulate": "sim-loop 4.3: hand-execution of each unit on its cases (MATCH/MISMATCH)",
    "simloop_refine": "sim-loop 4.4: refinement driven by the simulation mismatches",
    "repair_refine": "sim-loop 4.4: refinement driven by the simulation mismatches",
}


def _simloop_messages(inst_out: Path) -> list:
    """The 4.x steps as mini-swe-agent messages, banner-separated, in step order."""
    out: list = []
    for f in sorted(Path(inst_out).glob("4.[1-4]_simloop_*.traj.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        phase = d.get("phase") or f.stem
        out.append(sp._banner(phase, _SIMLOOP_DESC.get(phase, phase)))
        ms = d.get("messages") or []
        if not ms:
            # 4.1 in deterministic mode makes no model call, so it has no turns; render the
            # extracted units so the combined view still shows what the loop worked on.
            src = d.get("unit_source") or {}
            body = "\n\n".join(
                f"=== UNIT {u.get('id') or u.get('unit')} {u.get('file','?')} :: {u.get('symbol','?')} ===\n"
                + _sub._clip(src.get(u.get("id") or u.get("unit"), "(source not recorded)"), 2000)
                for u in (d.get("units") or []) if isinstance(u, dict) and (u.get("id") or u.get("unit"))) \
                or ("(reasoning review -- see 4.2_simloop_review)" if "review" in phase or
                    any("review" in u for u in (d.get("units") or []) if isinstance(u, dict))
                    else "(no units extracted)")
            ms = [{"role": "assistant",
                   "content": f"[{phase} | mode={d.get('mode', '?')}]\n{body}",
                   "extra": {"phase": phase}}]
        out += ms
    return out


def _plain_combined_trajectory(*args, **kwargs) -> dict:
    """Splice the sim-loop's 4.x turns into the combined trajectory, after the repair block.

    sp._combined_trajectory concatenates an explicit list of phases (explore/localize/repair/
    [repair_sites]/validate); the sim-loop is not one of them, so its steps were absent from the
    single-file view entirely.
    """
    traj = _orig_combined_trajectory(*args, **kwargs)
    io = _STATE.get("inst_out")
    if not isinstance(traj, dict) or io is None:
        return traj
    try:
        extra = _simloop_messages(io)
    except Exception as e:
        print(f"[simloop] combined-trajectory splice FAILED ({type(e).__name__}: {e})", flush=True)
        return traj
    if not extra:
        return traj
    msgs = traj.get("messages") or []
    # Insert directly after repair's block: before the next phase banner that follows it.
    idx, seen_repair = len(msgs), False
    for i, m in enumerate(msgs):
        pm = (m.get("extra") or {}).get("phase_marker")
        if pm == "repair":
            seen_repair = True
        elif seen_repair and pm in ("repair_sites", "validate"):
            idx = i
            break
    traj["messages"] = msgs[:idx] + extra + msgs[idx:]
    return traj


def _plain_run_pipeline(instance: dict, out_dir: Path, base_config: dict) -> dict:
    _STATE["problem_statement"] = (instance.get("problem_statement") or "")
    _STATE["findings"] = None
    # The sim-loop needs the spec fields and the per-instance output dir; run_pipeline builds
    # inst_out the same way (out_dir / instance_id) and unquotes the spec fields itself, so
    # capture the raw dict here and re-read the unquoted fields off it at loop time.
    _STATE["simloop_meta"] = None
    _STATE["instance"] = instance
    _STATE["inst_out"] = out_dir / instance["instance_id"]
    rec = _orig_run_pipeline(instance, out_dir, base_config)
    if isinstance(rec, dict):
        rec["variant"] = VARIANT
        # run_pipeline builds phases["repair"] from a FIXED key list, so repair["simloop"] never
        # reaches pipeline.json -- put it back and rewrite the file it already persisted.
        meta = _STATE.get("simloop_meta")
        if meta is not None:
            rec.setdefault("phases", {}).setdefault("repair", {})["simloop"] = meta
            io = _STATE.get("inst_out")
            if io is not None:
                try:
                    (Path(io) / "pipeline.json").write_text(
                        json.dumps(rec, indent=1, default=str))
                except Exception as e:
                    print(f"[simloop] pipeline.json rewrite FAILED "
                          f"({type(e).__name__}: {e})", flush=True)
    return rec


def install() -> None:
    """Apply the variant to the imported subagent_pipeline module (idempotent)."""
    sp.REPAIR_MODE = "classic"      # simfirst is a different phase-3 experiment entirely
    sp.REPAIR_SIM_RETRIES = 0       # the pre-fix simulation hook is what this variant removes
    sp.EXPLORE_TO_REPAIR = False
    sp.REPAIR_SITES_STEPS = 0   # skips phase 3b (see module docstring)
    sp._run_phase_agent = _plain_run_phase_agent
    sp._localization_reference = _capturing_localization_reference
    sp.run_pipeline = _plain_run_pipeline
    sp._combined_trajectory = _plain_combined_trajectory
    print(f"[variant] {VARIANT}: repair = default agent; "
          f"repair_sites={'on' if PLAIN_KEEP_REPAIR_SITES else 'off'}; "
          f"mech_audit={'SKIPPED' if PLAIN_SKIP_MECH else 'on'}; "
          f"spec_audit(behavioral)/audit_fix/validate unchanged", flush=True)


def main(argv: "list[str]") -> int:
    install()
    return sp.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
