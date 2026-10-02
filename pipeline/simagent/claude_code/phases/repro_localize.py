"""Arm R, call 1 -- LOCALIZE BY REPRODUCTION: explore the repository and write reproduction
tests that fail on the current code, then name the edit sites. Execution-based, no reasoning
step: the evidence is the failing test, not a trace.
"""
from __future__ import annotations

import json
import re
import shlex

from simagent import pipeline as sp
from simagent import localization as _loc

from .. import flow
from .localize import SITES_SCHEMA, _attach_spec_devices, _exists, _norm

_sub = sp._sub
REPRO_TURNS = 45

REPRO_SCHEMA = dict(SITES_SCHEMA)
REPRO_SCHEMA = {
    "type": "object",
    "properties": {
        **SITES_SCHEMA["properties"],
        "repro_files": {"type": "array", "items": {"type": "string"}},
        "repro_command": {"type": "string"},
        "reproduced": {"type": "boolean"},
        "observed_failure": {"type": "string"},
    },
    "required": ["root_cause", "sites", "repro_files", "repro_command", "reproduced", "observed_failure"],
}

PROMPT = """\
<pr_description>
{ps}
</pr_description>

# ROLE: LOCALIZE THE BUG BY REPRODUCING IT

Your job in this call is to find where the bug lives by making it FAIL in a test, not to fix it.

1. Explore the repository: find the modules, classes and functions the PR description implicates
   (grep for the symbols, error messages and API names it mentions) and read the relevant source.
2. Write one or more REPRODUCTION TESTS that exercise the behaviour the PR description asks for,
   through the real entry points (import and call the real function / method / view). {repro_files}
   Assert the behaviour the PR description REQUIRES, so the tests FAIL on the current code and
   will PASS once fixed.
3. RUN the reproduction tests and confirm they fail for the right reason (the missing behaviour,
   not an import error or a typo in your test). Record the exact command and the observed failure.
4. From the failing test's traceback and the source you read, decide the root cause and the set of
   sites (file :: symbol) a fix must edit; verify each symbol exists with grep.

Do NOT modify any non-test source file in this call. Stop when the reproduction fails for the
right reason and the sites are listed.
"""


def _repro_files_rule() -> str:
    if _sub.is_go():
        return ("Name the files\n   `repro_<topic>_test.go` in the SAME package directory as the code under test (package-internal\n"
                "   tests can reach unexported identifiers) and run them with `go test -run 'TestRepro' ./<pkg dir>/`.\n"
                "   Never edit an existing `*_test.go` file.")
    if _sub.is_js():
        return ("Name the files\n   `repro_<topic>.test.js` (or `.test.ts` / `.spec.ts`, matching the repository's own test\n"
                "   files) and put them where the repository's runner discovers tests for that module; run them\n"
                "   with the repository's own runner. Never edit an existing test or snapshot file.")
    return ("Name the\n   files `test_repro_<topic>.py` and put them next to the repository's existing tests for that\n"
            "   module (or under /tmp/repro if the repository has no test directory).")


def run(ctx: flow.Ctx):
    ps = _sub._clip(ctx.problem_statement, _sub.PS_CLIP)
    s = flow.Session(ctx, "localize", system_prompt="You are a careful software engineer localizing a bug by reproducing it.")
    r = s.step_structured("reproduce", PROMPT.format(ps=ps, repro_files=_repro_files_rule()) + flow.RUNTIME_NOTE.format(repo_path=ctx.repo_path),
                          tools=flow.FULL_TOOLS, max_turns=REPRO_TURNS, schema=REPRO_SCHEMA)
    so = r.structured_output or {}
    rc = so.get("root_cause") or {}
    sites, reasons, ruled = [], {}, set()
    for st in so.get("sites") or []:
        f = _norm(st.get("file", ""), ctx.repo_path)
        key = f"{f} :: {st.get('symbol','')}"
        if st.get("decision") == "RULED_OUT":
            ruled.add(key)
        elif f and st.get("symbol") and _exists(ctx, f, st["symbol"]):
            sites.append(key)
            reasons[key] = "reproduction: " + (st.get("why") or "")[:160]
    # harness: verify the reproduction actually fails on the current tree
    cmd = (so.get("repro_command") or "").strip()
    verified = None
    if cmd:
        try:
            out = ctx.env.execute({"command": f"cd {shlex.quote(ctx.repo_path)} && timeout 300 bash -c {shlex.quote(cmd)}; echo __RC=$?"},
                                  timeout=360)
            m = re.search(r"__RC=(\d+)", out.get("output") or "")
            verified = bool(m and m.group(1) != "0")
        except Exception:
            verified = None
    root_cause = f"{rc.get('function','')} | {_norm(rc.get('file',''), ctx.repo_path)} ({rc.get('why','')[:160]})" if rc.get("function") else ""
    fnd = _sub.SubAgentFindings(bug_function=rc.get("function", ""), bug_file=_norm(rc.get("file", ""), ctx.repo_path),
                                bug_line=int(rc.get("line") or 0), root_cause=root_cause, files_to_edit=sites,
                                ok=bool(sites), raw_plan=json.dumps(so))
    fnd.site_reasons = reasons
    fnd.demoted_sites = sorted(ruled)
    fnd.concrete_inputs = [x for x in (so.get("concrete_inputs") or []) if isinstance(x, str)][:6]
    fnd.mined_path = []
    fnd.input_directives = []
    fnd.repro = {"files": [f for f in (so.get("repro_files") or []) if isinstance(f, str)], "command": cmd,
                 "model_says_reproduced": bool(so.get("reproduced")), "harness_verified_failing": verified,
                 "observed_failure": (so.get("observed_failure") or "")[:600]}
    _attach_spec_devices(ctx, fnd)
    metrics = None
    if ctx.instance.get("patch"):
        try:
            gf, gfn = _loc.parse_gold_patch(ctx.instance["patch"])
            metrics = _loc.score_localization(fnd, gf, gfn)
        except Exception as e:  # noqa: BLE001
            ctx.log(f"[localize] scoring failed ({type(e).__name__}: {e})")
    ctx.findings = fnd
    ctx.record["localize"] = {"ok": fnd.ok, "arm": "repro", "bug_function": fnd.bug_function, "bug_file": fnd.bug_file,
                              "root_cause": root_cause, "files_to_edit": sites, "ruled_out": sorted(ruled),
                              "repro": fnd.repro, "metrics": metrics, "steps": s.steps, "cost": s.cost, "n_calls": s.n_calls}
    s.save(ctx.inst_out / "1_localize.traj.json", findings=fnd.to_dict() if hasattr(fnd, "to_dict") else {})
    ctx.log(f"[localize] (repro) root_cause={root_cause[:70]!r} sites={len(sites)} repro_files={fnd.repro['files']} "
            f"cmd={cmd[:60]!r} harness_verified_failing={verified} metrics={json.dumps(metrics)[:140] if metrics else None}")
    return fnd


def reference_block(fnd) -> str:
    lines = ["=== LOCALIZATION BY REPRODUCTION (from the localization call) ===",
             f"Root cause: {fnd.root_cause or '(none)'}", "Predicted edit sites:"]
    lines += [f"  - {k}  -- {fnd.site_reasons.get(k, '')}" for k in fnd.files_to_edit]
    rp = getattr(fnd, "repro", {}) or {}
    if rp.get("files"):
        lines.append(f"Reproduction test file(s): {', '.join(rp['files'])}")
    if rp.get("command"):
        lines.append(f"Reproduction command (fails on the current code): {rp['command']}")
        lines.append(f"Observed failure: {rp.get('observed_failure','')[:400]}")
        lines.append("After your fix, run the reproduction command again: it must pass. Do not edit the reproduction tests.")
    return "\n".join(lines)
