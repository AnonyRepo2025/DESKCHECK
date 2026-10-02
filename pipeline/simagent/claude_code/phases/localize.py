"""Call 1 -- LOCALIZE: explore -> predict anchors -> simulate execution paths -> predict edit sites.

Reasoning steps (L2, L3) run with no tools. Deterministic devices between steps: the static
call-chain tracer with frame source, path mining of the most likely end-to-end path, existence
check of every proposed site, frame-union (a frame the simulation names as MUST-CHANGE stays a
site unless the model explicitly ruled it out).
"""
from __future__ import annotations

import json
import re
import shlex

from simagent import pipeline as sp
from simagent import localization as _loc

from .. import flow

_sub = sp._sub

EXPLORE_TURNS = 35
ANCHOR_SCHEMA = {
    "type": "object",
    "properties": {
        "primary": {"type": "object", "properties": {
            "function": {"type": "string"}, "file": {"type": "string"}, "line": {"type": "integer"},
            "reason": {"type": "string"}}, "required": ["function", "file", "reason"]},
        "alternates": {"type": "array", "maxItems": 3, "items": {"type": "object", "properties": {
            "function": {"type": "string"}, "file": {"type": "string"}, "hypothesis": {"type": "string"}},
            "required": ["function", "file", "hypothesis"]}},
    },
    "required": ["primary", "alternates"],
}
SITES_SCHEMA = {
    "type": "object",
    "properties": {
        "root_cause": {"type": "object", "properties": {
            "function": {"type": "string"}, "file": {"type": "string"}, "line": {"type": "integer"},
            "why": {"type": "string"}}, "required": ["function", "file", "why"]},
        "sites": {"type": "array", "items": {"type": "object", "properties": {
            "file": {"type": "string"}, "symbol": {"type": "string"},
            "decision": {"type": "string", "enum": ["EDIT", "RULED_OUT"]}, "why": {"type": "string"}},
            "required": ["file", "symbol", "decision", "why"]}},
        "concrete_inputs": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["root_cause", "sites", "concrete_inputs"],
}

L2_PROMPT = """\
STEP 2 of the localization workflow -- PREDICT THE ANCHORS (reasoning only; you have no tools now).

From the issue and what you read in step 1, name the SINGLE most likely function (or method)
where the bug lives, plus up to 3 ALTERNATE candidates, each a competing hypothesis.

Prefer the PRODUCER, not the symptom surface: if the issue describes a WRONG / EXTRA / MISSING
VALUE in some output, name the function that CONSTRUCTS / PARSES that value (the data origin --
often a resolve / parse / build / __init__ step), NOT the outer API where the wrong value is merely
rendered. Three systematic traps: the issue author's SUGGESTED FIX SITE and the DEEPEST TRACEBACK
FRAME are both hypotheses, not facts; and for a MULTI-FAULT issue that lists several distinct
broken behaviours, name one candidate per remaining behaviour. Name functions that EXIST in the
repository's source now (you read the files). Give file paths relative to the repository root.
"""

L3_PROMPT = """\
STEP 3 of the localization workflow -- SIMULATE THE EXECUTION PATHS (reasoning only; no tools).

Below is the STATIC CALL CHAIN around your anchor(s): caller paths that reach the anchor, callee
paths it triggers, and the REAL SOURCE BODY of the key frames. Read each body and trace over the
real code -- never guess what a function does from its name. Do NOT write or run any code: this is
a mental trace, as if you were the interpreter.

{chains}

{mined}

Produce, in this order:

CONCRETE INPUT: one concrete, plausible input that exhibits the issue. If the issue gives no
example you MUST invent one from the frame source; refusing ("cannot construct", "no concrete
input") is a failure of this step.

EXECUTION SIMULATION: pick the caller path the input actually takes to reach the anchor, then
continue down the callee paths it triggers. For EACH function on the path state, in order, what it
receives, what it evaluates to, and how the value flows to the next hop. Track the WRONG VALUE to
its birthplace and call it out explicitly as `<value> is FIRST CONSTRUCTED here, in <function> at
<file:line>`. A later function that merely forwards, consumes or renders the value is NOT where it
goes wrong.
{mined_instruction}
PRODUCER CHECK: adversarial self-check -- the anchor is a HYPOTHESIS from step 2 and is often the
frame where the symptom SURFACES. Is the value already wrong when it ENTERS the anchor? Is the
wrong object constructed by a helper, another branch, or generic machinery in ANOTHER file? Weigh
each ALTERNATE against the anchor using its real source. Disagreeing with the anchor is a valid and
useful outcome. End the check with EXACTLY one line:
PRODUCER: <function> | <file>:<line> | <the wrong value it first constructs>

FRAMES THAT MUST CHANGE: every function your trace shows must be edited for the issue's scenario
to be fixed END-TO-END. Include a CALLER when the fix adds or changes a parameter/return that this
caller must pass or handle -- an optional parameter nobody passes fixes nothing. One per line:
- <function> | <file>
"""

L4_PROMPT = """\
STEP 4 of the localization workflow -- PREDICT THE EDIT SITES (read-only tools: Read, Grep, Glob,
and Bash for grep/sed only; do NOT modify any file).

Using your simulation, decide the set of sites a fix must edit. Rules:
1. PRIMARY = the root-cause function (the PRODUCER your trace identified).
2. ALSO consider callers and siblings that the FRAMES THAT MUST CHANGE list names, base classes
   whose behaviour the hidden tests may exercise directly, and every existing call site when the
   fix changes a signature (grep for the callers).
3. Treat the bug as a CLASS, not one line: grep for the same pattern in sibling modules.
4. DISPOSE OF EVERY NAMED FRAME explicitly: every function named under FRAMES THAT MUST CHANGE or
   in the PRODUCER line must appear as decision EDIT or be refuted as RULED_OUT with a reason.
   Silently dropped frames are re-added automatically.
5. Every site must EXIST: verify file and symbol with grep before listing it.
Also list the concrete inputs you traced (verbatim), so the later phases can reuse them as seeds.
"""

_PRODUCER_RE = re.compile(r"^\s*PRODUCER:\s*([^|\n]+?)\s*\|\s*([^|\n:]+?)(?::(\d+))?\s*\|", re.M)
_FRAME_RE = re.compile(r"^\s*-\s*`?([A-Za-z_][\w.]*)`?\s*\|\s*`?([\w./-]+\.\w+)`?", re.M)


def _frames(sim: str) -> list[tuple[str, str]]:
    i = sim.find("FRAMES THAT MUST CHANGE")
    if i < 0:
        return []
    return [(m.group(1), m.group(2)) for m in _FRAME_RE.finditer(sim[i:])][:8]


_TEST_PATH_RE = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_test\.(py|go|js|ts)$|\.(spec|test)\.[jt]sx?$")


def _is_test_path(relf: str) -> bool:
    return bool(_TEST_PATH_RE.search(relf or ""))


def _exists(ctx: flow.Ctx, relf: str, sym: str) -> bool:
    if _is_test_path(relf):
        return False   # a test file is never an edit site (b1 bf98f031: the repair edited a unit test)
    bare = sym.split(".")[-1]
    if _sub.is_go():   # func Name / func (r *T) Name / type Name / var|const Name (also inside a block)
        pat = (f"^\\s*func\\s+(\\([^)]*\\)\\s*)?{re.escape(bare)}\\b|^\\s*type\\s+{re.escape(bare)}\\b|"
               f"^\\s*(var|const)\\s+{re.escape(bare)}\\b|^\\s+{re.escape(bare)}(\\s+[^=]*)?\\s*=")
    elif _sub.is_js():
        # J1 (JS port): the Python `def|class` pattern matches nothing in a .js/.ts file, so every
        # edit site the localizer named was dropped as non-existent. Use the stock JS grammar
        # (function decl, const arrow, class/interface/type/enum, class method, object property).
        pat = _sub.def_pattern(bare)
    else:
        pat = f"^\\s*(async\\s+def|def|class)\\s+{re.escape(bare)}\\b"
    cmd = (f"cd {shlex.quote(ctx.repo_path)} && test -f {shlex.quote(relf)} && "
           f"grep -nE '{pat}' {shlex.quote(relf)} | head -1")
    try:
        out = ctx.env.execute({"command": cmd}, timeout=30)
        return bool((out.get("output") or "").strip())
    except Exception:
        return False


def _norm(path: str, repo_path: str) -> str:
    p = (path or "").strip()
    for pre in (repo_path.rstrip("/") + "/", "/app/", "/testbed/", "./"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p.lstrip("/")


def run(ctx: flow.Ctx) -> object:
    ps = _sub._clip(ctx.problem_statement, _sub.PS_CLIP)
    s = flow.Session(ctx, "localize",
                     system_prompt="You are a careful software engineer localizing a bug in a repository.")
    # L1 explore (read-only)
    # J1 (JS port): stock prompt bodies name Python tooling; _maybe_lang rewrites them per language
    # (no-op for Python). The per-step LANGUAGE NOTE alone left 'pytest'/'.py' phrasing in the body.
    explore = flow.strip_submit_protocol(flow.lang_text(sp.EXPLORE_BODY))
    r1 = s.step("explore", f"<pr_description>\n{ps}\n</pr_description>\n\n{explore}"
                + flow.RUNTIME_NOTE.format(repo_path=ctx.repo_path),
                tools=flow.READ_ONLY_TOOLS, max_turns=EXPLORE_TURNS)
    if ctx.baseline:
        return _run_baseline(ctx, s, ps)
    # L2 anchors (no tools)
    r2 = s.step_structured("anchors", L2_PROMPT, tools=flow.NO_TOOLS, max_turns=2, schema=ANCHOR_SCHEMA)
    anchors = r2.structured_output or {}
    prim = anchors.get("primary") or {}
    alts = anchors.get("alternates") or []
    if not prim.get("function"):
        ctx.log("[localize] no primary anchor -- falling back to text answer")
        prim = {"function": "", "file": "", "reason": r2.submission[:300]}
    # deterministic: tracer + path mining
    trace = _sub.make_trace_call_chain_runner(ctx.env, repo_path=ctx.repo_path)
    chains = []
    chain_texts = {}
    for i, a in enumerate([prim] + alts[:2]):
        if not a.get("function"):
            continue
        try:
            txt = trace(a["function"], _norm(a.get("file", ""), ctx.repo_path), int(a.get("line") or 0))
        except Exception as e:  # noqa: BLE001
            txt = f"[trace failed: {type(e).__name__}: {e}]"
        clip = (12000 if i == 0 else 3000) if True else (16000 if i == 0 else 5000)
        chain_texts[a["function"]] = txt
        hdr = "=== CALL CHAIN for the PRIMARY anchor ===" if i == 0 else \
            f"=== ALTERNATE CANDIDATE #{i}: '{a['function']}' [{a.get('file')}] ({a.get('hypothesis','')}) -- a COMPETING hypothesis; its call chain/source ==="
        chains.append(hdr + "\n" + _sub._clip(txt, clip))
    mined_txt, mined = "", []
    try:
        callers, callees = sp._parse_chain_paths(chain_texts.get(prim.get("function"), ""))
        mined = sp._mine_likely_path(callers, callees, ps) or []
    except Exception as e:  # noqa: BLE001
        ctx.log(f"[localize] path mining failed ({type(e).__name__}: {e})")
    if mined:
        mined_txt = ("=== MINED MOST-LIKELY END-TO-END PATH (deterministic consensus over all extracted "
                     "dependency paths) ===\n" + " -> ".join(mined))
    mined_instr = ("\nADDITIONAL SIMULATION ON THE MINED PATH: after the trace above, independently trace the "
                   "mined path end to end from the frame source alone (state VALID INPUT FOR THIS PATH, or "
                   "PATH INFEASIBLE with the reason); you may conclude it does not exhibit the bug.\n") if mined else ""
    r3 = s.step("simulate", L3_PROMPT.format(chains="\n\n".join(chains) or "(no call chain could be traced)",
                                             mined=mined_txt, mined_instruction=mined_instr),
                tools=flow.NO_TOOLS, max_turns=2)
    sim = r3.submission or ""
    prod = _PRODUCER_RE.search(sim)
    frames = _frames(sim)
    # L4 sites (read-only)
    r4 = s.step_structured("sites", L4_PROMPT + "\n(Answer from what you have already read; you have no tools in this step.)",
                           tools=flow.NO_TOOLS, max_turns=2, schema=SITES_SCHEMA)
    so0 = r4.structured_output or {}
    ok0 = [st for st in (so0.get("sites") or []) if st.get("decision") != "RULED_OUT"
           and _exists(ctx, _norm(st.get("file", ""), ctx.repo_path), st.get("symbol", ""))]
    if not ok0:
        ctx.log("[localize] no existing site from the tools-off sites step -- one read-only pass")
        r4 = s.step_structured("sites_verify", L4_PROMPT, tools=flow.READ_ONLY_TOOLS, max_turns=15, schema=SITES_SCHEMA)
    so = r4.structured_output or {}
    rc = so.get("root_cause") or {}
    sites, reasons, ruled = [], {}, set()
    for st in so.get("sites") or []:
        f = _norm(st.get("file", ""), ctx.repo_path)
        key = f"{f} :: {st.get('symbol','')}"
        if st.get("decision") == "RULED_OUT":
            ruled.add(key)
            continue
        if f and st.get("symbol") and _exists(ctx, f, st["symbol"]):
            sites.append(key)
            reasons[key] = "PLAN chose it: " + (st.get("why") or "")[:160]
        else:
            ctx.log(f"[localize] dropped non-existent site {key}")
    # frame-union: simulation frames neither edited nor ruled out are re-added
    for fn, f in frames:
        f = _norm(f, ctx.repo_path)
        key = f"{f} :: {fn}"
        if key in sites or key in ruled:
            continue
        if _exists(ctx, f, fn):
            sites.append(key)
            reasons[key] = "frame-union: the execution simulation named this frame as MUST-CHANGE / the producer"
    if prod:
        pf = _norm(prod.group(2), ctx.repo_path)
        key = f"{pf} :: {prod.group(1).strip()}"
        if key not in sites and key not in ruled and _exists(ctx, pf, prod.group(1).strip()):
            sites.append(key)
            reasons[key] = "value-origin: the PRODUCER CHECK named it as where the wrong value is first constructed"
    root_cause = ""
    if rc.get("function"):
        root_cause = f"{rc['function']} | {_norm(rc.get('file',''), ctx.repo_path)}" + (f":{rc['line']}" if rc.get("line") else "") + f" ({rc.get('why','')[:160]})"
    elif prod:
        root_cause = f"{prod.group(1).strip()} | {_norm(prod.group(2), ctx.repo_path)}"
    fnd = _sub.SubAgentFindings(
        bug_function=prim.get("function", ""), bug_file=_norm(prim.get("file", ""), ctx.repo_path),
        bug_line=int(prim.get("line") or 0), locate_reason=prim.get("reason", ""),
        call_chain="\n\n".join(chains)[:20000], simulation=sim, prediction="", root_cause=root_cause,
        files_to_edit=sites, ok=bool(prim.get("function")) and bool(sim),
        raw_locate=json.dumps(anchors), raw_plan=json.dumps(so),
        alt_candidates=[(a.get("function"), a.get("file"), a.get("hypothesis")) for a in alts])
    fnd.site_reasons = reasons
    _attach_spec_devices(ctx, fnd)
    fnd.demoted_sites = sorted(ruled)
    fnd.concrete_inputs = [x for x in (so.get("concrete_inputs") or []) if isinstance(x, str)][:6]
    fnd.mined_path = mined
    fnd.input_directives = []
    metrics = None
    if ctx.instance.get("patch"):
        try:
            gf, gfn = _loc.parse_gold_patch(ctx.instance["patch"])
            metrics = _loc.score_localization(fnd, gf, gfn)
        except Exception as e:  # noqa: BLE001
            ctx.log(f"[localize] scoring failed ({type(e).__name__}: {e})")
    ctx.findings = fnd
    ctx.record["localize"] = {"ok": fnd.ok, "bug_function": fnd.bug_function, "bug_file": fnd.bug_file,
                              "root_cause": root_cause, "files_to_edit": sites, "ruled_out": sorted(ruled),
                              "frames": frames, "mined_path": mined, "concrete_inputs": fnd.concrete_inputs,
                              "metrics": metrics, "steps": s.steps, "cost": s.cost, "n_calls": s.n_calls}
    s.save(ctx.inst_out / "1_localize.traj.json", findings=fnd.to_dict() if hasattr(fnd, "to_dict") else {})
    ctx.log(f"[localize] root_cause={root_cause[:80]!r} sites={len(sites)} ruled_out={len(ruled)} "
            f"metrics={json.dumps(metrics)[:160] if metrics else None}")
    return fnd


def reference_block(fnd) -> str:
    """The findings as a reference-only block for the repair call."""
    lines = ["=== LOCALIZATION FINDINGS (from the localization call; a hint, not a contract) ===",
             f"Root cause: {fnd.root_cause or '(none)'}",
             f"Symptom surface: {fnd.bug_function} ({fnd.bug_file})", "Predicted edit sites:"]
    for k in fnd.files_to_edit:
        lines.append(f"  - {k}  -- {getattr(fnd, 'site_reasons', {}).get(k, '')}")
    if getattr(fnd, "concrete_inputs", None):
        lines.append("Concrete inputs traced during localization (reuse them):")
        lines += [f"  - {x[:300]}" for x in fnd.concrete_inputs]
    i = (fnd.simulation or "").find("PRODUCER CHECK")
    if i >= 0:
        lines.append("Producer check from the execution simulation:\n" + fnd.simulation[i:i + 1200])
    lines.append("Verify anything you use against the real source; fix the issue where the source says it lives.")
    return "\n".join(lines)


L4_BASELINE_PROMPT = """\
STEP 2 of the localization workflow -- PREDICT THE EDIT SITES (read-only tools: Read, Grep, Glob,
and Bash for grep/sed only; do NOT modify any file).

From what you read, decide the set of sites a fix must edit: name the root-cause function and
every file :: symbol that must change, plus sites you considered and ruled out. Grep for callers
when the fix changes a signature, and for the same pattern in sibling modules. Every site must
EXIST: verify file and symbol with grep before listing it. List the concrete inputs (if any) you
would use to exercise the bug.
"""


def _run_baseline(ctx: flow.Ctx, s: flow.Session, ps: str):
    """Arm B localization: no anchors, no tracer, no execution simulation."""
    r4 = s.step_structured("sites", L4_BASELINE_PROMPT, tools=flow.READ_ONLY_TOOLS, max_turns=25, schema=SITES_SCHEMA)
    so = r4.structured_output or {}
    rc = so.get("root_cause") or {}
    sites, reasons, ruled = [], {}, set()
    for st in so.get("sites") or []:
        f = _norm(st.get("file", ""), ctx.repo_path)
        key = f"{f} :: {st.get('symbol','')}"
        if st.get("decision") == "RULED_OUT":
            ruled.add(key)
        elif f and st.get("symbol") and _exists(ctx, f, st["symbol"]):
            sites.append(key)
            reasons[key] = "PLAN chose it: " + (st.get("why") or "")[:160]
    root_cause = f"{rc.get('function','')} | {_norm(rc.get('file',''), ctx.repo_path)} ({rc.get('why','')[:160]})" if rc.get("function") else ""
    fnd = _sub.SubAgentFindings(bug_function=rc.get("function", ""), bug_file=_norm(rc.get("file", ""), ctx.repo_path),
                                bug_line=int(rc.get("line") or 0), root_cause=root_cause, files_to_edit=sites,
                                ok=bool(sites), raw_plan=json.dumps(so))
    fnd.site_reasons = reasons
    fnd.demoted_sites = sorted(ruled)
    fnd.concrete_inputs = [x for x in (so.get("concrete_inputs") or []) if isinstance(x, str)][:6]
    fnd.mined_path = []
    fnd.input_directives = []
    _attach_spec_devices(ctx, fnd)
    metrics = None
    if ctx.instance.get("patch"):
        try:
            gf, gfn = _loc.parse_gold_patch(ctx.instance["patch"])
            metrics = _loc.score_localization(fnd, gf, gfn)
        except Exception as e:  # noqa: BLE001
            ctx.log(f"[localize] scoring failed ({type(e).__name__}: {e})")
    ctx.findings = fnd
    ctx.record["localize"] = {"ok": fnd.ok, "baseline": True, "bug_function": fnd.bug_function, "bug_file": fnd.bug_file,
                              "root_cause": root_cause, "files_to_edit": sites, "ruled_out": sorted(ruled),
                              "concrete_inputs": fnd.concrete_inputs, "metrics": metrics, "steps": s.steps,
                              "cost": s.cost, "n_calls": s.n_calls}
    s.save(ctx.inst_out / "1_localize.traj.json", findings=fnd.to_dict() if hasattr(fnd, "to_dict") else {})
    ctx.log(f"[localize] (baseline) root_cause={root_cause[:80]!r} sites={len(sites)} metrics={json.dumps(metrics)[:160] if metrics else None}")
    return fnd


def _attach_spec_devices(ctx: flow.Ctx, fnd) -> None:
    """Spec-derived device inputs the audit/validate calls read off the findings (as the stock
    localize sets them): interface signature contracts -> I* checklist points, setter probes."""
    inst = dict(ctx.instance)
    for fld in ("requirements", "interface"):
        inst[fld] = sp._unquote_spec_field(inst.get(fld) or "")
    try:
        fnd.signature_contracts = sp._signature_contracts(inst)
    except Exception as e:  # noqa: BLE001
        ctx.log(f"[localize] signature contracts failed ({type(e).__name__}: {e})")
        fnd.signature_contracts = []
    for attr, fn in (("free_constants", lambda: sp._free_constants(ctx.env, ctx.repo_path, inst)),
                     ("message_phrases", lambda: sp._message_phrases(inst))):
        try:
            setattr(fnd, attr, fn())
        except Exception:
            setattr(fnd, attr, [])
