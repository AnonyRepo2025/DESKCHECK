"""Call 3 -- AUDIT: extract requirements -> simulate the patch per requirement -> fix violations.

A1 (no tools): the model turns the specification into testable points, merged with the
harness's deterministic checklist. A2 (no tools): per-point VERDICT with quoted EVIDENCE and a
concrete-input mental execution; harness grounds quotes, downgrades lazy passes, re-asks once,
then CONFIRMS every surviving VIOLATED verdict with an executed probe (model-authored script run
in the container) before it may trigger A3 (full tools), whose edits pass the mutation gate.
"""
from __future__ import annotations

import os
import re
from types import SimpleNamespace

from jinja2 import Template, Undefined

from simagent import pipeline as sp

from .. import flow

_sub = sp._sub
FIX_TURNS = int(os.getenv("AUDIT_FIX_STEPS", "50"))
REQ_SCHEMA = {"type": "object", "properties": {"points": {"type": "array", "maxItems": 8, "items": {
    "type": "object", "properties": {"text": {"type": "string"}, "quote": {"type": "string"}},
    "required": ["text", "quote"]}}}, "required": ["points"]}

A1_PROMPT = """\
AUDIT STEP 1 -- EXTRACT THE REQUIREMENTS (reasoning only; no tools).

Below are the issue and its specification. Turn them into a checklist of TESTABLE points: each
point states ONE observable behaviour, interface, signature, return shape, error message or
edge-case rule the fixed code must satisfy, and quotes VERBATIM the sentence it comes from. Do
not invent requirements the text does not state; do not restate the same sentence twice. Prefer
points that a hidden unit test could check with a concrete input.

=== ISSUE ===
{ps}

=== REQUIREMENTS ===
{req}

=== INTERFACE ===
{iface}
"""

REASK_NOTE = """

[RE-ASK] Your previous verdicts on the points below were REJECTED: their EVIDENCE quoted code that
is not in the pack/diff, or abstained UNVERIFIABLE on a point whose deciding facts ARE in the pack.
Re-verdict ONLY these points: {pids}. (1) EVIDENCE must consist of verbatim lines COPIED from the
EVIDENCE PACK or DIFF above -- they will be substring-checked; (2) SIMULATION must walk the quoted
lines on a concrete input; (3) UNVERIFIABLE is NOT a permitted verdict here.
"""


AUDIT_FIX_REQUIRE_TESTS = os.getenv("CCFLOW_AUDIT_FIX_REQUIRE_TESTS", "1") in ("1", "true", "yes")


def _enforce_executed_check(ctx: flow.Ctx, gate: dict, snap: str, patch_before: str, after: str) -> tuple[str, dict]:
    """An audit fix is kept only when the covering tests could actually be run on it. The probe
    confirms the auditor's READING of a requirement, not the hidden tests' reading (flow_haiku5
    ansible-77658704: R2 probe-confirmed, fix kept because the covering tests were unusable,
    33/33 -> 14/33 with a regression). Unverifiable edits are reverted."""
    if not AUDIT_FIX_REQUIRE_TESTS or not gate or not gate.get("changed") or not gate.get("kept"):
        return after, gate
    ec = gate.get("executed_check") or {}
    if ec.get("failed") is None:
        sp._tree_restore(ctx.env, ctx.repo_path, snap)
        gate = dict(gate, kept=False, reverted=True,
                    why=(gate.get("why") or "") + " -- REVERTED: covering tests could not be run, so the fix is unverifiable")
        ctx.log("[audit] fix reverted: covering tests unavailable, edit unverifiable")
        return patch_before, gate
    return after, gate


# Which verdicts may TRIGGER a fix. "interface" (default): deterministic interface points (I*
# from signature contracts), executed structural facts and W1; requirement-sentence points (R*)
# and model-extracted points are advisory -- the Luna analysis found R-point verdicts do not
# discriminate correct from wrong patches, and flow_haiku5 ansible-77658704 showed a
# probe-confirmed R2 "fix" turning 33/33 into 14/33: the probe validates the auditor's READING
# of an ambiguous sentence, which the hidden tests may read otherwise. "all": every confirmed
# VIOLATED verdict triggers.
AUDIT_FIX_SCOPE = os.getenv("CCFLOW_AUDIT_FIX_SCOPE", "interface")
PROBE_TRIGGERS = os.getenv("CCFLOW_AUDIT_PROBE_TRIGGERS", "1") in ("1", "true", "yes")
_MODULE_LEVEL_RE = re.compile(r"`([A-Za-z_][\w.]*)`\s*--\s*Inputs:.*MODULE-level function", re.S)


# G1 (Go batch 1, 2026-09-16): a requirement that NAMES a function / method / type to implement is
# an existence contract on Go -- the hidden *_test.go in that package references the name, and one
# missing symbol fails the whole package build (vuls-36456cb1 `searchCache`, flipt-3ef34d1f
# setProtobuf/getProtobuf: every F2P failed). On Go the execution probe cannot run, so a model
# VIOLATED verdict never triggers a fix; this makes the check mechanical.
_KIND = r"(?:functions?|methods?|helpers?(?:\s+(?:functions?|methods?))?|constructors?|types?|structs?|interfaces?)"
_ID = r"`([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)(?:\([^`]*\))?`"
# the sentence itself must call the backticked name a function / method / type ... (adjacency), so struct
# fields, parameters, enum values and plain words named in the same requirement are never extracted
_REQ_NAMED = [
    re.compile(_KIND + r"\s+(?:called\s+|named\s+)?" + _ID, re.I),                       # function named `X`
    re.compile(_ID + r"\s+(?:(?:exported|unexported|new|public|private|helper)\s+)?" + _KIND + r"\b", re.I),  # `X` method
]
_REQ_LIST = re.compile(r"\b(?:helper\s+)?(?:functions|methods|helpers|constructors|types)\b[^.:;]{0,60}?"
                       r"\b(?:implemented|defined|added|introduced|provided|exposed)\b[^.:;]{0,30}:\s*"
                       r"(`?[A-Za-z_]\w*`?(?:\s*,\s*(?:and\s+)?`?[A-Za-z_]\w*`?)+)", re.I)
_REQ_ACT_WORD = re.compile(r"\b(?:implement\w*|named|defin\w*|add\w*|introduc\w*|creat\w*|provid\w*|expos\w*|export\w*|must\s+exist)\b", re.I)
_GO_NOT_SYMBOL = {"string", "int", "int32", "int64", "uint", "bool", "byte", "rune", "error", "map", "any", "nil", "true",
                  "false", "func", "struct", "interface", "chan", "context", "float64", "float32"}


def go_required_symbols(text: str) -> "list[tuple[str, str]]":
    """(symbol, file hint) pairs a requirement sentence says must be implemented/defined, where the sentence
    itself calls the name a function / method / helper / constructor / type."""
    text = text or ""
    if not _REQ_ACT_WORD.search(text):
        return []
    names = [m.group(1).split(".")[-1] for rx in _REQ_NAMED for m in rx.finditer(text)]
    for m in _REQ_LIST.finditer(text):
        names += [x.strip(" `") for x in re.split(r"\s*,\s*(?:and\s+)?", m.group(1))]
    hint = next(iter(re.findall(r"([\w./-]+\.go)\b", text)), "")
    out, seen = [], set()
    for n in names:
        if len(n) < 3 or n.lower() in _GO_NOT_SYMBOL or n in seen or not re.fullmatch(r"[A-Za-z_]\w*", n):
            continue
        seen.add(n)
        out.append((n, hint))
    return out


def _go_declared_anywhere(ctx: flow.Ctx, name: str) -> bool:
    """True when some non-test .go file in the repository declares `name` (func, method, type, var, const,
    or an interface method / struct line). Errs towards True: a repo that cannot be listed never
    manufactures a violation."""
    import shlex as _shl
    n = re.escape(name)
    pat = (f"^func[[:space:]]+{n}\\b|^func[[:space:]]*\\([^)]*\\)[[:space:]]*{n}\\b|^type[[:space:]]+{n}\\b|"
           f"^[[:space:]]*(var|const)[[:space:]]+{n}\\b|^[[:space:]]+{n}[[:space:]]*(\\(|=)|^[[:space:]]+{n}[[:space:]]+[A-Za-z*\\[]")
    cmd = (f"cd {_shl.quote(ctx.repo_path)} 2>/dev/null || {{ echo __NOREPO; exit 0; }}; "
           f"F=$(git ls-files -co --exclude-standard -- '*.go' 2>/dev/null | grep -v '_test\\.go$' | grep -v '^vendor/'); "
           f"[ -z \"$F\" ] && {{ echo __NOFILES; exit 0; }}; "
           f"H=$(printf '%s\\n' \"$F\" | xargs -r grep -lE {_shl.quote(pat)} 2>/dev/null | head -1); "
           f"[ -n \"$H\" ] && echo \"__FOUND:$H\" || echo __ABSENT")
    try:
        out = ctx.env.execute({"command": cmd}, timeout=180).get("output") or ""
    except Exception:
        return True
    if "__ABSENT" in out and "__FOUND:" not in out:
        return False
    return True


def structural_facts(ctx: flow.Ctx, checklist, patch: str) -> dict:
    """Executed structural checks for interface points: a symbol the interface declares as a
    MODULE-level function must exist as a top-level `def` in a patched file (AST, not reading).
    Returns {pid: (ok, detail)}."""
    files = sp._DIFF_FILES_RE.findall(patch or "")
    out = {}
    # interface-declared (path, symbol) pairs: the symbol must exist at module level IN THAT FILE
    # (b1 1be7de78: the hidden tests import add_db_name from merge_marc as the interface declares;
    # the patch left it in utils and the whole test module failed to import)
    declared = {}
    try:
        for path, sym in sp._iface_declared_sites((ctx.instance.get("interface") or "").replace("\\n", "\n")):
            segs = set(re.sub(r"\.(py|go)$", "", path).split("/"))
            if not sym or sym.split(".")[-1] in segs or sym.split(".")[0] in segs:
                continue   # a path segment ("qutebrowser") is not a symbol
            declared.setdefault(sym, path)
    except Exception:
        pass
    for pid, txt in checklist:
        if not pid.startswith("I"):
            continue
        m = _MODULE_LEVEL_RE.search(txt)
        by_bare = {k.split(".")[-1]: k for k in declared}
        if not m:
            m2 = re.search(r"`([A-Za-z_][\w.]*)`", txt)
            key = (m2.group(1) if m2 and m2.group(1) in declared else (by_bare.get(m2.group(1).split(".")[-1]) if m2 else None))
            if key:
                dpath = declared[key]
                res = _module_level_in(ctx, key, [dpath])
                out[pid] = (res.startswith("FOUND"), f"declared at {dpath}: {key}: {res[:60]}")
            continue
        name = m.group(1).split(".")[-1]
        key = declared.get(m.group(1)) and m.group(1) or by_bare.get(name)
        if key:
            res = _module_level_in(ctx, key, [declared[key]])
            out[pid] = (res.startswith("FOUND"), f"declared at {declared[key]}: {key}: {res[:60]}")
            continue
        res = _module_level_in(ctx, name, files[:12])
        out[pid] = (res.startswith("FOUND"), f"module-level def {name}: {res[:80]}")
    if _sub.is_go():
        # only a MISSING symbol is recorded: an ok fact would silence the model's behavioural verdict on the point
        for pid, txt in checklist:
            if not pid.startswith("R") or pid in out:
                continue
            missing = [(n, h) for n, h in go_required_symbols(txt) if not _go_declared_anywhere(ctx, n)]
            if missing:
                out[pid] = (False, "GO_NAMED_SYMBOL: " + "; ".join(
                    f"`{n}` is declared nowhere in the repository" + (f" (the requirement places it in {h})" if h else "")
                    for n, h in missing[:6]))
    return out


def _module_level_in(ctx: flow.Ctx, name: str, files: list) -> str:
    """FOUND <file> if `name` is a module-level def/class in one of `files`, or, for a dotted
    `Class.method`, a method inside that module-level class; else ABSENT. (b1_reason_fix
    0fc6d110: the first version accepted only FunctionDef and flagged a class, its methods and
    a path segment as ABSENT, triggering a spurious 17-turn fix that the gate then reverted.)"""
    if _sub.is_go():
        # Go has no module-level def/class; a declared symbol exists when the file declares it as a
        # func, a method on the named receiver type, a type, or a var/const (a Python ast over .go
        # printed ABSENT for every declared symbol and forced a spurious structural fix)
        import shlex as _shl
        cls, _, meth = name.rpartition(".")
        n = re.escape(meth)
        if cls:
            pat = f"^func[[:space:]]*\\([^)]*\\*?{re.escape(cls)}(\\[[^]]*\\])?\\)[[:space:]]*{n}\\b"
        else:
            pat = (f"^func[[:space:]]+{n}\\b|^func[[:space:]]*\\([^)]*\\)[[:space:]]*{n}\\b|^type[[:space:]]+{n}\\b|"
                   f"^(var|const)[[:space:]]+{n}\\b|^[[:space:]]+{n}([[:space:]]+[^=]*)?[[:space:]]*=")
        cmd = (f"cd {ctx.repo_path} && for f in " + " ".join(_shl.quote(f) for f in files)
               + f"; do [ -f \"$f\" ] && grep -qE {_shl.quote(pat)} \"$f\" && {{ echo FOUND $f; exit 0; }}; done; echo ABSENT")
        try:
            return (ctx.env.execute({"command": cmd}, timeout=60).get("output") or "").strip()
        except Exception as e:  # noqa: BLE001
            return f"ERROR {type(e).__name__}"
    if _sub.is_js():
        # J1 (JS port): ast.parse over JavaScript/TypeScript raises for every file, so the Python
        # branch reported ABSENT for every declared symbol -- the same spurious-structural-fix
        # failure Go had. Use the stock JS definition grammar instead.
        import shlex as _shl
        _, _, meth = name.rpartition(".")
        pat = _sub.def_pattern(meth or name)
        cmd = (f"cd {ctx.repo_path} && for f in " + " ".join(_shl.quote(f) for f in files)
               + f"; do [ -f \"$f\" ] && grep -qE {_shl.quote(pat)} \"$f\" && {{ echo FOUND $f; exit 0; }}; done; echo ABSENT")
        try:
            return (ctx.env.execute({"command": cmd}, timeout=60).get("output") or "").strip()
        except Exception as e:  # noqa: BLE001
            return f"ERROR {type(e).__name__}"
    script = ("import ast,sys\nname=sys.argv[1]\ncls,_,meth=name.rpartition('.')\n"
              "for f in sys.argv[2:]:\n    try:\n        t=ast.parse(open(f).read())\n"
              "    except Exception:\n        continue\n"
              "    top=[n for n in t.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]\n"
              "    if not cls and any(n.name==meth for n in top):\n        print('FOUND', f); break\n"
              "    for c in top:\n        if isinstance(c,ast.ClassDef) and (not cls or c.name==cls) and cls and any(isinstance(m,(ast.FunctionDef,ast.AsyncFunctionDef)) and m.name==meth for m in c.body):\n"
              "            print('FOUND', f); break\n    else:\n        continue\n    break\nelse:\n    print('ABSENT')\n")
    b64 = __import__("base64").b64encode(script.encode()).decode()
    cmd = (f"cd {ctx.repo_path} && printf %s {b64} | base64 -d > /tmp/_struct.py && python3 /tmp/_struct.py "
           + __import__("shlex").quote(name) + " " + " ".join(__import__("shlex").quote(f) for f in files))
    try:
        res = (ctx.env.execute({"command": cmd}, timeout=60).get("output") or "").strip()
    except Exception as e:  # noqa: BLE001
        res = f"ERROR {type(e).__name__}"
    return res


def _render(t: str, **v) -> str:
    return Template(t, undefined=Undefined).render(**v)


def _diff_budget(patch: str, total: int, per_file_min: int = 2500) -> str:
    """Hunk-aware clip: every file of the patch keeps a share of the budget (b2 openlibrary-53d376b1:
    a 12.5 KB patch head-clipped at 6 KB hid the whole mock_infobase hunk and half of find_author from
    the tools-off audit -> 10/13 points UNVERIFIABLE 'truncated')."""
    if len(patch) <= total:
        return patch
    parts = re.split(r"(?m)^(?=diff --git )", patch)
    parts = [x for x in parts if x.strip()]
    if len(parts) <= 1:
        return _sub._clip(patch, total)
    share = max(per_file_min, total // len(parts))
    out = []
    for x in parts:
        out.append(x if len(x) <= share else x[:share] + f"\n...[file hunk truncated: {len(x) - share} more chars]...\n")
    return "".join(out)


_LITERAL_RE = re.compile(r"`[^`]{1,80}`|'[^']{1,60}'|\"[^\"]{1,60}\"|\b(?:None|True|False)\b")
_VALUE_WORD_RE = re.compile(r"\b(?:empty|exactly|equal(?:s|ed)?|return(?:s|ed)?|raise(?:s|d)?|tuple|list|dict|string|value|\d+)\b", re.I)


def _value_shaped(txt: str) -> bool:
    """A requirement whose satisfaction is an observable VALUE (literal / return / raise), i.e. one a
    source-literal reading can get wrong (b2 qutebrowser-305e7c96: spec says the third tuple field is
    `None`; the patch wrote the literal None, Qt returns '' -- verdicted SATISFIED off the literal)."""
    t = txt or ""
    return bool(_LITERAL_RE.search(t)) or len(_VALUE_WORD_RE.findall(t)) >= 2


_TRUNC_RE = re.compile(r"truncat|not (?:shown|included|visible|present|in the (?:evidence|pack|diff))|cannot (?:see|verify)"
                       r"|no(?:t)? (?:enough )?(?:evidence|source)|unavailable|elided|cut off", re.I)


def _probe_failure(env, rp: str, author_reply: str, patch: str = "") -> str:
    """Re-run the authored probe script and return the tail of its output (or a reason string) when it
    fails; "" when it ran fine (then the stock device failed for another reason)."""
    import base64
    import shlex
    lang = "python" if _sub.is_python() else ("js" if _sub.is_js() else "")
    if not lang:
        return f"the audit execution probe does not support this language ({_sub.LANG})"
    m = sp._PROBE_FENCE_RE[lang].search(author_reply or "")
    if not m:
        return f"no {lang} fenced block in the probe author reply"
    if lang == "js":
        out = sp._js_probe_output(env, rp, m.group(1), patch)
        if not out:
            return "the JS probe file could not be placed or run (no runner / whole-suite runner)"
        return "" if re.search(r"PROBE\[", out) else ("probe ran but printed no PROBE[<id>]: line\n" + out[-800:])
    try:
        from simagent.validation import python_profile
        profile = python_profile(env, rp)
        if not profile:
            return "no python profile for the container"
        b64 = base64.b64encode(m.group(1).encode()).decode()
        cmd = "import base64; exec(compile(base64.b64decode(" + repr(b64) + "), '<audit-probe>', 'exec'))"
        ex = env.execute({"command": "cd " + shlex.quote(rp) + " && " + profile["prefix"] + " -c " + shlex.quote(cmd)},
                         timeout=45)
        out = ex.get("output", "") or ""
        if ex.get("returncode") != 0:
            return f"exit code {ex.get('returncode')}\n" + out[-1500:]
        if not re.search(r"PROBE\[", out):
            return "script exited 0 but printed no PROBE[<id>]: line\n" + out[-800:]
        return ""
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def _probe_candidates(verdicts, checklist, struct_viol, ung_pass, violated, cap: int = 8) -> "tuple[list, dict]":
    """Points the execution probe should re-verdict, most decision-relevant first:
    VIOLATED (non-structural) -> UNVERIFIABLE because evidence was truncated/missing -> value-shaped
    SATISFIED (ungrounded first). Returns (pids, reason-by-pid)."""
    txt = dict(checklist)
    reasons: dict = {}
    for p in violated:
        if p not in struct_viol:
            reasons[p] = "violated"
    for p, v, body in verdicts:
        if v == "UNVERIFIABLE" and p not in reasons and _TRUNC_RE.search(body or ""):
            reasons[p] = "unverifiable_truncated"
    ung = {p for p, _ in ung_pass}
    for p, v, _ in sorted(verdicts, key=lambda t: (t[0] not in ung, t[0])):
        if v == "SATISFIED" and p not in reasons and _value_shaped(txt.get(p, "")):
            reasons[p] = "value_shaped_pass" + ("_ungrounded" if p in ung else "")
    pids = list(reasons)[:cap]
    return pids, {p: reasons[p] for p in pids}


def run(ctx: flow.Ctx) -> dict:
    env, rp = ctx.env, ctx.repo_path
    inst = dict(ctx.instance)
    for fld in ("requirements", "interface"):
        inst[fld] = sp._unquote_spec_field(inst.get(fld) or "")
    patch = ctx.patches.get("refined") or ctx.patches.get("initial") or ""
    meta: dict = {}
    if not patch.strip() or not (inst["requirements"] or inst["interface"]):
        ctx.log("[audit] nothing to audit (empty patch or no spec)")
        ctx.record["audit"] = {"skipped": True}
        return meta
    ps = _sub._clip(ctx.problem_statement, 6000)
    s = flow.Session(ctx, "audit", system_prompt=_sub._SUBAGENT_SYSTEM)
    # A1 requirements (model) + deterministic checklist (harness)
    checklist = list(sp._audit_checklist(inst, ctx.findings))
    r1 = None
    if not (True and len(checklist) >= 6):
        r1 = s.step_structured("requirements", A1_PROMPT.format(ps=ps, req=inst["requirements"] or "(none)",
                                                                iface=inst["interface"] or "(none)"),
                               tools=flow.NO_TOOLS, max_turns=2, schema=REQ_SCHEMA)
    have = " ".join(t for _, t in checklist).lower()
    # Model-extracted points continue the I* numbering: the stock verdict parser and the
    # grounding / probe machinery only recognise R*, I* and W* ids.
    n_i = sum(1 for pid, _ in checklist if pid.startswith("I"))
    model_pids: set = set()
    max_model = 4 if True else 8
    for p in ((r1.structured_output if r1 else None) or {}).get("points") or []:
        t = (p.get("text") or "").strip()
        if len(t) < 15 or t.lower()[:60] in have or len(model_pids) >= max_model:
            continue
        n_i += 1
        checklist.append((f"I{n_i}", f"{t}  [spec: \"{(p.get('quote') or '')[:200]}\"]"))
        model_pids.add(f"I{n_i}")
    meta["checklist"] = checklist
    if ctx.baseline:
        return _run_baseline(ctx, s, inst, patch, checklist, meta)
    pack = sp._audit_evidence_pack(env, rp, patch, checklist, ctx.findings, cap=14000 if True else 16000)
    corpus = pack + "\n" + patch
    # A2 simulate (no tools)
    prompt = flow.lang_text(sp._AUDIT_PURE_PROMPT).format(
        checklist="\n".join(f"[{pid}] {txt}" for pid, txt in checklist),
        diff=_diff_budget(patch, 14000 if True else 20000),
        pack=pack, mech="(mechanical audit disabled in this pipeline)", problem_statement=ps)
    r2 = s.step("simulate", prompt, tools=flow.NO_TOOLS, max_turns=2)
    text = r2.submission or ""
    verdicts = sp._parse_audit_verdicts(text)
    lazy = sp._lazy_passes(verdicts)
    ung_pass = sp._ungrounded_passes(verdicts, corpus)
    ung_viol = sp._ungrounded_verdicts(verdicts, corpus, "VIOLATED")
    have_v = {p for p, _, _ in verdicts}
    missing = [pid for pid, _ in checklist if pid not in have_v]
    reask = sorted({p for p, _ in ung_viol} | set(missing))
    meta["lean"] = {k: True for k in ("REASK_MIN", "TRIM")}
    meta["first"] = {"n_verdicts": len(verdicts), "violated": [p for p, v, _ in verdicts if v == "VIOLATED"],
                     "lazy": lazy, "ungrounded_pass": [p for p, _ in ung_pass], "ungrounded_viol": [p for p, _ in ung_viol],
                     "missing": missing}
    for rnd in range(2):
        if not reask:
            break
        note = REASK_NOTE.format(pids=", ".join(reask))
        if missing:
            note += (f"\nThe points {', '.join(missing)} received NO verdict block at all -- every checklist "
                     "point needs its own `[<id>] VERDICT: ...` block with EVIDENCE and SIMULATION.\n")
        r3 = s.step("reask" if rnd == 0 else "reask2", note, tools=flow.NO_TOOLS, max_turns=2)
        text2 = r3.submission or ""
        v2 = {p: (v, b) for p, v, b in sp._parse_audit_verdicts(text2)}
        merged = [(p, v2[p][0], v2[p][1]) if p in v2 else (p, v, b) for p, v, b in verdicts]
        merged += [(p, v2[p][0], v2[p][1]) for p in missing if p in v2]
        verdicts = merged
        have_v = {p for p, _, _ in verdicts}
        missing = [pid for pid, _ in checklist if pid not in have_v]
        reask = sorted(missing)   # second round only chases points still without a verdict
        meta.setdefault("reask_rounds", []).append({"asked": len(v2), "still_missing": missing})
    still = {p for p, _ in sp._ungrounded_verdicts(verdicts, corpus, "VIOLATED")}
    verdicts = [(p, ("UNVERIFIABLE" if (v == "VIOLATED" and p in still) else v), b) for p, v, b in verdicts]
    meta["reask"] = {"dropped_ungrounded_violated": sorted(still), "missing_after": missing}
    violated = [p for p, v, _ in verdicts if v == "VIOLATED"]
    # executed structural facts (module-level existence) decide their I-points outright
    facts = structural_facts(ctx, checklist, patch)
    meta["structural"] = {p: d for p, (ok, d) in facts.items()}
    struct_viol = [p for p, (ok, _) in facts.items() if not ok]
    struct_ok = {p for p, (ok, _) in facts.items() if ok}
    violated = [p for p in violated if p not in struct_ok] + [p for p in struct_viol if p not in violated]
    # execution probe gate: confirm VIOLATED verdicts off the real output
    confirmed = list(struct_viol)
    probe_pids, probe_reasons = _probe_candidates(verdicts, checklist, struct_viol, ung_pass, violated)
    probe_confirmed: list = []
    if probe_pids:
        orig = _sub._make_agent_query_fn
        try:
            _probe_note = ("\n\nPROBE DISCIPLINE: the script must exercise the requirement through the ENTRY POINT "
                           "the requirement is about (the module/function/class the user or test would call), on a "
                           "complete, realistic input -- not a helper in isolation. A helper's partial output is not "
                           "evidence about the module's behaviour. For a point that names a VALUE (a literal, a return, "
                           "a raise, a field), read that value the way the CALLER or a test would observe it -- through "
                           "the public accessor / the returned object / the model's data() -- and print THAT; printing the "
                           "literal you constructed or the source tuple proves nothing (a framework may convert it).\n")
            _qf = s.qf("probe")
            replies: list = []

            def _wrapped(prompt, **kw):
                out = _qf(prompt + _probe_note + (_probe_extra[0] if _probe_extra else ""))
                replies.append(out)
                return out
            _probe_extra: list = []
            _sub._make_agent_query_fn = lambda agent, on_usage=None: _wrapped
            rev = sp._execution_grounded_reverdict(env, rp, SimpleNamespace(), probe_pids[:8], checklist, patch, pack)
            if not rev:
                # the stock device returns "" silently when the probe script exits non-zero or prints no
                # PROBE[...] line (b2-fix qutebrowser-305e7c96 attempt 1: the author mocked PyQt5 and the
                # script crashed). Run the authored script once more ourselves to capture WHY, then give
                # the author one retry with the error tail.
                why = _probe_failure(env, rp, replies[0] if replies else "", patch)
                meta["probe_first_failure"] = why[:1500]
                if why:
                    ctx.log(f"[audit] probe attempt 1 failed: {why[:200]!r} -- retrying once with the error")
                    _probe_extra.append(
                        "\n\nYOUR PREVIOUS PROBE SCRIPT FAILED. Its output ended with:\n" + why[-1500:]
                        + "\nWrite a NEW script that fixes this. Rules: do NOT mock or stub framework/library modules "
                        "(PyQt, Django, Flask, web.py, ...) -- the container has the real dependencies installed and "
                        "the test-suite runs against them (QT_QPA_PLATFORM=offscreen is set); import the real modules and "
                        "only stub the minimal application objects the entry point needs. Wrap EVERY point in try/except "
                        "so the script always exits 0 and always prints one `PROBE[<id>]:` line per point.\n")
                    replies.clear()
                    rev = sp._execution_grounded_reverdict(env, rp, SimpleNamespace(), probe_pids[:8], checklist, patch, pack)
                    if not rev:
                        meta["probe_second_failure"] = _probe_failure(env, rp, replies[0] if replies else "", patch)[:1500]
        finally:
            _sub._make_agent_query_fn = orig
        rv = {p: v for p, v, _ in sp._parse_audit_verdicts(rev or "")}
        probe_confirmed = [p for p in probe_pids if rv.get(p) == "VIOLATED"]
        confirmed += probe_confirmed
        meta["probe"] = {"asked": probe_pids[:8], "reasons": probe_reasons, "reverdicts": rv, "confirmed": confirmed,
                         "no_probe_output": not bool(rev)}
        if not rev:
            ctx.log("[audit] probe could not be authored/run -- VIOLATED verdicts stay advisory")
    meta["verdicts"] = [(p, v) for p, v, _ in verdicts]
    triggering = [p for p in confirmed if (p.startswith("I") and p not in model_pids) or p in struct_viol or p == "W1"
                  or (PROBE_TRIGGERS and p in probe_confirmed)]
    advisory = [p for p in confirmed if p not in triggering]
    meta["trigger"] = {"scope": AUDIT_FIX_SCOPE, "triggering": triggering, "advisory": advisory}
    ctx.log(f"[audit] {len(verdicts)} verdicts, violated={violated}, confirmed={confirmed}, "
            f"triggering={triggering} advisory={advisory}")
    fixed = patch
    if triggering:
        confirmed = triggering + advisory   # advisory points are shown to the fix step, below the triggers
        snap = sp._tree_snapshot(env, rp)
        smoke_before = sp._import_smoke(env, rp, patch)[0]
        by = {p: (v, b) for p, v, b in verdicts}
        txt = {p: t for p, t in checklist}
        viol_txt = "\n\n".join(f"[{p}] {txt.get(p, '')}\nEVIDENCE:\n{(by.get(p) or ('', ''))[1][:1200]}"
                                + (f"\nSTRUCTURAL FACT (verified on the patched tree; NOT negotiable): {meta['structural'][p]}. "
                                   f"The specification requires this symbol by name. The hidden tests of that package reference "
                                   f"it by exactly this name, and in Go one missing symbol makes the whole test package fail to "
                                   f"compile. Declare it with that exact name (in the file the requirement names, if any), "
                                   f"implementing what the requirement describes, and keep existing callers working."
                                   if p in meta.get("structural", {}) and str(meta["structural"][p]).startswith("GO_NAMED_SYMBOL")
                                   else f"\nSTRUCTURAL FACT (AST-verified on the patched tree; NOT negotiable): "
                                   f"{meta['structural'][p]}. The specification refers to this symbol as a FUNCTION "
                                   f"('function `X`'), so the hidden tests access it as `module.X`; a def nested "
                                   f"inside another function is not reachable that way. 'No new interfaces are "
                                   f"introduced' is not a reason to keep it nested: making an existing function "
                                   f"importable at module level adds no new interface." if p in meta.get("structural", {}) else "")
                                for p in triggering)
        if any(str(meta.get("structural", {}).get(p, "")).startswith("GO_NAMED_SYMBOL") for p in triggering):
            viol_txt = ("HARD RULE: a STRUCTURAL FACT below means a symbol the specification requires BY NAME does not "
                        "exist anywhere in the repository. It outranks any reading of the PR prose; declare it with exactly "
                        "that name and implement it. Do not rename, inline or skip it.\n\n" + viol_txt)
        elif any(p in meta.get("structural", {}) for p in triggering):
            viol_txt = ("HARD RULE: a STRUCTURAL FACT below is an AST-verified violation of how the specification "
                        "names the symbol. It outranks any reading of the PR prose; implement it (define the symbol at "
                        "module level, keeping the existing behaviour and any inner usage working, e.g. by having the "
                        "enclosing function call the module-level one). Do not refuse it on precedence grounds.\n\n"
                        + viol_txt)
        if advisory:
            viol_txt += ("\n\n=== ADVISORY (requirement-sentence verdicts; no authority to change behaviour on their own -- "
                         "act only if a triggering point above requires the same change) ===\n"
                         + "\n\n".join(f"[{p}] {txt.get(p, '')}" for p in advisory))
        prompt = _render(flow.lang_text(sp.AUDIT_FIX_INSTANCE), task=ctx.instance["problem_statement"],
                         localization_reference=f"=== ROOT CAUSE (from localization) ===\n{getattr(ctx.findings, 'root_cause', '')}\nFix ONLY the violations listed below; do not re-audit other sites.",
                         diff_so_far=_sub._clip(patch, 4000), violations=viol_txt, phase_body=flow.lang_text(sp.AUDIT_FIX_BODY))
        prompt = flow.strip_submit_protocol(prompt) + flow.RUNTIME_NOTE.format(repo_path=rp)
        r4 = s.step("fix", prompt, tools=flow.FULL_TOOLS, max_turns=FIX_TURNS)
        exit_status = {"Completed": "Submitted"}.get(r4.exit_status, r4.exit_status)
        after, gate, _m, _s = sp._gate_phase_mutation(
            env, rp, inst, ctx.findings, "audit_fix", exit_status, patch, 0, smoke_before, snap_before=snap,
            check_tests_on_conflict=True, revert_on_test_regression=True, log=ctx.log)
        after, gate = _enforce_executed_check(ctx, gate, snap, patch, after)
        fixed = after
        meta["fix"] = {"exit": r4.exit_status, "turns": r4.n_calls, "gate": gate}
    (ctx.inst_out / "4_audited_patch.diff").write_text(fixed)
    ctx.patches["audited"] = fixed
    meta["changed"] = fixed != patch
    s.save(ctx.inst_out / "3_audit.traj.json", meta=meta)
    ctx.record["audit"] = {**meta, "steps": s.steps, "cost": s.cost, "n_calls": s.n_calls}
    return meta


A_BASELINE_PROMPT = """\
AUDIT STEP 2 (baseline arm) -- CHECK THE PATCH AGAINST THE CHECKLIST BY RUNNING CODE, THEN FIX.

A fix for the issue is applied in the repository (diff below). For EACH checklist point, verify
whether the patched code satisfies it by EXECUTING the code (a quick `python -c` probe, a small
/tmp script, or an existing test); do not decide a point by reading alone. Report one line per
point: `[<id>] SATISFIED|VIOLATED -- <what you ran and observed>`. Then FIX every VIOLATED point
by editing the source (never tests). Do not change anything a point does not require. Stop when
every point is SATISFIED or you have explained why a point cannot be verified.

=== CHECKLIST ===
{checklist}

=== DIFF ===
{diff}
"""


def _run_baseline(ctx: flow.Ctx, s: flow.Session, inst: dict, patch: str, checklist, meta: dict) -> dict:
    env, rp = ctx.env, ctx.repo_path
    snap = sp._tree_snapshot(env, rp)
    smoke_before = sp._import_smoke(env, rp, patch)[0]
    prompt = A_BASELINE_PROMPT.format(checklist="\n".join(f"[{pid}] {txt}" for pid, txt in checklist),
                                      diff=_sub._clip(patch, 6000)) + flow.RUNTIME_NOTE.format(repo_path=rp)
    r = s.step("verify_fix", prompt, tools=flow.FULL_TOOLS, max_turns=FIX_TURNS)
    exit_status = {"Completed": "Submitted"}.get(r.exit_status, r.exit_status)
    after, gate, _m, _s = sp._gate_phase_mutation(
        env, rp, inst, ctx.findings, "audit_fix", exit_status, patch, 0, smoke_before, snap_before=snap,
        check_tests_on_conflict=True, revert_on_test_regression=True, log=ctx.log)
    after, gate = _enforce_executed_check(ctx, gate, snap, patch, after)
    meta["baseline"] = True
    meta["verdicts_text"] = (r.submission or "")[:4000]
    meta["fix"] = {"exit": r.exit_status, "turns": r.n_calls, "gate": gate}
    (ctx.inst_out / "4_audited_patch.diff").write_text(after)
    ctx.patches["audited"] = after
    meta["changed"] = after != patch
    s.save(ctx.inst_out / "3_audit.traj.json", meta=meta)
    ctx.record["audit"] = {**meta, "steps": s.steps, "cost": s.cost, "n_calls": s.n_calls}
    return meta
