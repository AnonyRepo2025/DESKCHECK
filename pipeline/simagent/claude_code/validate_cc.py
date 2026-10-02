"""Milestone 5: the validate phase on Claude Code -- reason THEN act.

The stock pipeline enforces pre-validation reasoning with a turn-rollback hook (delete the
step that starts writing a test, inject VALIDATE_REMINDER, redo, score the redone reasoning
with default_newtest_compliance_checker, retry). A headless session cannot redo a turn, so the
same contract is split in two:

  1. REASON (tools-off query): produce the input -> expected-output table, with the corner
     cases mined explicitly, from the issue/spec/diff and the seeds the pipeline computed
     (traced inputs, required input classes, spec examples, new-branch / count probes).
     Scored by the SAME checker (>=3 of 4 criteria) plus a structural check (>=3 rows,
     >=2 corner-case rows, one FAILS-ON-UNPATCHED line per row); retried with the checker's
     own feedback up to VALIDATE_SIM_RETRIES times.
  2. ACT (Claude Code session): the accepted table is injected into the validate prompt as a
     BINDING block; the session writes the test that hard-codes those pairs and runs it.
     Two in-container hooks guard the act: PostToolUse(Bash) records whether an asserting
     probe / test run happened; Stop refuses (at most CC_STOP_BLOCKS times) to end the session
     until one has.

Everything after the session (discharge rounds, mutation gate with require_improvement,
regression guard) is the stock pipeline behind the seams.
"""
from __future__ import annotations

import json
import os
import re
import shlex

from simagent import pipeline as sp

_sub = sp._sub

CC_STOP_BLOCKS = int(os.getenv("CCPIPE_STOP_BLOCKS", "2"))
CC_VALIDATE_MIN_ROWS = int(os.getenv("CCPIPE_VALIDATE_MIN_ROWS", "3"))
CC_VALIDATE_MIN_CORNERS = int(os.getenv("CCPIPE_VALIDATE_MIN_CORNERS", "2"))

REASON_PROMPT = """\
<pr_description>
{task}
</pr_description>

{fix_reference}

=== PRE-VALIDATION REASONING (this step writes NO test and runs NO code) ===
A fix for the issue above is applied (diff and reference material above). Before any test is
written, design the validation by REASONING. The real grading tests are HIDDEN: your cases must
GENERALIZE from the issue and the specification, not from one reproduction.

STEP A -- INPUT -> EXPECTED-OUTPUT TABLE. For EACH case give:
  1. INPUT           -- the exact call / args / URL / value (a concrete input, written out)
  2. EXPECTED OUTPUT -- the exact literal the code SHOULD return / raise (write the literal;
                        no "the correct value", no paraphrase). If the issue or spec displays an
                        exact expected output, COPY it character for character.
  3. SOURCE          -- the ISSUE/spec sentence it comes from, or the corner-case class.
Derive every EXPECTED OUTPUT from the issue/spec by reasoning -- never from what the patched
code happens to produce (that is a tautology).
Each row uses the MINIMAL input that isolates its requirement: include only the fields / arguments
the requirement needs. A row about case-insensitive matching, wildcards, alternate names, defaults,
etc. must NOT also carry a field that would make the input succeed by a different path (e.g. dates
that let a surname lookup match) -- such a row passes for the wrong reason and proves nothing.

STEP B -- MINE CORNER CASES (mandatory). Enumerate the corner-case input classes the issue/spec
and the PATCHED code imply -- empty / zero / None / missing, single vs multiple elements,
duplicates, boundary and maximum values, type extremes, unresolvable / error paths, and an input
reaching EACH conditional branch the fix introduces -- then add at least {min_corners} such rows
to the table (SOURCE: corner case -- <class>). Hidden grading tests live disproportionately in
these classes; a table holding only the happy-path example does not generalize.

STEP C -- ORACLE CHECK. For EACH row state in one line why it FAILS on the un-patched code and
PASSES on the patched code (name the pre-fix behaviour it would exhibit).

Answer in EXACTLY this layout (plain text, no code fences):

CORNER-CASE CLASSES:
- <class>: <why it applies to this fix>
...

TABLE:
| # | INPUT | EXPECTED OUTPUT | SOURCE |
| 1 | <input> | <literal> | <issue sentence / corner case -- class> |
...

FAILS-ON-UNPATCHED:
- 1: <pre-fix behaviour> -> <post-fix behaviour>
...
"""

RETRY_SUFFIX = """

[REASONING INTERVENTION -- retry {attempt}/{max_retries}]
Your previous answer did NOT satisfy this step. Specifically, it still needs to:
{feedback}.
Rewrite the COMPLETE answer in the required layout (CORNER-CASE CLASSES / TABLE /
FAILS-ON-UNPATCHED). Your previous answer, for reference:
----
{previous}
----
"""

ACT_BLOCK = """

=== YOUR INPUT -> EXPECTED-OUTPUT TABLE (written by you in the pre-validation reasoning step; BINDING) ===
{table}
=== END OF TABLE ===
Write the test(s) from THIS table: every row becomes an assertion that HARD-CODES the literal
(input -> expected) pair and drives the REAL code under test through its actual entry point.
Do not drop the corner-case rows. If a row's expected value turns out to contradict the
specification on closer reading, say so explicitly and correct the row -- never silently
replace an expected literal with whatever the code returns.
"""

CC_VALIDATE_REMINDER = """\
[REASONING INTERVENTION -- pre-new-test phase reasoning]

NOTE: the real grading tests are HIDDEN and are NOT in this repo. A test that merely passes
locally proves nothing -- your tests must GENERALIZE from the issue, not from your one
reproduction.

STEP A (DONE): your input -> expected-output table, corner cases included, is in the BINDING
block above. Do not redo the analysis; if you must amend a row, state the amendment and why.

STEP B -- ACT: write the test whose assertions HARD-CODE those literal (input -> expected) pairs
AND drive the REAL code under test through its actual entry point (import & call the real
function / method / view, or a real request via the test client). NEVER re-implement / copy the
patched code into the test, NEVER compare a local "old vs new" pair, and NEVER mock the code
path you are validating -- those are tautologies that pass even when the fix is wrong.

PROBE DISCIPLINE + EXACTNESS (applies to EVERY probe you execute in this phase, including
one-off `python -c` branch/corner checks): commit the expected literal BEFORE running -- the
probe command itself must `assert result == expected, result`; a print() with no adjacent
assert is NOT validation, and an eyeballed repr hides near-misses. Hidden tests compare
EXACTLY: '  Pearson' != 'Pearson' -- if any returned string element differs from its .strip(),
or an empty-string element appears in a returned list, that is a SOURCE bug -- fix the source,
then re-run the probe.

HARNESS FIDELITY: the hidden tests are the repository's OWN test files for the changed code, updated.
Before writing, open the existing test file(s) for the module / component you changed and write your
test in THAT harness: the same setup and fixtures, the same mocks and spies, the same render / client
helpers, and the same way of driving interactions and async work. Drive a repeated or concurrent
action exactly as a user or the existing tests would -- e.g. two clicks back to back inside one
`act(...)` / the same tick, two calls before the first resolves -- never with an artificial delay
between them that lets state settle (ce554276: a 50 ms gap passed; the hidden double click did not).

Write the test ONCE from the table; the moment it passes on the patched code, stop -- do not
re-explore, rewrite, or run the repository's full test suite. If it FAILS, the source is at fault
unless the specification says otherwise: do not edit the test until it passes.
"""

# --- in-container hooks (bash + grep only: Go/JS images may lack python) --------------------
_POST_HOOK = r'''#!/bin/bash
# PostToolUse(Bash): record that an asserting probe / test run was executed.
IN=$(cat)
printf '%s\n' "$IN" >> "$CC_HOOK_DIR/post_inputs.jsonl" 2>/dev/null
if printf '%s' "$IN" | grep -q '"tool_name": *"Bash"'; then
  if printf '%s' "$IN" | grep -qiE 'assert|pytest|unittest|go test|npm test|npx jest|jest |mocha|vitest|require\.(Equal|Len|NoError|True|False)|expect\(|toBe|toEqual'; then
    touch "$CC_HOOK_DIR/tested"
  fi
fi
exit 0
'''

_STOP_HOOK = r'''#!/bin/bash
# Stop: refuse to end the validate session until an asserting probe / test run happened.
IN=$(cat)
printf '%s\n' "$IN" >> "$CC_HOOK_DIR/stop_inputs.jsonl" 2>/dev/null
if printf '%s' "$IN" | grep -q '"stop_hook_active": *true'; then exit 0; fi
[ -f "$CC_HOOK_DIR/tested" ] && exit 0
N=0; [ -f "$CC_HOOK_DIR/blocks" ] && N=$(cat "$CC_HOOK_DIR/blocks")
if [ "$N" -ge "$CC_STOP_BLOCKS" ]; then exit 0; fi
echo $((N+1)) > "$CC_HOOK_DIR/blocks"
cat >&2 <<'MSG'
[validate guard] This session has not EXECUTED any asserting probe or test yet. Validation
requires running the real code: write the test from your BINDING table (or a probe with an
explicit `assert result == expected, result`) and RUN it with the Bash tool. Then finish.
If the test fails, the source is at fault -- fix the source, do not weaken the test.
MSG
exit 2
'''


def _table_stats(text: str) -> tuple[int, int, int]:
    rows = [l for l in text.splitlines() if l.strip().startswith("|") and l.count("|") >= 4]
    rows = [l for l in rows if not re.match(r"^\|\s*#\s*\|", l.strip()) and not re.match(r"^\|[\s\-:|]+\|$", l.strip())]
    corner = [l for l in rows if re.search(r"corner|edge|boundar|empty|none|zero|missing|maximum|duplicate|error path|invalid|negative|single", l, re.I)]
    fails = len(re.findall(r"^\s*-\s*\d+\s*:", text, re.M))
    return len(rows), len(corner), fails


def structural_check(text: str) -> tuple[bool, str]:
    n, c, f = _table_stats(text)
    missing = []
    if n < CC_VALIDATE_MIN_ROWS:
        missing.append(f"give at least {CC_VALIDATE_MIN_ROWS} table rows (found {n})")
    if c < CC_VALIDATE_MIN_CORNERS:
        missing.append(f"include at least {CC_VALIDATE_MIN_CORNERS} corner-case rows with SOURCE 'corner case -- <class>' (found {c})")
    if f < max(1, n // 2) and os.getenv("CCFLOW_TABLE_LOOSE", os.getenv("CCFLOW_LEAN", "1")) not in ("1", "true", "yes"):
        missing.append("give one FAILS-ON-UNPATCHED line per row ('- <row>: <pre-fix> -> <post-fix>')")
    return (not missing), "; ".join(missing)


def reason_step(ask, *, task: str, fix_reference: str, max_retries: int, checker, log=print) -> dict:
    """Run the tools-off reasoning query with checker-driven retries. Returns
    {table, attempts:[{attempt, complied, feedback, answer_chars, cost}], complied}."""
    base = REASON_PROMPT.format(task=task, fix_reference=fix_reference, min_corners=CC_VALIDATE_MIN_CORNERS)
    prompt = base
    attempts = []
    prev = ""
    best = ""
    for attempt in range(0, max_retries + 1):
        res = ask(prompt, f"validate_reason{attempt}")
        text = (res.submission or "").strip()
        ok1, fb1 = checker(text, "", "") if text else (False, "the answer is empty")
        ok2, fb2 = structural_check(text)
        complied = bool(ok1 and ok2)
        feedback = "; ".join(x for x in (("" if ok1 else fb1), ("" if ok2 else fb2)) if x)
        attempts.append({"attempt": attempt, "complied": complied, "feedback": feedback,
                         "answer_chars": len(text), "cost": res.cost, "exit_status": res.exit_status})
        log(f"[validate:reason] attempt {attempt}: {'COMPLIED' if complied else 'retry -- ' + feedback[:160]}")
        if text:
            best = text
        if complied:
            break
        prev = text
        prompt = base + RETRY_SUFFIX.format(attempt=attempt + 1, max_retries=max_retries,
                                            feedback=feedback or "produce the required analysis",
                                            previous=prev[-6000:])
    return {"table": best, "attempts": attempts, "complied": bool(attempts and attempts[-1]["complied"])}


def install_hooks(env, label: str, hook_dir_in: str, put_file) -> tuple[str, ...]:
    """Write the two hook scripts into the container; return the extra `claude` args."""
    put_file(env, f"{hook_dir_in}/post.sh", _POST_HOOK)
    put_file(env, f"{hook_dir_in}/stop.sh", _STOP_HOOK)
    env.execute({"command": f"chmod +x {shlex.quote(hook_dir_in)}/*.sh && rm -f {shlex.quote(hook_dir_in)}/tested "
                            f"{shlex.quote(hook_dir_in)}/blocks"}, timeout=30)
    wrap = lambda s: f"CC_HOOK_DIR={shlex.quote(hook_dir_in)} CC_STOP_BLOCKS={CC_STOP_BLOCKS} {hook_dir_in}/{s}"
    settings = {"hooks": {
        "PostToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": wrap("post.sh")}]}],
        "Stop": [{"hooks": [{"type": "command", "command": wrap("stop.sh")}]}],
    }}
    return ("--settings", shlex.quote(json.dumps(settings)))


def hook_outcome(env, hook_dir_in: str) -> dict:
    out = env.execute({"command": f"cd {shlex.quote(hook_dir_in)} 2>/dev/null && "
                                  "echo tested=$([ -f tested ] && echo 1 || echo 0) blocks=$(cat blocks 2>/dev/null || echo 0) "
                                  "posts=$(wc -l < post_inputs.jsonl 2>/dev/null || echo 0)"}, timeout=30)
    m = re.search(r"tested=(\d) blocks=(\d+) posts=(\d+)", out.get("output", "") or "")
    return {"tested": bool(int(m.group(1))), "stop_blocks": int(m.group(2)), "bash_calls": int(m.group(3))} if m else {}


# ---------------------------------------------------------------------------------------------
# Literal preservation (b2 qutebrowser-305e7c96, both rolls): the BINDING table said the third
# tuple field is `None`; the generated test failed (Qt returns ''), and the agent rewrote the
# expectations (`Edit` on its own test / `sed -i "s/, None)/, '')/g"`) instead of the source. The
# rule against that is prompt-only; this makes it mechanical: table literals that an Edit / sed
# removed from a test the session wrote are reported, and the phase runs one bounded follow-up
# that treats the SOURCE as at fault.
_LIT_RE = re.compile(r"`([^`]{1,120})`|\b(None|True|False)\b|('(?:[^'\\]|\\.){1,80}')|(\"(?:[^\"\\]|\\.){1,80}\")")
_LIT_STOP = {"None", "True", "False", "''", '""', "[]", "{}", "()", "0", "1"}


def table_literals(table: str) -> "list[str]":
    """Expected-output literals of the BINDING table (3rd column), longest first."""
    lits: set = set()
    for line in (table or "").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] in ("#", "") or set(cells[0]) <= set("-"):
            continue
        exp = cells[2]
        for m in _LIT_RE.finditer(exp):
            lit = next(g for g in m.groups() if g)
            lit = lit.strip()
            if len(lit) >= 2 and lit not in _LIT_STOP:
                lits.add(lit)
        # bare None/True/False inside a longer literal count too (", None)")
        for kw in ("None", "True", "False"):
            if re.search(r"\b" + kw + r"\b", exp):
                lits.add(kw)
    return sorted(lits, key=len, reverse=True)


_PASS_RE = re.compile(r"\b\d+ passed\b|\bOK\b|ALL TESTS PASSED|All tests passed|\bPASSED\b(?!.*FAILED)", re.S)
_SENTINEL_LITS = ("None", "True", "False", "''", '""', "[]", "{}", "()")
_BARE_DATA_RE = re.compile(r"^[\(\[\{'\"]|^(?:expected\w*|result_expected|want)\s*=|\bassert")
_FAIL_RE = re.compile(r"AssertionError|\bFAILED\b|TEST FAILED|Traceback \(most recent|Expected:.*Got:|Exit code [1-9]", re.S)


def _same_site_rewrite(old: str, new: str, literals: "list[str]", any_literal: bool = False):
    """A line of `old` holding a table literal survives in `new` with the same prefix but WITHOUT the
    literal -> the expectation was rewritten in place (not a restructured / deleted block)."""
    new_lines = [l.strip() for l in (new or "").splitlines()]
    old_lines = (old or "").splitlines()
    for i, line in enumerate(old_lines):
        st = line.strip()
        if st.startswith("#") or "INPUT:" in st or st.startswith('"""') or st.startswith("'''"):
            continue
        # expectation context only: the line or one of the 8 lines above it asserts / names the
        # expected value (an input-construction or fixture line is not a rewritten expectation)
        ctxt = "\n".join(old_lines[max(0, i - 12):i + 1])
        if not _EXPECT_CTX_RE.search(ctxt) or _INPUT_LINE_RE.match(st):
            continue
        lits = list(literals)
        if any_literal and _BARE_DATA_RE.search(st):
            # after a failing run: the sentinel literals (None/True/False/empty) on an assertion /
            # expected / bare-data line count too -- the class a framework silently converts
            # (None -> ''), which is exactly what a test must not be bent to accept
            lits += [t for t in _SENTINEL_LITS if re.search(r"(?<![\w'\"])" + re.escape(t) + r"(?![\w'\"])", line)]
            lits = list(dict.fromkeys(lits))
        for lit in lits:
            if lit not in line:
                continue
            prefix = line.split(lit, 1)[0].strip()
            if len(prefix) < 8:
                continue
            for nl in new_lines:
                if nl.startswith(prefix) and lit not in nl:
                    rest = nl[len(prefix):].lstrip(" =:")
                    # only a literal -> literal swap counts (None -> '' ; 'a' -> 'b'); a swap to an
                    # identifier / call is a restructuring of the test, not a rewritten expectation
                    if _REPL_LIT_RE.match(rest):
                        return lit, line.strip()
    return None


_EXPECT_CTX_RE = re.compile(r"\bassert\b|\bexpected\w*|assert(?:Equal|Is|True|False|In|Raises|Dict|List|Tuple)|==|PROBE\["
                            r"|\bcheck\w*\(|_check\w*\(|\bverify\w*\(|\bshould\b")
_INPUT_LINE_RE = re.compile(r"^(?:result|res|out|output|actual|got|input|inp|data|args|kwargs|params|payload|cfg|config|self\.\w+)\s*=[^=]"
                            r"|^\w[\w.]*\.val\b|^\w+\([^=]*$")
_REPL_LIT_RE = re.compile(r"^(?:'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|None|True|False|-?\d+(?:\.\d+)?|\[|\{|\()")


def literal_rewrites(stream_path: str, literals: "list[str]") -> "list[dict]":
    """Edits / sed rewrites in the session stream that removed a table literal from a test or probe
    file the session itself wrote. Returns [{file, literal, via, snippet}]."""
    if not stream_path or not literals or not os.path.exists(stream_path):
        return []
    written: set = set()
    hits: list = []
    last_failed = False   # the most recent tool result showed a failing assertion / test
    try:
        for line in open(stream_path, errors="replace"):
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") == "user":
                for b in (ev.get("message") or {}).get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        c = b.get("content")
                        txt = c if isinstance(c, str) else json.dumps(c)
                        if _FAIL_RE.search(txt or ""):
                            last_failed = True          # sticky until a run passes
                        elif _PASS_RE.search(txt or ""):
                            last_failed = False
                continue
            if ev.get("type") != "assistant":
                continue
            for b in (ev.get("message") or {}).get("content") or []:
                if b.get("type") != "tool_use":
                    continue
                name, inp = b.get("name"), b.get("input") or {}
                if name == "Write" and inp.get("file_path"):
                    written.add(inp["file_path"])
                elif name == "Edit" and inp.get("file_path"):
                    fp = inp["file_path"]
                    old, new = inp.get("old_string") or "", inp.get("new_string") or ""
                    if not (fp in written or re.search(r"(^|/)(tests?|testing)/|/tmp/|test_[^/]*\.py$|_test\.(py|go)$", fp)):
                        continue
                    h = _same_site_rewrite(old, new, literals, any_literal=last_failed)
                    if h:
                        hits.append({"file": fp, "literal": h[0], "via": "Edit" + (" after failing run" if last_failed else ""),
                                     "snippet": h[1][:160]})
                elif name == "Bash":
                    cmd = inp.get("command") or ""
                    if re.search(r"\bsed\b.*-i", cmd) or re.search(r"\bperl\b.*-p?i", cmd):
                        for m in re.finditer(r"s([/|#@])((?:\\.|(?!\1).)*)\1((?:\\.|(?!\1).)*)\1", cmd):
                            pat, rep = m.group(2), m.group(3)
                            cand = list(literals)
                            if last_failed:
                                cand += [next(g for g in mm.groups() if g).strip() for mm in _LIT_RE.finditer(pat)]
                            lit = next((l for l in cand if len(l) >= 2 and l in pat and l not in rep), None)
                            if lit:
                                hits.append({"file": "", "literal": lit, "via": "sed", "snippet": cmd[:200]})
                                break
    except Exception:
        return hits
    return hits


# --- G2 (Go batch 1): the regression-fix step must repair SOURCE, never existing *_test.go ----------------
# On 10/60 Go instances the regression guard saw an existing test stop compiling; the fix step edited the
# test in 7 of them (Go test edits are stripped from the shipped diff) and all 10 fixes were reverted.
_TEST_GUARD_HOOK = r"""#!/bin/bash
# PreToolUse: deny edits to *_test.go files that already exist in HEAD (the graders compile the ORIGINAL ones).
IN=$(cat)
printf '%s\n' "$IN" >> "$CC_HOOK_DIR/pre_inputs.jsonl" 2>/dev/null
TOOL=$(printf '%s' "$IN" | grep -o '"tool_name": *"[^"]*"' | head -1 | sed 's/.*: *"//; s/"$//')
CANDS=""
case "$TOOL" in
  Edit|Write|MultiEdit|NotebookEdit)
    CANDS=$(printf '%s' "$IN" | grep -o '"file_path": *"[^"]*"' | head -1 | sed 's/.*: *"//; s/"$//') ;;
  Bash)
    CMD=$(printf '%s' "$IN" | grep -o '"command": *"[^"]*' | head -1)
    if printf '%s' "$CMD" | grep -qE '(sed|perl)[^|;&]* -[a-zA-Z]*i|>|tee |mv |cp |rm |git (checkout|restore|apply)'; then
      CANDS=$(printf '%s' "$CMD" | grep -oE '[A-Za-z0-9_./-]+_test\.go' | sort -u)
    fi ;;
esac
for F in $CANDS; do
  case "$F" in *_test.go) ;; *) continue ;; esac
  REL=${F#"$CC_REPO"/}; REL=${REL#./}
  if git -C "$CC_REPO" cat-file -e "HEAD:$REL" 2>/dev/null; then
    cat >&2 <<MSG
[regression-fix guard] $REL is an EXISTING test file. Edits to existing *_test.go files are stripped from the
submitted diff and the graders compile the ORIGINAL file against your source, so changing it cannot fix this.
Fix the SOURCE instead: keep (or restore) the function / method signatures and types that this test calls,
and adapt the implementation internally.
MSG
    exit 2
  fi
done
exit 0
"""


# JS/TS variant (J1, 2026-09-20): same rule, the runner's own test-file conventions.
_JS_TEST_GUARD_HOOK = _TEST_GUARD_HOOK.replace(
    "*_test.go) ;; *) continue ;;",
    "*.test.js|*.test.jsx|*.test.ts|*.test.tsx|*-test.js|*-test.ts|*.spec.js|*.spec.ts|*.spec.tsx|*.snap) ;; *) continue ;;"
).replace(
    "[A-Za-z0-9_./-]+_test\\.go",
    "[A-Za-z0-9_./-]+([.-]test|\\.spec)\\.[cm]?[jt]sx?|[A-Za-z0-9_./-]+\\.snap"
).replace("*_test.go files", "existing test/snapshot files").replace(
    "the graders compile the ORIGINAL file against your source", "the graders run THEIR copy of the test against your source")


def install_test_guard(env, hook_dir_in: str, repo_path: str, put_file) -> "tuple[str, ...]":
    """PreToolUse hook that denies edits to pre-existing test files; returns the extra `claude` args."""
    hook = _JS_TEST_GUARD_HOOK if _sub.is_js() else _TEST_GUARD_HOOK
    put_file(env, f"{hook_dir_in}/testguard.sh", hook)
    env.execute({"command": f"chmod +x {shlex.quote(hook_dir_in)}/testguard.sh"}, timeout=30)
    cmd = f"CC_HOOK_DIR={shlex.quote(hook_dir_in)} CC_REPO={shlex.quote(repo_path)} {hook_dir_in}/testguard.sh"
    settings = {"hooks": {"PreToolUse": [{"matcher": "Edit|Write|MultiEdit|NotebookEdit|Bash",
                                          "hooks": [{"type": "command", "command": cmd}]}]}}
    return ("--settings", shlex.quote(json.dumps(settings)))


_GO_ERR_SYMBOL = [re.compile(r"in (?:argument to|call to|return argument)\s+(?:[\w.]+\.)?(\w+)"),
                  re.compile(r"(?:not enough|too many) arguments in call to\s+(?:[\w.]+\.)?(\w+)"),
                  re.compile(r"undefined: (?:[\w.]+\.)?(\w+)"),
                  re.compile(r"\.(\w+) undefined \(type"),
                  re.compile(r"assignment mismatch: .*? (?:[\w.]+\.)?(\w+)\(\) returns")]


def go_compile_errors(output: str, spec_text: str) -> "tuple[list[str], dict]":
    """Compiler error lines from a `go test` run and the broken symbols, each marked named-in-spec or not."""
    lines = []
    for ln in (output or "").splitlines():
        ln = ln.strip()
        if re.match(r"\S+\.go:\d+:\d+: ", ln) and ln not in lines:
            lines.append(ln)
    syms: dict = {}
    for ln in lines:
        for rx in _GO_ERR_SYMBOL:
            for m in rx.findall(ln):
                syms.setdefault(m, bool(re.search(r"\b" + re.escape(m) + r"\b", spec_text or "")))
    return lines[:20], syms


GO_BUILD_FIX_PROMPT = """\
REGRESSION GUARD (Go): your change makes EXISTING test file(s) fail to COMPILE:

{regressions}

Compiler errors:
{errors}

The graders compile the repository's ORIGINAL *_test.go files against your source; edits to existing
test files are stripped from the submitted diff and are blocked in this step. A package whose tests do
not compile fails EVERY test in it, including the ones this task is graded on.

Broken symbols and whether the specification names them:
{symbols}

For each symbol the specification does NOT name: restore the signature / type exactly as the existing
test calls it and adapt the implementation internally (e.g. keep the old parameter type and convert at
the boundary). Do NOT keep the old shape by adding a differently-named wrapper for the new shape: the
hidden tests call the ORIGINAL name, so a wrapper compiles and still fails them (navidrome-677d9947).
A symbol the specification DOES name is not yours to restore at all -- this step is not run for those.
Run `go vet` on the affected package(s) to confirm the test files compile again, then stop.
"""


# --- G4: a literal-guard follow-up must actually restore an exact assertion --------------------------------
_WEAK_MATCH_RE = re.compile(
    r"\b(?:require|assert)\.(?:Contains|ErrorContains|Regexp|Subset|NotEmpty)\b|strings\.(?:Contains|HasPrefix|HasSuffix)\("
    r"|\bassertIn\b|\.startswith\(|\.endswith\(|\bre\.(?:search|match)\(|\btoContain\(|\btoMatch\("
    r"|\bassert\s+[\"'][^\"']+[\"']\s+in\s+")


def weak_literal_assertions(env, hits: "list[dict]") -> "list[dict]":
    """Lines in the hit files that still assert a rewritten literal through a partial matcher (Contains,
    HasPrefix, `in`, ...) instead of equality."""
    out = []
    for h in hits:
        f, lit = h.get("file") or "", (h.get("literal") or "").strip().strip("\"'`")
        if not f or len(lit) < 3:
            continue
        try:
            txt = env.execute({"command": f"grep -nF -- {shlex.quote(lit)} {shlex.quote(f)} 2>/dev/null | head -20"},
                              timeout=30).get("output") or ""
        except Exception:
            continue
        for ln in txt.splitlines():
            if _WEAK_MATCH_RE.search(ln):
                out.append({"file": f, "literal": lit, "line": ln.strip()[:200]})
    return out


def passing_run_after_last_edit(stream_path: str) -> bool:
    """True when the session's last source/test edit is followed by a tool result that shows a passing run."""
    if not stream_path or not os.path.exists(stream_path):
        return False
    last_edit, last_pass, i = -1, -1, 0
    pass_re = re.compile(_PASS_RE.pattern + r"|^ok\s|\n---\s*PASS|\bPASS\b", re.S | re.M)
    for line in open(stream_path, errors="replace"):
        try:
            ev = json.loads(line)
        except Exception:
            continue
        i += 1
        for b in (ev.get("message") or {}).get("content") or []:
            if not isinstance(b, dict):
                continue
            if ev.get("type") == "assistant" and b.get("type") == "tool_use" and b.get("name") in ("Edit", "Write", "MultiEdit"):
                last_edit = i
            if ev.get("type") == "user" and b.get("type") == "tool_result":
                c = b.get("content"); txt = c if isinstance(c, str) else json.dumps(c)
                if pass_re.search(txt or "") and not _FAIL_RE.search(txt or "") and "FAIL" not in (txt or ""):
                    last_pass = i
    return last_pass > last_edit


# --- V1 (Go baseline-only forensics 2026-09-19): the regression fix must not undo the specified behaviour ----
# vuls-fe8d252c: the fixer deleted the spec'd `return ..., errKernelVersionHeader` to satisfy TestViaHTTP, an EXISTING
# test that encodes the pre-fix behaviour the specification replaces. The gate compares regression COUNTS only
# (1 -> 0 looked like an improvement), so a patch that graded 1/1 shipped as 0/1.
def session_test_targets(stream_path: str, repo_path: str, go: bool) -> "list[str]":
    """Test files the validate session itself wrote (its spec-derived tests), as run targets:
    package dirs for Go, file paths otherwise. These encode the BINDING table, so they are the
    check that the specified behaviour survived the regression fix."""
    if not stream_path or not os.path.exists(stream_path):
        return []
    files: list = []
    try:
        for line in open(stream_path, errors="replace"):
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") != "assistant":
                continue
            for b in (ev.get("message") or {}).get("content") or []:
                if not isinstance(b, dict) or b.get("type") != "tool_use":
                    continue
                fp = (b.get("input") or {}).get("file_path") or ""
                if b.get("name") in ("Write", "Edit", "MultiEdit") and fp and fp not in files:
                    if re.search(r"_test\.go$|test_[^/]*\.py$|_test\.py$|\.(test|spec)\.[jt]sx?$", fp):
                        files.append(fp)
    except Exception:
        return []
    out: list = []
    for f in files:
        rel = f[len(repo_path):].lstrip("/") if f.startswith(repo_path) else f.lstrip("./")
        if f.startswith("/tmp") or not rel:
            continue
        t = ("./" + rel.rsplit("/", 1)[0]) if (go and "/" in rel) else ("./." if go else rel)
        if t not in out:
            out.append(t)
    return out[:6]


LITERAL_VERIFY_FOLLOWUP = """\
[REASONING INTERVENTION -- THE BINDING LITERAL IS STILL NOT ASSERTED EXACTLY]
After your follow-up, the test still checks the specification's value with a PARTIAL matcher, or no
passing run followed your last edit:

{weak}

A partial match hides exactly the near-misses the hidden tests catch (a missing leading "/", an extra
prefix, a different path). Do now:
  1. Replace each partial matcher above with an EQUALITY assertion on the COMPLETE expected value, built
     from the specification's exact form and the concrete input of the test (e.g. the full path your test
     passed in).
  2. Run the test. If it fails, the SOURCE is wrong: fix the source until the exact value matches.
     Never weaken the assertion again.
"""


LITERAL_FOLLOWUP = """\
[REASONING INTERVENTION -- BINDING TABLE LITERAL WAS REWRITTEN]
During validation you changed the expected value(s) of your own test / probe away from the BINDING
table instead of fixing the source:

{hits}

The table is binding: those literals come from the specification, not from what the code returns.
When the real code returns something else, the SOURCE is at fault. Do now, in order:
  1. Restore the table literal(s) in the test / probe (never keep an expectation copied from the
     code's current output).
  2. Diagnose WHY the real code returns a different value (trace the value through the layers it
     passes -- a framework may convert what the source stores, e.g. a model item turning None into
     '' -- and find the representation the specification's value requires at the point the caller /
     test observes it).
  3. Edit the SOURCE so the real code, driven through its actual entry point, returns exactly the
     table literal. Prefer the MINIMAL change: look at how existing code paths in this repository
     already produce that value at the same access point and do the same (e.g. omit an optional
     field / column instead of storing a placeholder for it); do not add new layers, overrides or
     sentinels when an existing convention yields the value. Never edit the repository's existing
     tests.
  4. Re-run the test / probe with `assert result == expected, result` and stop when it passes.
The only exception: if the specification text ITSELF states the value the code currently returns,
quote that sentence verbatim, keep the source, and restore the test to the specification's value.
A framework detail, a docstring, or "it is equivalent" is not such a sentence.
"""


# --- Fix 1 (JS batch1 baseline-only forensics 2026-09-21): a failing spec test must not disappear ------------
# In 5 of the 6 baseline-only JS instances the validate session's OWN tests failed on the behaviour the hidden
# test later failed on, and the pipeline dropped that signal: the session ended with the test still failing
# (a5afad27, 8168c6c4, 9a31cd0f) or edited the test until it passed without touching the source (ce554276
# renamed the failing test away after 7 edits; c5a2089c rewrote the failing rows). The harness now freezes each
# session test file as it was at the FIRST assertion failure, re-runs those frozen versions against the final
# tree, and -- when they still fail -- gives one bounded SOURCE-only fix round with the frozen tests read-only.
_SPEC_TEST_RE = re.compile(r"_test\.go$|(^|/)test_[^/]*\.py$|_test\.py$|(?:[.-]test|\.spec)\.[cm]?[jt]sx?$")
_RUN_FAIL_RE = re.compile(r"Tests:\s+\d+ failed|\b\d+ failing\b|^--- FAIL|^FAIL\s|\b\d+ failed\b", re.M)
_SETUP_ERR_RE = re.compile(r"Test suite failed to run|Cannot find module|SyntaxError|error TS\d+|ModuleNotFoundError"
                           r"|ImportError|ERROR collecting|collection error|\[build failed\]|cannot find package"
                           r"|no tests? (?:found|ran)|No tests found", re.I)


def _tool_text(block: dict) -> str:
    c = block.get("content")
    if isinstance(c, list):
        return "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c if isinstance(c, str) else json.dumps(c)


def frozen_failing_tests(env, stream_paths: "list[str]", repo_path: str) -> dict:
    """Reconstruct the session's test files from its Write/Edit/MultiEdit calls and freeze them at the first
    test run that failed on an ASSERTION (setup/import/compile failures are not a behavioural signal).

    Returns {"files": {rel: frozen_content}, "output": failing-run tail, "unreliable": [rel...]} -- empty
    "files" when no session test ever failed. A file whose reconstructed final content differs from the file
    on disk (edited through Bash, say) is dropped as unreliable rather than frozen from a wrong base."""
    contents: dict = {}
    broken: set = set()
    frozen: dict = {}
    frozen_out = ""

    def base(rel: str) -> "str | None":
        try:
            out = env.execute({"command": f"cd {shlex.quote(repo_path)} && git cat-file -e HEAD:{shlex.quote(rel)} 2>/dev/null "
                                          f"&& git show HEAD:{shlex.quote(rel)}"}, timeout=30)
            return out.get("output", "") if out.get("returncode", 0) == 0 else None
        except Exception:
            return None

    for sp_ in stream_paths:
        if not sp_ or not os.path.exists(sp_):
            continue
        for line in open(sp_, errors="replace"):
            try:
                ev = json.loads(line)
            except Exception:
                continue
            for b in (ev.get("message") or {}).get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use" and b.get("name") in ("Write", "Edit", "MultiEdit"):
                    inp = b.get("input") or {}
                    fp = inp.get("file_path") or ""
                    rel = fp[len(repo_path):].lstrip("/") if fp.startswith(repo_path) else ""
                    # suffix conventions, plus the language's test-dir conventions (NodeBB mocha: test/*.js)
                    if not rel or not (_SPEC_TEST_RE.search(rel) or _sub.is_test_path(rel)) or rel in broken:
                        continue
                    if b["name"] == "Write":
                        contents[rel] = inp.get("content", "")
                        continue
                    cur = contents.get(rel)
                    if cur is None:
                        cur = base(rel)
                    if cur is None:
                        broken.add(rel)
                        continue
                    edits = inp.get("edits") if b["name"] == "MultiEdit" else [inp]
                    for e in edits or []:
                        old, new = e.get("old_string", ""), e.get("new_string", "")
                        if old not in cur:
                            broken.add(rel)
                            break
                        cur = cur.replace(old, new) if e.get("replace_all") else cur.replace(old, new, 1)
                    if rel not in broken:
                        contents[rel] = cur
                elif b.get("type") == "tool_result" and not frozen:
                    txt = _tool_text(b)
                    if _RUN_FAIL_RE.search(txt) and not _SETUP_ERR_RE.search(txt) and contents:
                        frozen = {k: v for k, v in contents.items() if k not in broken}
                        frozen_out = txt[-3000:]
    unreliable = sorted(broken)
    for rel in list(frozen):
        disk = read_repo_file(env, repo_path, rel)
        # a file the session DELETED (scratch-test cleanup) is not evidence of an untracked edit: keep the
        # reconstruction. Only an existing file that differs from it was changed through Bash.
        if disk is False:
            continue
        if disk is None or (rel in contents and disk.rstrip("\n") != contents[rel].rstrip("\n")):
            if rel not in unreliable:
                unreliable.append(rel)
            frozen.pop(rel)
    return {"files": frozen, "output": frozen_out, "unreliable": unreliable}


def read_repo_file(env, repo_path: str, rel: str):
    """File content, False when the file does not exist, None when the read itself failed."""
    try:
        out = env.execute({"command": f"cd {shlex.quote(repo_path)} && if [ -f {shlex.quote(rel)} ]; then cat {shlex.quote(rel)}; "
                                      f"else echo __CCPIPE_NOFILE__; fi"}, timeout=30).get("output", "")
    except Exception:
        return None
    return False if out.strip() == "__CCPIPE_NOFILE__" else out


_PATH_GUARD_HOOK = r"""#!/bin/bash
# PreToolUse: deny edits to the frozen specification tests of the spec-fix round.
IN=$(cat)
TOOL=$(printf '%s' "$IN" | grep -o '"tool_name": *"[^"]*"' | head -1 | sed 's/.*: *"//; s/"$//')
case "$TOOL" in
  Edit|Write|MultiEdit|NotebookEdit)
    F=$(printf '%s' "$IN" | grep -o '"file_path": *"[^"]*"' | head -1 | sed 's/.*: *"//; s/"$//')
    REL=${F#"$CC_REPO"/}; REL=${REL#./}
    if grep -qxF "$REL" "$CC_HOOK_DIR/protected.txt"; then
      echo "[spec-fix guard] $REL is a FROZEN specification test: it is read-only in this round. Fix the SOURCE." >&2; exit 2
    fi ;;
  Bash)
    CMD=$(printf '%s' "$IN" | grep -o '"command": *"[^"]*' | head -1)
    if printf '%s' "$CMD" | grep -qE '(sed|perl)[^|;&]* -[a-zA-Z]*i|>|tee |mv |cp |rm |git (checkout|restore|apply|stash)'; then
      while read -r P; do
        [ -n "$P" ] && printf '%s' "$CMD" | grep -qF "$P" && {
          echo "[spec-fix guard] $P is a FROZEN specification test: it is read-only in this round. Fix the SOURCE." >&2; exit 2; }
      done < "$CC_HOOK_DIR/protected.txt"
    fi ;;
esac
exit 0
"""


def install_spec_fix_guard(env, hook_dir_in: str, repo_path: str, protected: "list[str]", put_file) -> "tuple[str, ...]":
    """PreToolUse hooks for the spec-fix round: the frozen spec tests AND pre-existing test files are read-only."""
    put_file(env, f"{hook_dir_in}/pathguard.sh", _PATH_GUARD_HOOK)
    put_file(env, f"{hook_dir_in}/protected.txt", "\n".join(protected) + "\n")
    put_file(env, f"{hook_dir_in}/testguard.sh", _JS_TEST_GUARD_HOOK if _sub.is_js() else _TEST_GUARD_HOOK)
    env.execute({"command": f"chmod +x {shlex.quote(hook_dir_in)}/pathguard.sh {shlex.quote(hook_dir_in)}/testguard.sh"}, timeout=30)
    pre = f"CC_HOOK_DIR={shlex.quote(hook_dir_in)} CC_REPO={shlex.quote(repo_path)} "
    settings = {"hooks": {"PreToolUse": [{"matcher": "Edit|Write|MultiEdit|NotebookEdit|Bash",
                                          "hooks": [{"type": "command", "command": pre + f"{hook_dir_in}/pathguard.sh"},
                                                    {"type": "command", "command": pre + f"{hook_dir_in}/testguard.sh"}]}]}}
    return ("--settings", shlex.quote(json.dumps(settings)))


SPEC_FIX_PROMPT = """\
[REASONING INTERVENTION -- YOUR OWN SPECIFICATION TEST FAILED AND WAS NEVER FIXED]
Earlier in this validation you wrote tests from the BINDING table, and the first time they ran they FAILED
on an assertion. Those test files are now restored EXACTLY as they were at that failure, and against the
current source they STILL FAIL:

{failed}

Test output (tail):
{tail}

These frozen test files are READ-ONLY in this round (edits to them are blocked), as are the repository's
existing tests. A failing test that encodes the required behaviour means the SOURCE is wrong. Do now:
  1. Read each failing assertion and name the requirement it encodes, quoting the specification.
  2. If an assertion CONTRADICTS the specification text (quote both), say so and stop without editing.
  3. Otherwise fix the SOURCE so these tests pass, keeping the specified interface (names, signatures)
     and the existing tests passing. Re-run the frozen tests and the covering tests, then stop.
"""
