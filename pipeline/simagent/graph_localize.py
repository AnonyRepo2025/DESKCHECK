"""Graph-pruning localization variant (``LOCALIZE_MODE=graph``).

The stock localize sub-agent (``simagent.localization`` + ``simagent.pipeline``'s path
mining) is ANCHOR-CENTRIC: it picks one primary function, traces its chain, simulates the
analyst-chosen path, then optionally ONE mined path.  Everything else the tracer extracted --
the alternate candidates' chains, the dozens of other caller/callee routes -- is only ever
*read* by the model, never executed, and the raw chain text is re-rendered into every prompt.

This variant is GRAPH-CENTRIC with explicit pruning.  Seven steps:

  1. LOCATE      -- one call; the model names FUNCTION plus ``ALT-FUNCTION`` candidates, held
                    inside [``GRAPH_ANCHOR_MIN``, ``GRAPH_ANCHOR_MAX``] (numbered slots put a
                    floor under breadth; one top-up query recovers a short answer).
  2. TRACE+GRAPH -- every anchor is traced; all caller/callee dependency paths are merged into
                    ONE deduplicated directed call graph (edge set union, redundant subpaths
                    dropped), carrying per-node file locations and frame source.
  3. MIN-COVER   -- enumerate end-to-end paths through the graph, then take the MINIMAL set of
                    them that covers every anchor (greedy set cover).
  4. TOP-K       -- for every unselected path, sum how many DISTINCT paths visit each of its
                    nodes; append the ``GRAPH_TOPK`` (default 1) paths with the highest such
                    TOTAL VISITED COUNT (the graph's most-traversed trunk).
  5. LLM-PICK    -- show the issue spec + the candidate paths; the model picks the path(s) most
                    likely to disclose the bug.
  6. SIMULATE    -- merge (3) + (4) + (5), dedupe, and simulate each surviving path with a
                    concrete input -- each prompt carries ONLY that path's frame source.
  7. SUMMARIZE   -- consolidate every trace into the root cause + the complete edit-site set.

Step 7's output is a ``SubAgentFindings`` with exactly the same contract as the stock pipeline
(``files_to_edit`` as ``file :: symbol``, ``site_reasons``, ``_loc._DEMOTED_SITES`` for the
advisory tier, ``FIRST CONSTRUCTED`` / ``FRAMES THAT MUST CHANGE`` markers in
``findings.simulation``), so every deterministic enricher in ``_run_localization`` -- the summary
harvest, the no-drop origin net, the spec rebalance, the sweeps, the contract layer -- runs on it
unchanged and the repair phase sees the same reference block.

Installed by monkeypatch (``install_graph_localization``); the stock chain-based localization is
untouched and remains the default.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict

from simagent import subagent as _sub
from simagent import localization as _loc

# --- knobs ------------------------------------------------------------------------------------
# Anchor-count BOUNDS (total anchors = primary + alternates). The first version left this
# entirely to the model ("you decide, no fixed number of slots"), which removed the structural
# floor the chain variant gets from its three hard-coded ALT-FUNCTION slots -- and the count then
# swung 1..21 across one cohort. Measured on unresolved_16 (graphloc_u16 -> u16_g1g4, same code,
# same settings): every instance whose alt count COLLAPSED lost a graph roughly half the size
# (77658704 3->1 alts, 73->44 nodes; 83fb24b9 4->1, 37->23 nodes) and flipped resolved ->
# unresolved, while the one that BALLOONED (6cc97447 8->11 alts, 149->204 nodes) grew its edit
# set 12->14 sites and its patch 6->9 files, breaking a pass_to_pass test. Free choice is a
# variance amplifier in both directions, so bound it and let the model choose inside the band.
GRAPH_ANCHOR_MIN = int(os.getenv("GRAPH_ANCHOR_MIN", "3"))
GRAPH_ANCHOR_MAX = int(os.getenv("GRAPH_ANCHOR_MAX", "6"))
# (4) how many extra paths, ranked by TOTAL VISITED COUNT (the sum over the path's nodes of how
# many distinct enumerated paths visit that node), are appended to the minimal cover.
GRAPH_TOPK = int(os.getenv("GRAPH_TOPK", "1"))
# path enumeration bounds (a dense graph has exponentially many root->leaf routes)
GRAPH_PATH_CAP = int(os.getenv("GRAPH_PATH_CAP", "600"))
GRAPH_PATH_DEPTH = int(os.getenv("GRAPH_PATH_DEPTH", "14"))
# (5) how many candidate paths the selector sees, and how many it may pick.
GRAPH_LLM_PATHS = int(os.getenv("GRAPH_LLM_PATHS", "40"))
GRAPH_LLM_PICK = int(os.getenv("GRAPH_LLM_PICK", "3"))
# (6) hard cap on simulations (cost); the cover paths are kept first, then top-k, then LLM picks.
GRAPH_SIM_MAX = int(os.getenv("GRAPH_SIM_MAX", "4"))
# per-simulation source budget: only the frames ON THAT PATH are inlined.
GRAPH_SRC_CLIP = int(os.getenv("GRAPH_SRC_CLIP", "10000"))
GRAPH_KEEP_FRAMES_MAX = int(os.getenv("GRAPH_KEEP_FRAMES_MAX", "4"))

# (4) ubiquitous helper names (``get``, ``format``, ``append``, ``str`` ...) sit on nearly every
# path, so they inflate every path's total visited count equally and decide nothing. They are
# skipped when scoring -- they stay IN the paths, they just do not vote.
GRAPH_SCORE_SKIP_NOISE = os.getenv("GRAPH_SCORE_SKIP_NOISE", "1") in ("1", "true", "yes")


def _bare(node: str) -> str:
    """Bare symbol of a node id: ``lib/api.py::GalaxyAPI._call`` -> ``_call``."""
    return node.rpartition("::")[2].rpartition(".")[2] or node


def _display(node: str) -> str:
    """Human-facing hop label: ``lib/api.py::GalaxyAPI._call`` -> ``GalaxyAPI._call``."""
    return node.rpartition("::")[2] or node


# =============================================================================================
# Step 1 -- LOCATE with an unbounded candidate list
# =============================================================================================
_GRAPH_LOCATE_PROMPT = """\
You are the GRAPH BUG-LOCALIZATION step of an automated localization sub-agent. Read the issue
AND the main agent's investigation so far, then name EVERY function (or method) that could
plausibly host this bug. Each one you name gets its caller/callee dependencies extracted and
merged into a single call graph, so a candidate that turns out to be innocent costs little --
but a real fix site you never name can never be recovered.

IMPORTANT -- prefer the PRODUCER, not the symptom surface: if the issue describes a WRONG /
EXTRA / MISSING VALUE in some output (a stray ``None``, a bad field, an incorrect URL/number),
name the function that CONSTRUCTS / PARSES that value (the data origin -- often a resolve /
parse / build / ``__init__`` step), NOT the outer API the issue calls where the wrong value is
merely rendered or reported. The wrapper the user invokes is usually the symptom; the producer
upstream is usually the bug.

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
    Requirements list, several bullet points), each behavior can live in a DIFFERENT function.
    Name one candidate per distinct behavior, so every described fault reaches the graph.
  - Name functions that EXIST in the repository's SOURCE now (surfaced by the exploration, a
    grep, a read file). A name that only appears in a TEST file or in the issue's wished-for API
    may not exist yet -- if you must name one, put it in an ALT-FUNCTION line, never as FUNCTION.

HOW MANY CANDIDATES: name AT LEAST {min_anchors} in total (the FUNCTION plus at least
{min_alts} ALT-FUNCTION lines) and AT MOST {max_anchors}. The slots below are not optional
padding: each one you fill gets its own dependency trace merged into the graph, so an unfilled
slot is coverage the later steps can never recover. Fill them in descending order of confidence
with genuinely DISTINCT hypotheses -- a different behavior the issue describes, an upstream
producer in another file, a sibling implementation in another backend, a base class. Only if the
issue is so narrow that you cannot ground a further candidate at all may you write "none".

Respond in EXACTLY this format, nothing else (no prose, no code blocks):
FUNCTION: <bare function or method name, no parentheses, no module path>
FILE: <relative path to the file if you can infer it, else: unknown>
LINE: <approximate line number in that file where the bug code lives, else: unknown>
REASON: <one sentence on why this is the likely bug site>
{slots}"""


# Asks for the missing candidates when LOCATE under-fills. One extra call, only when the answer
# came back below the floor -- measured cost of an under-filled list (u16_g1g4): 3 alts -> 1
# halved the graph (73 nodes/600 paths -> 44/299) on ansible-77658704, and 4 -> 1 halved it again
# on ansible-83fb24b9 (37/112 -> 23/54); both instances had resolved on the wider graph.
_GRAPH_LOCATE_MORE_PROMPT = """\
Your previous answer to the GRAPH BUG-LOCALIZATION step named only {have} candidate(s):
{named}

That is below the minimum of {min_anchors}. Each candidate gets its own dependency trace merged
into one call graph, so too few candidates yields a graph too narrow for the later steps to
recover from -- naming an innocent candidate costs almost nothing, missing a real fix site costs
everything.

=== ISSUE (problem statement) ===
{problem_statement}

=== WHAT THE MAIN AGENT HAS ALREADY DONE ===
{history}

Name {want} FURTHER candidate function(s)/method(s), DISTINCT from the ones above and from each
other: another behavior the issue describes, an upstream producer in a different file whose
output the ones above consume, a sibling implementation in another backend/subclass, or a base
class. They must EXIST in the repository's source now.

Respond with ONLY these lines, nothing else:
ALT-FUNCTION-{start}: <function> | <relative path or unknown> | <the distinct behavior or hypothesis>
"""


def _locate_anchors(self) -> "tuple[str, str, int, str, str, list]":
    """Run the BOUNDED LOCATE step; returns (function, file, line, reason, raw, alts).

    The candidate count is held inside [GRAPH_ANCHOR_MIN, GRAPH_ANCHOR_MAX]: explicit numbered
    slots put a floor under the breadth (an under-filled list halves the graph), one top-up
    query recovers a short answer, and the tail is truncated to the ceiling.
    """
    n_slots = max(1, GRAPH_ANCHOR_MAX - 1)
    slots = "\n".join(
        f"ALT-FUNCTION-{i}: <function> | <relative path or unknown> | "
        "<the distinct behavior or hypothesis>" for i in range(2, 2 + n_slots))
    text = self.query_fn(
        _GRAPH_LOCATE_PROMPT.format(
            problem_statement=_sub._clip(self.problem_statement, _sub.PS_CLIP),
            history=_sub._clip_tail(self.history, 10000),
            min_anchors=GRAPH_ANCHOR_MIN,
            min_alts=max(0, GRAPH_ANCHOR_MIN - 1),
            max_anchors=GRAPH_ANCHOR_MAX,
            slots=slots,
        )
    ) or ""
    fn_raw = _sub._line_field("FUNCTION", text)
    m = _sub._BARE_NAME_RE.search(fn_raw)
    function = m.group(0).split(".")[-1] if m else ""
    file = _sub._line_field("FILE", text)
    if file.lower() in ("unknown", "n/a", "none", ""):
        file = ""
    else:
        fm = _sub._PATH_TOKEN_RE.search(file)
        file = fm.group(0) if fm else file
    line = 0
    lm = re.search(r"\d{1,7}", _sub._line_field("LINE", text))
    if lm:
        line = int(lm.group(0))
    reason = _sub._line_field("REASON", text)
    alts = _sub._alt_candidates(text)
    # FLOOR: an under-filled list is unrecoverable downstream (the graph is built once), so top
    # it up with ONE extra query rather than tracing a graph we already know is too narrow.
    have = len(alts) + (1 if function else 0)
    if function and have < GRAPH_ANCHOR_MIN:
        want = GRAPH_ANCHOR_MIN - have
        named = "\n".join(f"  - {function} ({file or 'file unknown'})"
                          if i == 0 else f"  - {a} ({f or 'file unknown'})"
                          for i, (a, f, _h) in enumerate([(function, file, "")] + alts))
        self._log(f"    [graph] (1*) only {have} candidate(s) named (floor is "
                  f"{GRAPH_ANCHOR_MIN}); asking for {want} more")
        try:
            more = self.query_fn(
                _GRAPH_LOCATE_MORE_PROMPT.format(
                    have=have, named=named, min_anchors=GRAPH_ANCHOR_MIN, want=want,
                    start=have + 1,
                    problem_statement=_sub._clip(self.problem_statement, _sub.PS_CLIP),
                    history=_sub._clip_tail(self.history, 6000),
                )
            ) or ""
        except Exception as e:
            if type(e).__name__ == "_InstanceBudgetExceeded":
                raise
            more = ""
        seen = {function} | {a for a, _f, _h in alts}
        added = [c for c in _sub._alt_candidates(more) if c[0] not in seen]
        if added:
            alts += added
            text += "\n" + more
            self._log(f"    [graph] (1*) top-up added: {[a for a, _f, _h in added]}")
    # CEILING: each anchor costs a trace, and an over-broad anchor set measurably widens the edit
    # set and the patch (6cc97447: 11 alts -> 14 sites -> 9 files -> a broken pass_to_pass test).
    if GRAPH_ANCHOR_MAX > 0 and len(alts) > GRAPH_ANCHOR_MAX - 1:
        dropped = [a for a, _f, _h in alts[GRAPH_ANCHOR_MAX - 1:]]
        alts = alts[:GRAPH_ANCHOR_MAX - 1]
        self._log(f"    [graph] (1) anchor ceiling {GRAPH_ANCHOR_MAX}: dropped {dropped}")
    return function, file, line, reason, text, alts


# =============================================================================================
# Step 2 -- parse the rendered chains and merge them into one deduplicated call graph
# =============================================================================================
_SRC_BLOCK_RE = re.compile(r"^--- (.+?)\s+\((.*?)\) ---$", re.MULTILINE)
_QUAL_HOP_RE = None   # bound by _compile_lang_regexes (per-language source extensions)


@_sub.on_lang_change
def _compile_lang_regexes():
    global _QUAL_HOP_RE
    _QUAL_HOP_RE = re.compile(r"([\w./-]+\.(?:" + _sub.src_ext_alt() + r"))::([A-Za-z_][\w.]*)")


def _parse_chain(chain_text: str) -> dict:
    """Parse one rendered call chain into ``{caller_paths, callee_paths, locations, sources}``.

    Mirrors :func:`reproduce_intervention_subagent._format_chain`'s layout: the ``CALLER PATHS``
    / ``CALLEE PATHS`` arrow lines, the ``FUNCTION LOCATIONS`` map, and the per-frame
    ``--- <qual>  (<file:line>) ---`` source blocks. Self-contained (no pipeline import) so this
    module can be exercised standalone.

    NODE IDENTITY: when the chain carries the machine-readable ``QUALIFIED SITE PATHS`` section
    (the tracer's site-level ``[name, file, class]`` routes, emitted when CHAIN_QUALIFIED_PATHS=1)
    those paths WIN and every hop is a ``<file>::<Class.name>`` node id. Without it the parser
    falls back to the bare-name arrow lines -- which collapse every same-named method in the repo
    into one node, so a graph built from them grows false edges through shared helper names
    (measured on ansible-83909bf: ``... -> read -> __add__ -> __init__ -> parse -> ...`` stitched
    the galaxy API region to the CLI region through three unrelated definitions).
    """
    caller: "list[list[str]]" = []
    callee: "list[list[str]]" = []
    qual_caller: "list[list[str]]" = []
    qual_callee: "list[list[str]]" = []
    locations: "dict[str, str]" = {}
    displays: "dict[str, str]" = {}
    section = None
    for raw in (chain_text or "").splitlines():
        if raw.startswith("CALLER PATHS"):
            section = "caller"
            continue
        if raw.startswith("CALLEE PATHS"):
            section = "callee"
            continue
        if raw.startswith("QUALIFIED SITE PATHS"):
            section = "qual"
            continue
        if raw.startswith("FUNCTION LOCATIONS"):
            section = "loc"
            continue
        if raw and not raw[0].isspace():  # any other unindented header closes the section
            section = None
            continue
        if section == "qual" and "->" in raw:
            hops = []
            for f, q in _QUAL_HOP_RE.findall(raw):
                bare = q.rpartition(".")[2]
                nid = f"{f}::{bare}"
                hops.append(nid)
                if q != bare:  # keep the class only for DISPLAY, never for identity
                    displays.setdefault(nid, q)
            bucket = qual_caller if raw.lstrip().startswith("CALLER") else qual_callee
            if len(hops) >= 2 and hops not in bucket:
                bucket.append(hops)
        elif section in ("caller", "callee") and "->" in raw:
            names = []
            for part in raw.split("->"):
                m = re.match(r"\s*([A-Za-z_]\w*)", part)
                if m:
                    names.append(m.group(1))
            bucket = caller if section == "caller" else callee
            if len(names) >= 2 and names not in bucket:
                bucket.append(names)
        elif section == "loc" and ":" in raw:
            name, _, locs = raw.strip().partition(":")
            name = name.strip()
            first = locs.split(",")[0].strip()
            if re.fullmatch(r"[A-Za-z_]\w*", name) and first and name not in locations:
                locations[name] = first

    sources: "dict[str, tuple[str, str, str]]" = {}
    marks = list(_SRC_BLOCK_RE.finditer(chain_text or ""))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(chain_text)
        qual, loc = m.group(1).strip(), m.group(2).strip()
        bare = qual.split(".")[-1].strip()
        code = chain_text[m.end():end].strip("\n")
        if bare and bare not in sources:
            sources[bare] = (qual, loc, code)
    def _canon(paths):
        """Bare-name paths -> ``<file>::<name>`` ids using the location map, so a chain WITHOUT
        qualified paths (the Go tracer, an older cached chain) still unifies with one that has
        them. A name with no known location stays bare."""
        out = []
        for p in paths:
            hops = []
            for n in p:
                loc = locations.get(n) or ""
                f = re.sub(r":\d+.*$", "", loc).strip()
                hops.append(f"{f}::{n}" if f else n)
            if hops not in out:
                out.append(hops)
        return out

    return {"caller_paths": qual_caller or _canon(caller),
            "callee_paths": qual_callee or _canon(callee),
            "qualified": bool(qual_caller or qual_callee),
            "displays": displays,
            "locations": locations, "sources": sources}


def _strip_qualified_block(chain_text: str) -> str:
    """Remove the machine-readable QUALIFIED SITE PATHS section from a rendered chain."""
    out, skipping = [], False
    for raw in (chain_text or "").splitlines():
        if raw.startswith("QUALIFIED SITE PATHS"):
            skipping = True
            continue
        if skipping:
            if raw and not raw[0].isspace():
                skipping = False
            else:
                continue
        out.append(raw)
    return "\n".join(out)


def _anchor_node(anchor: str, parsed: dict) -> str:
    """The node id of the site the tracer actually anchored on, or '' when unqualified.

    The tracer renders CALLER routes that END at the target and CALLEE routes that START at it,
    so the anchor's own site is read off the qualified paths directly -- no re-resolution, no
    guessing which of the repo's same-named definitions was traced.
    """
    if not parsed.get("qualified") or not anchor:
        return ""
    for p in parsed.get("caller_paths") or []:
        if p and _bare(p[-1]) == anchor:
            return p[-1]
    for p in parsed.get("callee_paths") or []:
        if p and _bare(p[0]) == anchor:
            return p[0]
    return ""


def covers_anchor(path, anchor: str) -> bool:
    """Does ``path`` actually visit ``anchor``?

    A site-qualified anchor (``file.py::name``) must match the NODE -- matching it by bare name
    would let a route through ANY same-named function in the repo claim to cover it, which is
    exactly how ansible-83909bf reported a full 4/4 cover while never walking the fourth anchor.
    Bare anchors (unqualified Go chains) keep the old name comparison.
    """
    if "::" in anchor:
        return anchor in path
    return any(_bare(n) == anchor for n in path)


class CallGraph:
    """The deduplicated union of every anchor's caller/callee dependency paths.

    A node is a SITE (``<file>::<Class.name>``) whenever the tracer supplied qualified paths, so
    two same-named methods in different classes/files stay distinct. ``anchors`` are tracked as
    NODE IDS on such a graph and only fall back to the bare name when the tracer gave none (the
    Go tracer, an older cached chain) -- see :func:`covers_anchor` for why the distinction is
    load-bearing rather than cosmetic.
    """

    def __init__(self):
        self.edges: "set[tuple[str, str]]" = set()
        self.out: "dict[str, set]" = defaultdict(set)
        self.inn: "dict[str, set]" = defaultdict(set)
        self.nodes: "set[str]" = set()
        self.locations: "dict[str, str]" = {}
        self.displays: "dict[str, str]" = {}   # node id -> "Class.name" label
        self.sources: "dict[str, tuple[str, str, str]]" = {}
        self.anchors: "list[str]" = []          # anchors that actually landed in the graph
        self.dead_anchors: "list[str]" = []     # anchors with no extracted dependency at all
        self.seeds: "list[tuple[str, ...]]" = []  # per-anchor caller+callee joins (fallback)
        self.anchor_seed: "dict[str, tuple[str, ...]]" = {}  # anchor key -> its own traced route
        self.n_raw_paths = 0
        self.qualified = False  # True once any chain supplied site-qualified paths

    def add_chain(self, anchor: str, chain_text: str) -> bool:
        """Merge one traced chain. Returns True when it contributed at least one edge."""
        parsed = _parse_chain(chain_text)
        self.qualified = self.qualified or parsed["qualified"]
        for nid, label in parsed["displays"].items():
            self.displays.setdefault(nid, label)
        for name, loc in parsed["locations"].items():
            self.locations.setdefault(name, loc)
            # Path nodes carry only the leading identifier of a qualified name
            # ("Urlizer.__call__" -> "Urlizer"), so index the bare tail too or the node would
            # look location-less to file_of().
            self.locations.setdefault(name.split(".")[-1], loc)
        for bare, trio in parsed["sources"].items():
            self.sources.setdefault(bare, trio)
        callers, callees = parsed["caller_paths"], parsed["callee_paths"]
        self.n_raw_paths += len(callers) + len(callees)
        added = False
        for p in callers + callees:
            for u, v in zip(p, p[1:]):
                if u == v:
                    continue
                self.nodes.add(u)
                self.nodes.add(v)
                if (u, v) not in self.edges:
                    self.edges.add((u, v))
                    self.out[u].add(v)
                    self.inn[v].add(u)
                added = True
        # Fallback seed: the anchor's own best caller path joined to its best callee path.
        seed = ()
        if callers or callees:
            best_c = max(callers, key=len) if callers else []
            best_e = max(callees, key=len) if callees else []
            seed = tuple(best_c + best_e[1:]) if (best_c and best_e) else tuple(best_c or best_e)
            if len(seed) >= 2 and seed not in self.seeds:
                self.seeds.append(seed)
        # ANCHOR IDENTITY: on a site-qualified graph the anchor is the NODE the tracer actually
        # anchored on, read straight off the qualified routes (caller paths END at the target,
        # callee paths START at it). Recording the bare name instead lets every later step treat
        # any same-named function anywhere in the repo as "the anchor" -- measured on
        # ansible-83909bf, where the cover reported 4/4 anchors while the anchor
        # GalaxyLogin.__init__ was never on a selected path; a path through AnsibleError.__init__
        # satisfied it by name.
        nid = _anchor_node(anchor, parsed)
        on_graph = added and anchor and (
            nid in self.nodes if nid else any(_bare(n) == anchor for n in self.nodes))
        key = nid or anchor
        if on_graph and key not in self.anchors:
            self.anchors.append(key)
            if len(seed) >= 2:
                self.anchor_seed.setdefault(key, seed)
        elif not on_graph and anchor:
            self.dead_anchors.append(anchor)
        return added

    def file_of(self, node: str) -> str:
        """Repo-relative file for a node id -- read off the id itself when it is site-qualified,
        else looked up in the location map by bare name. '' when unknown."""
        head, sep, _tail = node.rpartition("::")
        if sep and head:
            return head
        bare = _bare(node)
        loc = self.locations.get(node) or self.locations.get(bare) or ""
        if not loc and bare in self.sources:
            loc = self.sources[bare][1]
        return re.sub(r":\d+.*$", "", loc).strip()

    def loc_of(self, node: str) -> str:
        """``file.py:12`` for a node, best effort."""
        bare = _bare(node)
        loc = self.locations.get(node) or self.locations.get(bare) or ""
        if not loc and bare in self.sources:
            loc = self.sources[bare][1]
        if loc:
            return loc
        return self.file_of(node) or "(unknown)"

    def display(self, node: str) -> str:
        """Readable hop label: the class-qualified name when the tracer gave one."""
        return self.displays.get(node) or _display(node)

    def source_of(self, node: str):
        return self.sources.get(_bare(node))

    def stats(self) -> dict:
        return {"nodes": len(self.nodes), "edges": len(self.edges),
                "raw_paths": self.n_raw_paths, "anchors": list(self.anchors),
                "unreachable_anchors": list(self.dead_anchors),
                "site_qualified_nodes": self.qualified}


def enumerate_paths(g: CallGraph, cap: int = GRAPH_PATH_CAP,
                    depth: int = GRAPH_PATH_DEPTH) -> "list[tuple[str, ...]]":
    """End-to-end simple paths through the merged graph, longest-first, deduplicated.

    Roots are the in-degree-0 nodes (true entry points); when the graph is fully cyclic every
    node with the minimum in-degree is used instead. Each DFS branch stops at a sink, at a
    revisit (cycle break), or at ``depth`` hops. Paths that are a contiguous SUBPATH of another
    are dropped -- that is the "redundant dependency" pruning.
    """
    roots = [n for n in g.nodes if not g.inn.get(n)]
    if not roots and g.nodes:
        lo = min(len(g.inn.get(n, ())) for n in g.nodes)
        roots = [n for n in g.nodes if len(g.inn.get(n, ())) == lo]
    found: "list[tuple[str, ...]]" = []

    def dfs(node: str, trail: "list[str]"):
        if len(found) >= cap:
            return
        nxt = [c for c in sorted(g.out.get(node, ())) if c not in trail]
        if not nxt or len(trail) >= depth:
            if len(trail) >= 2:
                t = tuple(trail)
                if t not in found:
                    found.append(t)
            return
        for c in nxt:
            if len(found) >= cap:
                return
            dfs(c, trail + [c])

    for r in sorted(roots):
        dfs(r, [r])
        if len(found) >= cap:
            break
    for s in g.seeds:  # guarantee every anchor keeps at least its own traced route
        if s not in found:
            found.append(s)
    # Drop contiguous subpaths (redundant dependencies).
    found.sort(key=len, reverse=True)
    kept: "list[tuple[str, ...]]" = []
    for p in found:
        needle = " -> " + " -> ".join(p) + " -> "
        if any(needle in (" -> " + " -> ".join(q) + " -> ") for q in kept):
            continue
        kept.append(p)
    return kept


# =============================================================================================
# Steps 3 + 4 -- minimal anchor cover, then top-k by total node-visit count
# =============================================================================================
def _issue_hits(path, issue_idents: set) -> int:
    return sum(1 for n in path if _bare(n) in issue_idents)


def minimal_cover(paths, anchors, issue_idents: set) -> "list[tuple[str, ...]]":
    """Greedy minimum set cover: fewest paths that together contain every anchor.

    Ties (equal newly-covered anchor count) break toward the path naming more issue identifiers,
    then the longer path -- both correlate with the route the reported scenario actually takes.

    Coverage is decided by :func:`covers_anchor`, so a site-qualified anchor is only covered by a
    path through THAT node -- never by a same-named function elsewhere in the repo.
    """
    hits = [{a for a in anchors if covers_anchor(p, a)} for p in paths]
    reachable = set().union(*hits) if hits else set()
    pool = list(range(len(paths)))
    chosen: "list[tuple[str, ...]]" = []
    need = set(reachable)
    while need and pool:
        best = max(pool, key=lambda i: (len(need & hits[i]),
                                        _issue_hits(paths[i], issue_idents), len(paths[i])))
        gain = need & hits[best]
        if not gain:
            break
        chosen.append(paths[best])
        need -= gain
        pool.remove(best)
    return chosen


def node_visit_counts(paths) -> "Counter":
    """How many DISTINCT enumerated paths visit each node (the node's traversal consensus)."""
    c: Counter = Counter()
    for p in paths:
        c.update(set(p))
    return c


def _scores_as_hop(node: str) -> bool:
    """Whether a hop gets a vote. Ubiquitous helpers (``get``, ``format``, ``append``, ``str``,
    ``list`` ...) are on nearly every path by construction, so counting them inflates every
    path's total equally and lets helper density -- not relevance -- pick the winner."""
    return not (GRAPH_SCORE_SKIP_NOISE and _bare(node) in _loc._NOISE_FUNCS)


def path_visit_score(path, visits: "Counter") -> int:
    """A path's TOTAL visited count: the sum of its (non-noise) nodes' visit counts.

    A path scores high when its hops are the ones the rest of the graph keeps routing through --
    i.e. it runs along the shared trunk of the extracted dependencies, not down a private branch.
    """
    return sum(visits[n] for n in set(path) if _scores_as_hop(n))


def top_by_visits(paths, exclude, k: int = GRAPH_TOPK,
                  universe=None) -> "list[tuple[str, ...]]":
    """The ``k`` unselected paths with the highest TOTAL visited count.

    Visit counts are computed over ``universe`` (default: every enumerated candidate path,
    INCLUDING the already-selected ones -- a node's centrality is a property of the whole graph,
    not of the leftovers). Ties break toward the longer path.
    """
    if k <= 0:
        return []
    visits = node_visit_counts(universe if universe is not None else paths)
    rest = [tuple(p) for p in paths if p not in exclude]
    rest.sort(key=lambda p: (path_visit_score(p, visits), len(p)), reverse=True)
    return rest[:k]


# =============================================================================================
# Step 5 -- the model picks the path(s) most likely to disclose the bug
# =============================================================================================
_PATH_SELECT_PROMPT = """\
You are the PATH-SELECTION step of an automated bug-localization sub-agent. A deterministic
analysis traced every candidate bug site named for this issue and merged ALL their caller/callee
dependencies into ONE call graph. Below are the distinct end-to-end execution paths through that
graph. Only a few of them can be simulated, so choose the ones that will actually EXPOSE the
defect the issue describes.

=== ISSUE (problem statement, including any Requirements / interface spec) ===
{problem_statement}

=== CANDIDATE BUG SITES that were traced (from the localization step) ===
{anchors}

=== CANDIDATE EXECUTION PATHS (arrow lines; "files:" gives where each hop is defined) ===
{paths}

Already scheduled for simulation by the deterministic pruning (minimal anchor cover + the
most-visited path) -- do NOT re-pick these, pick paths they leave UNCOVERED:
{already}

Choose the path(s) MOST LIKELY TO DISCLOSE THE BUG: the route the issue's own scenario drives
execution down, or the route whose hops touch the data/behaviour the issue says is wrong. Prefer
a path that reaches a PRODUCER (a parse/build/resolve/normalize hop) over one that only reaches
the reporting/rendering surface. Judge by what the issue actually describes -- not by path
length, and not by how central a name looks.

Pick between 1 and {max_pick} paths. Fewer is better if only one route is genuinely relevant.
Respond with ONE line per pick, nothing else (no prose, no code blocks):
PATH: <number> -- <one phrase: what this path would reveal about the issue>
"""


def _render_paths_block(paths, g: CallGraph, numbered: bool = True, cap_files: int = 8) -> str:
    """Arrow lines plus a compact per-hop file map. Only the SELECTABLE list is numbered --
    numbering the already-scheduled list too would collide with the pick indices."""
    out = []
    for i, p in enumerate(paths, start=1):
        files = []
        for n in p:
            f = g.file_of(n)
            if f:
                files.append(f"{g.display(n)}={f}")
        head = f"#{i}  " if numbered else "-  "
        line = f"{head}({len(p)} hops)  " + " -> ".join(g.display(n) for n in p)
        if files:
            line += "\n     files: " + ", ".join(files[:cap_files])
            if len(files) > cap_files:
                line += f", (+{len(files) - cap_files} more)"
        out.append(line)
    return "\n".join(out)


_PICK_RE = re.compile(r"^[\s>*\-`#]*PATH\s*:?\s*#?\s*(\d+)", re.IGNORECASE | re.MULTILINE)


def select_paths(self, g: CallGraph, candidates, already, problem_statement: str):
    """Ask the model which candidate paths to simulate; returns the picked paths."""
    shown = list(candidates)[:GRAPH_LLM_PATHS]
    if not shown:
        return []
    anchors = ", ".join(f"{g.display(a)} ({g.file_of(a) or 'file unknown'})"
                        for a in g.anchors) or "(none)"
    already_txt = _render_paths_block(already, g, numbered=False) if already else "(none)"
    text = self.query_fn(
        _PATH_SELECT_PROMPT.format(
            problem_statement=_sub._clip(problem_statement, _sub.PS_CLIP),
            anchors=anchors,
            paths=_sub._clip(_render_paths_block(shown, g), 9000),
            already=_sub._clip(already_txt, 2000),
            max_pick=GRAPH_LLM_PICK,
        )
    ) or ""
    picked = []
    for m in _PICK_RE.finditer(_sub._strip_md(text)):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(shown) and shown[idx] not in picked:
            picked.append(shown[idx])
        if len(picked) >= GRAPH_LLM_PICK:
            break
    return picked


# =============================================================================================
# Step 6 -- simulate each surviving path with a concrete input
# =============================================================================================
_GRAPH_SIM_PROMPT = """\
You are the PRUNED-PATH SIMULATION step of an automated bug-localization sub-agent. A call graph
was built from every candidate bug site's dependencies and pruned to a handful of execution
paths; you are given ONE of them, together with the real SOURCE of the frames on it. Trace it
with a concrete input. Do NOT write or run any code -- this is a mental trace.

=== ISSUE (problem statement) ===
{problem_statement}

=== THE PATH TO SIMULATE ({why}) ===
{path}

=== WHERE EACH HOP IS DEFINED ===
{locations}

=== SOURCE OF THE FRAMES ON THIS PATH (read these -- never guess a function's behaviour from
its name; a hop with no source shown was not captured, reason about it from its callers) ===
{sources}

Do BOTH:
  1. CONCRETE INPUT: give ONE concrete, valid input whose execution actually follows the path
     above, hop by hop. Use the issue's own example if that example really drives execution down
     this path; OTHERWISE INVENT a concrete, plausible one from the frame SOURCE above. A terse
     issue with NO example is NORMAL -- constructing the example is the point of this step. Do
     NOT refuse: "no concrete input", "cannot construct", "not possible" or "Unable to determine"
     is a FAILURE of this step. Never write "some input". If NO input can drive execution down
     this exact path, write "PATH INFEASIBLE:" plus a one-sentence reason, then trace the closest
     feasible variant instead.
  2. EXECUTION SIMULATION ALONG THE PATH: walk it end to end as if you were the interpreter --
     for EACH hop state what it receives, what its source evaluates to, and what it passes on,
     until the buggy line and on to the path's result. Name each function/line as you go.
     IMPORTANT -- track the WRONG VALUE to its birthplace: the moment a value first becomes wrong
     (a bad/extra/missing entry, a None that should be absent, an incorrect number/string), say
     so explicitly as "<value> is FIRST CONSTRUCTED here, in <function> at <file:line>". A later
     function that merely forwards, consumes, or renders that value is NOT where it goes wrong --
     the producer that constructed it is. If this path does NOT exhibit the bug, say so
     explicitly -- that is a valid and useful result.

Be concise: keep the whole answer under about 60 lines.

Respond with the clearly labelled sections below:
CONCRETE INPUT:
<the input>

EXECUTION SIMULATION:
<hop-by-hop trace, marking where each wrong value is FIRST CONSTRUCTED vs later used/rendered>

PRODUCER CHECK:
<adversarial self-check: is the value ALREADY wrong when it enters the hop you blame -- i.e. is
 the true producer an earlier hop, a helper off this path, or generic machinery in another file
 whose result this path merely consumes or crashes on? Ground the answer in the frame SOURCE.
 End with EXACTLY one line in this format (full function name, real file path):
 PRODUCER: <function> | <file>:<line> | <the wrong value it first constructs>
 -- or, when this path is clean: PRODUCER: none | - | this path does not exhibit the issue>
"""


def _sources_block(path, g: CallGraph, budget: int = GRAPH_SRC_CLIP) -> str:
    """Frame source for exactly the hops on this path (deduped, budgeted)."""
    chunks, used = [], 0
    for n in path:
        trio = g.source_of(n)
        if not trio:
            continue
        qual, loc, code = trio
        block = f"--- {qual}  ({loc}) ---\n{code}"
        if used + len(block) > budget:
            chunks.append(f"--- {qual}  ({loc}) --- (source omitted: budget)")
            continue
        chunks.append(block)
        used += len(block)
    return "\n\n".join(chunks) if chunks else "(no frame source was captured for this path)"


def simulate_path(self, g: CallGraph, path, why: str, problem_statement: str) -> str:
    """One path-directed simulation; returns the model's answer ('' when it refused/empty)."""
    locations = "\n".join(f"  {g.display(n)}: {g.loc_of(n)}" for n in path)
    text = self.query_fn(
        _GRAPH_SIM_PROMPT.format(
            problem_statement=_sub._clip(problem_statement, _sub.PS_CLIP),
            why=why,
            path=" -> ".join(g.display(n) for n in path),
            locations=locations,
            sources=_sources_block(path, g),
        )
    ) or ""
    if not text.strip() or _sub._simulation_refused(text):
        return ""
    return text


# =============================================================================================
# Step 7 -- consolidate the traces into the root cause + the complete edit-site set
# =============================================================================================
_GRAPH_SUMMARY_PROMPT = """\
You are the LOCALIZATION-SUMMARY step of an automated bug-localization sub-agent. Several
INDEPENDENT execution simulations were run over DIFFERENT pruned paths of this bug's call graph.
Consolidate them into the COMPLETE set of edit sites needed to FIX the issue. Do NOT add new
speculation -- only consolidate what the traces and the issue actually establish. Do NOT write or
run any script, test, or shell command.

=== ISSUE (problem statement, including any Requirements / interface spec) ===
{problem_statement}

=== WHAT THE MAIN AGENT HAS ALREADY DONE (its prior commands + observations) ===
{history}

=== CANDIDATE BUG SITES that were traced ===
{anchors}{untraced}

=== THE SIMULATED PATHS AND THEIR TRACES ===
{simulations}

Decide, reasoning from the traces' PRODUCER verdicts and FIRST CONSTRUCTED markers:
  1. The ROOT CAUSE: the deepest frame where the wrong value is FIRST CONSTRUCTED -- not the
     surface wrapper where the symptom merely surfaces. When the traces disagree, prefer the
     producer the majority of them reached, and say which trace you followed.
  2. Every OTHER function/method/class that must change for a COMPLETE fix: a CALLER that must
     pass or handle a changed parameter/return for the fix to take effect on its path (an
     optional parameter nobody passes fixes nothing), a SIBLING implementation of the same method
     in another backend/subclass with the same defect, or a shared BASE/abstract method (hidden
     tests often target the base class directly).
  3. Every candidate bug site above that your traces EXONERATED -- list it as RULED-OUT with a
     one-phrase reason grounded in its code. Do not silently omit one: a site you neither edit
     nor refute is treated as an oversight and re-added automatically.
Give the exact function/method/class NAME for each site (``Class.method`` for a method) and a
real relative file path. ERR TOWARD COMPLETENESS when a fix spans a producer plus its caller or
several backends -- but only list a site whose code must actually change.

Respond EXACTLY in this format, nothing else (no prose, no code blocks). The one-phrase evidence
after ``--`` is handed verbatim to the repair agent to prioritize its work:
ROOT CAUSE: <function> | <relative/path/file.py>:<line> | <the wrong value it first constructs>
EDIT: <relative/path/file.py> :: <function_or_method_or_class> -- <one-phrase why it must change>
EDIT: <relative/path/other.py> :: <Class.method> -- <one-phrase why it must change>
RULED-OUT: <relative/path/third.py> :: <symbol> -- <one-phrase reason it needs no change>
FRAMES THAT MUST CHANGE: <one "function | relative/path/file.py" per line, repeating every frame
  you listed as an EDIT above -- this machine-readable block is parsed downstream>
"""

_ROOT_CAUSE_LINE_RE = re.compile(r"^[\s>*\-`#]*ROOT\s+CAUSE\s*:\s*(.+?)\s*$",
                                 re.IGNORECASE | re.MULTILINE)


def summarize(self, g: CallGraph, sims, problem_statement: str, history: str,
              untraced: str = "") -> "tuple[str, str]":
    """Final consolidation call; returns ``(raw_text, root_cause)``."""
    blocks = []
    for i, (path, why, text) in enumerate(sims, start=1):
        blocks.append(
            f"=== SIMULATION #{i} on the pruned path ({why}): "
            f"{' -> '.join(g.display(n) for n in path)} ===\n"
            + _sub._clip(text, 4000)
        )
    anchors = ", ".join(f"{g.display(a)} ({g.file_of(a) or 'file unknown'})"
                        for a in g.anchors) or "(none)"
    text = self.query_fn(
        _GRAPH_SUMMARY_PROMPT.format(
            problem_statement=_sub._clip(problem_statement, _sub.PS_CLIP),
            history=_sub._clip_tail(history, 7000),
            anchors=anchors,
            untraced=untraced,
            simulations=_sub._clip("\n\n".join(blocks), 16000),
        )
    ) or ""
    rc = ""
    m = _ROOT_CAUSE_LINE_RE.search(_sub._strip_md(text))
    if m and m.group(1).strip().lower() not in ("<function>", "none", "unknown"):
        rc = m.group(1).strip()
    return text, rc


# =============================================================================================
# Orchestration -- the replacement SubAgentReproducer.run
# =============================================================================================
# A subclass of an IMMUTABLE builtin is constructed in ``__new__``; its ``__init__`` runs after
# the value already exists and cannot change it. Mutable builtins (dict/list/set/bytearray) are
# deliberately EXCLUDED -- for those ``__init__`` is a legitimate hook, and rewriting it would be
# wrong more often than right.
#
# Measured on ansible-6cc97447 (three runs, identical output): the summary named
#   lib/ansible/parsing/yaml/objects.py :: _AnsibleUnicode.__init__
# every time, while ``_AnsibleUnicode(str)`` can only be fixed in ``__new__`` -- as gold and all
# three generated patches in fact did. Repair had to silently translate the hook each time; two
# of the three got ``str`` semantics wrong in the process.
_IMMUTABLE_BUILTINS = {"str", "bytes", "tuple", "int", "float", "complex", "frozenset", "bool"}
_CLASS_DECL_RE = re.compile(r"^(\S+?\.py):\d+:\s*class\s+\w+\s*\(([^)]*)\)")


def _immutable_builtin_bases(relf: str, cls: str) -> "set[str]":
    """Immutable builtin base classes of ``cls`` as declared in ``relf`` (grep, no import)."""
    if _loc._REPO_SEARCH_TOOL is None or not cls:
        return set()
    try:
        hits = _loc._REPO_SEARCH_TOOL(
            r"^\s*class\s+" + re.escape(cls) + r"\s*\(", relf.rsplit("/", 1)[-1]) or ""
    except Exception:
        return set()
    for ln in hits.splitlines():
        m = _CLASS_DECL_RE.match(ln.strip())
        if not m or not _loc._file_match(m.group(1), relf):
            continue
        bases = {b.strip().split("[")[0].rsplit(".", 1)[-1] for b in m.group(2).split(",")}
        return bases & _IMMUTABLE_BUILTINS
    return set()


def _fix_builtin_ctor_hook(entry: str, reasons: "dict | None" = None) -> str:
    """``Cls.__init__`` -> ``Cls.__new__`` when Cls derives from an immutable builtin."""
    f, sym = _loc._split_site(entry)
    cls, _dot, tail = (sym or "").rpartition(".")
    if tail != "__init__" or not cls or not f:
        return entry
    bases = _immutable_builtin_bases(f, cls)
    if not bases:
        return entry
    fixed = f"{f} :: {cls}.__new__"
    if reasons is not None:
        why = (f"constructor hook corrected: {cls} subclasses {'/'.join(sorted(bases))}, which is "
               f"immutable -- its value is fixed in __new__ and __init__ cannot change it")
        reasons[fixed] = (reasons.pop(entry, "") + " | " if reasons.get(entry) else "") + why
    return fixed


def _normalize_site(entry: str) -> str:
    """Resolve a bare-basename site to a repo-relative path (bare paths die in reconciliation)."""
    f, sym = _loc._split_site(entry)
    if f and "/" not in f:
        try:
            relf = _loc._resolve_frame_file(f, (sym or "").split(".")[-1])
        except Exception:
            relf = ""
        if relf and "/" in relf:
            return f"{relf} :: {sym}" if sym else relf
    return entry


# The selection record of the RUN IN PROGRESS. Kept on the module (not only on the findings) so
# the pipeline can persist what was decided even when localization never returns -- an instance
# that dies on the budget cap inside the simulate loop still has its path selection on disk.
LAST_GRAPH_RECORD: "dict | None" = None


def _path_entry(g: CallGraph, path, source: str, why: str, rank: int,
                visits: "Counter | None" = None) -> dict:
    """One nominated path, in the form the pipeline record persists."""
    return {
        "rank": rank,
        "source": source,                       # cover | rescue | topk | llm
        "why": why,
        "hops": [g.display(n) for n in path],
        "nodes": list(path),                    # site ids -- the auditable form
        "files": [g.file_of(n) for n in path],
        "covers": [g.display(a) for a in g.anchors if covers_anchor(path, a)],
        "visit_score": path_visit_score(path, visits) if visits is not None else None,
        "simulated": False,                     # filled in at step 6
        "dropped_by_budget": False,
        "sim_chars": 0,
    }


def _graph_run(self) -> "_sub.SubAgentFindings":
    """Seven-step graph-pruning localization (drop-in for ``SubAgentReproducer.run``)."""
    global LAST_GRAPH_RECORD
    findings = _sub.SubAgentFindings()
    self.site_reasons = {}
    self.mined_path, self.mined_path_coverage, self.path_sim_ran = [], None, False
    _loc._DEMOTED_SITES = []
    _loc._DEMOTED_REASONS = {}
    reasons: dict = {}
    g = CallGraph()
    # Built up in place as the run proceeds and published on the findings immediately, so a run
    # that raises at step 5 still persists everything decided through step 4. ``stage`` says how
    # far it got.
    record: dict = {"stage": "locate", "topk": GRAPH_TOPK, "sim_max": GRAPH_SIM_MAX,
                    "path_selection": [], "anchor_coverage": {}}
    findings.graph_summary = record
    LAST_GRAPH_RECORD = record
    try:
        # -- (1) LOCATE: one primary + an unbounded candidate list ---------------------------
        (findings.bug_function, findings.bug_file, findings.bug_line,
         findings.locate_reason, findings.raw_locate, alts) = _locate_anchors(self)
        findings.alt_candidates = alts
        self._log(f"    [graph] (1) primary: {findings.bug_function or '?'} "
                  f"({findings.bug_file or '?'}:{findings.bug_line or '?'}); "
                  f"{len(alts)} alternate candidate(s): {[a for a, _f, _h in alts]}")

        # -- (2) TRACE every anchor, merge into ONE deduplicated call graph ------------------
        anchors: "list[tuple[str, str, int]]" = []
        if findings.bug_function:
            anchors.append((findings.bug_function, findings.bug_file, findings.bug_line))
        # Dedup on (name, file), matching _alt_candidates: two classes' same-named methods in
        # DIFFERENT files are two anchors, and dropping the second here would undo that parse.
        seen = {(findings.bug_function, findings.bug_file)}
        for afunc, afile, _hyp in alts:
            if not afunc or any(afunc == n and (afile == f or not f or not afile)
                                for n, f in seen):
                continue
            seen.add((afunc, afile))
            anchors.append((afunc, afile, 0))
        chains: "list[str]" = []
        for func, file, line in anchors:
            chain = self.trace(func, file, line)
            if not chain or "call chain unavailable" in chain[:80] or chain.startswith("["):
                self._log(f"    [graph] (2) '{func}': no chain extracted; skipped")
                g.dead_anchors.append(func)
                continue
            # A name with no definition anywhere is a wished-for API, not a graph anchor.
            if "has NO definition in the repo" in chain[:700]:
                self._log(f"    [graph] (2) '{func}' has NO definition in the repo; kept as a "
                          "note, not as a graph anchor")
                g.dead_anchors.append(func)
                chains.append(f"=== CANDIDATE '{func}' ===\n" + _sub._clip(chain, 1200))
                continue
            # A class or module-level data symbol (registry dict, constant) EXISTS but cannot be
            # a call-graph node; keep its definition + referencing functions as candidate
            # evidence and carry it as an untraced anchor instead of dropping it (measured on
            # qutebrowser-f8e7fea0, where discarding _WEBENGINE_SETTINGS here mislocalized).
            if ("is not a function -- it is a module-level DATA" in chain[:700]
                    or "is a CLASS defined at" in chain[:700]):
                self._log(f"    [graph] (2) '{func}' is a non-function symbol (class/data); "
                          "kept as an untraced candidate with its definition and referencing "
                          "functions")
                g.dead_anchors.append(func)
                chains.append(f"=== CANDIDATE '{func}' ===\n" + _sub._clip(chain, 2400))
                continue
            healed = _sub._chain_target(chain)
            if healed and healed != func:
                self._log(f"    [graph] (2) healed '{func}' -> '{healed}'")
                if func == findings.bug_function:
                    findings.bug_function = healed
                func = healed
            g.add_chain(func, chain)
            # The QUALIFIED SITE PATHS block is machine input for the graph builder above, not
            # something any prompt or reader should carry -- drop it from the stored chain.
            chains.append(f"=== CANDIDATE '{func}' ===\n" + _strip_qualified_block(chain))
        findings.call_chain = "\n\n".join(chains)
        st = g.stats()
        record.update(st)
        record["stage"] = "graph"
        self._log(f"    [graph] (2) merged call graph: {st['nodes']} nodes, {st['edges']} edges "
                  f"(deduplicated from {st['raw_paths']} traced dependency paths); "
                  f"anchors in graph: {st['anchors']}"
                  + (f"; unreachable: {st['unreachable_anchors']}"
                     if st["unreachable_anchors"] else ""))
        # Re-anchor the primary when it never made it into the graph but an alternate did.
        # (anchors are node ids on a qualified graph, so compare on the bare name.)
        if all(_bare(a) != findings.bug_function for a in g.anchors) and g.anchors:
            self._log(f"    [graph] (2*) primary '{findings.bug_function}' contributed no "
                      f"dependencies; re-anchoring on '{g.display(g.anchors[0])}'")
            findings.bug_function = _bare(g.anchors[0])
            findings.bug_file = g.file_of(g.anchors[0]) or findings.bug_file
            findings.bug_line = 0

        candidates = enumerate_paths(g) if g.edges else []
        issue_idents = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", self.problem_statement or ""))
        self._log(f"    [graph] (2) enumerated {len(candidates)} distinct end-to-end path(s) "
                  "after dropping redundant subpaths")
        record["candidate_paths"] = len(candidates)
        record["stage"] = "enumerate"

        # -- (3) MINIMAL COVER of all anchors ------------------------------------------------
        cover = minimal_cover(candidates, g.anchors, issue_idents)
        for p in cover:
            self._log(f"    [graph] (3) cover path ({len(p)} hops) covers "
                      f"{[g.display(a) for a in g.anchors if covers_anchor(p, a)]}: "
                      f"{' -> '.join(g.display(n) for n in p)}")
        if not cover:
            self._log("    [graph] (3) no path covers any anchor (empty/degenerate graph)")
        # An anchor no ENUMERATED path reaches is a silent coverage hole: it is in the graph, so
        # it is not reported unreachable, yet nothing will ever walk it. Say so, and rescue it
        # below with its own traced route.
        uncovered = [a for a in g.anchors if not any(covers_anchor(p, a) for p in cover)]
        record["cover_paths"] = [[g.display(n) for n in p] for p in cover]
        record["uncovered_by_cover"] = [g.display(a) for a in uncovered]
        record["stage"] = "cover"
        if uncovered:
            self._log(f"    [graph] (3!) {len(uncovered)} anchor(s) NOT covered by any enumerated "
                      f"path: {[g.display(a) for a in uncovered]}")

        # -- (4) APPEND the top-k most-visited paths -----------------------------------------
        visits = node_visit_counts(candidates)
        topk = top_by_visits(candidates, cover, GRAPH_TOPK, universe=candidates)
        record["top_visited_paths"] = [[g.display(n) for n in p] for p in topk]
        record["top_visited_scores"] = [path_visit_score(p, visits) for p in topk]
        record["stage"] = "topk"
        for p in topk:
            self._log(f"    [graph] (4) most-visited path (total visited count "
                      f"{path_visit_score(p, visits)} over {len(p)} hops): "
                      f"{' -> '.join(g.display(n) for n in p)}")

        # -- (5) LLM picks the path(s) most likely to disclose the bug -----------------------
        picked = []
        if candidates:
            try:
                picked = select_paths(self, g, candidates, cover + topk, self.problem_statement)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    raise
                self._log(f"    [graph] (5) path selection failed ({type(e).__name__}: {e})")
            for p in picked:
                self._log(f"    [graph] (5) model-picked path ({len(p)} hops): "
                          f"{' -> '.join(g.display(n) for n in p)}")
        record["selected_paths"] = [[g.display(n) for n in p] for p in picked]
        record["llm_picks_parsed"] = len(picked)
        record["llm_paths_shown"] = min(len(candidates), GRAPH_LLM_PATHS)
        record["stage"] = "llm"

        # -- (6) MERGE (3)+(4)+(5), dedupe, simulate each with a concrete input ---------------
        plan: "list[tuple[tuple, str]]" = []
        entries: "list[dict]" = []      # the persisted twin of `plan`, one dict per nomination

        def _nominate(path, source, why):
            if any(tuple(path) == tuple(q) for q, _w in plan):
                return False
            plan.append((path, why))
            entries.append(_path_entry(g, path, source, why, len(plan), visits))
            return True

        for p in cover:
            _nominate(p, "cover", "minimal cover of the candidate bug sites")
        # ANCHOR RESCUE, ranked right behind the cover so the budget cannot drop it: an anchor the
        # cover leaves unvisited is a candidate bug site the model named and nothing ever walks.
        # Its own traced caller->callee route always exists (the tracer produced it), so simulate
        # that instead of losing the candidate silently.
        for a in uncovered:
            seed = g.anchor_seed.get(a)
            if seed and len(seed) >= 2 and _nominate(
                    seed, "rescue", f"anchor rescue: no cover path visits {g.display(a)}, so its "
                                    "own traced route is walked instead"):
                self._log(f"    [graph] (3*) anchor rescue for '{g.display(a)}' ({len(seed)} hops): "
                          f"{' -> '.join(g.display(n) for n in seed)}")
        for p in topk:
            _nominate(p, "topk", "highest total visited count -- its hops are the ones the rest "
                                 "of the graph's paths keep routing through")
        for p in picked:
            _nominate(p, "llm", "model-selected as most likely to disclose the bug")
        if not plan and g.seeds:
            _nominate(g.seeds[0], "fallback", "fallback: the primary anchor's traced route")
        dropped = len(plan) - GRAPH_SIM_MAX
        if dropped > 0:
            self._log(f"    [graph] (6) simulation budget GRAPH_SIM_MAX={GRAPH_SIM_MAX} drops "
                      f"{dropped} merged path(s) (cover first, then anchor rescues, then "
                      "most-visited, then model picks)")
            plan = plan[:GRAPH_SIM_MAX]
            for e in entries[GRAPH_SIM_MAX:]:
                e["dropped_by_budget"] = True
        record["path_selection"] = entries
        record["simulations_dropped_by_budget"] = max(0, dropped)
        record["stage"] = "simulate"
        sims: "list[tuple[tuple, str, str]]" = []
        for i, (p, why) in enumerate(plan, start=1):
            text = simulate_path(self, g, p, why, self.problem_statement)
            if not text:
                self._log(f"    [graph] (6) simulation {i}/{len(plan)} refused/empty; skipped")
                entries[i - 1]["refused"] = True
                continue
            entries[i - 1]["simulated"] = True
            entries[i - 1]["sim_chars"] = len(text)
            sims.append((p, why, text))
            self._log(f"    [graph] (6) simulated path {i}/{len(plan)} ({len(text)} chars): "
                      f"{' -> '.join(g.display(n) for n in p)}")
        findings.simulation = "\n\n".join(
            f"=== SIMULATION on PRUNED PATH #{i} ({why}) ===\n"
            f"PATH: {' -> '.join(g.display(n) for n in p)}\n\n{t}"
            for i, (p, why, t) in enumerate(sims, start=1)
        )
        if plan:
            self.mined_path = [g.display(n) for n in plan[0][0]]
        self.path_sim_ran = bool(sims)

        # -- (7) SUMMARIZE into the root cause + the complete edit-site set ------------------
        raw_summary, rc = ("", "")
        # UNTRACED-ANCHOR carry-forward: LOCATE anchors that never became call-graph nodes (a
        # module-level dict, a class the def-index missed, or an invented name) used to vanish
        # from the SUMMARIZE candidate list entirely -- the summarizer was never asked to
        # adjudicate them (measured on qutebrowser-f8e7fea0, where the correct site
        # _WEBENGINE_SETTINGS was LOCATE's ALT-2 and died here). Surface each with its LOCATE
        # reason and require an explicit EDIT / RULED-OUT disposition.
        _anchor_reason = {}
        _anchor_file = {}
        if findings.bug_function:
            _anchor_reason[findings.bug_function] = (findings.locate_reason or "").strip()
            _anchor_file[findings.bug_function] = findings.bug_file or ""
        for _afunc, _afile, _hyp in (findings.alt_candidates or []):
            if _afunc:
                _anchor_reason.setdefault(_afunc, (_hyp or "").strip())
                _anchor_file.setdefault(_afunc, _afile or "")
        untraced = ""
        _dead = list(dict.fromkeys(g.dead_anchors))
        if _dead:
            _lines = []
            for func in _dead:
                _lines.append(
                    f"- {func} ({_anchor_file.get(func) or 'file unknown'}) -- "
                    f"{_anchor_reason.get(func) or '(no LOCATE reason recorded)'}")
                _bi = (findings.call_chain or "").find(f"=== CANDIDATE '{func}' ===")
                if _bi != -1:
                    _blk = (findings.call_chain or "")[_bi:]
                    _ni = _blk.find("=== CANDIDATE '", 20)
                    _blk = _blk if _ni == -1 else _blk[:_ni]
                    _lines.append("  " + _sub._clip(_blk.strip(), 900).replace("\n", "\n  "))
            untraced = (
                "\n\n=== ADDITIONAL CANDIDATES named at LOCATE but NOT traceable as call-graph "
                "nodes (a data structure, a class, or a name with no definition -- each shown "
                "with the evidence the tracer collected for it) ===\n" + "\n".join(_lines) +
                "\nAdjudicate EACH of these too: list it as an EDIT or as RULED-OUT exactly "
                "like the traced candidates. Do not skip one because no simulation walked it."
            )
        if sims:
            raw_summary, rc = summarize(self, g, sims, self.problem_statement, self.history,
                                        untraced=untraced)
        findings.raw_plan = raw_summary
        findings.root_cause = rc
        findings.prediction = ""  # no PREDICT step in this variant
        sites = _loc._parse_edit_sites(raw_summary, reasons)
        sites = [_normalize_site(s) for s in sites]
        try:
            fixed = [_fix_builtin_ctor_hook(s, reasons) for s in sites]
            if fixed != sites:
                self._log("    [graph] (7) constructor hook: "
                          f"{[(a, b) for a, b in zip(sites, fixed) if a != b]}")
                sites = list(dict.fromkeys(fixed))
        except Exception as e:
            self._log(f"    [graph] (7) builtin-ctor fix FAILED ({type(e).__name__}: {e})")
        ruled_reasons: dict = {}
        for s in _loc._parse_ruled_out(raw_summary, ruled_reasons):
            s = _normalize_site(s)
            if s not in _loc._DEMOTED_SITES and s not in sites:
                _loc._DEMOTED_SITES.append(s)
                _loc._DEMOTED_REASONS[s] = ("SUMMARY ruled it out: "
                                            + (ruled_reasons.get(s) or "(no reason given)"))
        # FRAME-UNION: the summary may only REMOVE a frame the traces established via an explicit
        # RULED-OUT line; anything it silently narrowed away is re-added (same Tier-1 rule the
        # chain variant enforces on PLAN).
        disposed = {(_loc._split_site(s)[1] or "").split(".")[-1]
                    for s in sites + _loc._DEMOTED_SITES}
        disposed.discard("")
        added = []
        for func, fpath in _loc._mandatory_frames(
                findings.simulation + "\n" + raw_summary, raw_summary):
            bare = func.split(".")[-1]
            if bare in disposed:
                continue
            relf = _loc._resolve_frame_file(fpath, bare) or g.file_of(bare)
            if not relf:
                continue
            entry = f"{relf} :: {func}"
            if entry not in sites:
                sites.append(entry)
                added.append(entry)
                disposed.add(bare)
                reasons.setdefault(entry, "frame-union: a pruned-path simulation named this "
                                          "frame as MUST-CHANGE / the producer of the wrong "
                                          "value (strong evidence -- prefer fixing producers)")
            if len(added) >= GRAPH_KEEP_FRAMES_MAX:
                break
        if added:
            self._log(f"    [graph] (7) frame-union: re-added {len(added)} frame(s) the "
                      f"simulations named but the summary dropped: {added}")
        # KEEP-UNLESS-REFUTED for LOCATE anchors: an anchor the tracer could not resolve to a
        # call-graph node is still a LOCATE-reasoned candidate; only an explicit RULED-OUT line
        # (or being edited already) disposes of it. Same Tier-1 asymmetry frame-union enforces
        # for sim-named frames -- LOCATE-named anchors deserve no less.
        _kept_untraced = []
        for func in _dead:
            bare = (func or "").split(".")[-1]
            if not bare:
                continue
            _disposed = sites + _loc._DEMOTED_SITES
            if any(bare == (_loc._split_site(s)[1] or "").split(".")[-1] for s in _disposed):
                continue
            relf = _loc._resolve_frame_file(_anchor_file.get(func, ""), bare)
            if not relf:
                self._log(f"    [graph] (7) untraced-anchor keep: no file resolvable for "
                          f"'{func}'; not added")
                continue
            entry = f"{relf} :: {func}"
            if entry in sites:
                continue
            _bi = (findings.call_chain or "").find(f"=== CANDIDATE '{func}' ===")
            _blk = (findings.call_chain or "")[_bi:_bi + 1500] if _bi != -1 else ""
            if "no function, class, or module-level assignment" in _blk:
                reasons[entry] = ("untraced-anchor keep: LOCATE named this symbol but a repo "
                                  "scan found NO definition at all -- either the fix must "
                                  "CREATE it, or the name is invented; verify before editing")
            else:
                reasons[entry] = ("untraced-anchor keep: LOCATE named this symbol (reason: "
                                  + (_anchor_reason.get(func) or "unrecorded")[:200]
                                  + ") but it is not a call-graph node (a data structure or "
                                  "class), so no simulation could walk it; kept because the "
                                  "summary did not explicitly rule it out")
            sites.append(entry)
            _kept_untraced.append(entry)
            if len(_kept_untraced) >= 4:
                break
        if _kept_untraced:
            self._log(f"    [graph] (7) untraced-anchor keep: re-added {len(_kept_untraced)} "
                      f"LOCATE anchor(s) that never became graph nodes: {_kept_untraced}")
        findings.files_to_edit = sites
        self.site_reasons = reasons
        # Append the summary under the header the downstream deterministic harvests key on, so
        # `_apply_summary_findings` folds FRAMES THAT MUST CHANGE / the root cause back in.
        if raw_summary.strip():
            findings.simulation += (
                f"\n\n=== SUMMARY OF THE SIMULATIONS (consolidated over {len(sims)} pruned "
                f"path(s)) ===\n" + raw_summary
            )
        self._log(f"    [graph] (7) edit sites: {findings.files_to_edit or '[]'}; "
                  f"root cause: {findings.root_cause or '(not isolated)'}")

        # ANCHOR LEDGER: for every candidate the model named, did anything actually WALK it?
        # This is the audit that was missing when 83909bf reported a full cover while
        # GalaxyLogin.__init__ was never on a simulated path.
        walked = [p for p, _w, _t in sims]
        record["anchor_coverage"] = {
            a: {"display": g.display(a),
                "file": g.file_of(a),
                "in_cover": any(covers_anchor(p, a) for p in cover),
                "simulated": any(covers_anchor(p, a) for p in walked),
                "simulated_by": [i for i, p in enumerate(walked, start=1)
                                 if covers_anchor(p, a)]}
            for a in g.anchors
        }
        record["anchors_never_simulated"] = [
            g.display(a) for a in g.anchors
            if not any(covers_anchor(p, a) for p in walked)]
        if record["anchors_never_simulated"]:
            self._log(f"    [graph] (6!) anchors named but never walked by any simulation: "
                      f"{record['anchors_never_simulated']}")
        record.update({
            **g.stats(),
            "candidate_paths": len(candidates),
            "cover_paths": [[g.display(n) for n in p] for p in cover],
            "top_visited_paths": [[g.display(n) for n in p] for p in topk],
            "top_visited_scores": [path_visit_score(p, visits) for p in topk],
            "selected_paths": [[g.display(n) for n in p] for p in picked],
            "simulated_paths": [[g.display(n) for n in p] for p in walked],
            "simulations_dropped_by_budget": max(0, dropped),
            "topk": GRAPH_TOPK,
            "stage": "done",
        })
        findings.ok = bool(findings.bug_function) and bool(
            findings.simulation or findings.raw_plan or findings.files_to_edit)
    except Exception as e:  # never let the sub-agent abort the main run
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        # A budget exhaustion must still propagate -- the pipeline handles it explicitly.
        if type(e).__name__ == "_InstanceBudgetExceeded":
            raise
        self._log(f"    [graph] pipeline error ({type(e).__name__}: {e}); "
                  "falling back to whatever was collected")
        findings.ok = False
        # Keep everything decided BEFORE the failure -- `stage` says where it stopped.
        record.update(g.stats())
        record["stage"] = f"error:{type(e).__name__}"
        record["error"] = f"{type(e).__name__}: {e}"
        findings.graph_summary = record
    return findings


def install_graph_localization() -> None:
    """Monkeypatch the sub-agent to run the seven-step graph-pruning localization."""
    if getattr(_sub.SubAgentReproducer, "_graph_localization_installed", False):
        return
    # Label the new prompts in recorded trajectories. Prepend: the generic
    # "SIMULATION step of an automated" marker is a substring of the pruned-path prompt.
    _loc._STEP_MARKERS.insert(0, ("PRUNED-PATH SIMULATION step", "SIMULATE_PATH"))
    _loc._STEP_MARKERS.insert(0, ("PATH-SELECTION step", "SELECT_PATHS"))
    _loc._STEP_MARKERS.insert(0, ("LOCALIZATION-SUMMARY step", "SUMMARIZE"))
    # Ask the tracer to also emit SITE-qualified paths ([name, file, class] per hop). Without
    # them every same-named method in the repo is ONE node and the enumeration invents routes
    # that stitch unrelated regions together through shared helper names.
    os.environ["CHAIN_QUALIFIED_PATHS"] = "1"
    _sub.SubAgentReproducer.run = _graph_run
    _sub.SubAgentReproducer._graph_localization_installed = True
