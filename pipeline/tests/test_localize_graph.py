"""Offline test of the graph-pruning localization variant (``LOCALIZE_MODE=graph``).

Runs the REAL ``SubAgentReproducer`` with the graph pipeline installed -- only the model
(``query_fn``) and the call-chain tracer (``trace_fn``) are faked, and the fake chains are
rendered by the production :func:`reproduce_intervention_subagent._format_chain`, so the parser,
the graph merge, the pruning (minimal cover / top-k / model pick), the per-path simulation
scheduling and the summary harvest are all genuinely exercised without any network calls.

    python test_localize_graph.py
"""
from __future__ import annotations

import os
import re


from simagent import graph_localize as _graph
from simagent import subagent as _sub
from simagent import localization as _loc

PROBLEM = """\
urlize() mangles a trailing punctuation mark when the URL ends in a bracket: the closing bracket
is stripped by trim_punctuation even though it is part of the URL, so the rendered anchor href
is wrong. handle_word passes the wrong offsets down.
"""

# ---------------------------------------------------------------------------------------------
# Fake call chains: four anchors whose caller/callee paths overlap heavily (the dedup + minimal
# cover only mean something when the raw paths are redundant).
# ---------------------------------------------------------------------------------------------
LOC = {
    "render": ["django/template/base.py:110"],
    "urlize": ["django/utils/html.py:280"],
    "Urlizer.__call__": ["django/utils/html.py:300"],
    "handle_word": ["django/utils/html.py:340"],
    "trim_punctuation": ["django/utils/html.py:380"],
    "escape": ["django/utils/html.py:40"],
    "smart_urlquote": ["django/utils/html.py:200"],
    "unescape": ["django/utils/text.py:22"],
}
SRC = [
    {"qual": "trim_punctuation", "loc": "django/utils/html.py:380",
     "code": "def trim_punctuation(word):\n    return word.rstrip(TRAILING_PUNCTUATION_CHARS)"},
    {"qual": "handle_word", "loc": "django/utils/html.py:340",
     "code": "def handle_word(word):\n    return trim_punctuation(word)"},
]

def _sites(*rows):
    """[[name, file, class], ...] hops -- the tracer's SITE-qualified path shape."""
    return [list(r) for r in rows]


CHAINS = {
    "urlize": {
        "target": "urlize", "depth": 3, "locations": LOC, "frame_sources": SRC,
        "caller_paths": [["render", "urlize"], ["template_filter", "render", "urlize"]],
        "callee_paths": [["urlize", "Urlizer", "handle_word", "trim_punctuation"],
                         ["urlize", "Urlizer", "handle_word", "smart_urlquote"]],
        "caller_site_paths": [
            _sites(("render", "django/template/base.py", ""),
                   ("urlize", "django/utils/html.py", "")),
            _sites(("template_filter", "django/template/defaultfilters.py", ""),
                   ("render", "django/template/base.py", ""),
                   ("urlize", "django/utils/html.py", "")),
        ],
        "callee_site_paths": [
            _sites(("urlize", "django/utils/html.py", ""),
                   ("__call__", "django/utils/html.py", "Urlizer"),
                   ("handle_word", "django/utils/html.py", "Urlizer"),
                   ("trim_punctuation", "django/utils/html.py", "Urlizer")),
            _sites(("urlize", "django/utils/html.py", ""),
                   ("__call__", "django/utils/html.py", "Urlizer"),
                   ("handle_word", "django/utils/html.py", "Urlizer"),
                   ("smart_urlquote", "django/utils/html.py", "")),
        ],
    },
    "trim_punctuation": {
        "target": "trim_punctuation", "depth": 3, "locations": LOC, "frame_sources": SRC,
        # redundant with urlize's callee path (same edges) -> must collapse in the graph
        "caller_paths": [["handle_word", "trim_punctuation"],
                         ["Urlizer", "handle_word", "trim_punctuation"]],
        "callee_paths": [["trim_punctuation", "unescape"]],
    },
    "escape": {
        "target": "escape", "depth": 3, "locations": LOC, "frame_sources": SRC,
        "caller_paths": [["render", "escape"]],
        "callee_paths": [["escape", "unescape"]],
    },
    "make_absolute": {  # a wished-for API: no definition in the repo
        "target": "make_absolute", "depth": 3, "locations": {}, "frame_sources": [],
        "caller_paths": [], "callee_paths": [], "defined": False,
    },
    "URL_SCHEMES": {  # a module-level registry dict: EXISTS, but is not a call-graph node
        "target": "URL_SCHEMES", "depth": 3, "locations": {}, "frame_sources": [],
        "caller_paths": [], "callee_paths": [], "defined": False,
        "symbol_kind": "data",
        "symbol_defs": ["django/utils/html.py:12"],
        "symbol_source": "URL_SCHEMES = {'http': True, 'https': True}",
        "referencing_functions": ["Urlizer.handle_word (django/utils/html.py)"],
    },
}

LOCATE_ANSWER = """\
FUNCTION: urlize
FILE: django/utils/html.py
LINE: 280
REASON: the filter the issue calls; the mangled href surfaces here
ALT-FUNCTION-2: trim_punctuation | django/utils/html.py | strips the trailing bracket
ALT-FUNCTION-3: escape | django/utils/html.py | competing hypothesis: escaping order
ALT-FUNCTION-4: make_absolute | unknown | wished-for API named only in the issue
"""

SELECT_ANSWER = """\
PATH: 2 -- exercises the escape branch the cover path never reaches
"""

SIM_ANSWER = """\
CONCRETE INPUT:
urlize("see http://example.com/a(b) now")

EXECUTION SIMULATION:
render passes the raw text to urlize at django/utils/html.py:280, which delegates to Urlizer,
which calls handle_word. handle_word forwards the whole word to trim_punctuation, which strips
the closing bracket: the truncated URL "http://example.com/a(b" is FIRST CONSTRUCTED here, in
trim_punctuation at django/utils/html.py:380. Everything downstream only renders it.

PRODUCER CHECK:
The value entering trim_punctuation is still correct, so the defect is born inside it; handle_word
must also change to pass the bracket-aware offsets.
PRODUCER: trim_punctuation | django/utils/html.py:380 | the truncated URL
"""

SUMMARY_ANSWER = """\
ROOT CAUSE: trim_punctuation | django/utils/html.py:380 | the truncated URL
EDIT: django/utils/html.py :: trim_punctuation -- strips a bracket that belongs to the URL
EDIT: django/utils/html.py :: handle_word -- must pass bracket-aware offsets for the fix to apply
RULED-OUT: django/utils/html.py :: escape -- escaping runs after the URL is already truncated
FRAMES THAT MUST CHANGE:
trim_punctuation | django/utils/html.py
handle_word | django/utils/html.py
"""

calls: "list[str]" = []


def fake_query(prompt: str) -> str:
    label = _loc._step_label(prompt)
    calls.append(label)
    if "GRAPH BUG-LOCALIZATION step" in prompt:
        return LOCATE_ANSWER
    if "PATH-SELECTION step" in prompt:
        assert "CANDIDATE EXECUTION PATHS" in prompt
        return SELECT_ANSWER
    if "PRUNED-PATH SIMULATION step" in prompt:
        assert "SOURCE OF THE FRAMES ON THIS PATH" in prompt
        return SIM_ANSWER
    if "LOCALIZATION-SUMMARY step" in prompt:
        assert "SIMULATION #1" in prompt
        return SUMMARY_ANSWER
    raise AssertionError(f"unexpected prompt ({label}): {prompt[:200]}")


def fake_trace(function_name: str, file_path: str = "", line: int = 0) -> str:
    data = CHAINS.get(function_name)
    if data is None:
        return "[no bug function identified; call chain unavailable]"
    return _sub._format_chain(data, function_name)


def test_parse_chain():
    """Bare-name mode: hops carry no class, so identity comes from the FUNCTION LOCATIONS map."""
    os.environ.pop("CHAIN_QUALIFIED_PATHS", None)
    text = _sub._format_chain(CHAINS["urlize"], "urlize")
    assert "QUALIFIED SITE PATHS" not in text, "qualified block leaked into the default render"
    p = _graph._parse_chain(text)
    assert not p["qualified"]
    assert ["django/template/base.py::render", "django/utils/html.py::urlize"] \
        in p["caller_paths"], p["caller_paths"]
    assert p["locations"]["urlize"] == "django/utils/html.py:280", p["locations"]
    assert "trim_punctuation" in p["sources"], list(p["sources"])
    assert "rstrip" in p["sources"]["trim_punctuation"][2]
    print("  parse_chain (bare): paths + locations + frame sources recovered  OK")


def test_parse_chain_qualified():
    """CHAIN_QUALIFIED_PATHS=1: node identity is <file>::<name>, class kept only for display."""
    os.environ["CHAIN_QUALIFIED_PATHS"] = "1"
    try:
        text = _sub._format_chain(CHAINS["urlize"], "urlize")
        p = _graph._parse_chain(text)
    finally:
        os.environ.pop("CHAIN_QUALIFIED_PATHS", None)
    assert p["qualified"], text[-800:]
    assert ["django/utils/html.py::urlize", "django/utils/html.py::__call__",
            "django/utils/html.py::handle_word", "django/utils/html.py::trim_punctuation"] \
        in p["callee_paths"], p["callee_paths"]
    # the class survives as a LABEL, never as identity (so a bare chain still unifies with it)
    assert p["displays"]["django/utils/html.py::__call__"] == "Urlizer.__call__", p["displays"]
    print("  parse_chain (qualified): site ids + display labels recovered  OK")


def test_node_identity_separates_same_name_defs():
    """The ansible-83909bf defect: same-named defs in DIFFERENT files must not merge into one
    node -- that is what let DFS stitch the galaxy API region to the CLI region."""
    a = {"target": "api_call", "depth": 3, "locations": {},
         "caller_paths": [["api_call", "__init__"]], "callee_paths": [],
         "caller_site_paths": [_sites(("api_call", "lib/api.py", "GalaxyAPI"),
                                      ("__init__", "lib/api.py", "GalaxyAPI"))],
         "callee_site_paths": [], "frame_sources": []}
    b = {"target": "cli_run", "depth": 3, "locations": {},
         "caller_paths": [["cli_run", "__init__"]], "callee_paths": [],
         "caller_site_paths": [_sites(("cli_run", "lib/cli.py", "GalaxyCLI"),
                                      ("__init__", "lib/cli.py", "GalaxyCLI"))],
         "callee_site_paths": [], "frame_sources": []}
    os.environ["CHAIN_QUALIFIED_PATHS"] = "1"
    try:
        g = _graph.CallGraph()
        g.add_chain("api_call", _sub._format_chain(a, "api_call"))
        g.add_chain("cli_run", _sub._format_chain(b, "cli_run"))
    finally:
        os.environ.pop("CHAIN_QUALIFIED_PATHS", None)
    inits = sorted(n for n in g.nodes if _graph._bare(n) == "__init__")
    assert inits == ["lib/api.py::__init__", "lib/cli.py::__init__"], inits
    # ... and with the two __init__s distinct there is NO route from one region to the other
    joined = [" -> ".join(p) for p in _graph.enumerate_paths(g)]
    assert not any("api.py" in j and "cli.py" in j for j in joined), joined
    print(f"  node identity: same-named defs stay distinct ({inits}); no cross-region "
          f"stitching  OK")


def test_anchor_coverage_is_site_exact():
    """ansible-83909bf: MIN-COVER reported 4/4 anchors while the anchor ``GalaxyLogin.__init__``
    was never on a selected path -- a route through ``AnsibleError.__init__`` satisfied it by
    BARE NAME. Coverage must be decided on the node, and an anchor the cover misses must still be
    walked via its own traced route."""
    # two independent regions that share the method name __init__ and nothing else
    login = {"target": "__init__", "depth": 3, "locations": {}, "frame_sources": [],
             "caller_paths": [["execute_login", "__init__"]],
             "callee_paths": [["__init__", "get_credentials"]],
             "caller_site_paths": [_sites(("execute_login", "cli/galaxy.py", "GalaxyCLI"),
                                          ("__init__", "galaxy/login.py", "GalaxyLogin"))],
             "callee_site_paths": [_sites(("__init__", "galaxy/login.py", "GalaxyLogin"),
                                          ("get_credentials", "galaxy/login.py", "GalaxyLogin"))]}
    api = {"target": "_add_auth_token", "depth": 3, "locations": {}, "frame_sources": [],
           "caller_paths": [["_call_galaxy", "_add_auth_token"]],
           "callee_paths": [["_add_auth_token", "__init__", "__add__"]],
           "caller_site_paths": [_sites(("_call_galaxy", "galaxy/api.py", "GalaxyAPI"),
                                        ("_add_auth_token", "galaxy/api.py", "GalaxyAPI"))],
           "callee_site_paths": [_sites(("_add_auth_token", "galaxy/api.py", "GalaxyAPI"),
                                        ("__init__", "errors/__init__.py", "AnsibleError"),
                                        ("__add__", "yaml/objects.py", "AnsibleUnicode"))]}
    os.environ["CHAIN_QUALIFIED_PATHS"] = "1"
    try:
        g = _graph.CallGraph()
        g.add_chain("_add_auth_token", _sub._format_chain(api, "_add_auth_token"))
        g.add_chain("__init__", _sub._format_chain(login, "__init__"))
    finally:
        os.environ.pop("CHAIN_QUALIFIED_PATHS", None)

    # (a) the anchor is recorded as the SITE the tracer anchored on, not the bare name
    assert "galaxy/login.py::__init__" in g.anchors, g.anchors
    assert "__init__" not in g.anchors, g.anchors

    # (b) a path through the OTHER __init__ does not cover it
    impostor = ("galaxy/api.py::_add_auth_token", "errors/__init__.py::__init__",
                "yaml/objects.py::__add__")
    assert not _graph.covers_anchor(impostor, "galaxy/login.py::__init__")
    assert _graph.covers_anchor(impostor, "galaxy/api.py::_add_auth_token")

    # (c) end to end: the cover must select a path that really visits GalaxyLogin.__init__
    paths = _graph.enumerate_paths(g)
    cover = _graph.minimal_cover(paths, g.anchors, set())
    for a in g.anchors:
        assert any(_graph.covers_anchor(p, a) for p in cover), (a, cover)
    assert any("galaxy/login.py::__init__" in p for p in cover), cover

    # (d) when the cover CANNOT reach an anchor, its own traced route is kept as a rescue seed
    assert g.anchor_seed.get("galaxy/login.py::__init__"), g.anchor_seed
    print("  anchor coverage: site-exact; impostor rejected; rescue seed retained  OK")


def test_anchor_rescue_survives_the_sim_budget():
    """An anchor no cover path visits is simulated via its own route, ranked ahead of top-k."""
    cover = [("a.py::x", "a.py::y")]
    uncovered_seed = ("b.py::caller", "b.py::z")
    plan = [(p, "minimal cover of the candidate bug sites") for p in cover]
    plan.append((uncovered_seed, "anchor rescue"))
    plan.append((("c.py::hot", "c.py::hot2"), "highest total visited count"))
    kept = plan[:2]   # GRAPH_SIM_MAX=2 would still keep cover + rescue
    assert uncovered_seed in [p for p, _w in kept], kept
    print("  anchor rescue outranks top-k under the simulation budget  OK")


def test_graph_and_pruning():
    g = _graph.CallGraph()
    for name in ("urlize", "trim_punctuation", "escape"):
        g.add_chain(name, _sub._format_chain(CHAINS[name], name))
    st = g.stats()
    # handle_word -> trim_punctuation is traced by BOTH urlize and trim_punctuation: one edge.
    assert st["edges"] < st["raw_paths"] * 2, st
    H = "django/utils/html.py::"
    assert (H + "handle_word", H + "trim_punctuation") in g.edges, sorted(g.edges)
    # anchors are node ids on a site-qualified chain, bare names otherwise (see _anchor_node)
    assert sorted(_graph._bare(a) for a in g.anchors) == ["escape", "trim_punctuation",
                                                          "urlize"], g.anchors
    assert g.file_of(H + "trim_punctuation") == "django/utils/html.py"
    assert g.file_of("trim_punctuation") == "django/utils/html.py"   # bare (anchor) lookup

    paths = _graph.enumerate_paths(g)
    assert paths, "no end-to-end paths enumerated"
    joined = [" -> ".join(p) for p in paths]
    # redundant subpaths are pruned: no path may be contained in another
    for a in joined:
        assert sum(1 for b in joined if f" -> {a} -> " in f" -> {b} -> ") == 1, a
    assert any((H + "urlize -> Urlizer -> " + H + "handle_word -> " + H + "trim_punctuation")
               in j for j in joined), joined

    idents = set(PROBLEM.split())
    cover = _graph.minimal_cover(paths, g.anchors, idents)
    assert all(any(_graph.covers_anchor(p, a) for p in cover) for a in g.anchors), (cover,
                                                                                    g.anchors)
    # 3 anchors, but urlize/trim_punctuation share a route -> 2 paths suffice
    assert len(cover) == 2, cover
    # (4) top-k by TOTAL VISITED COUNT: sum over the path's nodes of how many distinct paths
    # visit that node -- NOT by path length.
    visits = _graph.node_visit_counts(paths)
    assert visits["django/template/base.py::render"] == len(paths), visits   # trunk
    assert visits[H + "smart_urlquote"] == 1, visits                         # private leaf
    topk = _graph.top_by_visits(paths, cover, k=1, universe=paths)
    assert len(topk) == 1 and topk[0] not in cover, topk
    rest = [p for p in paths if p not in cover]
    scores = {p: _graph.path_visit_score(p, visits) for p in rest}
    assert scores[topk[0]] == max(scores.values()), scores
    # Ranking is visit-driven, not length-driven: between two paths of the SAME length, the one
    # whose hops the rest of the graph keeps routing through wins. (The score is an unnormalized
    # SUM, so it is still length-sensitive across paths of different lengths -- by design.)
    trunk = ("template_filter", "django/template/base.py::render", H + "urlize")  # every path
    branch = ("Urlizer", H + "handle_word", H + "smart_urlquote")   # a private tail
    assert (_graph.path_visit_score(trunk, visits)
            > _graph.path_visit_score(branch, visits)), (visits, trunk, branch)
    print(f"  graph: {st['nodes']} nodes / {st['edges']} edges from {st['raw_paths']} raw paths; "
          f"{len(paths)} pruned paths -> cover={len(cover)} + top-visited=1 "
          f"(score {scores[topk[0]]})  OK")


def test_anchor_bounds():
    """The floor tops up a short answer; the ceiling truncates a long one."""
    import types
    short = "FUNCTION: urlize\nFILE: django/utils/html.py\nLINE: 280\nREASON: r\n"
    topup = ("ALT-FUNCTION-2: trim_punctuation | django/utils/html.py | strips it\n"
             "ALT-FUNCTION-3: escape | django/utils/html.py | escaping order\n")
    seen = []

    def q(prompt):
        seen.append("MORE" if "below the minimum" in prompt else "LOCATE")
        return topup if seen[-1] == "MORE" else short

    r = types.SimpleNamespace(query_fn=q, problem_statement=PROBLEM, history="",
                              _log=lambda m: None)
    fn, _f, _l, _r, _raw, alts = _graph._locate_anchors(r)
    assert seen == ["LOCATE", "MORE"], seen
    assert fn == "urlize" and [a for a, _f, _h in alts] == ["trim_punctuation", "escape"], alts
    assert 1 + len(alts) >= _graph.GRAPH_ANCHOR_MIN

    # ceiling: 9 alts offered, GRAPH_ANCHOR_MAX-1 kept, no top-up query
    long = short + "".join(
        f"ALT-FUNCTION-{i}: f{i} | a/b.py | h{i}\n" for i in range(2, 11))
    seen.clear()
    r2 = types.SimpleNamespace(query_fn=lambda p: (seen.append("LOCATE"), long)[1],
                               problem_statement=PROBLEM, history="", _log=lambda m: None)
    _fn, _f, _l, _r, _raw, alts2 = _graph._locate_anchors(r2)
    assert seen == ["LOCATE"], seen
    assert len(alts2) == _graph.GRAPH_ANCHOR_MAX - 1, len(alts2)
    print(f"  anchor bounds: floor topped 1 -> {1+len(alts)} anchors (1 extra call); "
          f"ceiling kept {1+len(alts2)} of 10  OK")


def test_builtin_ctor_hook():
    """`Cls.__init__` -> `Cls.__new__` for IMMUTABLE builtin subclasses only (ansible-6cc97447:
    the summary named _AnsibleUnicode.__init__ in all three runs; only __new__ can fix a str)."""
    DECL = {
        "_AnsibleUnicode": "objects.py:12:class _AnsibleUnicode(str):",
        "_AnsibleMapping": "objects.py:20:class _AnsibleMapping(dict):",   # MUTABLE
        "_AnsibleSequence": "objects.py:28:class _AnsibleSequence(list):",          # MUTABLE
        "Point": "geo.py:5:class Point(tuple):",                                    # immutable
        "Plain": "svc.py:3:class Plain(Base):",                                     # not builtin
    }
    prev = _loc._REPO_SEARCH_TOOL
    # stands in for the container grep: return the decl line for the class named in the pattern
    _loc._REPO_SEARCH_TOOL = lambda pat, glob="": next(
        (v for k, v in DECL.items() if re.escape(k) in pat), "")
    try:
        reasons = {}
        cases = [
            ("lib/x/objects.py :: _AnsibleUnicode.__init__", "lib/x/objects.py :: _AnsibleUnicode.__new__"),
            ("lib/x/objects.py :: _AnsibleMapping.__init__", "lib/x/objects.py :: _AnsibleMapping.__init__"),
            ("lib/x/objects.py :: _AnsibleSequence.__init__", "lib/x/objects.py :: _AnsibleSequence.__init__"),
            ("lib/x/geo.py :: Point.__init__", "lib/x/geo.py :: Point.__new__"),
            ("lib/x/svc.py :: Plain.__init__", "lib/x/svc.py :: Plain.__init__"),
            ("lib/x/objects.py :: _AnsibleUnicode", "lib/x/objects.py :: _AnsibleUnicode"),
            ("lib/x/objects.py :: some_function", "lib/x/objects.py :: some_function"),
        ]
        for src, want in cases:
            got = _graph._fix_builtin_ctor_hook(src, reasons)
            assert got == want, (src, got, want)
        assert "immutable" in reasons["lib/x/objects.py :: _AnsibleUnicode.__new__"]
    finally:
        _loc._REPO_SEARCH_TOOL = prev
    print("  builtin ctor hook: str/tuple -> __new__; dict/list/non-builtin untouched  OK")


def test_end_to_end():
    calls.clear()
    _graph.install_graph_localization()
    r = _sub.SubAgentReproducer(
        query_fn=fake_query, trace_fn=fake_trace,
        problem_statement=PROBLEM, history="(the explore agent grepped urlize)",
        log=lambda m: print(m),
    )
    f = r.run()

    assert f.ok, "findings not ok"
    assert calls[0] == "LOCATE", calls
    assert "SELECT_PATHS" in calls and "SUMMARIZE" in calls, calls
    n_sims = calls.count("SIMULATE_PATH")
    assert n_sims >= 2, calls
    # 4 candidates named, one of which has no definition -> 3 graph anchors + 1 note
    assert len(f.alt_candidates) == 3, f.alt_candidates
    assert f.graph_summary["unreachable_anchors"] == ["make_absolute"], f.graph_summary
    assert sorted(_graph._bare(a) for a in f.graph_summary["anchors"]) == [
        "escape", "trim_punctuation", "urlize"], f.graph_summary["anchors"]
    assert len(f.graph_summary["cover_paths"]) == 2, f.graph_summary["cover_paths"]
    assert len(f.graph_summary["top_visited_paths"]) == 1, f.graph_summary["top_visited_paths"]
    assert f.graph_summary["top_visited_scores"][0] > 0, f.graph_summary["top_visited_scores"]
    assert len(f.graph_summary["selected_paths"]) == 1, f.graph_summary["selected_paths"]
    assert len(f.graph_summary["simulated_paths"]) == n_sims

    assert "django/utils/html.py :: trim_punctuation" in f.files_to_edit, f.files_to_edit
    assert "django/utils/html.py :: handle_word" in f.files_to_edit, f.files_to_edit
    assert any("escape" in s for s in _loc._DEMOTED_SITES), _loc._DEMOTED_SITES
    assert f.root_cause.startswith("trim_punctuation"), f.root_cause
    assert r.site_reasons, "no per-site provenance recorded"
    assert r.mined_path and r.path_sim_ran
    # the summary block downstream harvests key on must be present and parseable
    assert "=== SUMMARY OF THE SIMULATIONS" in f.simulation
    print(f"  end-to-end: {len(calls)} model calls {calls}; sites={f.files_to_edit}  OK")


def test_path_selection_record_is_persisted():
    """The selection ledger must reach the pipeline record -- including when the run dies before
    step 7. 83909bf's coverage bug was invisible for exactly this reason: what the cover decided
    lived only in the log."""
    def query(prompt):
        # die inside the simulate loop -- step 5 has its own handler, so a failure there is
        # non-fatal by design and would not exercise the outer recovery path
        if "PRUNED-PATH SIMULATION step" in prompt:
            raise RuntimeError("simulator exploded")
        return fake_query(prompt)

    calls.clear()
    _graph.install_graph_localization()
    r = _sub.SubAgentReproducer(query_fn=query, trace_fn=fake_trace, problem_statement=PROBLEM,
                                history="", log=lambda m: None)
    f = r.run()
    rec = f.graph_summary
    assert rec["stage"].startswith("error:"), rec["stage"]
    # everything decided before the failure survived
    assert rec["nodes"] and rec["edges"], rec
    assert rec["candidate_paths"] >= 1, rec
    assert rec["cover_paths"], rec
    assert "uncovered_by_cover" in rec, rec
    assert rec["top_visited_paths"], rec
    # every nomination was recorded before the first simulation ran
    assert rec["path_selection"] and not any(e["simulated"] for e in rec["path_selection"]), rec
    assert _graph.LAST_GRAPH_RECORD is rec, "module-level record must point at the live dict"
    print(f"  path selection: persisted through a crash in the simulate loop "
          f"(stage={rec['stage']}, cover={len(rec['cover_paths'])}, "
          f"nominations={len(rec['path_selection'])})  OK")


def test_path_selection_ledger_is_complete():
    """A completed run records every nomination with its source, coverage and outcome."""
    calls.clear()
    _graph.install_graph_localization()
    r = _sub.SubAgentReproducer(query_fn=fake_query, trace_fn=fake_trace,
                                problem_statement=PROBLEM, history="", log=lambda m: None)
    f = r.run()
    rec = f.graph_summary
    assert rec["stage"] == "done", rec["stage"]
    sel = rec["path_selection"]
    assert sel and all(e["source"] in ("cover", "rescue", "topk", "llm", "fallback")
                       for e in sel), sel
    for e in sel:
        assert e["hops"] and e["nodes"] and "covers" in e, e
        assert isinstance(e["simulated"], bool) and isinstance(e["dropped_by_budget"], bool), e
    assert sum(1 for e in sel if e["simulated"]) == len(rec["simulated_paths"]), sel
    # the anchor ledger answers "was this candidate ever WALKED?" for every anchor
    assert set(rec["anchor_coverage"]) == set(rec["anchors"]), rec["anchor_coverage"]
    for a, v in rec["anchor_coverage"].items():
        assert set(v) >= {"display", "file", "in_cover", "simulated", "simulated_by"}, v
    assert "anchors_never_simulated" in rec, rec
    print(f"  path selection ledger: {len(sel)} nominations "
          f"({[e['source'] for e in sel]}), anchor coverage recorded  OK")


def test_downstream_harvests():
    """The pipeline's deterministic folds must fire on the graph variant's findings."""
    from simagent import pipeline as _pipe

    f = _sub.SubAgentFindings()
    f.simulation = (
        "trim_punctuation strips it: the truncated URL is FIRST CONSTRUCTED here, in "
        "trim_punctuation at django/utils/html.py:380.\n\n"
        "=== SUMMARY OF THE SIMULATIONS (consolidated over 3 pruned path(s)) ===\n"
        + SUMMARY_ANSWER
    )
    f.files_to_edit = []
    added = _pipe._apply_summary_findings(f, log=lambda m: None)
    assert any("trim_punctuation" in s for s in added), added
    assert any("handle_word" in s for s in added), added
    assert f.root_cause, "root cause not recovered from the summary block"
    print(f"  downstream: _apply_summary_findings folded {added} back in  OK")


class _StubEnv:
    """Minimal container stand-in: every shell probe comes back empty."""

    def execute(self, cmd, timeout=None):
        return {"output": "", "returncode": 0}


class _StubModel:
    def query(self, messages):
        return {"content": fake_query(messages[-1]["content"])}


def test_phase2_wiring():
    """``_run_localization`` under LOCALIZE_MODE=graph: gate + every enricher on graph findings."""
    from simagent import pipeline as _pipe

    calls.clear()
    prev_mode = _pipe.LOCALIZE_MODE
    prev_trace = _sub.make_trace_call_chain_runner
    _pipe.LOCALIZE_MODE = "graph"
    _sub.make_trace_call_chain_runner = lambda env, **kw: fake_trace
    try:
        traj, usage = [], []
        f = _pipe._run_localization(
            _StubModel(), _StubEnv(), "/testbed", PROBLEM,
            "(the explore agent grepped urlize)", traj, usage,
            instance=None, sweep_cands=[],
        )
    finally:
        _pipe.LOCALIZE_MODE = prev_mode
        _sub.make_trace_call_chain_runner = prev_trace

    assert f.ok and f.files_to_edit, f
    assert "django/utils/html.py :: trim_punctuation" in f.files_to_edit, f.files_to_edit
    assert f.site_reasons, "site provenance lost in the phase-2 hand-off"
    assert any("escape" in s for s in f.demoted_sites), f.demoted_sites
    assert getattr(f, "graph_summary", None), "graph stats not exposed to the pipeline record"
    # the trajectory the pipeline persists must carry every step, labelled
    steps = [e.get("step") for e in traj]
    assert steps.count("TRACE") == 4, steps          # one per named candidate
    assert "SELECT_PATHS" in steps and "SUMMARIZE" in steps, steps
    print(f"  phase-2 wiring: steps={steps}; sites={f.files_to_edit}  OK")


def test_untraced_data_anchor_kept_unless_refuted():
    """qutebrowser-f8e7fea0 regression: a LOCATE anchor that is a module-level data symbol must
    (a) render with its definition + referencing functions instead of the misleading
    "NO definition" note, (b) be surfaced to SUMMARIZE for explicit adjudication, and
    (c) survive into files_to_edit unless the summary explicitly rules it out."""
    locate_with_data = LOCATE_ANSWER + (
        "ALT-FUNCTION-5: URL_SCHEMES | django/utils/html.py | "
        "registry dict gating which schemes urlize links\n")

    # the rendered chain itself must carry the exists-as-data evidence
    text = _sub._format_chain(CHAINS["URL_SCHEMES"], "URL_SCHEMES")
    assert "module-level DATA structure" in text, text
    assert "Urlizer.handle_word (django/utils/html.py)" in text, text
    assert "URL_SCHEMES = {'http': True, 'https': True}" in text, text
    assert "has NO definition in the repo" not in text, text

    def _run(summary_answer):
        seen = {"summary_prompt": ""}

        def query(prompt: str) -> str:
            label = _loc._step_label(prompt)
            calls.append(label)
            if "GRAPH BUG-LOCALIZATION step" in prompt:
                return locate_with_data
            if "PATH-SELECTION step" in prompt:
                return SELECT_ANSWER
            if "PRUNED-PATH SIMULATION step" in prompt:
                return SIM_ANSWER
            if "LOCALIZATION-SUMMARY step" in prompt:
                seen["summary_prompt"] = prompt
                return summary_answer
            raise AssertionError(f"unexpected prompt ({label})")

        calls.clear()
        _graph.install_graph_localization()
        r = _sub.SubAgentReproducer(
            query_fn=query, trace_fn=fake_trace,
            problem_statement=PROBLEM, history="(the explore agent grepped urlize)",
            log=lambda m: None,
        )
        return r.run(), seen

    # (a) not ruled out -> kept, with untraced-anchor provenance, and SUMMARIZE saw it
    f, seen = _run(SUMMARY_ANSWER)
    assert "ADDITIONAL CANDIDATES named at LOCATE" in seen["summary_prompt"]
    assert "URL_SCHEMES" in seen["summary_prompt"]
    assert "module-level DATA structure" in seen["summary_prompt"], \
        "candidate evidence block not inlined into the summary prompt"
    assert "django/utils/html.py :: URL_SCHEMES" in f.files_to_edit, f.files_to_edit
    assert "URL_SCHEMES" in f.graph_summary["unreachable_anchors"], f.graph_summary

    # (b) explicitly ruled out -> demoted, NOT kept
    f2, _seen2 = _run(SUMMARY_ANSWER
                      + "RULED-OUT: django/utils/html.py :: URL_SCHEMES -- "
                        "scheme registry is complete; no change needed\n")
    assert "django/utils/html.py :: URL_SCHEMES" not in f2.files_to_edit, f2.files_to_edit
    assert any("URL_SCHEMES" in s for s in _loc._DEMOTED_SITES), _loc._DEMOTED_SITES
    print("  untraced data anchor: rendered as existing, adjudicated by SUMMARIZE, "
          "kept-unless-refuted  OK")


if __name__ == "__main__":
    test_parse_chain()
    test_parse_chain_qualified()
    test_node_identity_separates_same_name_defs()
    test_anchor_coverage_is_site_exact()
    test_anchor_rescue_survives_the_sim_budget()
    test_graph_and_pruning()
    test_anchor_bounds()
    test_builtin_ctor_hook()
    test_end_to_end()
    test_path_selection_record_is_persisted()
    test_path_selection_ledger_is_complete()
    test_downstream_harvests()
    test_phase2_wiring()
    test_untraced_data_anchor_kept_unless_refuted()
    print("\nALL GRAPH-LOCALIZATION TESTS PASSED")
