"""Adversarial coverage for heterogeneous requirement-list serialization."""

from types import SimpleNamespace

from simagent import pipeline as sp


def test_markdown_markers_and_short_obligations():
    text = """Requirements:
- Must be idempotent.
* The parser should retain values.
+ Callers shall receive a tuple.
"""
    assert sp._requirement_bullets(text) == [
        "Must be idempotent.",
        "The parser should retain values.",
        "Callers shall receive a tuple.",
    ]


def test_inline_no_space_bullets_are_split():
    text = (
        "Requirements: -The parser must preserve order. "
        "-The purge path must remove stale members. "
        "-`map_obj_to_commands` should compact consecutive ranges."
    )
    assert sp._requirement_bullets(text) == [
        "The parser must preserve order.",
        "The purge path must remove stale members.",
        "`map_obj_to_commands` should compact consecutive ranges.",
    ]


def test_code_span_leading_bullets_are_not_dropped():
    text = """
- `PlayIterator` must insert a noop task in lockstep mode.
- `_set_failed_state` should preserve the handler phase.
"""
    assert [b.split()[0] for b in sp._requirement_bullets(text)] == [
        "`PlayIterator`", "`_set_failed_state`"]


def test_numbered_task_unicode_and_lettered_lists():
    text = """
1. The first rule must hold.
2) The second rule should hold.
(3) The third rule is required.
- [x] The checked rule must hold.
\u2022 The Unicode rule shall hold.
a) The lettered rule must hold.
"""
    bullets = sp._requirement_bullets(text)
    assert len(bullets) == 6, bullets
    assert bullets[0].startswith("The first")
    assert bullets[-1].startswith("The lettered")


def test_continuations_fold_but_headings_do_not_leak():
    text = """
- The parser must preserve values across
  wrapped continuation lines and punctuation.

Notes:
- The result should remain ordered.
"""
    assert sp._requirement_bullets(text) == [
        "The parser must preserve values across wrapped continuation lines and punctuation.",
        "The result should remain ordered.",
    ]


def test_json_quoted_and_literal_newline_field():
    text = '"- The parser must retain \\"quoted\\" values.\\n\\n- The result should be stable."'
    assert sp._requirement_bullets(text) == [
        'The parser must retain "quoted" values.',
        "The result should be stable.",
    ]


def test_prose_hyphens_urls_rules_and_fenced_code_are_not_items():
    text = """
- The parser must preserve input - output mappings and https://host/foo-bar.
---
```python
- This apparent bullet must not be audited.
```
- The value should remain between -1 and -5.
"""
    bullets = sp._requirement_bullets(text)
    assert bullets == [
        "The parser must preserve input - output mappings and https://host/foo-bar.",
        "The value should remain between -1 and -5.",
    ]


def test_unbulleted_normative_paragraph_fallback():
    text = "The implementation must preserve insertion order across every supported input."
    assert sp._requirement_bullets(text) == [text]


def test_consumers_share_the_generalized_parser():
    instance = {
        "requirements": (
            "-`parse_pairs` must process each `:` delimiter. "
            "-`parse_pairs` should accept the parameters `first`, `second`, and `third`."
        ),
        "interface": "",
    }
    ledger = sp._obligation_ledger(instance)
    assert ledger and ledger[0]["symbol"] == "parse_pairs"
    assert len(ledger[0]["obligations"]) == 2
    checklist = sp._audit_checklist(instance, SimpleNamespace(
        synth_vectors=[], free_constants=[], signature_contracts=[]))
    assert [pid for pid, _ in checklist[:2]] == ["R1", "R2"]


def test_vector_synthesis_grounds_specials_against_requirements(monkeypatch):
    rule = "Git inputs with `#` must retain the subdirectory separator."
    reply = f"RULE: {rule}\nVECTOR: repo#subdir"
    monkeypatch.setattr(sp._sub, "_make_agent_query_fn", lambda _ns: lambda _p: reply)
    monkeypatch.setattr(sp, "_structural_rules", lambda _instance: [])
    got = sp._synth_vectors(object(), {"requirements": "- " + rule})
    assert got == [{"rule": rule, "vectors": ["repo#subdir"]}]
