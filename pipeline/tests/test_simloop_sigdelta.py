"""The sim-loop must REQUIRE a default-path case when a signature gains optional parameters.

ansible-6cc97447: `__new__(cls, value)` became `__new__(cls, object='', encoding=None,
errors=None)`. All four generated cases supplied `encoding=` explicitly, so the only input that
discriminates correct from incorrect behaviour -- `_AnsibleUnicode(b'Hello')`, where `str` gives
the repr `"b'Hello'"` rather than a decode -- was never tested. The gained parameters are
derivable from the diff, so that case is now mandatory.

    python test_simloop_sigdelta.py
"""
from __future__ import annotations
from simagent import plainrepair as pr

REAL = """\
diff --git a/lib/ansible/parsing/yaml/objects.py b/lib/ansible/parsing/yaml/objects.py
--- a/lib/ansible/parsing/yaml/objects.py
+++ b/lib/ansible/parsing/yaml/objects.py
@@ -12,22 +12,34 @@
-    def __new__(cls, value):
-        return _datatag.AnsibleTagHelper.tag_copy(value, str(value))
+    def __new__(cls, object='', encoding=None, errors=None):
+        if isinstance(object, (bytes, bytearray)):
+            object = _converters.to_text(object)
+        return _datatag.AnsibleTagHelper.tag_copy(object, str(object))
"""

UNITS = [{"id": "U2", "file": "lib/ansible/parsing/yaml/objects.py",
          "symbol": "_AnsibleUnicode.__new__", "code": "def __new__(cls, object='', ...): ..."}]


def test_delta_detected():
    d = pr._signature_deltas(REAL)
    assert "__new__" in d, d
    assert d["__new__"]["gained_optional"] == ["encoding", "errors", "object"], d
    assert d["__new__"]["old"] == ["value"], d
    print(f"  delta: old={d['__new__']['old']} gained={d['__new__']['gained_optional']}  OK")


def test_mandatory_case_emitted():
    blk = pr._required_case_block(UNITS, REAL)
    assert "U2#G" in blk and "WITHOUT any of the gained parameters" in blk, blk
    assert "`encoding`" in blk and "before this patch it took (value)" in blk, blk
    print(f"  mandatory slot: {blk[:96]}...  OK")


def test_prompt_carries_it():
    p = pr._CASES_PROMPT_BATCH.format(
        context="C", problem_statement="P", requirements="R", interface="I",
        units=pr._render_units(UNITS), required=pr._required_case_block(UNITS, REAL))
    assert "MANDATORY CASES" in p and "U2#G" in p and "not optional" in p, p[:300]
    print("  4.2 prompt carries the mandatory-case section  OK")


def test_silent_when_no_delta():
    """A patch that changes a body but not a signature must add no mandatory case."""
    body_only = REAL.replace("-    def __new__(cls, value):", "-    x = 1").replace(
        "+    def __new__(cls, object='', encoding=None, errors=None):", "+    x = 2")
    assert pr._signature_deltas(body_only) == {}, pr._signature_deltas(body_only)
    assert pr._required_case_block(UNITS, body_only) == "(none)"
    # and a signature that gains a REQUIRED (non-default) param is not a default-path case
    req = REAL.replace("object='', encoding=None, errors=None", "value, extra")
    assert pr._signature_deltas(req) == {}, pr._signature_deltas(req)
    print("  silent on body-only edits and on newly-REQUIRED params  OK")


if __name__ == "__main__":
    test_delta_detected(); test_mandatory_case_emitted()
    test_prompt_carries_it(); test_silent_when_no_delta()
    print("\nALL SIGNATURE-DELTA TESTS PASSED")
