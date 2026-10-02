"""Small, repository-independent evidence and scope policies for pipeline phases."""

import re
from dataclasses import dataclass


SCOPE_NOTE = """
Scope clarification: application-owned configuration schemas, defaults, and resources are
implementation files when the issue directly requires changing their behavior. A generic
configuration-file restriction protects unrelated environment, build, dependency and test-runner
configuration; it does not prohibit those task-required application changes. Honor any explicit
issue-specific prohibition. Do not modify existing tests or submit temporary probes.
Check each requirement end to end: declaration/registration, producer, consumer, and externally
observable result. Reading a missing option with a fallback does not implement its declaration.
Test-only interfaces describe validation, not production classes to add to the shipped patch.
Production code must behave the same with tests and real callers. Never detect mocks, test
framework identity, or an active test runner to choose a different implementation merely to
satisfy assertions. Preserve contracts through ordinary runtime interfaces; report incompatible
specification/test expectations rather than introducing a test-specific execution path.
"""


def clarify_scope(template):
    """Reconcile generic boilerplate, never rewrite an issue's specific restrictions."""
    if SCOPE_NOTE in template:
        return template
    return template.rstrip() + "\n\n" + SCOPE_NOTE


_LOC_PREFIX_RE = re.compile(r"^(?:[\w./-]+\.\w{1,4}:\d+\s*[:\-\u2013\u2014]?\s*|[Ll]ine\s+\d+\s*[:\-\u2013\u2014]\s*|L?\d{1,5}[:|]\s*)")


def quote_text(text):
    """Strip Markdown decoration and normalize whitespace for source citation checks.

    Tolerates the citation styles models use for a quoted line: a leading location
    (``filepathcategory.py:140 `def data(...)```, ``Line 505: `if key_str ...```), a backticked
    span followed by commentary (the span is the quote), and a bare ``NNN:`` line number
    (ccpipe b1_two_arms: on 3 of 4 reasoning-arm losses every lens finding was dropped as
    "no quoted TRACE line" for these reasons, two of them naming the hidden-test failure)."""
    text = (text or "").strip()
    if text.startswith("> "):
        text = text[2:].strip()
    spans = re.findall(r"`([^`\n]{6,})`", text)
    if spans:
        text = max(spans, key=len)
    else:
        text = _LOC_PREFIX_RE.sub("", text)
    if len(text) >= 2 and text.startswith("`") and text.endswith("`"):
        text = text[1:-1]
    return " ".join(text.split())


@dataclass(frozen=True)
class TestEvidence:
    status: str
    failures: frozenset
    executed: int
    returncode: object
    reason: str = ""


def pytest_evidence(output, returncode=None):
    """Require affirmative execution evidence; an empty failure regex is not success."""
    failures = frozenset(re.findall(r"^(?:FAILED|ERROR)\s+(\S+\.py(?:::\S+)?)", output, re.M))
    counts = {}
    # Only pytest summary lines, not arbitrary assertion strings containing 'N passed'.
    for line in output.splitlines():
        if re.search(r"\b(?:passed|failed|error|errors|skipped|deselected)\b.*\bin\s+[\d.]+s", line):
            for count, kind in re.findall(r"\b(\d+)\s+(passed|failed|errors?|skipped|deselected)\b", line):
                counts[kind.rstrip("s")] = int(count)
    executed = counts.get("passed", 0) + counts.get("failed", 0)
    # rstrip above leaves passed/failed intact (neither ends with s).
    fatal = (returncode not in (None, 0, 1) or
             re.search(r"INTERNALERROR|unrecognized arguments|Fatal Python error|no tests ran", output))
    if fatal:
        return TestEvidence("invalid", failures, executed, returncode, "runner did not complete usable tests")
    if failures:
        return TestEvidence("failed", failures, executed, returncode)
    if returncode == 0 and executed > 0 and not counts.get("error") and not counts.get("failed"):
        return TestEvidence("passed", failures, executed, returncode)
    return TestEvidence("invalid", failures, executed, returncode, "no affirmative successful execution evidence")
