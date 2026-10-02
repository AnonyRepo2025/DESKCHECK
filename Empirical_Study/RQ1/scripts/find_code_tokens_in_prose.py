from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from find_execution_reasoning import (
    DEFAULT_RESOLUTIONS,
    DEFAULT_TRAJS_DIR,
    extract_context,
    iter_assistant_thoughts,
    load_resolutions,
    strip_code_blocks,
    _list_traj_files,
)


PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # `identifier`, `dotted.path`, `func()` quoted with backticks in prose.
    (re.compile(r"`[A-Za-z_][\w\.]*(?:\([^`]{0,80}\))?`"),
     "backtick-identifier"),

    # Inline function / method call with parentheses: `foo(...)`, `obj.bar()`.
    (re.compile(
        r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*"
        r"\((?:[^()\n]{0,120})\)"),
     "function-call"),

    # Attribute / dotted access without a call: `obj.attr`, `module.sub.name`
    (re.compile(
        r"\b(?!e\.g\b)[a-z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*"
        r"(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*\b"),
     "attribute-access"),

    # Indexed / subscripted access: `arr[0]`, `kwargs['x']`, `m["key"]`.
    (re.compile(
        r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*"
        r"\[\s*[`'\"]?[\w\-+]+[`'\"]?\s*\]"),
     "indexed-access"),

    # Assignment expression in prose: `x = 5`, `count = None`, `kwargs = {}`
    (re.compile(
        r"\b[a-zA-Z_]\w*\s*=\s*"
        r"(?!=)"
        r"(?:[`'\"][^`'\"\n]{1,40}[`'\"]|\d+(?:\.\d+)?|"
        r"True|False|None|\[[^\n\]]{0,40}\]|\{[^\n\}]{0,40}\}|"
        r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*(?:\([^()\n]{0,40}\))?)"),
     "assignment-expression"),

    # Comparison in prose: `x == 5`, `n != 0`, `i < len(arr)`.
    (re.compile(
        r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*\s*"
        r"(?:==|!=|<=|>=|<|>)\s*"
        r"(?:[`'\"][^`'\"\n]{0,40}[`'\"]|\d+(?:\.\d+)?|"
        r"True|False|None|[a-zA-Z_]\w*(?:\([^()\n]{0,40}\))?)"),
     "comparison-expression"),

    # Type / sentinel literals shown verbatim: True, False, None.
    # Word-boundary + case-sensitive — "true" mid-sentence shouldn't count.
    (re.compile(r"\b(?:True|False|None)\b"),
     "type-literal"),

    # Decorators: @cached_property, @staticmethod, @app.route.
    (re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z_]\w*(?:\.\w+)*"),
     "decorator"),

    # Numeric / arrow-style operators in prose: `->`, `=>`, `+=`, `-=`,
    # `**`, `//`. Common when the model is showing transitions.
    (re.compile(r"(?:->|=>|\+=|-=|\*=|/=|//=|\*\*=|//|\*\*)"),
     "operator-token"),

    # Lambda expression: `lambda x: x + 1`.
    (re.compile(r"\blambda\s+[a-zA-Z_]\w*(?:\s*,\s*[a-zA-Z_]\w*)*\s*:"),
     "lambda-expression"),

    (re.compile(
        r"(?<![A-Za-z0-9_/])"
        r"[A-Za-z0-9_-][A-Za-z0-9_./-]{0,199}\.(?:py|js|ts|tsx|jsx|cpp|cc|c|h|hpp|"
        r"java|go|rs|rb|sh|json|yaml|yml|toml|md)\b"),
     "file-path"),

    # snake_case identifier in prose: `my_var`, `some_helper_fn`.
    (re.compile(r"(?<![A-Za-z0-9_])[a-z][a-z0-9]*(?:_[a-z0-9]+)+(?![A-Za-z0-9_])"),
     "snake-case-identifier"),

    # camelCase identifier in prose: `myFunc`, `parseInputArgs`.
    (re.compile(r"(?<![A-Za-z0-9_])[a-z]+(?:[A-Z][a-z0-9]+){1,}(?![A-Za-z0-9_])"),
     "camel-case-identifier"),

    # SCREAMING_SNAKE_CASE constant: `MAX_RETRIES`, `API_TIMEOUT_S`.
    (re.compile(r"(?<![A-Za-z0-9_])[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+(?![A-Za-z0-9_])"),
     "constant-identifier"),

    # PascalCase class-like identifier in prose: `BaseHandler`, `MyClass`.
    # Two-letter exclusions (`OK`, `IO`) prevent acronyms from matching.
    (re.compile(r"(?<![A-Za-z0-9_])[A-Z][a-z]+[A-Z][A-Za-z0-9]*(?![A-Za-z0-9_])"),
     "pascal-case-identifier"),

    # Single-quoted / double-quoted short string literal that looks like
    (re.compile(
        r"(?<![A-Za-z0-9])"
        r"(?:'[A-Za-z_./][\w./\-]{0,40}'|\"[A-Za-z_./][\w./\-]{0,40}\")"
        r"(?![A-Za-z0-9])"),
     "code-string-literal"),

    # Regex / bytes prefix on a string: r'...', b'...', rb"...".
    (re.compile(r"(?<![A-Za-z0-9_])[rRbBfFuU]{1,2}[`'\"]"),
     "string-prefix"),
]


# SWE-bench-style task prompts wrap the issue text in
# `<pr_description>...</pr_description>`. If the model echoes that block back
# (or the trajectory format inlines it into a "thought"), the literal PR text
# is not the model's own execution reasoning and would feed false positives
# into every regex pass. Strip it before any pattern sees it.
PR_DESCRIPTION_RE = re.compile(
    r"<pr_description\b[^>]*>.*?</pr_description\s*>",
    re.DOTALL | re.IGNORECASE,
)


def _strip_pr_description(text: str) -> str:
    """Drop `<pr_description>...</pr_description>` blocks (closing-tag-anchored).
    A standalone newline is left behind so adjacent prose doesn't fuse."""
    return PR_DESCRIPTION_RE.sub("\n", text)


# A "prose-context" probe: the surrounding window should look like English,
# not like more code. We count short, common function words. If we see at
# least N of them, the match really is sitting inside natural language.
PROSE_CONTEXT_RE = re.compile(
    r"\b(?:the|is|are|was|were|be|been|being|"
    r"this|that|these|those|it|its|"
    r"a|an|of|in|on|to|for|with|from|by|at|as|if|"
    r"and|or|but|so|because|since|when|while|then|"
    r"we|you|i|they|he|she|"
    r"will|would|should|could|can|may|might|must|"
    r"have|has|had|do|does|did|"
    r"not|no|yes|"
    r"which|what|who|where|how|why|"
    r"means|need|note|seems|appears|expects|expected|"
    r"call|called|calls|use|used|using|return|returns|"
    r"check|checks|see|look|set|sets|get|gets)\b",
    re.I,
)


MIN_THOUGHT_CHARS = 50
DEFAULT_MIN_SCORE = 1
DEFAULT_PROSE_HITS = 1
DEFAULT_MIN_QUALIFYING_TURNS = 1
DEFAULT_MAX_TOKENS_PER_LABEL = 50


# ---------------------------------------------------------------------------
# Pass 2: variable-state patterns
# ---------------------------------------------------------------------------
STATE_PATTERNS: list[tuple[str, str]] = [
    # `X` is None / `X` is True / `X` is `something` / `X` is 0
    (r"`\w[\w\.\[\]]*`\s+(?:is|was|will be|would be|should be|becomes?|"
     r"remains?|stays?|equals?)\s+"
     r"(?:None|null|nil|empty|True|False|undefined|"
     r"`[^`]+`|\"[^\"]+\"|'[^']+'|-?\d+(?:\.\d+)?)",
     "exact-variable-state-identification"),

    # X is set/reset/initialized/assigned/updated/cleared to/with/as Y
    (r"`?\w[\w\.\[\]]*`?\s+(?:is|gets?|was|are|were)\s+"
     r"(?:set|reset|initialized|assigned|updated|cleared|overwritten|"
     r"incremented|decremented|toggled|populated|filled)\s+"
     r"(?:to|with|as|by|from)\b",
     "variable-state-update"),

    # set X to Y / sets X to None / setting X to a list
    (r"\bsets?\s+`?\w[\w\.\[\]]*`?\s+to\s+"
     r"(?:None|null|empty|True|False|`[^`]+`|\d|the\s+\w+|a\s+\w+|an\s+\w+)",
     "variable-state-update"),

    # When/after/if X is/becomes/holds/equals ...
    # The negative lookahead drops English pronouns from the variable slot —
    # "if this is a", "when it was None" tend to reference meta-context, not
    # an actual identifier whose state we're tracking.
    (r"\b(?:when|after|once|while|if|until|before)\s+"
     r"(?!(?:it|this|that|there|these|those|he|she|they|we|you|i)\b)"
     r"`?\w[\w\.\[\]]*`?\s+"
     r"(?:is|was|becomes?|gets?|equals?|holds?|contains?)\s+"
     r"(?:None|null|empty|True|False|set|reset|`|\d|the\s+|a\s+|an\s+)",
     "hypothetical-variable-state-assumption"),

    # X holds / X contains / X stores ... — restricted to DYNAMIC state.
    # The naive form ("anything contains the anything") fired on static
    # structure talk: "the file that contains the `delete_cookie` method",
    # "the patch only contains the single-line change". Two precise forms:
    # (a) backticked subject — a runtime entity by construction.
    (r"`\w[\w\.\[\]]*`\s+(?:(?:still|now|only|also|always|never)\s+)?"
     r"(?:holds?|contains?|stores?)\s+"
     r"(?:(?:the\s+)?(?:value|reference|None|null|empty|a\s+|an\s+|the\s+)|"
     r"`[^`\n]+`|'[^'\n]{0,40}'|\"[^\"\n]{0,40}\")",
     "exact-variable-state-identification"),
    # (b) plain-word subject, but the subject must not be a relative pronoun
    # or a static container (file/patch/module/...), and the object must be
    # a value-shaped thing (backticked, sentinel, or a runtime-data noun).
    (r"\b(?!(?:to|might|may|could|would|will|can|cannot|must|that|which|"
     r"who|likely|probably|also|only|file|files|patch|diff|module|modules|"
     r"directory|folder|repo|repository|document|page|script|test|tests|"
     r"class|method|function)\b)"
     r"\w[\w\.\[\]]*\s+(?:(?:still|now|only|also|always|never)\s+)?"
     r"(?:holds?|contains?|stores?)\s+"
     r"(?:the\s+|a\s+|an\s+)?"
     r"(?:`[^`\n]+`|'[^'\n]{0,40}'|\"[^\"\n]{0,40}\"|"
     r"values?\b|references?\b|None\b|null\b|empty\b|keys?\b|"
     r"entr(?:y|ies)\b|elements?\b|items?\b|duplicates?\b|lists?\b|"
     r"dicts?\b|dictionar(?:y|ies)\b|tuples?\b|strings?\b|integers?\b|"
     r"numbers?\b|cop(?:y|ies)\b|instances?\b)",
     "exact-variable-state-identification"),

    # X is/was coerced / cast / converted / rounded to <concrete value> —
    # the value requirement keeps this on runtime data ("was coerced to
    # `5`") and off generic refactor talk ("converted to a classmethod").
    (r"\b(?:is|was|are|were|gets?|got|being)\s+"
     r"(?:coerced|cast|converted|truncated|rounded|normalized|"
     r"serialized|encoded|decoded)\s+to\s+"
     r"(?:`[^`\n]{1,60}`|None|True|False|-?\d+(?:\.\d+)?|"
     r"'[^'\n]{0,40}'|\"[^\"\n]{0,40}\")",
     "variable-state-update"),

    # X is an `Y` instance / subclass — runtime type-state of a value
    (r"\bis\s+an?\s+`[^`\n]{1,60}`\s+(?:instance|object|subclass)\b",
     "exact-variable-state-identification"),

    # value of X is / current value of X.
    # Excluded targets: literal True/False/None (those are values themselves,
    # not variables whose state we'd be discussing).
    (r"\b(?:current\s+)?value\s+of\s+"
     r"(?!(?:`?(?:True|False|None|null)`?)\b)"
     r"`?\w[\w\.\[\]]*`?\s+(?:is|was|becomes?)\b",
     "exact-variable-state-identification"),

    # X currently is / has / holds / contains ...
    (r"`?\w[\w\.\[\]]*`?\s+currently\s+(?:is|has|holds?|contains?)\b",
     "exact-variable-state-identification"),

    # default value of X / initial value for X
    (r"\b(?:default|initial)\s+value\s+(?:of|for)\s+`?\w[\w\.\[\]]*`?\b",
     "variable-state-initialization"),

    # X starts as / X is initially / X begins empty.
    (r"`?\w[\w\.\[\]]*`?\s+(?:starts?|is\s+initially|begins?)\s+"
     r"(?:as|out|empty|None)",
     "variable-state-initialization"),

    # (Generic mutation-verb pattern was removed — it produced too many
    # false positives like "change the behavior" / "update the function".
    # Real value-mutation talk is already covered by variable-state-update
    # and exact-variable-state-identification.)

    # X is now None / X is no longer empty
    (r"`?\w[\w\.\[\]]*`?\s+is\s+(?:no longer|now)\s+"
     r"(?:None|null|empty|set|`|the|a |an )",
     "exact-variable-state-identification"),

    # defaults to `X` / defaults to None / defaults to 'email'
    (r"\bdefaults?\s+to\s+"
     r"(?:`[^`\n]{1,60}`|None|True|False|-?\d+|"
     r"'[^'\n]{0,40}'|\"[^\"\n]{0,40}\")",
     "variable-state-initialization"),

    # hardcoded value / hard-coded to / hardcoded as — a fixed-value statement
    (r"\b(?:hardcoded|hard-coded)\s+(?:value|to|as)\b",
     "variable-state-initialization"),

    # both forms share the same dict / object / reference — aliasing state
    (r"\b(?:shares?|sharing|share)\s+the\s+same\s+"
     r"(?:dict|dictionary|list|object|instance|reference|state)\b",
     "exact-variable-state-identification"),

    # X is set/updated/assigned unconditionally / directly / twice / again —
    # a state mutation whose manner (not target value) carries the reasoning.
    (r"\b(?:is|are|was|were|gets?)\s+"
     r"(?:set|updated|assigned|initialized|overwritten|cleared|"
     r"mutated|modified|reset)\s+"
     r"(?:unconditionally|directly|twice|again|first|early|late|"
     r"eagerly|lazily|implicitly|explicitly)\b",
     "variable-state-update"),

    # sets/overwrites/forces/resets the <name> to <value> — like the bare
    # "sets X to Y" pattern above but tolerant of an article-led subject
    # ("explicitly sets the offset to 0") and of mutation-verb variants.
    (r"\b(?:sets?|setting|overwrites?|overwrote|overwriting|forces?|forcing|"
     r"resets?|hardcodes?)\s+"
     r"(?:the\s+|a\s+|an\s+|its\s+)?`?\w[\w\.\[\]]*`?\s+to\s+"
     r"(?:`[^`\n]{1,60}`|None|True|False|-?\d+|empty|zero|"
     r"'[^'\n]{0,40}'|\"[^\"\n]{0,40}\")",
     "variable-state-update"),

    # ensures it is at least 1 / ensures `X` is not None — value-bound talk
    (r"\bensures?\s+(?:that\s+)?(?:it|`[^`\n]{1,40}`|\w[\w.]*)\s*(?:'?s)?\s+"
     r"(?:is\s+)?(?:at\s+least|at\s+most|always|never|non-?empty|"
     r"not\s+None|positive)\b",
     "exact-variable-state-identification"),
]

# Kinds considered high-confidence (everything except generic "mutation-verb").
STRONG_KINDS: set[str] = {kind for _, kind in STATE_PATTERNS if kind != "mutation-verb"}

_COMPILED_STATE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), kind) for p, kind in STATE_PATTERNS
]


def find_state_phrases(snippet: str) -> tuple[list[tuple[str, str]], set[str]]:
    """Run every state regex against `snippet` and return (phrases, kinds).

    `phrases` is a list of (kind, matched-substring) in pattern order.
    `kinds` is the distinct set of kinds that fired.
    """
    phrases: list[tuple[str, str]] = []
    kinds: set[str] = set()
    for cre, kind in _COMPILED_STATE_PATTERNS:
        for mo in cre.finditer(snippet):
            phrases.append((kind, mo.group(0)))
            kinds.add(kind)
    return phrases, kinds


# ---------------------------------------------------------------------------
# Pass 3: call-dependency patterns
# ---------------------------------------------------------------------------
CALL_DEP_PATTERNS: list[tuple[str, str]] = [
    # `Y` calls `X` / `Y` invokes `X` / `Y` triggers `X`
    (r"`\w[\w\.]*(?:\([^()`]{0,80}\))?`\s+"
     r"(?:calls?|invokes?|triggers?|fires?|kicks?\s+off|"
     r"is\s+calling|will\s+call)\s+"
     r"`\w[\w\.]*(?:\([^()`]{0,80}\))?`",
     "forward-call"),

    # `X` is called / `X` gets invoked / `X` was triggered (no source side)
    (r"`\w[\w\.]*(?:\([^()`]{0,80}\))?`\s+"
     r"(?:is|gets?|was|are|were|will\s+be|would\s+be|"
     r"is\s+being|gets?\s+being)\s+"
     r"(?:called|invoked|triggered|dispatched|forwarded|reached)\b",
     "backward-call"),

    # called/invoked from/by/via `Y` — passive form with explicit source
    (r"\b(?:called|invoked|triggered|dispatched|forwarded|reached)\s+"
     r"(?:by|from|via|in|inside)\s+"
     r"`\w[\w\.]*(?:\([^()`]{0,80}\))?`",
     "backward-call"),

    # When `X` is called / after `X` gets invoked
    (r"\b(?:when|after|once|before|while|until|if)\s+"
     r"`\w[\w\.]*(?:\([^()`]{0,80}\))?`\s+"
     r"(?:is|gets?)\s+"
     r"(?:called|invoked|triggered|dispatched|reached)\b",
     "backward-call"),

    # `Y` delegates to `X` / `Y` dispatches to `X` / `Y` forwards to `X`
    (r"`\w[\w\.]*`\s+"
     r"(?:delegates?|dispatches?|forwards?|defers?|hands?\s+off)\s+"
     r"(?:to\s+)?"
     r"`\w[\w\.]*(?:\([^()`]{0,80}\))?`",
     "forward-call"),

    # which / it / that / this method  → calls `X` (anaphoric link).
    # The subject is a pronoun, but the target must still be backticked.
    (r"\b(?:which|it|that)\s+"
     r"(?:calls?|invokes?|triggers?|fires?|"
     r"dispatches?\s+to|delegates?\s+to|forwards?\s+to|defers?\s+to)\s+"
     r"`\w[\w\.]*(?:\([^()`]{0,80}\))?`",
     "forward-call"),

    # Y goes through `X` / routes through `X` — only the target needs to be
    # backticked; the source can be a noun phrase ("the loopback goes through
    # `ICRS`"). The verb stem keeps it specific enough to avoid prose noise.
    (r"\b\w[\w\.]*\s+"
     r"(?:goes?|routes?|flows?|runs?)\s+"
     r"(?:currently\s+|now\s+|always\s+)?"
     r"through\s+"
     r"`\w[\w\.]*(?:\([^()`]{0,80}\))?`",
     "forward-call"),

    # `X` is passed to `Y` / `X` gets passed into `Y`
    (r"`\w[\w\.]*`\s+"
     r"(?:is|gets?|was)\s+"
     r"passed\s+(?:to|into)\s+"
     r"`\w[\w\.]*(?:\([^()`]{0,80}\))?`",
     "forward-call"),

    # passes these parameters to runshell_db() / sends the value into `Y` —
    # active-voice value flow with a loose (not necessarily backticked)
    # subject; the explicit call-shaped target keeps it precise.
    (r"\b(?:passes?|passed|passing|sends?|sent|forwards?|hands?)\s+"
     r"(?:these\s+|those\s+|the\s+|its\s+|this\s+|all\s+|both\s+)?"
     r"[\w`'\".\[\]]{1,40}(?:\s+\w+){0,2}\s+"
     r"(?:to|into)\s+"
     r"`?\w[\w\.]*(?:\([^()`\n]{0,60}\))?`?",
     "forward-call"),

    # when/if/after we|it|the code call(s) ... — pronoun-subject call narration
    (r"\b(?:when|if|after|once|before)\s+"
     r"(?:we|you|it|they|the\s+\w+)\s+calls?\b",
     "backward-call"),

    # instead of using/calling `X` — contrastive call-path reasoning
    (r"\binstead\s+of\s+(?:using|calling|invoking|going\s+through)\s+"
     r"(?:the\s+)?`\w[\w\.\(\)]*`",
     "forward-call"),

    # tries to access `X` / tried to call `Y` — attempted call/lookup
    (r"\btr(?:y|ies|ied|ying)\s+to\s+"
     r"(?:access|call|invoke|look\s*up|read|fetch|import|load|find|open)\s+"
     r"`[^`\n]+`",
     "forward-call"),

    # adds/passes `X` to the `Y` call / constructor — argument wiring
    (r"\b(?:adds?|added|adding|passes?|passed|passing)\s+"
     r"`[^`\n]{1,60}`\s+to\s+the\s+`[^`\n]{1,60}`\s+"
     r"(?:call|method|function|constructor)\b",
     "forward-call"),

    # X is instantiated/constructed/called with `args` — call-site detail
    (r"\b(?:is|are|was|were|gets?)\s+"
     r"(?:instantiated|constructed|created|initialized|invoked|called)\s+"
     r"with\s+`[^`\n]+`",
     "backward-call"),

    # before/after setting / calling / checking ... — execution-order
    # narration between two code actions.
    (r"\b(?:before|after)\s+"
     r"(?:setting|calling|checking|assigning|updating|validating|"
     r"returning|raising|iterating|appending|writing|reading)\b",
     "execution-order"),
]

_COMPILED_CALL_DEP_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), kind) for p, kind in CALL_DEP_PATTERNS
]


def find_call_dep_phrases(snippet: str) -> tuple[list[tuple[str, str]], set[str]]:
    """Run every call-dependency regex against `snippet`."""
    phrases: list[tuple[str, str]] = []
    kinds: set[str] = set()
    for cre, kind in _COMPILED_CALL_DEP_PATTERNS:
        for mo in cre.finditer(snippet):
            phrases.append((kind, mo.group(0)))
            kinds.add(kind)
    return phrases, kinds


# ---------------------------------------------------------------------------
# Output-of-method/code patterns
# ---------------------------------------------------------------------------

# Subpattern: a value an output expression can produce — backticked atom,
# Python sentinel, number, quoted string, or one of a handful of noun phrases
# that show up around "the output is …".
_VALUE_RE = (
    r"(?:`[^`\n]{1,80}`"
    r"|None|null|nil|True|False|NotImplemented|nothing|empty"
    r"|-?\d+(?:\.\d+)?"
    r"|\"[^\"\n]{0,60}\""
    r"|'[^'\n]{0,60}'"
    r"|the\s+(?:value|result|string|list|tuple|dict|set|empty|same)\b"
    r"|an?\s+(?:value|result|string|list|tuple|dict|set|empty)\b)"
)

OUTPUT_PATTERNS: list[tuple[str, str]] = [
    # `X` returns / yields / emits / produces / generates / prints / outputs
    # / gives <VALUE>
    (r"`\w[\w\.]*(?:\([^()`]{0,80}\))?`\s+"
     r"(?:returns?|returned|will\s+return|would\s+return|"
     r"yields?|emits?|produces?|generates?|prints?|outputs?|"
     r"evaluates?\s+to|gives?(?:\s+(?:us|me|back|you))?)\s+"
     + _VALUE_RE,
     "active-returns"),

    # X is/gets returned / yielded / emitted / produced / generated /
    # printed / output. The verb list itself is specific enough that we
    # don't need backticks on the subject ("values are yielded",
    # "no warnings are emitted").
    (r"\b(?:`\w[\w\.]*(?:\([^()`]{0,80}\))?`|\w[\w_]*)\s+"
     r"(?:is|gets?|was|are|were|will\s+be|would\s+be|"
     r"is\s+being|gets?\s+being)\s+"
     r"(?:returned|yielded|emitted|produced|generated|printed|output)\b",
     "passive-returns"),

    # the (expected/actual/computed/...) result / output / return value
    # (of `X`) is <VALUE>. One optional adjective is allowed before the
    # head noun.
    (r"\bthe\s+"
     r"(?:\w+\s+)?"
     r"(?:result|output|return\s+value|returned\s+value)\s+"
     r"(?:of\s+`\w[\w\.\(\)]*`\s+)?"
     r"(?:is|was|becomes?|will\s+be|would\s+be|equals?)\s+"
     + _VALUE_RE,
     "result-of"),

    # X evaluates to <VALUE> — backtick on subject is optional, but the
    # value side must be backticked / a literal, which keeps this precise.
    (r"`?\w[\w\.\[\]]*`?\s+evaluates?\s+to\s+" + _VALUE_RE,
     "evaluates-to"),

    # this / it / that  → returns / gives us / yields / prints <VALUE>
    (r"\b(?:this|it|that|which)\s+"
     r"(?:returns?|yields?|emits?|produces?|generates?|prints?|outputs?|"
     r"gives?(?:\s+(?:us|me|back|you))?|evaluates?\s+to)\s+"
     + _VALUE_RE,
     "anaphoric-returns"),

    # `X` ends up returning / yielding / as / with <VALUE>
    (r"`?\w[\w\.]*`?\s+ends?\s+up\s+"
     r"(?:returning|yielding|emitting|producing|"
     r"as|with|being)\s+"
     + _VALUE_RE,
     "ends-up"),

    # calling `X` returns / gives <VALUE>
    (r"\bcalling\s+`\w[\w\.\(\)]*`\s+"
     r"(?:returns?|yields?|emits?|produces?|prints?|"
     r"gives?(?:\s+(?:us|me|back|you))?)\s+"
     + _VALUE_RE,
     "calling-returns"),

    # `X` returns the <noun phrase> — value described as a plain noun phrase
    # ("`get_email_field_name()` returns the email field name").
    (r"`\w[\w\.]*(?:\([^()`]{0,80}\))?`\s+"
     r"(?:returns?|returned|will\s+return|would\s+return|"
     r"yields?|emits?|produces?|gives?|outputs?|prints?|evaluates?\s+to)\s+"
     r"(?:the|an?|its|their)\s+\w+",
     "active-returns"),

    # splits / converts / maps ... (in)to `Y` — value transformation with a
    # concrete (backticked or list-shaped) result.
    (r"\b(?:splits?|converts?|turns?|transforms?|parses?|expands?|"
     r"collapses?|coerces?|casts?|maps?)\s+"
     r"[^.\n]{1,60}\s+(?:in)?to\s+"
     r"(?:`[^`\n]{1,80}`|\[[^\]\n]{0,60}\])",
     "value-transformation"),

    # produces `X` instead of `Y` — concrete value contrast on both sides.
    # The lookbehinds drop "use/using `X` instead of `Y`" — that's a static
    # fix prescription (which code to write), not an observed runtime value.
    (r"(?<!\buse\s)(?<!uses\s)(?<!used\s)(?<!using\s)(?<!mport\s)"
     r"`[^`\n]{1,60}`\s+instead\s+of\s+`[^`\n]{1,60}`",
     "value-contrast"),

    # ===== runtime-exception: the runtime effect of executing the code =====
    # causes/triggers/results in a TypeError / a crash / infinite recursion
    (r"\b(?:causes?|causing|caused|triggers?|triggering|results?\s+in|"
     r"resulting\s+in|leads?\s+to|leading\s+to)\s+"
     r"(?:an?\s+|the\s+)?"
     r"(?:(?-i:[A-Z]\w*(?:Error|Exception|Warning))|"
     r"(?:an?\s+)?(?:crash|exception|error|infinite\s+loop|overflow|"
     r"recursion))",
     "runtime-exception"),

    # causes/triggers `X` where `X` is a backticked code entity rather than
    # an exception name — a runtime effect ("triggers the `__set__` method",
    # "causes `references_column` to always return false").
    (r"\b(?:causes?|causing|caused|triggers?|triggering|results?\s+in|"
     r"resulting\s+in|leads?\s+to|leading\s+to)\s+"
     r"(?:an?\s+|the\s+)?`[^`\n]{1,60}`",
     "runtime-effect"),

    # fails / crashes with a TypeError / with `X`
    (r"\b(?:fails?|failed|failing|errors?|crashes?|crashed)\s+with\s+"
     r"(?:an?\s+|the\s+)?"
     r"(?:(?-i:[A-Z]\w*(?:Error|Exception|Warning))|`[^`\n]{1,60}`)",
     "runtime-exception"),

    # won't be caught / is never caught — exception propagation reasoning
    (r"\b(?:won'?t|isn'?t|not|never|doesn'?t\s+get)\s+(?:be\s+)?caught\b",
     "runtime-exception"),

    # named runtime hazards: division by zero, infinite recursion, ...
    (r"\b(?:division\s+by\s+zero|infinite\s+(?:loop|recursion)|off-by-one|"
     r"out\s+of\s+(?:range|bounds)|integer\s+overflow|race\s+condition|"
     r"deadlock)\b",
     "runtime-exception"),

    # backticked `return ...` / `raise ...` statement quoted in prose,
    # anchored to an EXECUTION verb. Unanchored, this matched static edit
    # talk ("should be changed to `return NotImplemented`", "insert the
    # methods after `return name`") far more often than control flow.
    # (a) execution verb before the quoted statement
    (r"\b(?:reach(?:es|ed|ing)?|hits?|hitting|executes?|executing|"
     r"skips?|skipping|falls?\s+through\s+to|stops?\s+at|ends?\s+at|"
     r"evaluates?|never\s+reach(?:es|ed)?)\s+"
     r"(?:the\s+)?`(?:return|raise)\s[^`\n]{1,120}`",
     "inline-return-statement"),
    # (b) quoted statement followed by an execution predicate
    (r"`(?:return|raise)\s[^`\n]{1,120}`\s+"
     r"(?:is\s+(?:reached|executed|hit|skipped|never)|"
     r"executes?|runs?|fires?|never\s+(?:runs?|executes?|fires?))\b",
     "inline-return-statement"),

    # ===== verified-behavior: observed execution outcome =====
    # confirms/confirmed/verified the bug / issue / behavior
    (r"\b(?:confirms?|confirmed|verified|verifies)\s+"
     r"(?:that\s+)?(?:the\s+)?(?:bug|issue|fix|error|behavior|problem)",
     "verified-behavior"),

    # (successfully) reproduced the issue / this reproduces the bug.
    # Past/3rd-person forms only: "Let me write a script TO REPRODUCE the
    # issue" is a plan, not an observed execution outcome, so the bare
    # infinitive is deliberately not matched.
    (r"\b(?:successfully\s+)?reproduce[sd]\s+the\s+"
     r"(?:issue|bug|error|problem|crash|failure)\b",
     "verified-behavior"),

    # preserves / maintains (insertion) order — output-property statement
    (r"\b(?:preserv(?:es?|ing)|maintain(?:s|ing)?)\s+"
     r"(?:insertion\s+)?order\b",
     "verified-behavior"),

    # all/the/existing tests pass / the test fails — an observed run outcome.
    # The determiner keeps hypothetical "if tests pass" style prose out.
    (r"\b(?:all|the|existing|new|both|these|those|every)\s+"
     r"(?:\w+\s+)?tests?\s+(?:now\s+)?(?:pass(?:es|ed)?|fail(?:s|ed)?)\b",
     "verified-behavior"),
]

_COMPILED_OUTPUT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), kind) for p, kind in OUTPUT_PATTERNS
]


def find_output_phrases(snippet: str) -> tuple[list[tuple[str, str]], set[str]]:
    """Run every output-of-method regex against `snippet`."""
    phrases: list[tuple[str, str]] = []
    kinds: set[str] = set()
    for cre, kind in _COMPILED_OUTPUT_PATTERNS:
        for mo in cre.finditer(snippet):
            phrases.append((kind, mo.group(0)))
            kinds.add(kind)
    return phrases, kinds


# ---------------------------------------------------------------------------
# Loop-execution patterns
# ---------------------------------------------------------------------------
LOOP_EXEC_PATTERNS: list[tuple[str, str]] = [
    # X iterates over Y / X loops through Y / X cycles through Y
    (r"\b(?:`\w[\w\.]*`|\w[\w_]*)\s+"
     r"(?:iterates?|iterating|loops?|looping|cycles?|cycling)\s+"
     r"(?:over|through|across|down)\s+"
     r"(?:the\s+|each\s+|every\s+|all\s+|its\s+)?"
     r"`?\w[\w\.]*(?:\([^()`]{0,80}\))?`?",
     "post-loop"),

    # walks/traverses (through) Y — subject is optional ("by walking through
    # each case", "traverses the MRO").
    (r"\b(?:walks?|walking|traverses?|traversing)\s+"
     r"(?:through\s+|down\s+|across\s+)?"
     r"(?:the|each|every|all|its)\s+"
     r"(?:`\w[\w\.]*`|(?-i:\w*[A-Z][A-Za-z0-9]*)|\w*_\w+)",
     "post-loop"),

    # for each iteration
    (r"\bfor\s+each\s+iteration\b",
     "iteration-level"),

    # First / next / last / each iteration
    (r"\b(?:first|second|third|next|last|previous|each|every|same|current|"
     r"final|prior|nth|n-?th|inner|outer)\s+iteration\b",
     "iteration-level"),

    # On/at/during/after the Nth iteration
    (r"\b(?:on|at|in|during|after|before)\s+"
     r"(?:the\s+)?"
     r"(?:first|second|third|next|last|each|every|previous|current|final)\s+"
     r"iteration\b",
     "iteration-level"),

    # In/inside/during the (for|while|nested|inner|outer) loop
    (r"\b(?:in|inside|within|throughout|during|later\s+in|earlier\s+in)\s+"
     r"(?:the|this|that|a|each|inner|outer|nested)\s+"
     r"(?:for\s+|while\s+|inner\s+|outer\s+|nested\s+)?"
     r"loops?\b",
     "iteration-level"),

    # Loop-control flow: enters/exits/breaks out of/continues/skips the loop
    (r"\b(?:enters?|enter\s+into|exits?|leaves?|terminates?|"
     r"breaks?\s+out\s+of|breaks?\s+from|"
     r"continues?\s+(?:to\s+(?:the\s+)?next|with|in)|"
     r"skips?|completes?\s+(?:without|with))\s+"
     r"(?:the|this|that|a)?\s*"
     r"(?:for\s+|while\s+|inner\s+|outer\s+|nested\s+)?"
     r"loops?\b",
     "post-loop"),

    # The/this for/while loop iterates / processes / executes / completes
    (r"\b(?:the|this|that|each)\s+"
     r"(?:for\s+|while\s+|inner\s+|outer\s+|nested\s+)?"
     r"loop\s+"
     r"(?:iterates?|loops?|processes?|executes?|runs?|continues?|"
     r"terminates?|exits?|completes?|finishes?|stops?|"
     r"begins?|starts?|breaks?)\b",
     "post-loop"),

    # a/the/this for|while loop (construct mention)
    (r"\b(?:a|the|this|that|each|nested|inner|outer)\s+"
     r"(?:for|while)\s+loops?\b",
     "post-loop"),

    # Loop concepts: loop body / condition / counter / variable / index
    (r"\bloop\s+"
     r"(?:body|condition|counter|variable|index|invariant|iterator|increment)\b",
     "post-loop"),

    # iterating gives / yields / produces ...
    (r"\biterating\s+(?:over\s+`?\w[\w\.]*`?\s+)?"
     r"(?:gives?|yields?|produces?|returns?)\b",
     "post-loop"),

    # escapes / converts / checks each element — element-wise iteration talk.
    # Third-person -s forms only: the imperative/infinitive base form
    # ("Let me CHECK EACH of these calls") is the agent narrating its own
    # workflow, not the code's iteration behavior.
    (r"\b(?:escapes|marks|converts|wraps|copies|checks|validates|"
     r"processes|transforms|deduplicates|normalizes)\s+each\s+\w+",
     "iteration-level"),

    # stops / truncates at the shortest sequence — iteration termination
    (r"\b(?:stops?|stopped|stopping|truncates?|cuts\s+off)\s+at\s+the\s+\w+",
     "post-loop"),

    # backticked `for ...` statement quoted in prose, anchored to an
    # execution verb — unanchored it matched patch descriptions ("the change
    # from `for dim in concatenated.dims:` to `for dim in concat_dims:`").
    # (a) iteration verb shortly before the quoted statement
    (r"\b(?:iterat\w+|loops?|looping|enters?|executes?|executing|runs?|"
     r"running|hits?|reach\w+)\s+[^`\n]{0,30}"
     r"`for\s[^`\n]{1,120}`",
     "post-loop"),
    # (b) quoted statement followed by a loop/execution predicate
    (r"`for\s[^`\n]{1,120}`\s*[-—:,]?\s*"
     r"(?:loop|iterates?|executes?|runs?|"
     r"(?:this\s+loop\s+)?(?:doesn'?t|never)\s+(?:execute|run)s?)\b",
     "post-loop"),
]

_COMPILED_LOOP_EXEC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), kind) for p, kind in LOOP_EXEC_PATTERNS
]


def find_loop_phrases(snippet: str) -> tuple[list[tuple[str, str]], set[str]]:
    """Run every loop-execution regex against `snippet`."""
    phrases: list[tuple[str, str]] = []
    kinds: set[str] = set()
    for cre, kind in _COMPILED_LOOP_EXEC_PATTERNS:
        for mo in cre.finditer(snippet):
            phrases.append((kind, mo.group(0)))
            kinds.add(kind)
    return phrases, kinds


# ---------------------------------------------------------------------------
# Conditional-execution patterns
# ---------------------------------------------------------------------------
COND_EXEC_PATTERNS: list[tuple[str, str]] = [
    # take/hit/enter/fall into the if/else/elif branch
    (r"\b(?:takes?|took|taking|hits?|hitting|enters?|entered|entering|"
     r"falls?|fell|falling|fallen)\s+"
     r"(?:into\s+|to\s+|through\s+)?"
     r"(?:the|this|that|a|an)\s+"
     r"(?:if|else|elif|then|else\s+if)\s+"
     r"(?:branch|clause|case|block|path|leg|arm)",
     "branch-taken"),

    # in/inside/within the if/else branch
    (r"\b(?:in|inside|within|throughout)\s+"
     r"(?:the|this|that|each)\s+"
     r"(?:if|else|elif|then)\s+"
     r"(?:branch|clause|case|block|path|leg|arm)",
     "branch-taken"),

    # this/the (else|if|elif|then) branch is taken / gets executed / will run
    (r"\b(?:the|this|that|each|either|both|first|second|inner|outer)\s+"
     r"(?:(?:if|else|elif|then|else\s+if)\s+)?"
     r"(?:branch|clause|case|path|leg|arm)\s+"
     r"(?:is\s+taken|gets?\s+taken|will\s+be\s+taken|was\s+taken|"
     r"is\s+chosen|gets?\s+chosen|"
     r"gets?\s+executed|is\s+executed|will\s+(?:run|execute|fire))",
     "branch-taken"),

    # falls through to the else / next case / default / fallback
    (r"\bfalls?\s+through\s+(?:to\s+)?"
     r"(?:the\s+)?(?:else|elif|next\s+case|default|fallback|next\s+\w+)",
     "branch-taken"),

    # goes/proceeds to/into the else/then branch
    (r"\b(?:goes?|going|proceeds?|jumps?)\s+"
     r"(?:into|to|through|down)\s+"
     r"(?:the\s+)?"
     r"(?:if|else|elif|then)\s+"
     r"(?:branch|clause|case|block|path)?",
     "branch-taken"),

    # The/this condition is True/False/met/satisfied/violated/already-X.
    # Head noun does NOT include "test" — in this domain "the test passes"
    # almost always means the unit test, not a predicate.
    (r"\b(?:the|this|that|first|second|inner|outer|original)\s+"
     r"(?:condition|check|guard|predicate|invariant)\s+"
     r"(?:is|was|becomes?|will\s+be|would\s+be)\s+"
     r"(?:True|False|None|met|satisfied|violated|unmet|"
     r"already\s+(?:satisfied|true|false|met))\b",
     "condition-evaluates"),

    # the condition `X` is / evaluates to / holds / fails / passes / matches
    # / would be / will be. Head noun excludes "test" for the same reason.
    (r"\b(?:the\s+)?(?:condition|check|guard|predicate|invariant)\s+"
     r"`[^`\n]{1,120}`\s+"
     r"(?:is|was|evaluates?\s+to|holds?|fails?|passes?|matches?|"
     r"becomes?|short-?circuits?|"
     r"would\s+be|will\s+be|could\s+be|should\s+be)\b",
     "condition-evaluates"),

    # the/this condition fails / passes / holds / matches / short-circuits.
    # Head noun excludes "test" — see comments above.
    (r"\b(?:the|this|that|first|second|inner|outer)\s+"
     r"(?:condition|check|guard|predicate|invariant)s?\s+"
     r"(?:fails?|passes?|holds?|succeeds?|matches?|"
     r"short-?circuits?|always\s+(?:fails?|passes?|holds?))",
     "condition-evaluates"),

    # short-circuit evaluation / X short-circuits and ...
    (r"\bshort-?circuits?\b",
     "condition-evaluates"),

    # returns/exits/breaks/skips/bails early — early-exit conditional pattern
    (r"\b(?:returns?|exits?|breaks?|skips?|bails?(?:\s+out)?)\s+early\b",
     "branch-taken"),

    # we hit / reach the else / `else`
    (r"\b(?:hits?|hit|reach(?:es|ed)?|reaching)\s+"
     r"(?:the\s+)?(?:`else`|else)\b",
     "branch-taken"),

    # The if/elif statement evaluates / runs / matches
    (r"\bthe\s+(?:if|elif|else|else\s+if|match|switch|ternary)\s+"
     r"(?:statement|expression|block|chain)\s+"
     r"(?:evaluates?|runs?|matches?|fires?|executes?|tests?)",
     "condition-evaluates"),

    # backticked `if ...:` / `while ...` / `except ...` statement quoted in
    # prose, anchored to an EXECUTION verb. Without the anchor this fired on
    # static edit talk ("change line 178 from `if dim != addend_dim:` to …",
    # "wrap it inside the `if formset.is_valid():` block").
    # (a) execution verb before the quoted guard
    (r"\b(?:checks?|checking|evaluates?|evaluating|enters?|entering|"
     r"hits?|hitting|reach(?:es|ed|ing)?|fails?|passes?|satisfies|"
     r"triggers?|skips?|due\s+to)\s+"
     r"(?:the\s+|a\s+|on\s+)?"
     r"`(?:if|elif|while|assert|except)\s[^`\n]{1,120}`",
     "condition-evaluates"),
    # (b) quoted guard followed by an evaluation predicate
    (r"`(?:if|elif|while|assert|except)\s[^`\n]{1,120}`\s+"
     r"(?:condition|check|guard|evaluates?|fails?|passes?|matches?|holds?|"
     r"short-?circuits?|succeeds?|triggers?|"
     r"is\s+(?:True|False|met|satisfied)|"
     r"would\s+be|will\s+be|never|always)\b",
     "condition-evaluates"),

    # enters / hits / reaches the `else` branch — like the first
    # branch-taken pattern above but tolerant of backticked keywords and of
    # except/try/finally blocks.
    (r"\b(?:enters?|entered|takes?|took|hits?|hit|reaches?|reached|"
     r"falls?\s+into|goes?\s+(?:in)?to)\s+"
     r"(?:the|this|that)\s+"
     r"`?(?:if|else|elif|except|try|finally|first|second)`?\s+"
     r"(?:branch|block|clause|case|path)\b",
     "branch-taken"),

    # the check / condition for `X` — a named, backticked predicate
    (r"\b(?:condition|check|guard)\s+(?:for|in|on)\s+`[^`\n]+`",
     "condition-evaluates"),

    # only runs / fires / applies if|when ... — guarded-execution narration
    (r"\bonly\s+(?:runs?|executes?|fires?|applies|triggers?|happens?|"
     r"matches?|raises?)\s+(?:if|when|for|once)\b",
     "branch-taken"),
]

_COMPILED_COND_EXEC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), kind) for p, kind in COND_EXEC_PATTERNS
]


def find_cond_phrases(snippet: str) -> tuple[list[tuple[str, str]], set[str]]:
    """Run every conditional-execution regex against `snippet`."""
    phrases: list[tuple[str, str]] = []
    kinds: set[str] = set()
    for cre, kind in _COMPILED_COND_EXEC_PATTERNS:
        for mo in cre.finditer(snippet):
            phrases.append((kind, mo.group(0)))
            kinds.add(kind)
    return phrases, kinds


# ---------------------------------------------------------------------------
# "OTHERS" — a catch-all category for execution-process reasoning that the
# state / call-dep / output / loop / cond detectors miss. Three rule families:
#
#   Rule 1: runtime exception / failure reasoning
#       — "fails with a TypeError", "raises ValueError", "X is raised",
#         "doesn't work on", "would fail"
#   Rule 2: expectation / contract mismatch
#       — "returns `X` instead of `Y`", "expects X but gets Y",
#         "incorrectly serializes the path"
#   Rule 3: dispatch coverage / missed cases
#       — "doesn't handle Compare nodes", "doesn't consider fields",
#         "only handles BinOp", "fails to recognize them"
# ---------------------------------------------------------------------------
# Exception names like `TypeError` / `ValueError` are CamelCase ending in
# Error|Exception|Warning. Under IGNORECASE we wrap the class-name subpattern
# in (?-i:...) so the leading uppercase is enforced.
_EXCEPTION_NAME_RE = r"(?-i:[A-Z]\w*(?:Error|Exception|Warning))"

OTHERS_PATTERNS: list[tuple[str, str]] = [
    # ===== Rule 1: exception / runtime-failure reasoning =====

    # raises [a/the] TypeError / raises `MyError`
    (r"\b(?:raises?|raised|raising)\s+(?:an?\s+|the\s+)?"
     r"(?:" + _EXCEPTION_NAME_RE + r"|`[^`\n]{1,80}`)",
     "raises-exception"),

    # throws [a] RuntimeException / threw `X`
    (r"\b(?:throws?|threw|throwing)\s+(?:an?\s+|the\s+)?"
     r"(?:" + _EXCEPTION_NAME_RE + r"|`[^`\n]{1,80}`)",
     "raises-exception"),

    # An|the|this <Error|Exception|Warning> is raised / thrown / emitted
    (r"\b(?:an?|the|this|that)\s+"
     + _EXCEPTION_NAME_RE +
     r"\s+(?:is|was|will\s+be|would\s+be|gets?|got)\s+"
     r"(?:raised|thrown|emitted|triggered|generated)\b",
     "raises-exception"),

    # fails with [a] TypeError / fails with `X`
    (r"\bfails?\s+with\s+(?:an?\s+|the\s+)?"
     r"(?:" + _EXCEPTION_NAME_RE + r"|`[^`\n]{1,80}`)",
     "raises-exception"),

    # would fail / will crash / could raise / might fail
    (r"\b(?:would|will|could|might|may)\s+(?:fail|crash|raise|error\s+out)\b",
     "raises-exception"),

    # X doesn't / won't / didn't work — runtime-behavior failure
    (r"\b(?:doesn'?t|don'?t|won'?t|wouldn'?t|didn'?t)\s+work\b",
     "raises-exception"),

    # incorrectly returns / serializes / parses / treats / sets / handles ...
    (r"\bincorrect(?:ly)?\s+"
     r"(?:returns?|serializes?|parses?|produces?|computes?|treats?|"
     r"sets?|handles?|stores?|emits?|raises?|yields?|outputs?|"
     r"formats?|interprets?|encodes?|decodes?|maps?|assigns?)\b",
     "raises-exception"),

    # fails to recognize / detect / handle / consider <X>
    (r"\bfails?\s+to\s+"
     r"(?:recognize|detect|handle|consider)\s+"
     r"(?:`[^`\n]{1,80}`|\w+)",
     "raises-exception"),
]

_COMPILED_OTHERS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), kind) for p, kind in OTHERS_PATTERNS
]


def find_others_phrases(snippet: str) -> tuple[list[tuple[str, str]], set[str]]:
    """Run every "others" regex (exception / mismatch / coverage)
    against `snippet`."""
    phrases: list[tuple[str, str]] = []
    kinds: set[str] = set()
    for cre, kind in _COMPILED_OTHERS_PATTERNS:
        for mo in cre.finditer(snippet):
            phrases.append((kind, mo.group(0)))
            kinds.add(kind)
    return phrases, kinds


OTHERS_ENABLED = True


# ---------------------------------------------------------------------------
# Pass 7: prose-execution narration on no-code-token turns.
#
# Designed to fire on the visible-response turns the code-token pass missed
# entirely — turns where the model summarizes the bug or fix in plain English
# without any backticked identifier, snake_case, or file path. Each pattern
# is anchored on a structural cue (contrastive verb pair, "when X <verb>"
# opener, RFC/spec citation, numbered fix-plan list, "during execution"
# framing) so we don't drift into the noisy bare-English false positives.
# ---------------------------------------------------------------------------

# Common code-action verbs reused across multiple patterns.
_CODE_VERB = (
    r"(?:converts?|treats?|returns?|yields?|generates?|produces?|emits?|"
    r"outputs?|subtracts?|adds?|removes?|drops?|skips?|stores?|sets?|"
    r"assigns?|reads?|writes?|raises?|throws?|takes?|enters?|exits?|"
    r"runs?|executes?|parses?|matches?|computes?|maps?|encodes?|decodes?|"
    r"validates?|filters?|sorts?|merges?|splits?|joins?|loads?|saves?|"
    r"sends?|receives?|fetches?|wraps?|unwraps?|registers?|dispatches?|"
    r"triggers?|fires?|handles?|resolves?|finds?|iterates?|loops?|"
    r"becomes?|holds?|contains?|fails?|gets?|"
    r"converted|treated|returned|yielded|generated|produced|emitted|"
    r"stored|set|assigned|raised|thrown|called|invoked|executed|"
    r"parsed|matched|computed|mapped|encoded|decoded|loaded|saved|"
    r"sent|received|fetched|wrapped|registered|dispatched|handled|"
    r"resolved|split|joined|sorted|filtered|merged|"
    r"accessed|updated|inserted|deleted|modified|created|destroyed|"
    r"closed|opened|initialized|serialized|deserialized|cached|cleared)"
)

PROSE_EXEC_PATTERNS: list[tuple[str, str]] = [
    # 1) Buggy-behavior contrast: "the (code|method|grammar|fix|...) <verb> X
    # (but|instead of|rather than|however) <verb> Y". The code-noun head plus
    # a *contrastive* marker keeps the pattern precise — standalone "then"
    # was dropped because it also matches benign temporal sequences like
    # "do X and then do Y".
    (r"\bthe\s+(?:code|method|function|loop|logic|implementation|grammar|"
     r"parser|regex|pattern|fix|patch|check|behavior|algorithm)\b"
     r"[^.\n]{0,40}?\b" + _CODE_VERB + r"\b"
     r"[^.\n]{1,160}"
     r"\b(?:but|instead\s+of|rather\s+than|however)\b"
     r"[^.\n]{1,160}",
     "buggy-behavior-contrast"),

    # 1b) Bare "<verb-ing> ... instead of <verb-ing>" — for cases where the
    # subject isn't a "the code" head. Both verb slots must be code-actions.
    (r"\b" + _CODE_VERB +
     r"[^.\n]{1,80}\binstead\s+of\b[^.\n]{0,40}\b" + _CODE_VERB,
     "verb-instead-of-verb"),

    # 2) Conditional execution narrative: "when <subject> <code-verb>" or
    # bare passive opener "when <code-verb-ed>". Subject (when present) is
    # 1–4 word-shaped tokens, hyphens allowed; the code-verb tail keeps this
    # from matching "when the user decides" style social prose.
    (r"\bwhen\s+"
     r"(?:(?:the\s+|a\s+|an\s+|each\s+|this\s+|that\s+|any\s+|some\s+)?"
     r"[\w-]+(?:\s+[\w-]+){0,3}\s+(?:is\s+|are\s+|gets?\s+)?)?"
     r"\b" + _CODE_VERB + r"\b",
     "conditional-when-verb"),

    # 3a) Spec-referenced bug ("according to RFC X / violates spec Y / per
    # the protocol").
    (r"\b(?:according\s+to|as\s+per|per\s+the|violates?|conforms?\s+to|"
     r"as\s+specified\s+in|complies?\s+with|comply\s+with|"
     r"follow(?:s|ed|ing)?)\s+"
     r"(?:the\s+)?"
     r"(?:RFC\s*\d+|PEP\s*\d+|ISO\s*\d+|HTTP\s*\d?(?:\.\d)?|BCP\s*\d+|"
     r"[\w-]+(?:\s+[\w-]+){0,2}\s+"
     r"(?:spec|standard|protocol|contract|RFC|PEP|convention))",
     "spec-violation"),

    # 3b) Bare "RFC 7231" / "PEP 257" citation in prose.
    (r"\b(?:RFC|PEP|ISO|BCP)\s*\d+\b",
     "rfc-pep-citation"),

    # 3c) "<X>'s contract|spec|standard|protocol <verb>" framing — captures
    # phrasing like "Python's hash contract only requires …".
    (r"\b\w+(?:'s)?\s+(?:contract|spec|standard|protocol)\s+"
     r"(?:is|says|requires|specifies|states|mandates|allows|expects|enforces|"
     r"only\s+\w+|does\s+not\s+\w+)\b",
     "contract-mention"),

    # 4) Numbered fix-plan: at least two items ("1. … 2. …") within ≤8 lines
    # of each other. MULTILINE is set globally on compile.
    (r"^\s*1\.\s+\S[^\n]{0,200}"
     r"(?:\n[^\n]{0,300}){0,8}"
     r"\n\s*2\.\s+\S",
     "numbered-fix-plan"),

    # 5) Order-of-execution narrative ("X is loaded before Y") — load/run/
    # init verbs are specific enough to avoid generic "before this happens"
    # prose. Subject is 1–4 hyphenated/word tokens.
    (r"\b(?:the\s+)?[\w-]+(?:\s+[\w-]+){0,3}\s+is\s+"
     r"(?:loaded|called|registered|executed|run|invoked|initialized|"
     r"dispatched|fired|set\s+up|wired\s+up|loaded\s+up)\s+"
     r"(?:before|after|prior\s+to|following|once)\s+[\w-]+",
     "order-of-execution"),

    # 6) "during <execution-noun>" framing — pure-prose loop / lifecycle
    # narration with no anchor on backticks.
    (r"\b(?:during|while|throughout)\s+"
     r"(?:iteration|iterating|traversal|traversing|enumeration|"
     r"deletion|insertion|loading|registering|parsing|matching|"
     r"serialization|deserialization|teardown|setup|initialization|"
     r"dispatch|dispatching|evaluation|evaluating)\b",
     "during-execution"),
]

_COMPILED_PROSE_EXEC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE | re.MULTILINE), kind)
    for p, kind in PROSE_EXEC_PATTERNS
]


PROSE_EXEC_ENABLED = False


def find_prose_exec_phrases(snippet: str) -> tuple[list[tuple[str, str]], set[str]]:
    """Run the prose-execution regex family (buggy-behavior contrast,
    conditional 'when X <verb>', spec citations, numbered fix plan, ...)
    against `snippet`. Intended for visible-response turns that the
    code-token scan flagged as no-match."""
    phrases: list[tuple[str, str]] = []
    kinds: set[str] = set()
    for cre, kind in _COMPILED_PROSE_EXEC_PATTERNS:
        for mo in cre.finditer(snippet):
            phrases.append((kind, mo.group(0)))
            kinds.add(kind)
    return phrases, kinds


OBSERVATION_KINDS: set[str] = {"verified-behavior"}


DEFAULT_MIN_REASONING_SNIPPETS = 2


def annotate_snippets(
    instances: list[dict],
    include_mutation_verb_only: bool = False,
    min_reasoning_snippets: int = DEFAULT_MIN_REASONING_SNIPPETS,
    require_core_reasoning: bool = True,
) -> dict:
    """Walk every snippet under instances[*].qualifying_turns[*].matches[*]
    once and annotate it inline with variable-state, call-dependency, and
    output-talk findings.

    Per match:
        state_phrases / state_kinds / is_state_strong
        call_dep_phrases / call_dep_kinds / has_call_dep
        output_phrases / output_kinds / has_output_talk

    Per turn (union over its matches):
        state_kinds, has_state_strong
        call_dep_kinds, has_call_dep
        output_kinds, has_output_talk

    Per instance (union across the trajectory, deduped at snippet level):
        has_variable_state_talk, n_state_match_snippets,
        n_state_strong_snippets, state_kinds
        has_call_dep_talk, n_call_dep_snippets, call_dep_kinds
        has_output_talk, n_output_snippets, output_kinds

    Returns aggregate counts for all three annotation kinds.
    """
    state_n_total = 0
    state_n_strong = 0
    state_instances: set[str] = set()
    state_kind_counts: dict[str, int] = defaultdict(int)
    # Per-subcategory instance counts: how many distinct instances each kind
    # fired on (vs. kind_counts above, which counts snippet hits).
    state_kind_inst_counts: dict[str, int] = defaultdict(int)

    cd_n_total = 0
    cd_instances: set[str] = set()
    cd_kind_counts: dict[str, int] = defaultdict(int)
    cd_kind_inst_counts: dict[str, int] = defaultdict(int)

    out_n_total = 0
    out_instances: set[str] = set()
    out_kind_counts: dict[str, int] = defaultdict(int)
    out_kind_inst_counts: dict[str, int] = defaultdict(int)

    loop_n_total = 0
    loop_instances: set[str] = set()
    loop_kind_counts: dict[str, int] = defaultdict(int)
    loop_kind_inst_counts: dict[str, int] = defaultdict(int)

    cond_n_total = 0
    cond_instances: set[str] = set()
    cond_kind_counts: dict[str, int] = defaultdict(int)
    cond_kind_inst_counts: dict[str, int] = defaultdict(int)

    others_n_total = 0
    others_instances: set[str] = set()
    others_kind_counts: dict[str, int] = defaultdict(int)
    others_kind_inst_counts: dict[str, int] = defaultdict(int)

    for inst in instances:
        inst_id = inst["instance_id"]
        inst_state_kinds: set[str] = set()
        # Strong-only state kinds for this instance (mirrors the strong gating
        # used by state_kind_counts), so per-subcategory instance counts line up.
        inst_state_strong_kinds: set[str] = set()
        inst_cd_kinds: set[str] = set()
        inst_out_kinds: set[str] = set()
        inst_loop_kinds: set[str] = set()
        inst_cond_kinds: set[str] = set()
        inst_others_kinds: set[str] = set()

        n_inst_state_total = 0
        n_inst_state_strong = 0
        n_inst_cd = 0
        n_inst_out = 0
        n_inst_loop = 0
        n_inst_cond = 0
        n_inst_others = 0

        seen_state: set[str] = set()
        seen_cd: set[str] = set()
        seen_out: set[str] = set()
        seen_loop: set[str] = set()
        seen_cond: set[str] = set()
        seen_others: set[str] = set()
        # Cross-family evidence (deduped at snippet level) for the
        # instance-level decision: any annotated snippet, and "core"
        # snippets whose kinds go beyond pure outcome observation.
        seen_any: set[str] = set()
        seen_core: set[str] = set()

        for turn in inst.get("qualifying_turns", []):
            turn_state_kinds: set[str] = set()
            turn_cd_kinds: set[str] = set()
            turn_out_kinds: set[str] = set()
            turn_loop_kinds: set[str] = set()
            turn_cond_kinds: set[str] = set()
            turn_others_kinds: set[str] = set()
            turn_has_state_strong = False
            turn_has_cd = False
            turn_has_out = False
            turn_has_loop = False
            turn_has_cond = False
            turn_has_others = False

            for m in turn.get("matches", []):
                snippet = m.get("snippet") or ""

                # --- variable-state annotation ---
                s_phrases, s_kinds = find_state_phrases(snippet)
                m["state_phrases"] = [[k, p] for k, p in s_phrases]
                m["state_kinds"] = sorted(s_kinds)
                is_strong = bool(s_kinds) and (
                    include_mutation_verb_only
                    or any(k in STRONG_KINDS for k in s_kinds)
                )
                m["is_state_strong"] = is_strong

                if s_kinds:
                    turn_state_kinds.update(s_kinds)
                    if is_strong:
                        turn_has_state_strong = True
                        inst_state_strong_kinds.update(s_kinds)
                    if snippet not in seen_state:
                        seen_state.add(snippet)
                        n_inst_state_total += 1
                        if is_strong:
                            n_inst_state_strong += 1
                            for k in s_kinds:
                                state_kind_counts[k] += 1

                # --- call-dependency annotation ---
                cd_phrases, cd_kinds = find_call_dep_phrases(snippet)
                m["call_dep_phrases"] = [[k, p] for k, p in cd_phrases]
                m["call_dep_kinds"] = sorted(cd_kinds)
                m["has_call_dep"] = bool(cd_kinds)

                if cd_kinds:
                    turn_cd_kinds.update(cd_kinds)
                    turn_has_cd = True
                    if snippet not in seen_cd:
                        seen_cd.add(snippet)
                        n_inst_cd += 1
                        for k in cd_kinds:
                            cd_kind_counts[k] += 1

                # --- output-of-method annotation ---
                o_phrases, o_kinds = find_output_phrases(snippet)
                m["output_phrases"] = [[k, p] for k, p in o_phrases]
                m["output_kinds"] = sorted(o_kinds)
                m["has_output_talk"] = bool(o_kinds)

                if o_kinds:
                    turn_out_kinds.update(o_kinds)
                    turn_has_out = True
                    if snippet not in seen_out:
                        seen_out.add(snippet)
                        n_inst_out += 1
                        for k in o_kinds:
                            out_kind_counts[k] += 1

                # --- loop-execution annotation ---
                l_phrases, l_kinds = find_loop_phrases(snippet)
                m["loop_phrases"] = [[k, p] for k, p in l_phrases]
                m["loop_kinds"] = sorted(l_kinds)
                m["has_loop_talk"] = bool(l_kinds)

                if l_kinds:
                    turn_loop_kinds.update(l_kinds)
                    turn_has_loop = True
                    if snippet not in seen_loop:
                        seen_loop.add(snippet)
                        n_inst_loop += 1
                        for k in l_kinds:
                            loop_kind_counts[k] += 1

                # --- conditional-execution annotation ---
                co_phrases, co_kinds = find_cond_phrases(snippet)
                m["cond_phrases"] = [[k, p] for k, p in co_phrases]
                m["cond_kinds"] = sorted(co_kinds)
                m["has_cond_talk"] = bool(co_kinds)

                if co_kinds:
                    turn_cond_kinds.update(co_kinds)
                    turn_has_cond = True
                    if snippet not in seen_cond:
                        seen_cond.add(snippet)
                        n_inst_cond += 1
                        for k in co_kinds:
                            cond_kind_counts[k] += 1

                # --- cross-family evidence for the instance decision ---
                decision_kinds: set[str] = set()
                if is_strong:
                    decision_kinds |= s_kinds
                decision_kinds |= cd_kinds | o_kinds | l_kinds | co_kinds
                if decision_kinds:
                    seen_any.add(snippet)
                    if decision_kinds - OBSERVATION_KINDS:
                        seen_core.add(snippet)

                # --- "others" (exception / mismatch / coverage) annotation ---
                # Disabled via OTHERS_ENABLED. Patterns + find_others_phrases
                # remain importable for ad-hoc use.
                if OTHERS_ENABLED:
                    ot_phrases, ot_kinds = find_others_phrases(snippet)
                    m["others_phrases"] = [[k, p] for k, p in ot_phrases]
                    m["others_kinds"] = sorted(ot_kinds)
                    m["has_others_talk"] = bool(ot_kinds)

                    if ot_kinds:
                        turn_others_kinds.update(ot_kinds)
                        turn_has_others = True
                        if snippet not in seen_others:
                            seen_others.add(snippet)
                            n_inst_others += 1
                            for k in ot_kinds:
                                others_kind_counts[k] += 1

            turn["state_kinds"] = sorted(turn_state_kinds)
            turn["has_state_strong"] = turn_has_state_strong
            turn["call_dep_kinds"] = sorted(turn_cd_kinds)
            turn["has_call_dep"] = turn_has_cd
            turn["output_kinds"] = sorted(turn_out_kinds)
            turn["has_output_talk"] = turn_has_out
            turn["loop_kinds"] = sorted(turn_loop_kinds)
            turn["has_loop_talk"] = turn_has_loop
            turn["cond_kinds"] = sorted(turn_cond_kinds)
            turn["has_cond_talk"] = turn_has_cond
            if OTHERS_ENABLED:
                turn["others_kinds"] = sorted(turn_others_kinds)
                turn["has_others_talk"] = turn_has_others
            inst_state_kinds.update(turn_state_kinds)
            inst_cd_kinds.update(turn_cd_kinds)
            inst_out_kinds.update(turn_out_kinds)
            inst_loop_kinds.update(turn_loop_kinds)
            inst_cond_kinds.update(turn_cond_kinds)
            inst_others_kinds.update(turn_others_kinds)

        inst["state_kinds"] = sorted(inst_state_kinds)
        inst["n_state_match_snippets"] = n_inst_state_total
        inst["n_state_strong_snippets"] = n_inst_state_strong
        inst["has_variable_state_talk"] = n_inst_state_strong > 0

        inst["call_dep_kinds"] = sorted(inst_cd_kinds)
        inst["n_call_dep_snippets"] = n_inst_cd
        inst["has_call_dep_talk"] = n_inst_cd > 0

        inst["output_kinds"] = sorted(inst_out_kinds)
        inst["n_output_snippets"] = n_inst_out
        inst["has_output_talk"] = n_inst_out > 0

        inst["loop_kinds"] = sorted(inst_loop_kinds)
        inst["n_loop_snippets"] = n_inst_loop
        inst["has_loop_talk"] = n_inst_loop > 0

        inst["cond_kinds"] = sorted(inst_cond_kinds)
        inst["n_cond_snippets"] = n_inst_cond
        inst["has_cond_talk"] = n_inst_cond > 0

        if OTHERS_ENABLED:
            inst["others_kinds"] = sorted(inst_others_kinds)
            inst["n_others_snippets"] = n_inst_others
            inst["has_others_talk"] = n_inst_others > 0

        # An instance "has code reasoning" when the execution-reasoning
        # annotators produced ENOUGH evidence:
        #   (a) at least `min_reasoning_snippets` distinct annotated snippets
        #       across all families — one stray phrase in a long trajectory
        #       does not make the trajectory "reasoning about execution"; and
        #   (b) when `require_core_reasoning`, at least one snippet whose
        #       kinds go beyond pure outcome observation (OBSERVATION_KINDS,
        #       e.g. "all tests pass") — observing a result is not the same
        #       as reasoning about the execution process.
        # Merely having code tokens in prose is NOT sufficient.
        inst["n_reasoning_snippets"] = len(seen_any)
        inst["n_core_reasoning_snippets"] = len(seen_core)
        inst["has_code_reasoning"] = (
            len(seen_any) >= min_reasoning_snippets
            and (not require_core_reasoning or len(seen_core) >= 1)
        )

        state_n_total += n_inst_state_total
        state_n_strong += n_inst_state_strong
        if n_inst_state_strong:
            state_instances.add(inst_id)
        for k in inst_state_strong_kinds:
            state_kind_inst_counts[k] += 1
        cd_n_total += n_inst_cd
        if n_inst_cd:
            cd_instances.add(inst_id)
        for k in inst_cd_kinds:
            cd_kind_inst_counts[k] += 1
        out_n_total += n_inst_out
        if n_inst_out:
            out_instances.add(inst_id)
        for k in inst_out_kinds:
            out_kind_inst_counts[k] += 1
        loop_n_total += n_inst_loop
        if n_inst_loop:
            loop_instances.add(inst_id)
        for k in inst_loop_kinds:
            loop_kind_inst_counts[k] += 1
        cond_n_total += n_inst_cond
        if n_inst_cond:
            cond_instances.add(inst_id)
        for k in inst_cond_kinds:
            cond_kind_inst_counts[k] += 1
        if OTHERS_ENABLED:
            others_n_total += n_inst_others
            if n_inst_others:
                others_instances.add(inst_id)
            for k in inst_others_kinds:
                others_kind_inst_counts[k] += 1

    return {
        "state": {
            "n_total_hits": state_n_total,
            "n_strong_hits": state_n_strong,
            "n_instances_with_hits": len(state_instances),
            "kind_counts": dict(sorted(state_kind_counts.items(), key=lambda x: -x[1])),
            "kind_instance_counts": dict(state_kind_inst_counts),
            "include_mutation_verb_only": include_mutation_verb_only,
        },
        "call_dep": {
            "n_total_hits": cd_n_total,
            "n_instances_with_hits": len(cd_instances),
            "kind_counts": dict(sorted(cd_kind_counts.items(), key=lambda x: -x[1])),
            "kind_instance_counts": dict(cd_kind_inst_counts),
        },
        "output": {
            "n_total_hits": out_n_total,
            "n_instances_with_hits": len(out_instances),
            "kind_counts": dict(sorted(out_kind_counts.items(), key=lambda x: -x[1])),
            "kind_instance_counts": dict(out_kind_inst_counts),
        },
        "loop": {
            "n_total_hits": loop_n_total,
            "n_instances_with_hits": len(loop_instances),
            "kind_counts": dict(sorted(loop_kind_counts.items(), key=lambda x: -x[1])),
            "kind_instance_counts": dict(loop_kind_inst_counts),
        },
        "cond": {
            "n_total_hits": cond_n_total,
            "n_instances_with_hits": len(cond_instances),
            "kind_counts": dict(sorted(cond_kind_counts.items(), key=lambda x: -x[1])),
            "kind_instance_counts": dict(cond_kind_inst_counts),
        },
        **({
            "others": {
                "n_total_hits": others_n_total,
                "n_instances_with_hits": len(others_instances),
                "kind_counts": dict(sorted(others_kind_counts.items(), key=lambda x: -x[1])),
                "kind_instance_counts": dict(others_kind_inst_counts),
            },
        } if OTHERS_ENABLED else {}),
    }


def score_thought(
    thought: str,
    prose_hits_required: int = DEFAULT_PROSE_HITS,
    max_tokens_per_label: int = DEFAULT_MAX_TOKENS_PER_LABEL,
) -> tuple[int, list[dict]]:
    """Return (distinct-label score, [match-dict, ...]).

    Each match dict explicitly reports the code tokens it identified:
        {
            "label":       "<pattern label>",
            "tokens":      ["sep_matrix", "CompoundModel", ...],   # up to max_tokens_per_label
            "n_tokens":    <total distinct token count>,
            "snippet":     "<surrounding-prose context for the strongest hit>",
            "prose_hits":  <strongest prose-context score on this label>,
        }

    "Distinct" still counts one label per turn (a thought with eight
    backtick identifiers and nothing else scores 1), but the tokens list
    surfaces the actual variables / calls / statements that were matched.
    """
    if not thought or len(thought) < MIN_THOUGHT_CHARS:
        return 0, []

    # Skip pathologically long thoughts. Real reasoning is well under 100K
    # chars; >500K is essentially always a degenerate output (e.g. a model
    # that emitted a multi-million-character single line of dots) and the
    # regex passes would either hit catastrophic backtracking or burn
    # minutes for no real signal.
    if len(thought) > 500_000:
        return 0, []

    cleaned = strip_code_blocks(_strip_pr_description(thought))
    if len(cleaned) < MIN_THOUGHT_CHARS:
        return 0, []

    matches: list[dict] = []
    for pat, label in PATTERNS:
        tokens: list[str] = []
        seen_tokens: set[str] = set()
        best_snippet: str | None = None
        best_phits = 0
        for m in pat.finditer(cleaned):
            snippet = extract_context(cleaned, m)
            phits = len(PROSE_CONTEXT_RE.findall(snippet))
            if phits < prose_hits_required:
                continue
            tok = m.group(0).strip()
            if tok not in seen_tokens:
                seen_tokens.add(tok)
                if len(tokens) < max_tokens_per_label:
                    tokens.append(tok)
            if phits > best_phits:
                best_phits = phits
                best_snippet = snippet
        if tokens:
            matches.append({
                "label": label,
                "tokens": tokens,
                "n_tokens": len(seen_tokens),
                "snippet": best_snippet,
                "prose_hits": best_phits,
            })

    matches.sort(key=lambda x: -x["prose_hits"])
    return len(matches), matches


def parse_thoughts(
    trajs_dir: Path,
    top_level_only: bool = True,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for tf in _list_traj_files(trajs_dir):
        iid = tf.stem.replace(".traj", "")
        out[iid] = {
            "file": str(tf),
            "thoughts": list(iter_assistant_thoughts(tf, top_level_only=top_level_only)),
        }
    return out


# Official SWE-bench Verified ships a human-annotated `difficulty` column
# (estimated time-to-fix) keyed by instance_id. We join it in so every
# instance result carries the canonical difficulty alongside its resolution.
DEFAULT_DIFFICULTY_DATASET = "princeton-nlp/SWE-bench_Verified"

# The official annotation has four time-to-fix buckets; we collapse them into
# three coarse categories used everywhere downstream:
#   easy   = "<15 min fix"
#   medium = "15 min - 1 hour"
#   hard   = "1-4 hours" or ">4 hours"
_DIFFICULTY_3CAT = {
    "<15 min fix": "easy",
    "15 min - 1 hour": "medium",
    "1-4 hours": "hard",
    ">4 hours": "hard",
}

# Canonical ordering / short labels for the three collapsed categories.
DIFFICULTY_ORDER = ["easy", "medium", "hard"]
_DIFFICULTY_ABBR = {
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
}


def _normalize_difficulty(raw) -> str | None:
    """Collapse an official SWE-bench difficulty label into easy/medium/hard.

    Unrecognized values pass through unchanged (so a dataset that already
    uses the 3-category scheme, or an unexpected bucket, is preserved)."""
    if raw is None:
        return None
    raw = str(raw)
    return _DIFFICULTY_3CAT.get(raw, raw)


def load_difficulties(
    dataset: str | None = DEFAULT_DIFFICULTY_DATASET,
    json_path: Path | None = None,
) -> dict[str, str]:
    """Return {instance_id -> difficulty} from the official SWE-bench annotation.

    Resolution order:
      1. `json_path` — a local mapping file, when given and present. Accepts
         {iid: "difficulty"}, {iid: {"difficulty": ...}}, or a list of row
         dicts each carrying `instance_id` + `difficulty`. Offline & fast.
      2. `dataset` — a HuggingFace dataset id (default the SWE-bench Verified
         split, which ships a per-instance `difficulty` column). Requires the
         `datasets` library and uses its local HF cache when offline.

    Never raises: on any failure it warns and returns whatever it has
    (possibly empty) so the rest of the scan still runs.
    """
    # 1. Local JSON mapping wins (reproducible, no network / library needed).
    if json_path is not None and Path(json_path).exists():
        try:
            with open(json_path) as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"WARNING: could not read difficulty JSON {json_path}: {exc}")
            data = None
        out: dict[str, str] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                d = _normalize_difficulty(
                    v.get("difficulty") if isinstance(v, dict) else v
                )
                if d is not None:
                    out[str(k)] = d
        elif isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    iid = row.get("instance_id") or row.get("instance")
                    d = _normalize_difficulty(row.get("difficulty"))
                    if iid and d is not None:
                        out[str(iid)] = d
        if out:
            return out

    # 2. HuggingFace dataset (cached locally after first download).
    if not dataset:
        return {}
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from datasets import load_dataset
            ds = load_dataset(dataset, split="test")
    except Exception as exc:
        print(f"WARNING: could not load difficulty dataset '{dataset}': {exc}\n"
              f"  (install `datasets` or pass --difficulty-json; "
              f"continuing without difficulty labels.)")
        return {}
    if "difficulty" not in ds.column_names:
        print(f"WARNING: dataset '{dataset}' has no 'difficulty' column "
              f"(columns: {ds.column_names}); continuing without labels.")
        return {}
    return {
        str(r["instance_id"]): _normalize_difficulty(r["difficulty"])
        for r in ds
        if r.get("instance_id") and r.get("difficulty") is not None
    }


def score_one(
    parsed: dict[str, dict],
    resolutions: dict[str, dict],
    min_score: int,
    prose_hits_required: int,
    min_qualifying_turns: int,
    max_first_msg_idx: int | None = None,
    max_api_calls: int | None = None,
    difficulties: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict]]:
    difficulties = difficulties or {}
    instance_results: list[dict] = []
    qualifying_steps: list[dict] = []

    for iid, parsed_inst in parsed.items():
        per_turn: list[dict] = []
        max_score = 0
        all_labels: set[str] = set()
        # Tokens collected across the whole trajectory, grouped by label.
        # Lets a reader see at a glance which identifiers / calls / etc.
        # the model actually surfaced in prose during this run.
        all_tokens_by_label: dict[str, set[str]] = {}
        thoughts = parsed_inst["thoughts"]

        # Pass-7 collector: prose-execution hits on turns where the code-token
        # pass found nothing. Keeps the per-instance summary auditable.
        prose_exec_turns: list[dict] = []
        prose_exec_kinds_inst: set[str] = set()

        for msg_idx, thought in thoughts:
            score, matches = score_thought(thought, prose_hits_required)
            if score >= min_score:
                turn = {
                    "message_index": msg_idx,
                    "score": score,
                    "matches": matches,
                }
                per_turn.append(turn)
                qualifying_steps.append({
                    "instance_id": iid,
                    "file": parsed_inst["file"],
                    "message_index": msg_idx,
                    "score": score,
                    "matches": matches,
                })
                max_score = max(max_score, score)
                for m in matches:
                    all_labels.add(m["label"])
                    bucket = all_tokens_by_label.setdefault(m["label"], set())
                    bucket.update(m["tokens"])
            elif PROSE_EXEC_ENABLED:
                # No code tokens fired in this turn. Check whether the visible
                # response is still narrating execution process in pure English
                # using the prose-execution patterns.
                if not thought or len(thought) < MIN_THOUGHT_CHARS:
                    continue
                cleaned = strip_code_blocks(_strip_pr_description(thought))
                if len(cleaned) < MIN_THOUGHT_CHARS:
                    continue
                pe_phrases, pe_kinds = find_prose_exec_phrases(cleaned)
                if pe_kinds:
                    prose_exec_turns.append({
                        "message_index": msg_idx,
                        "prose_exec_kinds": sorted(pe_kinds),
                        "prose_exec_phrases": [[k, p] for k, p in pe_phrases[:8]],
                        "thought_preview": cleaned[:400].replace("\n", " ").strip(),
                    })
                    prose_exec_kinds_inst.update(pe_kinds)

        res = resolutions.get(iid, {})
        first_idx = per_turn[0]["message_index"] if per_turn else None
        api_calls = res.get("api_calls")

        flagged = len(per_turn) >= min_qualifying_turns
        if flagged and max_first_msg_idx is not None and first_idx is not None:
            if first_idx > max_first_msg_idx:
                flagged = False
        if flagged and max_api_calls is not None and isinstance(api_calls, (int, float)):
            if api_calls > max_api_calls:
                flagged = False

        tokens_by_label_sorted = {
            lbl: sorted(toks) for lbl, toks in sorted(all_tokens_by_label.items())
        }
        instance_results.append({
            "instance_id": iid,
            "file": parsed_inst["file"],
            "resolved": res.get("resolved"),
            "difficulty": difficulties.get(iid),
            "cost": res.get("cost"),
            "api_calls": api_calls,
            "has_code_tokens_in_prose": flagged,
            # Decided later by annotate_snippets (OR of the annotator families);
            # default False so the field exists even if the state scan is off.
            "has_code_reasoning": False,
            "first_qualifying_msg_idx": first_idx,
            "n_assistant_turns_with_thought": len(thoughts),
            "n_qualifying_turns": len(per_turn),
            "max_turn_score": max_score,
            "distinct_labels": sorted(all_labels),
            "n_distinct_tokens": sum(len(v) for v in tokens_by_label_sorted.values()),
            "tokens_by_label": tokens_by_label_sorted,
            "qualifying_turns": per_turn,
            "n_prose_exec_only_turns": len(prose_exec_turns),
            "prose_exec_kinds": sorted(prose_exec_kinds_inst),
            "has_prose_exec_only_talk": len(prose_exec_turns) > 0,
            "prose_exec_only_turns": prose_exec_turns,
        })

    return instance_results, qualifying_steps


def analyze_trajectories(
    trajs_dir: Path,
    resolutions: dict[str, dict] | None = None,
    min_score: int = DEFAULT_MIN_SCORE,
    prose_hits_required: int = DEFAULT_PROSE_HITS,
    min_qualifying_turns: int = DEFAULT_MIN_QUALIFYING_TURNS,
    max_first_msg_idx: int | None = None,
    max_api_calls: int | None = None,
    parsed: dict[str, dict] | None = None,
    top_level_only: bool = True,
    run_state_scan: bool = True,
    include_mutation_verb_only: bool = False,
    difficulties: dict[str, str] | None = None,
    min_reasoning_snippets: int = DEFAULT_MIN_REASONING_SNIPPETS,
    require_core_reasoning: bool = True,
) -> dict:
    resolutions = resolutions or {}
    difficulties = difficulties or {}
    if parsed is None:
        parsed = parse_thoughts(trajs_dir, top_level_only=top_level_only)

    instance_results, qualifying_steps = score_one(
        parsed, resolutions,
        min_score=min_score,
        prose_hits_required=prose_hits_required,
        min_qualifying_turns=min_qualifying_turns,
        max_first_msg_idx=max_first_msg_idx,
        max_api_calls=max_api_calls,
        difficulties=difficulties,
    )

    qualifying_steps.sort(key=lambda r: -r["score"])

    # Run the execution-reasoning annotators UP FRONT so every instance carries
    # has_code_reasoning before we aggregate. An instance counts as code
    # reasoning only when an annotator family fired on its code-token snippets;
    # raw code-token presence (has_code_tokens_in_prose) no longer decides it.
    agg = None
    if run_state_scan:
        agg = annotate_snippets(
            instance_results,
            include_mutation_verb_only=include_mutation_verb_only,
            min_reasoning_snippets=min_reasoning_snippets,
            require_core_reasoning=require_core_reasoning,
        )
    for inst in instance_results:
        inst.setdefault("has_code_reasoning", False)

    crosstab: dict[str, dict[str, int]] = {
        "resolved=True":  {"code_reasoning=True": 0, "code_reasoning=False": 0},
        "resolved=False": {"code_reasoning=True": 0, "code_reasoning=False": 0},
        "resolved=None":  {"code_reasoning=True": 0, "code_reasoning=False": 0},
    }
    for inst in instance_results:
        rkey = f"resolved={inst['resolved']}"
        if rkey not in crosstab:
            crosstab[rkey] = {"code_reasoning=True": 0, "code_reasoning=False": 0}
        ckey = f"code_reasoning={bool(inst['has_code_reasoning'])}"
        crosstab[rkey][ckey] = crosstab[rkey].get(ckey, 0) + 1

    n_with_resolution = sum(1 for i in instance_results if i["resolved"] is not None)
    n_resolved = sum(1 for i in instance_results if i["resolved"] is True)
    missing_resolution = [
        i["instance_id"] for i in instance_results if i["resolved"] is None
    ]

    # Difficulty breakdown (official SWE-bench annotation). Per bucket we track
    # the instance count, how many resolved, and how many showed code reasoning
    # — so difficulty can be cross-read against both signals. We additionally
    # break the code-reasoning subset down by resolution outcome
    # (resolved / unresolved / unknown), which directly answers "on each
    # difficulty, for instances WITH code reasoning, how many issues are
    # resolved vs. not resolved".
    n_with_difficulty = sum(1 for i in instance_results if i.get("difficulty"))
    difficulty_breakdown: dict[str, dict] = {}
    for inst in instance_results:
        d = inst.get("difficulty") or "unknown"
        b = difficulty_breakdown.setdefault(
            d,
            {
                "n": 0,
                "n_resolved": 0,
                "n_code_reasoning": 0,
                # Code-reasoning subset, split by resolution outcome.
                "n_code_reasoning_resolved": 0,
                "n_code_reasoning_unresolved": 0,
                "n_code_reasoning_unknown": 0,
                # Complementary no-code-reasoning subset, same split — handy
                # for a side-by-side resolution-rate comparison.
                "n_no_code_reasoning_resolved": 0,
                "n_no_code_reasoning_unresolved": 0,
                "n_no_code_reasoning_unknown": 0,
            },
        )
        b["n"] += 1
        resolved = inst.get("resolved")
        if resolved is True:
            b["n_resolved"] += 1
        if inst.get("has_code_reasoning"):
            b["n_code_reasoning"] += 1
            if resolved is True:
                b["n_code_reasoning_resolved"] += 1
            elif resolved is False:
                b["n_code_reasoning_unresolved"] += 1
            else:
                b["n_code_reasoning_unknown"] += 1
        else:
            if resolved is True:
                b["n_no_code_reasoning_resolved"] += 1
            elif resolved is False:
                b["n_no_code_reasoning_unresolved"] += 1
            else:
                b["n_no_code_reasoning_unknown"] += 1
    # Order canonically (the four official buckets, then any extras, then
    # "unknown" last) so downstream readers get a stable layout.
    diff_order = [d for d in DIFFICULTY_ORDER if d in difficulty_breakdown]
    diff_order += sorted(
        d for d in difficulty_breakdown
        if d not in DIFFICULTY_ORDER and d != "unknown"
    )
    if "unknown" in difficulty_breakdown:
        diff_order.append("unknown")
    difficulty_breakdown = {d: difficulty_breakdown[d] for d in diff_order}
    difficulty_counts = {d: b["n"] for d, b in difficulty_breakdown.items()}

    # Pass-7 aggregation: prose-execution narration on no-code-token turns.
    pe_n_turns = 0
    pe_kind_counts: dict[str, int] = defaultdict(int)
    pe_kind_inst_counts: dict[str, int] = defaultdict(int)
    pe_instances: set[str] = set()
    for inst in instance_results:
        turns = inst.get("prose_exec_only_turns") or []
        if turns:
            pe_instances.add(inst["instance_id"])
            pe_n_turns += len(turns)
            inst_pe_kinds: set[str] = set()
            for t in turns:
                for k in t.get("prose_exec_kinds", []):
                    pe_kind_counts[k] += 1
                    inst_pe_kinds.add(k)
            for k in inst_pe_kinds:
                pe_kind_inst_counts[k] += 1

    result = {
        "trajs_dir": str(trajs_dir),
        "n_files": len(parsed),
        "min_score": min_score,
        "prose_hits_required": prose_hits_required,
        "min_qualifying_turns": min_qualifying_turns,
        "top_level_only": top_level_only,
        "n_instances_with_code_reasoning": sum(
            1 for i in instance_results if i["has_code_reasoning"]
        ),
        "n_instances_with_code_tokens_in_prose": sum(
            1 for i in instance_results if i["has_code_tokens_in_prose"]
        ),
        "n_with_resolution_data": n_with_resolution,
        "n_resolved": n_resolved,
        "n_missing_resolution": len(missing_resolution),
        "missing_resolution_ids": missing_resolution[:20],
        "n_with_difficulty_data": n_with_difficulty,
        "difficulty_counts": difficulty_counts,
        "difficulty_breakdown": difficulty_breakdown,
        "crosstab_resolved_by_code_reasoning": crosstab,
        "instances": instance_results,
        "top_steps": qualifying_steps[:50],
        "prose_exec_n_total_turns": pe_n_turns,
        "prose_exec_n_instances_with_hits": len(pe_instances),
        "prose_exec_kind_counts": dict(sorted(pe_kind_counts.items(),
                                              key=lambda x: -x[1])),
        "prose_exec_kind_instance_counts": dict(pe_kind_inst_counts),
    }

    # Pass 2 aggregates (computed up front by annotate_snippets above) lift to
    # the top level so the whole scan is one flat result.
    if agg is not None:
        s = agg["state"]
        result["state_n_total_hits"] = s["n_total_hits"]
        result["state_n_strong_hits"] = s["n_strong_hits"]
        result["state_n_instances_with_hits"] = s["n_instances_with_hits"]
        result["state_kind_counts"] = s["kind_counts"]
        result["state_kind_instance_counts"] = s["kind_instance_counts"]
        result["state_include_mutation_verb_only"] = s["include_mutation_verb_only"]

        c = agg["call_dep"]
        result["call_dep_n_total_hits"] = c["n_total_hits"]
        result["call_dep_n_instances_with_hits"] = c["n_instances_with_hits"]
        result["call_dep_kind_counts"] = c["kind_counts"]
        result["call_dep_kind_instance_counts"] = c["kind_instance_counts"]

        o = agg["output"]
        result["output_n_total_hits"] = o["n_total_hits"]
        result["output_n_instances_with_hits"] = o["n_instances_with_hits"]
        result["output_kind_counts"] = o["kind_counts"]
        result["output_kind_instance_counts"] = o["kind_instance_counts"]

        l = agg["loop"]
        result["loop_n_total_hits"] = l["n_total_hits"]
        result["loop_n_instances_with_hits"] = l["n_instances_with_hits"]
        result["loop_kind_counts"] = l["kind_counts"]
        result["loop_kind_instance_counts"] = l["kind_instance_counts"]

        co = agg["cond"]
        result["cond_n_total_hits"] = co["n_total_hits"]
        result["cond_n_instances_with_hits"] = co["n_instances_with_hits"]
        result["cond_kind_counts"] = co["kind_counts"]
        result["cond_kind_instance_counts"] = co["kind_instance_counts"]

        if "others" in agg:
            ot = agg["others"]
            result["others_n_total_hits"] = ot["n_total_hits"]
            result["others_n_instances_with_hits"] = ot["n_instances_with_hits"]
            result["others_kind_counts"] = ot["kind_counts"]
            result["others_kind_instance_counts"] = ot["kind_instance_counts"]

    return result


def _fmt_resolved(value) -> str:
    if value is True:
        return "RESOLVED"
    if value is False:
        return "FAILED  "
    return "UNKNOWN "


def _fmt_difficulty(value) -> str:
    if not value:
        return "?"
    return _DIFFICULTY_ABBR.get(value, value)


def print_report(result: dict, top_n: int = 30) -> None:
    n_files = result["n_files"]
    n_reason = result.get("n_instances_with_code_reasoning", 0)
    n_hits = result["n_instances_with_code_tokens_in_prose"]
    n_with_res = result.get("n_with_resolution_data", 0)
    n_resolved = result.get("n_resolved", 0)

    print(f"Scanned {n_files} trajectory file(s) under {result['trajs_dir']}")
    print(f"min_score = {result['min_score']}, "
          f"prose_hits_required = {result['prose_hits_required']}")
    print(f"Trajectories with CODE REASONING (annotator hit): {n_reason} / {n_files}")
    print(f"  (of which have code tokens in prose:           {n_hits} / {n_files})")
    if n_with_res:
        print(f"Instances with resolution data:           "
              f"{n_with_res} / {n_files}  (resolved: {n_resolved})")
        if result.get("n_missing_resolution"):
            print(f"  (missing resolution data for "
                  f"{result['n_missing_resolution']} instances; "
                  f"first few: {result['missing_resolution_ids'][:5]})")
    n_with_diff = result.get("n_with_difficulty_data", 0)
    if n_with_diff:
        print(f"Instances with difficulty labels:         "
              f"{n_with_diff} / {n_files}  (official SWE-bench annotation)")
    print()

    breakdown = result.get("difficulty_breakdown") or {}
    if breakdown:
        print("=" * 100)
        print("DIFFICULTY BREAKDOWN (official SWE-bench)  ×  resolved  ×  code-reasoning")
        print("=" * 100)
        print(f"  {'difficulty':<16s} {'n':>6s} "
              f"{'resolved':>10s} {'(rate)':>8s} "
              f"{'code-reason':>12s} {'(rate)':>8s}")
        tot_n = tot_res = tot_cr = 0
        for diff, b in breakdown.items():
            n = b["n"]
            nr = b["n_resolved"]
            nc = b["n_code_reasoning"]
            tot_n += n
            tot_res += nr
            tot_cr += nc
            print(f"  {diff:<16s} {n:>6d} "
                  f"{nr:>10d} {nr / n:>7.0%} "
                  f"{nc:>12d} {nc / n:>7.0%}")
        if tot_n:
            print(f"  {'total':<16s} {tot_n:>6d} "
                  f"{tot_res:>10d} {tot_res / tot_n:>7.0%} "
                  f"{tot_cr:>12d} {tot_cr / tot_n:>7.0%}")
        print()

        # Among code-reasoning instances only: how many resolved vs. not
        # resolved, per difficulty. "rate" is resolved / (resolved+unresolved)
        # so unknown-resolution instances don't drag the denominator down.
        print("-" * 100)
        print("  CODE-REASONING instances only  →  resolved vs. not resolved, per difficulty")
        print("-" * 100)
        print(f"  {'difficulty':<16s} {'code-reason':>12s} "
              f"{'resolved':>10s} {'unresolved':>11s} {'unknown':>8s} "
              f"{'(res-rate)':>10s}")
        tot_cr = tot_crr = tot_cru = tot_crk = 0
        for diff, b in breakdown.items():
            nc = b["n_code_reasoning"]
            nrr = b["n_code_reasoning_resolved"]
            nru = b["n_code_reasoning_unresolved"]
            nrk = b["n_code_reasoning_unknown"]
            tot_cr += nc
            tot_crr += nrr
            tot_cru += nru
            tot_crk += nrk
            denom = nrr + nru
            rate = f"{nrr / denom:>9.0%}" if denom else f"{'n/a':>10s}"
            print(f"  {diff:<16s} {nc:>12d} "
                  f"{nrr:>10d} {nru:>11d} {nrk:>8d} {rate:>10s}")
        if tot_cr:
            denom = tot_crr + tot_cru
            rate = f"{tot_crr / denom:>9.0%}" if denom else f"{'n/a':>10s}"
            print(f"  {'total':<16s} {tot_cr:>12d} "
                  f"{tot_crr:>10d} {tot_cru:>11d} {tot_crk:>8d} {rate:>10s}")
        print()

    crosstab = result.get("crosstab_resolved_by_code_reasoning") or {}
    if crosstab:
        print("=" * 100)
        print("CROSSTAB: resolution × code-reasoning")
        print("=" * 100)
        col_keys = ["code_reasoning=True", "code_reasoning=False"]
        row_order = ["resolved=True", "resolved=False", "resolved=None"]
        for rk in list(crosstab.keys()):
            if rk not in row_order:
                row_order.append(rk)
        header = f"  {'':<18s} " + "  ".join(f"{c:>20s}" for c in col_keys) + "    total"
        print(header)
        col_totals = {c: 0 for c in col_keys}
        grand = 0
        for rk in row_order:
            cells = crosstab.get(rk, {})
            row_total = sum(cells.get(c, 0) for c in col_keys)
            grand += row_total
            for c in col_keys:
                col_totals[c] += cells.get(c, 0)
            print(f"  {rk:<18s} "
                  + "  ".join(f"{cells.get(c, 0):>20d}" for c in col_keys)
                  + f"    {row_total:>5d}")
        print(f"  {'total':<18s} "
              + "  ".join(f"{col_totals[c]:>20d}" for c in col_keys)
              + f"    {grand:>5d}")
        print()

    print("=" * 100)
    print("PER-INSTANCE SUMMARY (instances with code reasoning)")
    print("=" * 100)
    flagged = [i for i in result["instances"] if i.get("has_code_reasoning")]
    flagged.sort(key=lambda r: (-r["max_turn_score"], -r["n_qualifying_turns"]))
    for inst in flagged[:top_n]:
        labels = ", ".join(inst["distinct_labels"][:6])
        if len(inst["distinct_labels"]) > 6:
            labels += f", … (+{len(inst['distinct_labels']) - 6})"
        print(f"  [{_fmt_resolved(inst['resolved'])}] "
              f"{inst['instance_id']:<40s}"
              f"  diff={_fmt_difficulty(inst.get('difficulty')):<7s}"
              f"  turns={inst['n_qualifying_turns']:>3d}"
              f"  max_score={inst['max_turn_score']:>2d}"
              f"  labels=[{labels}]")
    if len(flagged) > top_n:
        print(f"  … (+{len(flagged) - top_n} more)")
    print()

    print("=" * 100)
    print(f"TOP {top_n} TURN-LEVEL EXAMPLES (highest score)")
    print("=" * 100)
    for rank, step in enumerate(result["top_steps"][:top_n], start=1):
        print(f"\n{'-' * 100}")
        print(f"  #{rank}  score={step['score']}  "
              f"instance={step['instance_id']}  msg_idx={step['message_index']}")
        print(f"  file: {step['file']}")
        for m in step["matches"][:4]:
            snippet = m["snippet"]
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            tokens_preview = ", ".join(m["tokens"][:8])
            if m["n_tokens"] > 8:
                tokens_preview += f", … (+{m['n_tokens'] - 8} more)"
            print(f"    [{m['label']}] (prose={m['prose_hits']}, "
                  f"n_tokens={m['n_tokens']})")
            print(f"      tokens : {tokens_preview}")
            print(f"      context: \"{snippet}\"")

    print()
    print("=" * 100)
    print("PATTERN LABEL FREQUENCY (across qualifying turns)")
    print("=" * 100)
    label_counts: dict[str, int] = {}
    for inst in result["instances"]:
        for turn in inst["qualifying_turns"]:
            for m in turn["matches"]:
                label_counts[m["label"]] = label_counts.get(m["label"], 0) + 1
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<32s} {count:>5d}")

    # ------------------------------------------------------------------
    # Pass-2 state-scan section (reads inline annotations on each match).
    # ------------------------------------------------------------------
    if "state_n_total_hits" in result:
        print()
        print("=" * 100)
        print("VARIABLE-STATE SCAN (annotations live inline on each match)")
        print("=" * 100)
        print(f"  Total snippets matching any state pattern : "
              f"{result['state_n_total_hits']}")
        print(f"  Strong hits (value-oriented kinds)        : "
              f"{result['state_n_strong_hits']}")
        print(f"  Distinct instances with state talk         : "
              f"{result['state_n_instances_with_hits']}")
        if result.get("state_include_mutation_verb_only"):
            print("  (--include-mutation-verb-only is ON: "
                  "noisy 'we change X' snippets included)")
        print()
        print("  Kind distribution (strong hits)  "
              "[hits = snippet matches, inst = distinct instances]:")
        state_kind_inst = result.get("state_kind_instance_counts") or {}
        for kind, count in (result.get("state_kind_counts") or {}).items():
            print(f"    {count:>5d} hits  {state_kind_inst.get(kind, 0):>4d} inst  "
                  f"{kind}")

        # Build exemplars on the fly from the inline annotations: one strongest
        # match per instance, ranked by (#kinds, #phrases).
        exemplars: list[tuple[dict, dict, dict]] = []
        for inst in result["instances"]:
            best: tuple[int, int, dict, dict, dict] | None = None
            for turn in inst.get("qualifying_turns", []):
                for m in turn.get("matches", []):
                    if not m.get("is_state_strong"):
                        continue
                    score = (len(m.get("state_kinds") or []),
                             len(m.get("state_phrases") or []))
                    cand = (score[0], score[1], inst, turn, m)
                    if best is None or cand[:2] > best[:2]:
                        best = cand
            if best is not None:
                exemplars.append((best[2], best[3], best[4]))
        exemplars.sort(key=lambda t: (-len(t[2]["state_kinds"]), t[0]["instance_id"]))

        if exemplars and top_n:
            print()
            print(f"  Top {min(top_n, len(exemplars))} per-instance state exemplars:")
            for inst, turn, m in exemplars[:top_n]:
                kinds = ",".join(m["state_kinds"])
                phrases_preview = "; ".join(
                    f"({k}) {p!r}" for k, p in m["state_phrases"][:3]
                )
                snip = m["snippet"]
                if len(snip) > 240:
                    snip = snip[:240] + "..."
                print(f"    [{inst['instance_id']}] msg {turn['message_index']}  "
                      f"kinds={kinds}")
                print(f"      phrases: {phrases_preview}")
                print(f"      snippet: {snip}")

    # ------------------------------------------------------------------
    # Pass-3 call-dependency-scan section
    # ------------------------------------------------------------------
    if "call_dep_n_total_hits" in result:
        print()
        print("=" * 100)
        print("CALL-DEPENDENCY SCAN (annotations live inline on each match)")
        print("=" * 100)
        print(f"  Total snippets matching any call-dep pattern : "
              f"{result['call_dep_n_total_hits']}")
        print(f"  Distinct instances with call-dep talk         : "
              f"{result['call_dep_n_instances_with_hits']}")
        print()
        print("  Kind distribution  "
              "[hits = snippet matches, inst = distinct instances]:")
        cd_kind_inst = result.get("call_dep_kind_instance_counts") or {}
        for kind, count in (result.get("call_dep_kind_counts") or {}).items():
            print(f"    {count:>5d} hits  {cd_kind_inst.get(kind, 0):>4d} inst  "
                  f"{kind}")

        # Pick one strongest match per instance.
        cd_exemplars: list[tuple[dict, dict, dict]] = []
        for inst in result["instances"]:
            best = None
            for turn in inst.get("qualifying_turns", []):
                for m in turn.get("matches", []):
                    if not m.get("has_call_dep"):
                        continue
                    score = (len(m.get("call_dep_kinds") or []),
                             len(m.get("call_dep_phrases") or []))
                    cand = (score[0], score[1], inst, turn, m)
                    if best is None or cand[:2] > best[:2]:
                        best = cand
            if best is not None:
                cd_exemplars.append((best[2], best[3], best[4]))
        cd_exemplars.sort(
            key=lambda t: (-len(t[2]["call_dep_kinds"]), t[0]["instance_id"])
        )

        if cd_exemplars and top_n:
            print()
            print(f"  Top {min(top_n, len(cd_exemplars))} per-instance "
                  "call-dep exemplars:")
            for inst, turn, m in cd_exemplars[:top_n]:
                kinds = ",".join(m["call_dep_kinds"])
                phrases_preview = "; ".join(
                    f"({k}) {p!r}" for k, p in m["call_dep_phrases"][:3]
                )
                snip = m["snippet"]
                if len(snip) > 240:
                    snip = snip[:240] + "..."
                print(f"    [{inst['instance_id']}] msg {turn['message_index']}  "
                      f"kinds={kinds}")
                print(f"      phrases: {phrases_preview}")
                print(f"      snippet: {snip}")

    # ------------------------------------------------------------------
    # Output-of-method-scan section
    # ------------------------------------------------------------------
    if "output_n_total_hits" in result:
        print()
        print("=" * 100)
        print("OUTPUT-OF-METHOD SCAN (annotations live inline on each match)")
        print("=" * 100)
        print(f"  Total snippets matching any output pattern : "
              f"{result['output_n_total_hits']}")
        print(f"  Distinct instances with output talk         : "
              f"{result['output_n_instances_with_hits']}")
        print()
        print("  Kind distribution  "
              "[hits = snippet matches, inst = distinct instances]:")
        out_kind_inst = result.get("output_kind_instance_counts") or {}
        for kind, count in (result.get("output_kind_counts") or {}).items():
            print(f"    {count:>5d} hits  {out_kind_inst.get(kind, 0):>4d} inst  "
                  f"{kind}")

        out_exemplars: list[tuple[dict, dict, dict]] = []
        for inst in result["instances"]:
            best = None
            for turn in inst.get("qualifying_turns", []):
                for m in turn.get("matches", []):
                    if not m.get("has_output_talk"):
                        continue
                    score = (len(m.get("output_kinds") or []),
                             len(m.get("output_phrases") or []))
                    cand = (score[0], score[1], inst, turn, m)
                    if best is None or cand[:2] > best[:2]:
                        best = cand
            if best is not None:
                out_exemplars.append((best[2], best[3], best[4]))
        out_exemplars.sort(
            key=lambda t: (-len(t[2]["output_kinds"]), t[0]["instance_id"])
        )

        if out_exemplars and top_n:
            print()
            print(f"  Top {min(top_n, len(out_exemplars))} per-instance "
                  "output exemplars:")
            for inst, turn, m in out_exemplars[:top_n]:
                kinds = ",".join(m["output_kinds"])
                phrases_preview = "; ".join(
                    f"({k}) {p!r}" for k, p in m["output_phrases"][:3]
                )
                snip = m["snippet"]
                if len(snip) > 240:
                    snip = snip[:240] + "..."
                print(f"    [{inst['instance_id']}] msg {turn['message_index']}  "
                      f"kinds={kinds}")
                print(f"      phrases: {phrases_preview}")
                print(f"      snippet: {snip}")

    # ------------------------------------------------------------------
    # Loop-execution-scan section
    # ------------------------------------------------------------------
    if "loop_n_total_hits" in result:
        print()
        print("=" * 100)
        print("LOOP-EXECUTION SCAN (annotations live inline on each match)")
        print("=" * 100)
        print(f"  Total snippets matching any loop pattern : "
              f"{result['loop_n_total_hits']}")
        print(f"  Distinct instances with loop talk         : "
              f"{result['loop_n_instances_with_hits']}")
        print()
        print("  Kind distribution  "
              "[hits = snippet matches, inst = distinct instances]:")
        loop_kind_inst = result.get("loop_kind_instance_counts") or {}
        for kind, count in (result.get("loop_kind_counts") or {}).items():
            print(f"    {count:>5d} hits  {loop_kind_inst.get(kind, 0):>4d} inst  "
                  f"{kind}")

        loop_exemplars: list[tuple[dict, dict, dict]] = []
        for inst in result["instances"]:
            best = None
            for turn in inst.get("qualifying_turns", []):
                for m in turn.get("matches", []):
                    if not m.get("has_loop_talk"):
                        continue
                    score = (len(m.get("loop_kinds") or []),
                             len(m.get("loop_phrases") or []))
                    cand = (score[0], score[1], inst, turn, m)
                    if best is None or cand[:2] > best[:2]:
                        best = cand
            if best is not None:
                loop_exemplars.append((best[2], best[3], best[4]))
        loop_exemplars.sort(
            key=lambda t: (-len(t[2]["loop_kinds"]), t[0]["instance_id"])
        )

        if loop_exemplars and top_n:
            print()
            print(f"  Top {min(top_n, len(loop_exemplars))} per-instance "
                  "loop exemplars:")
            for inst, turn, m in loop_exemplars[:top_n]:
                kinds = ",".join(m["loop_kinds"])
                phrases_preview = "; ".join(
                    f"({k}) {p!r}" for k, p in m["loop_phrases"][:3]
                )
                snip = m["snippet"]
                if len(snip) > 240:
                    snip = snip[:240] + "..."
                print(f"    [{inst['instance_id']}] msg {turn['message_index']}  "
                      f"kinds={kinds}")
                print(f"      phrases: {phrases_preview}")
                print(f"      snippet: {snip}")

    # ------------------------------------------------------------------
    # Conditional-execution-scan section
    # ------------------------------------------------------------------
    if "cond_n_total_hits" in result:
        print()
        print("=" * 100)
        print("CONDITIONAL-EXECUTION SCAN (annotations live inline on each match)")
        print("=" * 100)
        print(f"  Total snippets matching any cond pattern : "
              f"{result['cond_n_total_hits']}")
        print(f"  Distinct instances with cond talk         : "
              f"{result['cond_n_instances_with_hits']}")
        print()
        print("  Kind distribution  "
              "[hits = snippet matches, inst = distinct instances]:")
        cond_kind_inst = result.get("cond_kind_instance_counts") or {}
        for kind, count in (result.get("cond_kind_counts") or {}).items():
            print(f"    {count:>5d} hits  {cond_kind_inst.get(kind, 0):>4d} inst  "
                  f"{kind}")

        cond_exemplars: list[tuple[dict, dict, dict]] = []
        for inst in result["instances"]:
            best = None
            for turn in inst.get("qualifying_turns", []):
                for m in turn.get("matches", []):
                    if not m.get("has_cond_talk"):
                        continue
                    score = (len(m.get("cond_kinds") or []),
                             len(m.get("cond_phrases") or []))
                    cand = (score[0], score[1], inst, turn, m)
                    if best is None or cand[:2] > best[:2]:
                        best = cand
            if best is not None:
                cond_exemplars.append((best[2], best[3], best[4]))
        cond_exemplars.sort(
            key=lambda t: (-len(t[2]["cond_kinds"]), t[0]["instance_id"])
        )

        if cond_exemplars and top_n:
            print()
            print(f"  Top {min(top_n, len(cond_exemplars))} per-instance "
                  "cond exemplars:")
            for inst, turn, m in cond_exemplars[:top_n]:
                kinds = ",".join(m["cond_kinds"])
                phrases_preview = "; ".join(
                    f"({k}) {p!r}" for k, p in m["cond_phrases"][:3]
                )
                snip = m["snippet"]
                if len(snip) > 240:
                    snip = snip[:240] + "..."
                print(f"    [{inst['instance_id']}] msg {turn['message_index']}  "
                      f"kinds={kinds}")
                print(f"      phrases: {phrases_preview}")
                print(f"      snippet: {snip}")

    # ------------------------------------------------------------------
    # "Others" scan section — exception / mismatch / coverage
    # ------------------------------------------------------------------
    if "others_n_total_hits" in result:
        print()
        print("=" * 100)
        print("OTHERS SCAN (exception / mismatch / coverage; inline on each match)")
        print("=" * 100)
        print(f"  Total snippets matching any 'others' pattern: "
              f"{result['others_n_total_hits']}")
        print(f"  Distinct instances with others talk         : "
              f"{result['others_n_instances_with_hits']}")
        print()
        print("  Kind distribution  "
              "[hits = snippet matches, inst = distinct instances]:")
        others_kind_inst = result.get("others_kind_instance_counts") or {}
        for kind, count in (result.get("others_kind_counts") or {}).items():
            print(f"    {count:>5d} hits  {others_kind_inst.get(kind, 0):>4d} inst  "
                  f"{kind}")

        others_exemplars: list[tuple[dict, dict, dict]] = []
        for inst in result["instances"]:
            best = None
            for turn in inst.get("qualifying_turns", []):
                for m in turn.get("matches", []):
                    if not m.get("has_others_talk"):
                        continue
                    score = (len(m.get("others_kinds") or []),
                             len(m.get("others_phrases") or []))
                    cand = (score[0], score[1], inst, turn, m)
                    if best is None or cand[:2] > best[:2]:
                        best = cand
            if best is not None:
                others_exemplars.append((best[2], best[3], best[4]))
        others_exemplars.sort(
            key=lambda t: (-len(t[2]["others_kinds"]), t[0]["instance_id"])
        )

        if others_exemplars and top_n:
            print()
            print(f"  Top {min(top_n, len(others_exemplars))} per-instance "
                  "others exemplars:")
            for inst, turn, m in others_exemplars[:top_n]:
                kinds = ",".join(m["others_kinds"])
                phrases_preview = "; ".join(
                    f"({k}) {p!r}" for k, p in m["others_phrases"][:3]
                )
                snip = m["snippet"]
                if len(snip) > 240:
                    snip = snip[:240] + "..."
                print(f"    [{inst['instance_id']}] msg {turn['message_index']}  "
                      f"kinds={kinds}")
                print(f"      phrases: {phrases_preview}")
                print(f"      snippet: {snip}")

    # ------------------------------------------------------------------
    # Pass-7: prose-execution narration on no-code-token turns.
    # ------------------------------------------------------------------
    if "prose_exec_n_total_turns" in result:
        print()
        print("=" * 100)
        print("PROSE-EXECUTION SCAN (turns with no code tokens, English-only)")
        print("=" * 100)
        print(f"  Total no-code-token turns matching any prose-exec pattern: "
              f"{result['prose_exec_n_total_turns']}")
        print(f"  Distinct instances with prose-exec hits                  : "
              f"{result['prose_exec_n_instances_with_hits']}")
        print()
        print("  Kind distribution  "
              "[turns = matching turns, inst = distinct instances]:")
        pe_kind_inst = result.get("prose_exec_kind_instance_counts") or {}
        for kind, count in (result.get("prose_exec_kind_counts") or {}).items():
            print(f"    {count:>5d} turns  {pe_kind_inst.get(kind, 0):>4d} inst  "
                  f"{kind}")

        # Per-instance exemplars: ranked by (#kinds, #turns).
        pe_exemplars: list[tuple[dict, dict]] = []
        for inst in result["instances"]:
            turns = inst.get("prose_exec_only_turns") or []
            if not turns:
                continue
            # Pick the turn with the most kinds, ties → first by msg_idx.
            best = max(turns, key=lambda t: (len(t["prose_exec_kinds"]),
                                             -t["message_index"]))
            pe_exemplars.append((inst, best))
        pe_exemplars.sort(
            key=lambda t: (-len(t[1]["prose_exec_kinds"]), t[0]["instance_id"])
        )

        if pe_exemplars and top_n:
            print()
            print(f"  Top {min(top_n, len(pe_exemplars))} per-instance "
                  "prose-exec exemplars:")
            for inst, t in pe_exemplars[:top_n]:
                kinds = ",".join(t["prose_exec_kinds"])
                phrases_preview = "; ".join(
                    f"({k}) {p!r}" for k, p in t["prose_exec_phrases"][:3]
                )
                snip = t["thought_preview"]
                if len(snip) > 240:
                    snip = snip[:240] + "..."
                print(f"    [{inst['instance_id']}] msg {t['message_index']}  "
                      f"kinds={kinds}")
                print(f"      phrases: {phrases_preview}")
                print(f"      thought: {snip}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajs-dir", type=Path, default=DEFAULT_TRAJS_DIR,
                    help="Directory holding *.traj.json (recursively).")
    ap.add_argument("--resolutions", type=Path, default=DEFAULT_RESOLUTIONS,
                    help="Path to per_instance_details.json mapping "
                         "instance_id -> {resolved, cost, api_calls}. "
                         "Pass an empty string or 'none' to skip.")
    ap.add_argument("--difficulty-dataset", default=DEFAULT_DIFFICULTY_DATASET,
                    help="HuggingFace dataset id carrying the official "
                         "SWE-bench per-instance 'difficulty' column "
                         "(default: princeton-nlp/SWE-bench_Verified). "
                         "Pass an empty string or 'none' to skip.")
    ap.add_argument("--difficulty-json", type=Path, default=None,
                    help="Optional local JSON mapping instance_id -> difficulty "
                         "(or rows carrying instance_id + difficulty). Takes "
                         "precedence over --difficulty-dataset when present.")
    ap.add_argument("--no-difficulty", action="store_true",
                    help="Skip loading the official SWE-bench difficulty labels.")
    ap.add_argument("--min-score", type=int, default=DEFAULT_MIN_SCORE,
                    help="Distinct code-token categories required per turn "
                         "to count (default: 1). Raise to demand multiple "
                         "kinds of code tokens in the same turn.")
    ap.add_argument("--prose-hits", type=int, default=DEFAULT_PROSE_HITS,
                    help="Common English words required in the surrounding "
                         "context window for a match to count (default: 3). "
                         "This is the 'really inside prose, not inline code' "
                         "knob — raise it to demand a more sentence-like "
                         "context.")
    ap.add_argument("--min-qualifying-turns", type=int,
                    default=DEFAULT_MIN_QUALIFYING_TURNS,
                    help="Qualifying turns required per trajectory to flag it "
                         "(default: 1).")
    ap.add_argument("--max-first-msg-idx", type=int, default=None,
                    help="If set, demote a flagged trajectory to unflagged "
                         "when its first qualifying turn happens after this "
                         "msg_idx.")
    ap.add_argument("--max-api-calls", type=int, default=None,
                    help="If set, demote a flagged trajectory to unflagged "
                         "when its total api_calls exceeds this number.")
    # Default: scan only the model's visible response text.
    # Skipping CoT surfaces avoids the ~2x duplication that LiteLLM-style
    # providers introduce by mirroring the same chain-of-thought into both
    # `message.thinking_blocks` and
    # `message.extra.response.choices[*].message.reasoning_content`.
    # Use --no-top-level-only to also include thinking / reasoning surfaces.
    ap.add_argument("--top-level-only", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="(default) Scan only the visible response text and "
                         "skip extended-thinking / chain-of-thought surfaces. "
                         "Pass --no-top-level-only to include them.")
    ap.add_argument("--top", type=int, default=30,
                    help="How many top examples to print.")
    ap.add_argument("--out", type=Path, default=None,
                    help="If set, write the combined result (pass-1 code-token "
                         "scan + pass-2 variable-state scan, the latter under "
                         "'variable_state_scan') as a single JSON file.")

    ap.add_argument("--min-reasoning-snippets", type=int,
                    default=DEFAULT_MIN_REASONING_SNIPPETS,
                    help="Distinct annotated snippets required before an "
                         "instance counts as code reasoning (default: 2). "
                         "Set to 1 to restore the old any-single-hit rule.")
    ap.add_argument("--allow-observation-only", action="store_true",
                    help="Count instances whose only annotated snippets are "
                         "outcome observations (e.g. 'all tests pass') as "
                         "code reasoning. By default at least one snippet "
                         "must reason about the execution process itself.")
    ap.add_argument("--no-state-scan", action="store_true",
                    help="Skip the pass-2 variable-state scan over the "
                         "code-token snippets.")
    ap.add_argument("--include-mutation-verb-only", action="store_true",
                    help="Keep state-scan hits whose only matching kind is the "
                         "generic 'mutation-verb' (noisy). Default: drop them.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.trajs_dir.exists():
        raise SystemExit(f"trajs-dir does not exist: {args.trajs_dir}")

    res_path = args.resolutions
    if res_path is not None and str(res_path).lower() in ("", "none"):
        res_path = None
    if res_path is not None and not Path(res_path).exists():
        print(f"WARNING: resolutions file not found: {res_path} "
              f"-- continuing without resolution data.")
        res_path = None

    resolutions = load_resolutions(res_path)
    if resolutions:
        print(f"Loaded resolution data for {len(resolutions)} instance(s) "
              f"from {res_path}\n")

    if args.no_difficulty:
        difficulties: dict[str, str] = {}
    else:
        diff_dataset = args.difficulty_dataset
        if diff_dataset is not None and str(diff_dataset).lower() in ("", "none"):
            diff_dataset = None
        difficulties = load_difficulties(
            dataset=diff_dataset, json_path=args.difficulty_json
        )
        if difficulties:
            src = args.difficulty_json or diff_dataset
            print(f"Loaded difficulty labels for {len(difficulties)} "
                  f"instance(s) from {src}\n")

    result = analyze_trajectories(
        args.trajs_dir,
        resolutions=resolutions,
        min_score=args.min_score,
        prose_hits_required=args.prose_hits,
        min_qualifying_turns=args.min_qualifying_turns,
        max_first_msg_idx=args.max_first_msg_idx,
        max_api_calls=args.max_api_calls,
        top_level_only=args.top_level_only,
        run_state_scan=not args.no_state_scan,
        include_mutation_verb_only=args.include_mutation_verb_only,
        difficulties=difficulties,
        min_reasoning_snippets=args.min_reasoning_snippets,
        require_core_reasoning=not args.allow_observation_only,
    )
    print_report(result, top_n=args.top)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nWrote combined results to {args.out}")
        print(f"  - total instances:            {len(result['instances'])}")
        print(f"  - code-reasoning instances:   "
              f"{result.get('n_instances_with_code_reasoning', 0)}")
        if "state_n_strong_hits" in result:
            n_inst_state = sum(
                1 for i in result["instances"] if i.get("has_variable_state_talk")
            )
            print(f"  - strong state hits:          {result['state_n_strong_hits']}")
            print(f"  - instances w/ state talk:    {n_inst_state}")
        if "call_dep_n_total_hits" in result:
            n_inst_cd = sum(
                1 for i in result["instances"] if i.get("has_call_dep_talk")
            )
            print(f"  - call-dep hits:              {result['call_dep_n_total_hits']}")
            print(f"  - instances w/ call-dep talk: {n_inst_cd}")
        if "output_n_total_hits" in result:
            n_inst_out = sum(
                1 for i in result["instances"] if i.get("has_output_talk")
            )
            print(f"  - output hits:                {result['output_n_total_hits']}")
            print(f"  - instances w/ output talk:   {n_inst_out}")
        if "loop_n_total_hits" in result:
            n_inst_loop = sum(
                1 for i in result["instances"] if i.get("has_loop_talk")
            )
            print(f"  - loop hits:                  {result['loop_n_total_hits']}")
            print(f"  - instances w/ loop talk:     {n_inst_loop}")
        if "cond_n_total_hits" in result:
            n_inst_cond = sum(
                1 for i in result["instances"] if i.get("has_cond_talk")
            )
            print(f"  - cond hits:                  {result['cond_n_total_hits']}")
            print(f"  - instances w/ cond talk:     {n_inst_cond}")
        if "others_n_total_hits" in result:
            n_inst_others = sum(
                1 for i in result["instances"] if i.get("has_others_talk")
            )
            print(f"  - others hits:                {result['others_n_total_hits']}")
            print(f"  - instances w/ others talk:   {n_inst_others}")
        if "prose_exec_n_total_turns" in result:
            n_inst_pe = sum(
                1 for i in result["instances"] if i.get("has_prose_exec_only_talk")
            )
            print(f"  - prose-exec turns (no-code): "
                  f"{result['prose_exec_n_total_turns']}")
            print(f"  - instances w/ prose-exec:    {n_inst_pe}")
        ann_kinds = "{state_*,call_dep_*,output_*,loop_*,cond_*"
        if OTHERS_ENABLED:
            ann_kinds += ",others_*"
        ann_kinds += "}"
        print("  - per-match annotations live inline at "
              f"instances[*].qualifying_turns[*].matches[*].{ann_kinds}")


if __name__ == "__main__":
    main()
