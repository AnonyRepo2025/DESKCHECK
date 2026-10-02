from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

DEFAULT_TRAJS_DIR = Path(
    "experiments/evaluation/bash-only/"
    "20260217_mini-v2.0.0_claude-4-6-opus/trajs"
)
DEFAULT_RESOLUTIONS = Path(
    "experiments/evaluation/bash-only/"
    "20260217_mini-v2.0.0_claude-4-6-opus/per_instance_details.json"
)

PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Tracing variable values
    (re.compile(
        r"(?:the\s+)?(?:value\s+of\s+)?[`'\"]?\w+[`'\"]?\s+"
        r"(?:starts?\s+(?:as|at|with)|is\s+initially)\s+[`'\"]?\w",
        re.I), "variable-starts-as"),
    (re.compile(
        r"(?:the\s+)?(?:value\s+of\s+)?[`'\"]?\w+[`'\"]?\s+"
        r"(?:becomes?|is\s+(?:now|set\s+to|updated\s+to|changed\s+to))\s+[`'\"]?\w",
        re.I), "variable-becomes"),

    # Function call + return with specific values
    (re.compile(r"(?:when|if)\s+.*?\s+is\s+called\s+with\b", re.I),
     "when-called-with"),
    (re.compile(
        r"calling\s+[`'\"]?\w+[`'\"]?\s*\(.*?\)\s*"
        r"(?:returns?|gives?|yields?|produces?)",
        re.I), "calling-X-returns"),
    # `returns` (third-person) — the imperative `return X` is too easily
    # matched on raw source code the agent pastes verbatim.
    (re.compile(
        r"returns\s+[`'\"]?(?:True|False|None|\d+|an?\s+\w+|the\s+\w+)[`'\"]?",
        re.I), "returns-specific-value"),

    # Evaluation / result
    (re.compile(r"evaluates?\s+to\b", re.I), "evaluates-to"),
    (re.compile(r"results?\s+in\b", re.I), "results-in"),

    # Execution flow: first … then … finally
    (re.compile(
        r"first[,:]?\s+.{10,80}?\bthen\b.{10,80}?"
        r"\b(?:finally|next|after\s+that)\b",
        re.I | re.S), "first-then-finally"),

    # Passing values between functions
    (re.compile(r"pass(?:es|ed|ing)\s+.{1,60}?\bto\s+[`'\"]?\w+", re.I),
     "passes-to"),

    # "so the output is", "which gives us", "this means … will …"
    (re.compile(
        r"so\s+the\s+(?:output|result|return\s+value)\s+"
        r"(?:is|will\s+be|would\s+be)\b", re.I), "so-output-is"),
    (re.compile(r"which\s+gives?\s+(?:us|back|the)\b", re.I),
     "which-gives-us"),
    (re.compile(
        r"this\s+means?\s+.{1,60}?"
        r"\bwill\s+(?:be|return|raise|fail|produce|call|invoke)\b",
        re.I), "this-means-will"),

    # Trace / walk / step / work / think through code (or what/how X happens).
    (re.compile(
        r"(?:trac(?:e|ing)|walk(?:ing)?|step(?:ping)?|"
        r"work(?:ing)?|think(?:ing)?|go(?:ing)?|simulat(?:e|ing))\s+through\s+"
        r"(?:the\s+|this\s+|that\s+|what\s+|how\s+|each\s+)?"
        r"(?:code|execution|logic|function|method|flow|call|"
        r"iteration|loop|happens?|works?|conditional|branch|cases?|scenario)",
        re.I), "trace-through"),

    # Numbered execution steps
    (re.compile(r"(?:step\s+1\b.{10,300}?step\s+2\b)", re.I | re.S),
     "numbered-steps"),

    # "x = <value>" then later describes change
    (re.compile(
        r"[`'\"]?\w+[`'\"]?\s*=\s*[`'\"]?\w+[`'\"]?\s*[\.\,;]\s*"
        r"(?:then|so|after|next|which)\b", re.I), "assignment-then"),

    (re.compile(
        r"will\s+(?:try\s+to|attempt\s+to)\s+"
        r"(?:call|invoke|access|use|execute|run)\b", re.I), "will-try-to"),

    # "after the loop / call / iteration … becomes / equals / returns"
    (re.compile(
        r"after\s+(?:the\s+)?(?:loop|iteration|call(?:ing)?|"
        r"executing|running)\b.{1,80}?"
        r"\b(?:becomes?|is\s+now|equals?|will\s+be|returns?)\b",
        re.I | re.S), "after-loop-becomes"),

    # "if we pass / supply / provide X … then"
    (re.compile(
        r"if\s+(?:we|you|one)\s+(?:pass|supply|provide|give|send|input)\b"
        r".{1,100}?\bthen\b", re.I | re.S), "if-we-pass-then"),

    # "the function returns" with specifics
    (re.compile(
        r"the\s+(?:function|method|call)\s+"
        r"(?:returns?|yields?|produces?|outputs?)\s+[`'\"]?\w",
        re.I), "the-function-returns"),

    # Execution reasoning with code references like "self.x returns" or "obj.method() becomes".
    (re.compile(
        r"(?:self|cls|obj)\.\w+\s*(?:\(.*?\))?\s+"
        r"(?:returns?|becomes?|evaluates?\s+to|"
        r"will\s+(?:be|return|raise))\b", re.I), "self-dot-returns"),

    # "since X is …, … will …"
    (re.compile(
        r"since\s+[`'\"]?\w+[`'\"]?\s+is\s+.{1,80}?,\s*.{0,40}?"
        r"\bwill\s+(?:be|return|raise|fail|call|get|produce)\b",
        re.I | re.S), "since-X-is-will"),

    # Numbered/bulleted execution-trace step list: a list line where a backtick-quoted identifier (often dotted) gets assigned, set, or returns.
    # Strong signal that the model is enumerating successive states.
    (re.compile(
        r"^\s*(?:\d+[\.\)]\s+|[-*+]\s+)"
        r"`[^`]+`\s+"
        r"(?:is\s+(?:set\s+to|now|equal\s+to|assigned|None|True|False|the)|"
        r"becomes?|returns?|gets?\s+(?:set|assigned|called)|holds?|"
        r"points?\s+to|=\s+[`'\"]?\w|→|->)\b",
        re.I | re.M),
     "numbered-step-state-change"),

    # Indexed-element traces: A[2][0] = 0, kwargs['x'] = 5, arr[i] is None
    # The agent is pinning down a concrete value at a concrete index.
    (re.compile(
        r"\b\w+(?:\.\w+)*\s*\[[^\]]{1,40}\]\s*"
        r"(?:=\s*[`'\"]?(?:\d+|None|True|False|\w)|"
        r"is\s+[`'\"]?(?:\d+|None|True|False|now|set\s+to)|"
        r"->|→|becomes?|returns?\s+[`'\"]?\w)",
        re.I),
     "indexed-element-value"),

    # Explicit step / iteration / pass labels: "Iteration 1:", "Step 2 -",
    # "Round #3:", "Pass 1." Followed by content (the trace itself).
    (re.compile(
        r"\b(?:iteration|step|pass|round|cycle)\s+#?\d+\s*[:\-—.]\s+\S",
        re.I), "iteration-label"),

    # Arrow-result notation: f(x) -> value, obj.method() → result
    # The agent writes execution as call → return.
    (re.compile(
        r"\b\w+(?:\.\w+)*\s*\([^)]{0,80}\)\s*"
        r"(?:->|=>|→|⟹)\s*"
        r"[`'\"]?(?:\d+|None|True|False|\w)",
        re.I),
     "call-arrow-result"),
]


QUALITY_RE = re.compile(
    r"(?:`[^`]+`|'[^']+'|\"[^\"]+\"|\b\d+\b|\bNone\b|\bTrue\b|\bFalse\b"
    r"|\bself\.\w+|\bcls\.\w+|\w+\.\w+\()",
    re.I,
)


# Match (and remove) markdown-fenced code blocks: ```...``` or ~~~...~~~.
# Non-greedy, dotall — pairs the nearest closing fence so adjacent blocks
# don't get joined together.
FENCED_CODE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.S)

# A single line of the form "  123:  <content>" or "  123\t<content>" —
# the shape of `cat -n` / `sed -n` / grep-with-line-numbers output that
# the model regularly pastes back into its reasoning. One such line is
# fine (e.g. a prose reference like "see line 42:"); a run of two or
# more is almost certainly file content.
LINE_NUMBERED_RE = re.compile(r"^\s*\d+[\t:]\s")


def strip_code_blocks(text: str) -> str:
    """Drop code-snippet-shaped regions so patterns only match prose.

    The patterns above are tuned for natural-language execution reasoning
    ("X becomes 0, then we call f(...)"), but they fire just as readily
    inside source code the agent has quoted verbatim — e.g. the
    `iteration-label` regex matches `pass 178:` inside cat output, and
    `numbered-steps` matches `print('Step 1: ...')` inside a script.
    Stripping these regions before matching removes that contamination.
    """
    text = FENCED_CODE_RE.sub("\n", text)

    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if LINE_NUMBERED_RE.match(lines[i]):
            j = i
            while j < len(lines) and LINE_NUMBERED_RE.match(lines[j]):
                j += 1
            if j - i >= 2:
                i = j
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


MIN_THOUGHT_CHARS = 50
DEFAULT_MIN_SCORE = 1               
DEFAULT_QUALITY_HITS = 2           
DEFAULT_MIN_QUALIFYING_TURNS = 1 


def extract_thought(
    message: dict,
    top_level_only: bool = False,
    provider_uses_content: bool = True,
) -> str:
    """Pull the assistant's thought text out of one message.

    With ``top_level_only=True``, return the LLM's response text.
    Concretely: prefer ``content`` (Anthropic ``text`` blocks, DeepSeek
    ``content``, OpenAI ``output_text``) and skip thinking / CoT
    surfaces. But some providers (e.g. Gemini-3-flash via LiteLLM) emit
    ``content=None`` everywhere and route the model's actual response
    into ``reasoning_content`` / ``thinking_blocks``. For those, the
    "thinking" surfaces *are* the response, so we fall back to them.

    The caller is responsible for telling us which kind of provider we
    have — pass ``provider_uses_content=False`` to get the Gemini-style
    fallback behavior. ``iter_assistant_thoughts`` detects this once per
    trajectory; here it defaults to True (the structured-response case).
    """
    parts: list[str] = []

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        parts.append(content.strip())
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif (not top_level_only and btype == "thinking"
                  and isinstance(block.get("thinking"), str)):
                parts.append(block["thinking"])

    # In top-level mode skip thinking surfaces, except when the provider
    # doesn't use the content surface at all — in which case those
    # surfaces are the only response we have.
    skip_thinking = top_level_only and provider_uses_content

    if not skip_thinking:
        for block in message.get("thinking_blocks") or []:
            if isinstance(block, dict) and block.get("type") == "thinking":
                text = block.get("thinking")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())

        # OpenRouter / minimax shape
        rd_blocks = message.get("reasoning_details")
        appended_from_rd = False
        if isinstance(rd_blocks, list):
            for block in rd_blocks:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                        appended_from_rd = True
        if not appended_from_rd:
            reasoning = message.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                parts.append(reasoning.strip())

        # DeepSeek-reasoner shape
        extra = message.get("extra")
        if isinstance(extra, dict):
            resp = extra.get("response")
            if isinstance(resp, dict):
                for choice in resp.get("choices") or []:
                    if not isinstance(choice, dict):
                        continue
                    inner = choice.get("message")
                    if not isinstance(inner, dict):
                        continue
                    rc = inner.get("reasoning_content")
                    if isinstance(rc, str) and rc.strip():
                        parts.append(rc.strip())

    # OpenAI Responses-API shape — `message` items are the response;
    # `reasoning` items are the CoT summary. In top-level mode, drop the
    # reasoning summary unless the provider clearly doesn't use the
    # message surface (mirrors the content/reasoning_content fallback
    # logic above).
    output = message.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "message":
                for sub in item.get("content") or []:
                    if isinstance(sub, dict) and sub.get("type") == "output_text":
                        text = sub.get("text")
                        if isinstance(text, str) and text.strip():
                            parts.append(text.strip())
            elif itype == "reasoning":
                if top_level_only and provider_uses_content:
                    continue
                for sub in item.get("summary") or []:
                    if isinstance(sub, dict):
                        text = sub.get("text")
                        if isinstance(text, str) and text.strip():
                            parts.append(text.strip())

    return "\n\n".join(parts).strip()


def _provider_uses_content(messages: list) -> bool:
    """True if any assistant message in this trajectory has a populated
    ``content`` (string or non-empty list) or OpenAI ``output[message]``.
    False for providers like Gemini-3-flash where every assistant turn
    has ``content=None`` and the response lives in ``reasoning_content``.
    """
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        is_asst = msg.get("role") == "assistant"
        is_resp = msg.get("role") is None and isinstance(msg.get("output"), list)
        if not (is_asst or is_resp):
            continue
        c = msg.get("content")
        if isinstance(c, str) and c.strip():
            return True
        if isinstance(c, list) and any(isinstance(b, dict) for b in c):
            return True
        out = msg.get("output")
        if isinstance(out, list) and any(
            isinstance(it, dict) and it.get("type") == "message" for it in out
        ):
            return True
    return False


def iter_assistant_thoughts(
    traj_path: Path,
    top_level_only: bool = False,
) -> Iterable[tuple[int, str]]:
    """Yield (message_index, thought_text) for every assistant turn."""
    try:
        with open(traj_path) as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return

    messages = data.get("messages") or []
    # Some traj formats use "trajectory" with explicit "thought" fields.
    if not messages and "trajectory" in data:
        for i, step in enumerate(data["trajectory"]):
            t = step.get("thought") or ""
            if t:
                yield i, t
        return

    uses_content = _provider_uses_content(messages) if top_level_only else True

    for i, msg in enumerate(messages):
        is_chat_assistant = msg.get("role") == "assistant"
        is_response_object = msg.get("role") is None and isinstance(msg.get("output"), list)
        if not (is_chat_assistant or is_response_object):
            continue
        thought = extract_thought(
            msg,
            top_level_only=top_level_only,
            provider_uses_content=uses_content,
        )
        if thought:
            yield i, thought



def extract_context(text: str, match: re.Match[str], context_chars: int = 300) -> str:
    half = context_chars // 2
    start = max(0, match.start() - half)
    end = min(len(text), match.end() + half)
    snippet = text[start:end].replace("\n", " ").strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def score_thought(
    thought: str,
    quality_hits_required: int = DEFAULT_QUALITY_HITS,
) -> tuple[int, list[tuple[str, str, int]]]:
    """Return (distinct-label score, [(label, snippet, quality_hits), ...])."""
    if not thought or len(thought) < MIN_THOUGHT_CHARS:
        return 0, []

    cleaned = strip_code_blocks(thought)
    if len(cleaned) < MIN_THOUGHT_CHARS:
        return 0, []

    matches: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for pat, label in PATTERNS:
        for m in pat.finditer(cleaned):
            if label in seen:
                break
            snippet = extract_context(cleaned, m)
            qhits = len(QUALITY_RE.findall(snippet))
            if qhits >= quality_hits_required:
                seen.add(label)
                matches.append((label, snippet, qhits))
                break  # one example per label is enough

    matches.sort(key=lambda x: -x[2])
    return len(matches), matches



def load_resolutions(path: Path | None) -> dict[str, dict]:
    if path is None or not Path(path).exists():
        return {}
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    if isinstance(data, list):
        out: dict[str, dict] = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            iid = entry.get("instance_id") or entry.get("instance")
            if iid:
                out[str(iid)] = entry
        return out
    return {}


def _list_traj_files(trajs_dir: Path) -> list[Path]:
    files = sorted(trajs_dir.rglob("*.traj.json")) + sorted(trajs_dir.rglob("*.traj"))
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in files:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


def parse_thoughts(
    trajs_dir: Path,
    top_level_only: bool = False,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for tf in _list_traj_files(trajs_dir):
        iid = tf.stem.replace(".traj", "")
        out[iid] = {
            "file": str(tf),
            "thoughts": list(iter_assistant_thoughts(tf, top_level_only=top_level_only)),
        }
    return out


def score_one(
    parsed: dict[str, dict],
    resolutions: dict[str, dict],
    min_score: int,
    quality_hits_required: int,
    min_qualifying_turns: int,
    max_first_msg_idx: int | None = None,
    max_api_calls: int | None = None,
) -> tuple[list[dict], list[dict]]:
    instance_results: list[dict] = []
    qualifying_steps: list[dict] = []

    for iid, parsed_inst in parsed.items():
        per_turn: list[dict] = []
        max_score = 0
        all_labels: set[str] = set()
        thoughts = parsed_inst["thoughts"]

        for msg_idx, thought in thoughts:
            score, matches = score_thought(thought, quality_hits_required)
            if score >= min_score:
                turn = {
                    "message_index": msg_idx,
                    "score": score,
                    "matches": [
                        {"label": lbl, "quality_hits": q, "snippet": snip}
                        for lbl, snip, q in matches
                    ],
                }
                per_turn.append(turn)
                qualifying_steps.append({
                    "instance_id": iid,
                    "file": parsed_inst["file"],
                    "message_index": msg_idx,
                    "score": score,
                    "matches": turn["matches"],
                })
                max_score = max(max_score, score)
                all_labels.update(m[0] for m in matches)

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

        instance_results.append({
            "instance_id": iid,
            "file": parsed_inst["file"],
            "resolved": res.get("resolved"),
            "cost": res.get("cost"),
            "api_calls": api_calls,
            "has_execution_reasoning": flagged,
            "first_qualifying_msg_idx": first_idx,
            "n_assistant_turns_with_thought": len(thoughts),
            "n_qualifying_turns": len(per_turn),
            "max_turn_score": max_score,
            "distinct_labels": sorted(all_labels),
            "qualifying_turns": per_turn,
        })

    return instance_results, qualifying_steps


def analyze_trajectories(
    trajs_dir: Path,
    resolutions: dict[str, dict] | None = None,
    min_score: int = DEFAULT_MIN_SCORE,
    quality_hits_required: int = DEFAULT_QUALITY_HITS,
    min_qualifying_turns: int = DEFAULT_MIN_QUALIFYING_TURNS,
    max_first_msg_idx: int | None = None,
    max_api_calls: int | None = None,
    parsed: dict[str, dict] | None = None,
    top_level_only: bool = False,
) -> dict:
    resolutions = resolutions or {}
    if parsed is None:
        parsed = parse_thoughts(trajs_dir, top_level_only=top_level_only)

    instance_results, qualifying_steps = score_one(
        parsed, resolutions,
        min_score=min_score,
        quality_hits_required=quality_hits_required,
        min_qualifying_turns=min_qualifying_turns,
        max_first_msg_idx=max_first_msg_idx,
        max_api_calls=max_api_calls,
    )

    qualifying_steps.sort(key=lambda r: -r["score"])

    # Cross-tab: resolved (yes/no/unknown) × has_execution_reasoning (yes/no)
    crosstab: dict[str, dict[str, int]] = {
        "resolved=True":  {"reasoning=True": 0, "reasoning=False": 0},
        "resolved=False": {"reasoning=True": 0, "reasoning=False": 0},
        "resolved=None":  {"reasoning=True": 0, "reasoning=False": 0},
    }
    for inst in instance_results:
        rkey = f"resolved={inst['resolved']}"
        if rkey not in crosstab:
            crosstab[rkey] = {"reasoning=True": 0, "reasoning=False": 0}
        ckey = f"reasoning={bool(inst['has_execution_reasoning'])}"
        crosstab[rkey][ckey] = crosstab[rkey].get(ckey, 0) + 1

    n_with_resolution = sum(1 for i in instance_results if i["resolved"] is not None)
    n_resolved = sum(1 for i in instance_results if i["resolved"] is True)
    missing_resolution = [
        i["instance_id"] for i in instance_results if i["resolved"] is None
    ]

    return {
        "trajs_dir": str(trajs_dir),
        "n_files": len(parsed),
        "min_score": min_score,
        "quality_hits_required": quality_hits_required,
        "min_qualifying_turns": min_qualifying_turns,
        "top_level_only": top_level_only,
        "n_instances_with_execution_reasoning": sum(
            1 for i in instance_results if i["has_execution_reasoning"]
        ),
        "n_with_resolution_data": n_with_resolution,
        "n_resolved": n_resolved,
        "n_missing_resolution": len(missing_resolution),
        "missing_resolution_ids": missing_resolution[:20],
        "crosstab_resolved_by_reasoning": crosstab,
        "instances": instance_results,
        "top_steps": qualifying_steps[:50],
    }


def run_sweep(
    trajs_dir: Path,
    resolutions: dict[str, dict],
    quality_hits_grid: list[int],
    min_score_grid: list[int],
    min_qualifying_turns_grid: list[int],
) -> list[dict]:
    parsed = parse_thoughts(trajs_dir)
    rows: list[dict] = []
    for q in quality_hits_grid:
        for s in min_score_grid:
            for t in min_qualifying_turns_grid:
                instances, _ = score_one(
                    parsed, resolutions,
                    min_score=s, quality_hits_required=q,
                    min_qualifying_turns=t,
                )
                n = len(instances)
                w = [i for i in instances if i["has_execution_reasoning"]]
                wo = [i for i in instances if not i["has_execution_reasoning"]]
                w_res = sum(1 for i in w if i["resolved"] is True)
                wo_res = sum(1 for i in wo if i["resolved"] is True)
                rate_w = (w_res / len(w)) if w else None
                rate_wo = (wo_res / len(wo)) if wo else None
                lift = (rate_w - rate_wo) if (rate_w is not None and rate_wo is not None) else None
                rows.append({
                    "quality_hits": q,
                    "min_score": s,
                    "min_qualifying_turns": t,
                    "n": n,
                    "n_flagged": len(w),
                    "flag_pct": len(w) / n if n else 0.0,
                    "n_with_resolved": w_res,
                    "n_without_resolved": wo_res,
                    "rate_with": rate_w,
                    "rate_without": rate_wo,
                    "lift": lift,
                })
    return rows


def print_sweep(rows: list[dict], trajs_dir: str) -> None:
    print(f"\nSweep over definition knobs for: {trajs_dir}")
    print(f"  (q = quality_hits, s = min_score, t = min_qualifying_turns)")
    print(f"  → looser as any knob ↓ ; stricter as any knob ↑")
    print()
    print(f"  {'q':>2s} {'s':>2s} {'t':>2s}"
          f"  {'flagged':>10s}  {'r/with':>14s}  {'r/without':>14s}  {'lift':>8s}")
    print(f"  {'-' * 70}")
    for r in rows:
        if r["rate_with"] is None or r["rate_without"] is None:
            tail = "  (insufficient data)"
        else:
            tail = (f"  {r['n_with_resolved']:>3d}/{r['n_flagged']:<3d}={r['rate_with']:>5.1%}"
                    f"  {r['n_without_resolved']:>3d}/{r['n']-r['n_flagged']:<3d}={r['rate_without']:>5.1%}"
                    f"  {r['lift']:>+7.1%}")
        print(f"  {r['quality_hits']:>2d} {r['min_score']:>2d} {r['min_qualifying_turns']:>2d}"
              f"  {r['n_flagged']:>4d}/{r['n']:<4d}={r['flag_pct']:>5.1%}"
              + tail)


def _fmt_resolved(value) -> str:
    if value is True:
        return "RESOLVED"
    if value is False:
        return "FAILED  "
    return "UNKNOWN "


def print_report(result: dict, top_n: int = 30) -> None:
    n_files = result["n_files"]
    n_hits = result["n_instances_with_execution_reasoning"]
    n_with_res = result.get("n_with_resolution_data", 0)
    n_resolved = result.get("n_resolved", 0)

    print(f"Scanned {n_files} trajectory file(s) under {result['trajs_dir']}")
    print(f"min_score = {result['min_score']}, "
          f"quality_hits_required = {result['quality_hits_required']}")
    print(f"Trajectories with code-execution reasoning: {n_hits} / {n_files}")
    if n_with_res:
        print(f"Instances with resolution data:           "
              f"{n_with_res} / {n_files}  (resolved: {n_resolved})")
        if result.get("n_missing_resolution"):
            print(f"  (missing resolution data for "
                  f"{result['n_missing_resolution']} instances; "
                  f"first few: {result['missing_resolution_ids'][:5]})")
    print()

    crosstab = result.get("crosstab_resolved_by_reasoning") or {}
    if crosstab:
        print("=" * 100)
        print("CROSSTAB: resolution × code-execution reasoning")
        print("=" * 100)
        col_keys = ["reasoning=True", "reasoning=False"]
        row_order = ["resolved=True", "resolved=False", "resolved=None"]
        for rk in list(crosstab.keys()):
            if rk not in row_order:
                row_order.append(rk)
        header = f"  {'':<18s} " + "  ".join(f"{c:>16s}" for c in col_keys) + "    total"
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
                  + "  ".join(f"{cells.get(c, 0):>16d}" for c in col_keys)
                  + f"    {row_total:>5d}")
        print(f"  {'total':<18s} "
              + "  ".join(f"{col_totals[c]:>16d}" for c in col_keys)
              + f"    {grand:>5d}")

        with_reasoning = {
            "resolved": crosstab.get("resolved=True", {}).get("reasoning=True", 0),
            "failed":   crosstab.get("resolved=False", {}).get("reasoning=True", 0),
        }
        without_reasoning = {
            "resolved": crosstab.get("resolved=True", {}).get("reasoning=False", 0),
            "failed":   crosstab.get("resolved=False", {}).get("reasoning=False", 0),
        }
        wr_tot = with_reasoning["resolved"] + with_reasoning["failed"]
        wo_tot = without_reasoning["resolved"] + without_reasoning["failed"]
        if wr_tot:
            print(f"\n  resolution rate WITH code-execution reasoning: "
                  f"{with_reasoning['resolved']}/{wr_tot} "
                  f"= {with_reasoning['resolved'] / wr_tot:.1%}")
        if wo_tot:
            print(f"  resolution rate WITHOUT code-execution reasoning: "
                  f"{without_reasoning['resolved']}/{wo_tot} "
                  f"= {without_reasoning['resolved'] / wo_tot:.1%}")
        print()

    print("=" * 100)
    print("PER-INSTANCE SUMMARY (only instances flagged as having execution reasoning)")
    print("=" * 100)
    flagged = [i for i in result["instances"] if i["has_execution_reasoning"]]
    flagged.sort(key=lambda r: (-r["max_turn_score"], -r["n_qualifying_turns"]))
    for inst in flagged[:top_n]:
        labels = ", ".join(inst["distinct_labels"][:6])
        if len(inst["distinct_labels"]) > 6:
            labels += f", … (+{len(inst['distinct_labels']) - 6})"
        print(f"  [{_fmt_resolved(inst['resolved'])}] "
              f"{inst['instance_id']:<40s}"
              f"  turns={inst['n_qualifying_turns']:>2d}"
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
        for m in step["matches"][:3]:
            snippet = m["snippet"]
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            print(f"    [{m['label']}] (q={m['quality_hits']}) "
                  f"\"{snippet}\"")

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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajs-dir", type=Path, default=DEFAULT_TRAJS_DIR,
                    help="Directory holding *.traj.json (recursively).")
    ap.add_argument("--resolutions", type=Path, default=DEFAULT_RESOLUTIONS,
                    help="Path to per_instance_details.json mapping "
                         "instance_id -> {resolved, cost, api_calls}. "
                         "Pass an empty string or 'none' to skip.")
    ap.add_argument("--min-score", type=int, default=DEFAULT_MIN_SCORE,
                    help="Distinct pattern labels required per turn to count "
                         "(default: 2).")
    ap.add_argument("--quality-hits", type=int, default=DEFAULT_QUALITY_HITS,
                    help="Concrete references required around each match "
                         "(default: 2).")
    ap.add_argument("--min-qualifying-turns", type=int,
                    default=DEFAULT_MIN_QUALIFYING_TURNS,
                    help="Qualifying turns required per trajectory to flag it "
                         "as having code-execution reasoning (default: 1). "
                         "Raise to demand sustained tracing across multiple "
                         "turns; this is the trajectory-level tightness knob.")
    ap.add_argument("--max-first-msg-idx", type=int, default=None,
                    help="If set, demote a flagged trajectory to unflagged "
                         "when its first qualifying turn happens after this "
                         "msg_idx. Empirically separates 'reasoning-while-"
                         "stuck' (late) from 'reasoning-then-fix' (early). "
                         "Try 30.")
    ap.add_argument("--max-api-calls", type=int, default=None,
                    help="If set, demote a flagged trajectory to unflagged "
                         "when its total api_calls exceeds this number. "
                         "Difficulty / struggle proxy. Try 40.")
    ap.add_argument("--top-level-only", action="store_true",
                    help="Score only the model's committed response text "
                         "(Anthropic `text`, DeepSeek `content`, OpenAI "
                         "`output_text`). Skip extended-thinking / "
                         "chain-of-thought surfaces. Useful for separating "
                         "what the model 'commits to' from its private "
                         "monologue.")
    ap.add_argument("--top", type=int, default=30,
                    help="How many top examples to print.")
    ap.add_argument("--out", type=Path, default=None,
                    help="If set, write the full result as JSON to this path.")
    ap.add_argument("--sweep", action="store_true",
                    help="Sweep a grid of (quality_hits × min_score × "
                         "min_qualifying_turns) and print one summary row per "
                         "definition. Skips per-instance reporting.")
    ap.add_argument("--sweep-quality-hits", type=str, default="1,2,3",
                    help="Comma-separated values for the sweep grid "
                         "(default: 1,2,3).")
    ap.add_argument("--sweep-min-score", type=str, default="1,2,3",
                    help="Comma-separated values for the sweep grid "
                         "(default: 1,2,3).")
    ap.add_argument("--sweep-min-qualifying-turns", type=str, default="1,2,3",
                    help="Comma-separated values for the sweep grid "
                         "(default: 1,2,3).")
    return ap.parse_args()


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


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

    if args.sweep:
        rows = run_sweep(
            args.trajs_dir,
            resolutions,
            quality_hits_grid=_parse_int_list(args.sweep_quality_hits),
            min_score_grid=_parse_int_list(args.sweep_min_score),
            min_qualifying_turns_grid=_parse_int_list(args.sweep_min_qualifying_turns),
        )
        print_sweep(rows, str(args.trajs_dir))
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as fh:
                json.dump({"trajs_dir": str(args.trajs_dir), "sweep": rows},
                          fh, indent=2)
            print(f"\nWrote sweep results to {args.out}")
        return

    result = analyze_trajectories(
        args.trajs_dir,
        resolutions=resolutions,
        min_score=args.min_score,
        quality_hits_required=args.quality_hits,
        min_qualifying_turns=args.min_qualifying_turns,
        max_first_msg_idx=args.max_first_msg_idx,
        max_api_calls=args.max_api_calls,
        top_level_only=args.top_level_only,
    )
    print_report(result, top_n=args.top)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nWrote full results to {args.out}")


if __name__ == "__main__":
    main()
