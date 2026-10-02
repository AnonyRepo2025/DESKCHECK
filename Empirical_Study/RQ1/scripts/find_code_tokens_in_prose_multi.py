from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import find_code_tokens_in_prose as base
from find_execution_reasoning import (
    _provider_uses_content,
    extract_thought,
    load_resolutions,
)


# --------------------------------------------------------------------------
# OpenHands
# --------------------------------------------------------------------------
def _text_blocks(blocks) -> list[dict]:
    out = []
    for b in blocks or []:
        if isinstance(b, dict) and isinstance(b.get("text"), str) and b["text"].strip():
            out.append({"type": "text", "text": b["text"]})
    return out


def openhands_event_to_message(ev: dict) -> dict | None:
    """Map one OpenHands ActionEvent / agent MessageEvent onto a chat message
    dict shaped so `find_execution_reasoning.extract_thought` reads it."""
    kind = ev.get("kind")
    if ev.get("source") != "agent":
        return None
    if kind == "MessageEvent":
        lm = ev.get("llm_message") or {}
        if lm.get("role") != "assistant":
            return None
        msg: dict = {"role": "assistant", "content": _text_blocks(lm.get("content"))}
        tb = lm.get("thinking_blocks") or []
        if tb:
            msg["thinking_blocks"] = tb
        elif isinstance(lm.get("reasoning_content"), str):
            msg["reasoning"] = lm["reasoning_content"]
        return msg
    if kind != "ActionEvent":
        return None
    msg = {"role": "assistant", "content": _text_blocks(ev.get("thought"))}
    tb = [b for b in (ev.get("thinking_blocks") or []) if isinstance(b, dict)]
    if tb:
        msg["thinking_blocks"] = tb
    elif isinstance(ev.get("reasoning_content"), str) and ev["reasoning_content"].strip():
        # Same text OpenHands would also put in thinking_blocks; keep one copy.
        msg["reasoning"] = ev["reasoning_content"]
    rri = ev.get("responses_reasoning_item")
    if isinstance(rri, dict) and rri.get("summary"):
        msg["output"] = [{"type": "reasoning", "summary": rri["summary"]}]
    return msg


def iter_openhands_instances(jsonl: Path):
    """Yield (instance_id, source_ref, messages) per line of output.jsonl."""
    with open(jsonl) as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = rec.get("instance_id")
            if not iid:
                continue
            msgs = []
            for ev in rec.get("history") or []:
                if isinstance(ev, dict):
                    m = openhands_event_to_message(ev)
                    if m is not None:
                        msgs.append(m)
            yield iid, f"{jsonl}#L{ln}", msgs


# --------------------------------------------------------------------------
# Sonar Foundation Agent
# --------------------------------------------------------------------------
def sonar_message(m: dict) -> dict | None:
    if m.get("role") != "assistant":
        return None
    content = []
    for b in m.get("blocks") or []:
        if not isinstance(b, dict):
            continue
        bt = b.get("block_type") or b.get("type")
        if bt == "text" and isinstance(b.get("text"), str) and b["text"].strip():
            content.append({"type": "text", "text": b["text"]})
        elif bt == "thinking":
            t = b.get("content") if isinstance(b.get("content"), str) else b.get("thinking")
            if isinstance(t, str) and t.strip():
                content.append({"type": "thinking", "thinking": t})
    return {"role": "assistant", "content": content}


def iter_sonar_instances(trajs_dir: Path):
    for f in sorted(trajs_dir.glob("*.json")):
        try:
            data = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, list):
            continue
        msgs = [sonar_message(m) for m in data if isinstance(m, dict)]
        yield f.stem, str(f), [m for m in msgs if m is not None]



# --------------------------------------------------------------------------
# Claude Code (SWE-bench Pro, GPT-5.4)
# --------------------------------------------------------------------------
CLAUDECODE_ROLE_TO_PHASE = {"localizer": "localization", "patcher": "patch",
                            "validator": "validation"}


def claudecode_traj_file(inst_dir: Path) -> tuple[str, Path] | None:
    """(layout, path) for one `trajs/<iid>/` dir; None when it holds no trajectory."""
    iid = inst_dir.name
    if (inst_dir / f"{iid}.traj.json").is_file():
        return "traj.json", inst_dir / f"{iid}.traj.json"
    for name in ("trajectory.jsonl", f"{iid}.traj.jsonl"):
        if (inst_dir / name).is_file():
            return "jsonl", inst_dir / name
    return None


def claudecode_step_message(step: dict) -> dict:
    """Merged-run step -> chat message: `thinking` as a thinking block,
    `response` as a text block (either may be empty)."""
    content = []
    th = step.get("thinking")
    if isinstance(th, str) and th.strip():
        content.append({"type": "thinking", "thinking": th})
    resp = step.get("response")
    if isinstance(resp, str) and resp.strip():
        content.append({"type": "text", "text": resp})
    return {"role": "assistant", "content": content}


def claudecode_jsonl_message(row: dict) -> dict | None:
    """Raw Claude Code event row -> chat message (assistant rows only). The
    session log stores content blocks without a `type` key, so classify by
    which key is present."""
    if row.get("type") != "assistant":
        return None
    content = []
    for b in row.get("content") or []:
        if not isinstance(b, dict):
            continue
        if isinstance(b.get("thinking"), str) and b["thinking"].strip():
            content.append({"type": "thinking", "thinking": b["thinking"]})
        elif isinstance(b.get("text"), str) and b["text"].strip():
            content.append({"type": "text", "text": b["text"]})
    return {"role": "assistant", "content": content}


def load_claudecode_instance(inst_dir: Path):
    """(layout, path, messages, phases) for one instance, or None.

    `messages[i]` is the chat message for message_index i (empty-content
    placeholders keep indices aligned with the raw file); `phases[i]` is the
    pipeline phase of that turn (`localization|patch|validation` from the
    merged run's `role`, or `single_session` for raw session logs)."""
    found = claudecode_traj_file(inst_dir)
    if found is None:
        return None
    layout, path = found
    msgs: list[dict] = []
    phases: list[str] = []
    try:
        if layout == "traj.json":
            data = json.load(open(path))
            for step in data.get("trajectory") or []:
                if not isinstance(step, dict):
                    step = {}
                msgs.append(claudecode_step_message(step))
                role = step.get("role") or ""
                phases.append(CLAUDECODE_ROLE_TO_PHASE.get(role, role or "unknown"))
        else:
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    try:
                        row = json.loads(line) if line else {}
                    except json.JSONDecodeError:
                        row = {}
                    m = claudecode_jsonl_message(row) if isinstance(row, dict) else None
                    msgs.append(m if m is not None else {"role": "user", "content": []})
                    phases.append("single_session")
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return layout, path, msgs, phases


def iter_claudecode_instances(trajs_dir: Path):
    """Yield (instance_id, source_ref, messages) per `trajs/<iid>/` dir."""
    root = trajs_dir / "trajs" if (trajs_dir / "trajs").is_dir() else trajs_dir
    for inst_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        loaded = load_claudecode_instance(inst_dir)
        if loaded is None:
            continue
        _layout, path, msgs, _phases = loaded
        yield inst_dir.name, str(path), msgs


# --------------------------------------------------------------------------
# ExpeRepair timelines (SWE-bench Lite, Claude 4 Sonnet)
# --------------------------------------------------------------------------
EXPEREPAIR_PHASE_TO_PHASE = {"localization": "localization", "generation": "patch",
                             "validation": "validation",
                             "reproduction": "reproduction"}


# ExpeRepair's reproduction loop asks the model for a `<test_analysis>` block
# (its reading of the reproduction script's stdout/stderr); the same text is
# re-serialised as a `'test-analysis': '...'` entry inside check_repro records
# and episodic-memory lines. `--experepair-strip-test-analysis` drops both.
_TEST_ANALYSIS_XML_RE = re.compile(r"<test[_-]analysis>.*?</test[_-]analysis>", re.S | re.I)
_TEST_ANALYSIS_KEY_RE = re.compile(
    r"""(['"])test[_-]analysis\1\s*:\s*(?:'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")""", re.S | re.I)


def strip_test_analysis(text: str) -> str:
    text = _TEST_ANALYSIS_XML_RE.sub(" ", text)
    return _TEST_ANALYSIS_KEY_RE.sub(" ", text)


def load_experepair_instance(path: Path, strip_test_analysis_blocks: bool = False):
    """(instance_id, messages, phases) for one `<iid>.timeline.json`, or None.

    One assistant message per non-empty `reasoning` text, enumerated in
    timeline order (steps -> files -> reasoning); `phases[i]` is the logical
    phase of message i (generation is mapped to `patch`). With
    `strip_test_analysis_blocks`, test-analysis text is removed from each turn
    before it is scanned (the turn keeps its index; it may become empty)."""
    try:
        data = json.load(open(path))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    iid = data.get("instance_id") or path.name.replace(".timeline.json", "")
    msgs: list[dict] = []
    phases: list[str] = []
    for step in data.get("steps") or []:
        raw_phase = step.get("phase") or "unknown"
        phase = EXPEREPAIR_PHASE_TO_PHASE.get(raw_phase, raw_phase)
        for f in step.get("files") or []:
            for text in f.get("reasoning") or []:
                if not (isinstance(text, str) and text.strip()):
                    continue
                if strip_test_analysis_blocks:
                    text = strip_test_analysis(text)
                msgs.append({"role": "assistant",
                             "content": [{"type": "text", "text": text}]
                             if text.strip() else []})
                phases.append(phase)
    return iid, msgs, phases


def iter_experepair_instances(trajs_dir: Path, strip_test_analysis_blocks: bool = False):
    for f in sorted(trajs_dir.glob("*.timeline.json")):
        loaded = load_experepair_instance(f, strip_test_analysis_blocks)
        if loaded is None:
            continue
        iid, msgs, _phases = loaded
        yield iid, str(f), msgs


# --------------------------------------------------------------------------
def messages_to_thoughts(msgs: list[dict], top_level_only: bool) -> list[tuple[int, str]]:
    uses_content = _provider_uses_content(msgs) if top_level_only else True
    out = []
    for i, m in enumerate(msgs):
        t = extract_thought(m, top_level_only=top_level_only, provider_uses_content=uses_content)
        if t:
            out.append((i, t))
    return out


def parse_thoughts(fmt: str, path: Path, top_level_only: bool,
                  strip_test_analysis_blocks: bool = False) -> dict[str, dict]:
    if fmt == "openhands":
        if path.is_dir():
            src = path / "output.jsonl" if (path / "output.jsonl").exists() else path / "run" / "output.jsonl"
        else:
            src = path
        it = iter_openhands_instances(src)
    elif fmt == "sonar":
        src = path / "trajs" if (path / "trajs").is_dir() else path
        it = iter_sonar_instances(src)
    elif fmt == "claudecode":
        it = iter_claudecode_instances(path)
    elif fmt == "experepair":
        it = iter_experepair_instances(path, strip_test_analysis_blocks)
    else:
        raise SystemExit(f"unknown --format {fmt}")
    parsed: dict[str, dict] = {}
    for iid, ref, msgs in it:
        parsed[iid] = {"file": ref, "thoughts": messages_to_thoughts(msgs, top_level_only)}
    return parsed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", required=True, choices=["openhands", "sonar", "claudecode", "experepair"])
    ap.add_argument("--trajs-dir", type=Path, required=True,
                    help="openhands: run dir (or output.jsonl); sonar: dir holding trajs/ (or trajs/ itself); "
                         "claudecode: dir holding trajs/<iid>/ (or trajs/ itself); "
                         "experepair: dir of <iid>.timeline.json files.")
    ap.add_argument("--experepair-strip-test-analysis", action="store_true",
                    help="experepair only: drop <test_analysis> blocks and 'test-analysis': "
                         "values from every turn before scanning.")
    ap.add_argument("--resolutions", type=Path, default=None,
                    help="JSON {instance_id: {resolved: bool}} (see prepare_empirical_inputs.py).")
    ap.add_argument("--difficulty-dataset", default=base.DEFAULT_DIFFICULTY_DATASET)
    ap.add_argument("--difficulty-json", type=Path, default=None)
    ap.add_argument("--no-difficulty", action="store_true")
    ap.add_argument("--min-score", type=int, default=base.DEFAULT_MIN_SCORE)
    ap.add_argument("--prose-hits", type=int, default=base.DEFAULT_PROSE_HITS)
    ap.add_argument("--min-qualifying-turns", type=int, default=base.DEFAULT_MIN_QUALIFYING_TURNS)
    ap.add_argument("--max-first-msg-idx", type=int, default=None)
    ap.add_argument("--max-api-calls", type=int, default=None)
    ap.add_argument("--top-level-only", action=argparse.BooleanOptionalAction, default=True,
                    help="(default) visible response text only; --no-top-level-only also scans thinking/CoT surfaces.")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-reasoning-snippets", type=int, default=base.DEFAULT_MIN_REASONING_SNIPPETS)
    ap.add_argument("--allow-observation-only", action="store_true")
    ap.add_argument("--no-state-scan", action="store_true")
    ap.add_argument("--include-mutation-verb-only", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.trajs_dir.exists():
        raise SystemExit(f"trajs-dir does not exist: {args.trajs_dir}")
    resolutions = load_resolutions(args.resolutions) if args.resolutions else {}
    if resolutions:
        print(f"Loaded resolution data for {len(resolutions)} instance(s) from {args.resolutions}\n")
    if args.no_difficulty:
        difficulties: dict[str, str] = {}
    else:
        dd = args.difficulty_dataset
        if dd is not None and str(dd).lower() in ("", "none"):
            dd = None
        difficulties = base.load_difficulties(dataset=dd, json_path=args.difficulty_json)
        if difficulties:
            print(f"Loaded difficulty labels for {len(difficulties)} instance(s) from "
                  f"{args.difficulty_json or dd}\n")

    parsed = parse_thoughts(args.format, args.trajs_dir, args.top_level_only,
                            strip_test_analysis_blocks=args.experepair_strip_test_analysis)
    print(f"Parsed {len(parsed)} {args.format} trajectories from {args.trajs_dir} "
          f"(top_level_only={args.top_level_only})\n")
    result = base.analyze_trajectories(
        args.trajs_dir,
        resolutions=resolutions,
        min_score=args.min_score,
        prose_hits_required=args.prose_hits,
        min_qualifying_turns=args.min_qualifying_turns,
        max_first_msg_idx=args.max_first_msg_idx,
        max_api_calls=args.max_api_calls,
        parsed=parsed,
        top_level_only=args.top_level_only,
        run_state_scan=not args.no_state_scan,
        include_mutation_verb_only=args.include_mutation_verb_only,
        difficulties=difficulties,
        min_reasoning_snippets=args.min_reasoning_snippets,
        require_core_reasoning=not args.allow_observation_only,
    )
    result["trajectory_format"] = args.format
    result["experepair_strip_test_analysis"] = bool(args.experepair_strip_test_analysis)
    base.print_report(result, top_n=args.top)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nWrote combined results to {args.out}")
        print(f"  - total instances:            {len(result['instances'])}")
        print(f"  - code-reasoning instances:   {result.get('n_instances_with_code_reasoning', 0)}")


if __name__ == "__main__":
    main()
