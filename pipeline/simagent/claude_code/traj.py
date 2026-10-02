"""stream-json (Claude Code headless) -> mini-swe-agent-1.1 message list.

Every Claude Code tool call becomes an assistant message with a `bash`-shaped tool_call
whose command is the Bash command itself or a shell-equivalent rendering of the file tool
(Read -> sed -n, Edit/Write -> a `cc-edit`/`cc-write` pseudo-command). The raw tool_use block
is preserved under extra.cc_tool_use so nothing is lost; downstream metrics that only look at
role/content/extra.actions keep working.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Iterable

from simagent.usage import anthropic_usage, cc_event_record  # noqa: E402


def _as_command(name: str, inp: dict) -> str:
    """Shell-equivalent rendering of a tool call. Never raises: a malformed tool input (seen:
    Read offset "140, 200") must not take the phase down, so anything odd falls back to the
    generic cc-tool rendering."""
    try:
        return _as_command_strict(name, inp)
    except Exception:
        return f"cc-tool {name} {shlex.quote(json.dumps(inp, sort_keys=True, default=str))}"


def _as_command_strict(name: str, inp: dict) -> str:
    if name == "Bash":
        return inp.get("command", "")
    if name == "Read":
        fp = shlex.quote(str(inp.get("file_path", "")))
        off, lim = inp.get("offset"), inp.get("limit")
        if off or lim:
            a = int(off or 1)
            b = a + int(lim or 2000) - 1
            return f"sed -n '{a},{b}p' {fp}"
        return f"cat {fp}"
    if name == "Edit":
        return (f"cc-edit {shlex.quote(str(inp.get('file_path', '')))} "
                f"--old {shlex.quote(str(inp.get('old_string', '')))} "
                f"--new {shlex.quote(str(inp.get('new_string', '')))}"
                + (" --all" if inp.get("replace_all") else ""))
    if name == "Write":
        return f"cc-write {shlex.quote(str(inp.get('file_path', '')))} <<'CC_EOF'\n{inp.get('content', '')}\nCC_EOF"
    if name == "Grep":
        parts = ["grep", "-rn"]
        if inp.get("-i"):
            parts.append("-i")
        if inp.get("glob"):
            parts.append(f"--include={shlex.quote(str(inp['glob']))}")
        parts += [shlex.quote(str(inp.get("pattern", ""))), shlex.quote(str(inp.get("path", ".")))]
        return " ".join(parts)
    if name == "Glob":
        return f"find {shlex.quote(str(inp.get('path', '.')))} -path {shlex.quote(str(inp.get('pattern', '*')))}"
    return f"cc-tool {name} {shlex.quote(json.dumps(inp, sort_keys=True))}"


def _tool_result_text(block: dict) -> str:
    c = block.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
    return "" if c is None else str(c)


def convert(events: Iterable[dict], *, system_prompt: str, user_prompt: str) -> tuple[list[dict], dict]:
    """Return (messages, summary). summary has session_id, num_turns, cost, usage, result,
    is_error, structured_output, n_assistant_messages, n_tool_calls."""
    msgs: list[dict] = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}]
    summary: dict = {"session_id": None, "num_turns": 0, "cost": 0.0, "usage": {}, "result": "",
                     "is_error": False, "structured_output": None, "stop_reason": None,
                     "n_assistant_messages": 0, "n_tool_calls": 0, "model": None,
                     "permission_denials": []}
    cur: dict | None = None      # assistant message being assembled (grouped by API message id)
    cur_id = None
    tool_names: dict[str, str] = {}

    def flush():
        nonlocal cur, cur_id
        if cur is not None:
            msgs.append(cur)
            summary["n_assistant_messages"] += 1
        cur, cur_id = None, None

    for ev in events:
        t = ev.get("type")
        if t == "system" and ev.get("subtype") == "init":
            summary["session_id"] = ev.get("session_id")
            summary["model"] = ev.get("model")
        elif t == "assistant":
            m = ev.get("message") or {}
            mid = m.get("id") or ev.get("uuid")
            if ev.get("parent_tool_use_id"):
                continue  # subagent-forwarded text: not part of the main transcript
            if cur is None or mid != cur_id:
                flush()
                cur_id = mid
                cur = {"role": "assistant", "content": "", "tool_calls": [],
                       "extra": {"actions": [], "cc_tool_use": [], "thinking": "",
                                 "usage": anthropic_usage(m["usage"]) if m.get("usage") else None,
                                 "model": m.get("model"),
                                 "message_id": mid}}
            content = m.get("content")
            blocks = content if isinstance(content, list) else [{"type": "text", "text": str(content or "")}]
            for b in blocks:
                bt = b.get("type")
                if bt == "text":
                    cur["content"] += b.get("text", "")
                elif bt == "thinking":
                    cur["extra"]["thinking"] += b.get("thinking", "") or ""
                elif bt == "tool_use":
                    name, inp = b.get("name", ""), b.get("input") or {}
                    tool_names[b.get("id", "")] = name
                    cmd = _as_command(name, inp)
                    cur["tool_calls"].append({
                        "id": b.get("id"), "type": "function",
                        "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}})
                    cur["extra"]["actions"].append({"command": cmd, "tool": name})
                    cur["extra"]["cc_tool_use"].append({"id": b.get("id"), "name": name, "input": inp})
                    summary["n_tool_calls"] += 1
            if m.get("usage"):
                cur["extra"]["usage"] = anthropic_usage(m["usage"])
            if m.get("stop_reason"):
                cur["extra"]["stop_reason"] = m["stop_reason"]
        elif t == "user":
            m = ev.get("message") or {}
            content = m.get("content")
            blocks = content if isinstance(content, list) else [{"type": "text", "text": str(content or "")}]
            had_result = False
            for b in blocks:
                if b.get("type") == "tool_result":
                    flush()
                    had_result = True
                    msgs.append({"role": "tool", "content": _tool_result_text(b),
                                 "tool_call_id": b.get("tool_use_id"),
                                 "extra": {"tool": tool_names.get(b.get("tool_use_id", ""), ""),
                                           "is_error": bool(b.get("is_error"))}})
            if not had_result:
                text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                if text.strip() and text.strip() != user_prompt.strip():
                    flush()
                    msgs.append({"role": "user", "content": text, "extra": {"injected": True}})
        elif t == "result":
            flush()
            summary["num_turns"] = ev.get("num_turns", 0)
            summary["cost"] = ev.get("total_cost_usd", 0.0) or 0.0
            # the step's complete token/cost aggregate (built from modelUsage)
            summary["usage"] = cc_event_record(ev)["usage"]
            summary["result"] = ev.get("result", "") or ""
            summary["is_error"] = bool(ev.get("is_error"))
            summary["structured_output"] = ev.get("structured_output")
            summary["stop_reason"] = ev.get("stop_reason") or ev.get("subtype")
            summary["terminal_reason"] = ev.get("terminal_reason")
            summary["permission_denials"] = ev.get("permission_denials") or []
            if not summary["session_id"]:
                summary["session_id"] = ev.get("session_id")
    flush()
    return msgs, summary


def append_stream_record(raw_path, record_path) -> None:
    """Append the stream record of one session to ``record_path``: each raw stream-json event
    becomes ``usage_record.cc_event_record(ev)`` -- the fields ccpipe reads, with usage built
    from token counts. Raw lines that are not JSON are carried over as-is."""
    raw_path, record_path = Path(raw_path), Path(record_path)
    if not raw_path.exists():
        return
    lines = []
    with open(raw_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.dumps(cc_event_record(json.loads(line)), ensure_ascii=False))
            except json.JSONDecodeError:
                lines.append(line)
    with open(record_path, "a", encoding="utf-8") as out:
        for line in lines:
            out.write(line + "\n")


def read_stream(path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
