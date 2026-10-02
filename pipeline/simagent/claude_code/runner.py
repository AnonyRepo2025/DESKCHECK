"""Run one Claude Code headless session inside the task container and collect it.

A session = one `claude -p` process executed via `docker exec`, streaming stream-json to a
host file. The prompt goes in over stdin (no arg-length limits), the system prompt via a
file. The result is converted to a mini-swe-agent-shaped phase dict so the rest of the
pipeline (and the metrics tooling) sees the same structure as a DefaultAgent phase.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import traj
from .env import WORK_DIR_IN

DEFAULT_TOOLS = "Read,Edit,Write,Bash,Grep,Glob"
RATE_LIMIT_RE = re.compile(r"rate.?limit|usage limit|session limit|limit reached|hit your .*limit|too many requests"
                           r"|\b429\b|overloaded|resets \d", re.I)
_RESET_RE = re.compile(r"resets?\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*\(?(UTC|GMT)?\)?", re.I)


def seconds_until_reset(error_text: str, default: int = 1800, cap: int = 6 * 3600) -> int:
    """Parse "resets 5:20am (UTC)" out of a limit message into a wait in seconds (+60s slack)."""
    import datetime as _dt
    m = _RESET_RE.search(error_text or "")
    if not m:
        return default
    hh = int(m.group(1)) % 12 if m.group(3) else int(m.group(1))
    if m.group(3) and m.group(3).lower() == "pm":
        hh += 12
    mm = int(m.group(2) or 0)
    now = _dt.datetime.now(_dt.timezone.utc)
    target = now.replace(hour=hh % 24, minute=mm, second=0, microsecond=0)
    if target <= now:
        target += _dt.timedelta(days=1)
    return int(min(cap, max(60, (target - now).total_seconds() + 60)))


def wait_for_reset(error_text: str, log=print, label: str = "") -> None:
    secs = seconds_until_reset(error_text)
    log(f"[cc:{label}] subscription limit: {error_text[:100]!r} -- sleeping {secs//60} min until reset")
    while secs > 0:
        step = min(secs, 600)
        time.sleep(step)
        secs -= step
        if secs > 0:
            log(f"[cc:{label}] still waiting for limit reset ({secs//60} min left)")


@dataclass
class SessionResult:
    phase: str
    exit_status: str
    submission: str
    n_calls: int
    cost: float
    messages: list = field(default_factory=list)
    session_id: str | None = None
    usage: dict = field(default_factory=dict)
    structured_output: object = None
    is_error: bool = False
    error_text: str = ""
    duration: float = 0.0
    returncode: int | None = None
    stream_path: str = ""
    n_tool_calls: int = 0
    model: str | None = None

    def as_phase(self) -> dict:
        """The dict shape `_run_phase_agent` returns (plus ccpipe extras)."""
        return {"phase": self.phase, "exit_status": self.exit_status, "submission": self.submission,
                "n_calls": self.n_calls, "cost": self.cost, "messages": self.messages, "agent": None,
                "hook": None, "session_id": self.session_id, "usage": self.usage,
                "structured_output": self.structured_output, "is_error": self.is_error,
                "error_text": self.error_text, "duration": self.duration,
                "n_tool_calls": self.n_tool_calls, "model": self.model}


def _docker_exec_input(env, cmd: str, data: str, timeout: int = 120) -> None:
    p = subprocess.run(["docker", "exec", "-i", env.container_id, "bash", "-c", cmd],
                       input=data.encode("utf-8"), capture_output=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"docker exec write failed: {p.stderr.decode(errors='replace')[:300]}")


def put_file(env, path_in: str, data: str) -> None:
    _docker_exec_input(env, f"mkdir -p {shlex.quote(os.path.dirname(path_in))} && cat > {shlex.quote(path_in)}", data)


def build_cmd(*, label: str, model: str, tools: str, max_turns: int, system_file: str | None,
              prompt_file: str, json_schema: dict | None, resume: str | None,
              append_system_prompt: str | None, effort: str | None, extra_args: tuple[str, ...],
              cwd: str) -> str:
    parts = ["claude", "-p", "--output-format", "stream-json", "--verbose",
             "--model", shlex.quote(model), "--tools", shlex.quote(tools),
             "--permission-mode", "bypassPermissions", "--setting-sources", '""',
             "--disable-slash-commands", "--strict-mcp-config", "--max-turns", str(max_turns)]
    if system_file:
        parts += ["--system-prompt", f'"$(cat {shlex.quote(system_file)})"']
    if append_system_prompt is not None:
        parts += ["--append-system-prompt", shlex.quote(append_system_prompt)]
    if json_schema is not None:
        parts += ["--json-schema", shlex.quote(json.dumps(json_schema))]
    if resume:
        parts += ["--resume", shlex.quote(resume)]
    if effort:
        parts += ["--effort", shlex.quote(effort)]
    parts += list(extra_args)
    return f"cd {shlex.quote(cwd)} && {' '.join(parts)} < {shlex.quote(prompt_file)}"


def run_session(env, *, label: str, prompt: str, system_prompt: str | None, out_dir: Path,
                model: str = "sonnet", tools: str = DEFAULT_TOOLS, max_turns: int = 200,
                cwd: str = "/app", timeout: int = 3600, json_schema: dict | None = None,
                resume: str | None = None, append_system_prompt: str | None = None,
                effort: str | None = None, extra_args: tuple[str, ...] = (),
                log=print) -> SessionResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = f"{WORK_DIR_IN}/{label}.prompt"
    put_file(env, prompt_file, prompt)
    system_file = None
    if system_prompt is not None:
        system_file = f"{WORK_DIR_IN}/{label}.system"
        put_file(env, system_file, system_prompt)
    cmd = build_cmd(label=label, model=model, tools=tools, max_turns=max_turns, system_file=system_file,
                    prompt_file=prompt_file, json_schema=json_schema, resume=resume,
                    append_system_prompt=append_system_prompt, effort=effort, extra_args=extra_args,
                    cwd=cwd)
    (out_dir / f"{label}.cmd").write_text(cmd)
    stream_path = out_dir / f"{label}.stream.jsonl"
    err_path = out_dir / f"{label}.stderr"
    log(f"[cc:{label}] starting (model={model} tools={tools} max_turns={max_turns}"
        f"{' resume=' + resume[:8] if resume else ''})")
    t0 = time.time()
    rc: int | None = None
    # Claude Code's raw stream-json goes to a scratch capture outside the run dir; the persisted
    # <label>.stream.jsonl gets the stream record built from it (traj.append_stream_record).
    fd, raw_path = tempfile.mkstemp(prefix=f"ccpipe_{label}_", suffix=".stream.jsonl")
    os.close(fd)
    try:
        with open(raw_path, "wb") as so, open(err_path, "ab") as se:
            p = subprocess.Popen(["docker", "exec", "-w", cwd, env.container_id, "bash", "-c", cmd],
                                 stdout=so, stderr=se, stdin=subprocess.DEVNULL)
            try:
                rc = p.wait(timeout=timeout)
            except BaseException as e:   # SIGALRM wall deadlines (plainrepair) / KeyboardInterrupt: never leak the session
                if not isinstance(e, subprocess.TimeoutExpired):
                    subprocess.run(["docker", "exec", env.container_id, "bash", "-c", "pkill -f 'claude -p' || true"],
                                   capture_output=True, timeout=60)
                    p.kill()
                    raise
                log(f"[cc:{label}] WALL TIMEOUT after {timeout}s -- killing")
                subprocess.run(["docker", "exec", env.container_id, "bash", "-c", "pkill -f 'claude -p' || true"],
                               capture_output=True, timeout=60)
                try:
                    rc = p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
                    rc = -9
    finally:   # also on interrupt: whatever the session streamed is recorded
        traj.append_stream_record(raw_path, stream_path)
        os.unlink(raw_path)
    dur = time.time() - t0
    events = traj.read_stream(stream_path)
    msgs, summ = traj.convert(events, system_prompt=system_prompt or "", user_prompt=prompt)
    stderr_text = err_path.read_text(errors="replace")[-4000:] if err_path.exists() else ""
    err_text = (summ["result"] if summ["is_error"] else "") or ("" if rc == 0 else stderr_text)
    if (summ.get("terminal_reason") or "").endswith("max_turns") or summ.get("stop_reason") in ("max_turns", "error_max_turns"):
        status = "LimitsExceeded"   # Claude Code reports the turn cap as is_error/subtype=error_max_turns
    elif summ["is_error"] or (rc not in (0, None) and not events):
        status = "RateLimited" if RATE_LIMIT_RE.search(err_text or "") else "Error"
    elif rc == -9 or (rc is None):
        status = "WallTimeout"
    elif summ["num_turns"] >= max_turns and summ.get("stop_reason") == "tool_use" and not summ.get("structured_output"):
        status = "LimitsExceeded"   # cut off mid tool-use
    elif not events:
        status = "Error"
    else:
        status = "Completed"
    if status == "LimitsExceeded":
        summ["is_error"] = False
        err_text = ""
    res = SessionResult(phase=label, exit_status=status, submission=summ["result"],
                        n_calls=summ["n_assistant_messages"] or summ["num_turns"], cost=summ["cost"],
                        messages=msgs, session_id=summ["session_id"], usage=summ["usage"],
                        structured_output=summ["structured_output"], is_error=summ["is_error"],
                        error_text=(err_text or "")[:2000], duration=dur, returncode=rc,
                        stream_path=str(stream_path), n_tool_calls=summ["n_tool_calls"],
                        model=summ.get("model"))
    (out_dir / f"{label}.result.json").write_text(json.dumps(
        {k: v for k, v in res.__dict__.items() if k != "messages"}, indent=1, default=str))
    log(f"[cc:{label}] done exit={status} turns={res.n_calls} tools={res.n_tool_calls} "
        f"cost=${res.cost:.4f} ({dur:.0f}s)" + (f" err={res.error_text[:160]!r}" if res.error_text else ""))
    return res


def ask(env, *, label: str, prompt: str, system_prompt: str | None, out_dir: Path, model: str = "sonnet",
        json_schema: dict | None = None, timeout: int = 900, log=print, effort: str | None = None) -> SessionResult:
    """Tool-free single query (the `_ask`/`qf` surface of the old pipeline)."""
    return run_session(env, label=label, prompt=prompt, system_prompt=system_prompt, out_dir=out_dir,
                       model=model, tools="", max_turns=1, timeout=timeout, json_schema=json_schema,
                       log=log, effort=effort)
