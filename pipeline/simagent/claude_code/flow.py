"""Resume-per-step workflow runner: one Claude Code conversation per phase, N steps.

A phase is a `Session`; each `step()` is a resumed invocation of the same conversation with
its own prompt, tool policy (`tools=""` = pure reasoning, no tools at all), optional JSON
schema, and turn cap. Between steps the harness runs deterministic devices and gates. The
transcript of the whole conversation is assembled in mini-swe-agent-1.1 message shape.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import runner

READ_ONLY_TOOLS = "Read,Grep,Glob,Bash"


def lean(name: str) -> bool:
    """Token-saving options, each a flag defaulting to the measured behaviour (off).
    CCFLOW_LEAN=1 turns all on; CCFLOW_<NAME>=0/1 overrides one. Names: MERGE_LENSES,
    REASK_MIN, SITES_NOTOOLS, TRIM, TABLE_LOOSE."""
    v = os.getenv(f"CCFLOW_{name}")
    if v is not None:
        return v in ("1", "true", "yes")
    return os.getenv("CCFLOW_LEAN", "1") in ("1", "true", "yes")
FULL_TOOLS = runner.DEFAULT_TOOLS
NO_TOOLS = ""


@dataclass
class Ctx:
    env: object
    repo_path: str
    instance: dict
    inst_out: Path
    model: str
    log: object
    effort: str | None = None
    problem_statement: str = ""
    findings: object = None
    patches: dict = field(default_factory=dict)
    record: dict = field(default_factory=dict)
    cost: float = 0.0
    session_timeout: int = int(os.getenv("CCPIPE_SESSION_TIMEOUT", "5400"))
    query_timeout: int = int(os.getenv("CCPIPE_QUERY_TIMEOUT", "900"))
    baseline: bool = False   # arm B: no code-reasoning / execution-simulation steps


def lang_text(text: str) -> str:
    """Language-adapt a stock prompt body for JS/TS instances only.

    The Go batches ran without this rewrite (they rely on the per-step LANGUAGE NOTE), so the Go
    path stays byte-identical to the completed runs; Python is a no-op inside _maybe_lang anyway.
    """
    from simagent import pipeline as _sp
    return _sp._maybe_lang(text) if _sp._sub.is_js() else text

class Session:
    """One Claude Code conversation driven step by step."""

    def __init__(self, ctx: Ctx, phase: str, system_prompt: str | None = None):
        self.ctx = ctx
        self.phase = phase
        self.system_prompt = system_prompt
        self.session_id: str | None = None
        self.steps: list[dict] = []
        self.messages: list[dict] = []
        self.cost = 0.0
        self.n_calls = 0
        self.n_tool_calls = 0

    def step(self, name: str, prompt: str, *, tools: str, max_turns: int, schema: dict | None = None,
             extra_args: tuple = (), timeout: int | None = None) -> runner.SessionResult:
        k = len(self.steps) + 1
        label = f"{self.phase}_{k:02d}_{name}"
        first = self.session_id is None
        if tools != NO_TOOLS and not getattr(self, "_lang_noted", False):
            # the stock pipeline appends the Go/JS language note to every agentic prompt (go test, never
            # edit existing *_test.go, ...); "" for Python, so Python prompts are byte-identical
            from simagent import pipeline as _sp
            note = _sp._lang_note()
            if note:
                prompt = prompt + note
            self._lang_noted = True
        if timeout is None:
            timeout = self.ctx.query_timeout if tools == NO_TOOLS else self.ctx.session_timeout
        t0 = time.time()
        for attempt in range(int(os.getenv("CCPIPE_LIMIT_RETRIES", "6")) + 1):
            res = runner.run_session(
                self.ctx.env, label=label if attempt == 0 else f"{label}_r{attempt}", prompt=prompt,
                system_prompt=self.system_prompt if first else None, out_dir=self.ctx.inst_out / "cc",
                model=self.ctx.model, tools=tools, max_turns=max_turns, cwd=self.ctx.repo_path,
                timeout=timeout, json_schema=schema, resume=self.session_id, effort=self.ctx.effort,
                extra_args=extra_args, log=self.ctx.log)
            if res.exit_status != "RateLimited":
                break
            runner.wait_for_reset(res.error_text, log=self.ctx.log, label=label)
        if first and res.session_id:
            self.session_id = res.session_id
        msgs = res.messages if first else [m for m in res.messages if m.get("role") != "system"]
        for m in msgs:
            m.setdefault("extra", {})["flow_step"] = label
        self.messages += msgs
        self.cost += res.cost or 0.0
        self.n_calls += res.n_calls or 0
        self.n_tool_calls += res.n_tool_calls or 0
        self.ctx.cost += res.cost or 0.0
        self.steps.append({"step": name, "label": label, "tools": tools, "max_turns": max_turns,
                           "schema": bool(schema), "exit_status": res.exit_status, "turns": res.n_calls,
                           "tool_calls": res.n_tool_calls, "cost": res.cost, "duration": round(time.time() - t0),
                           "answer_chars": len(res.submission or ""), "structured": res.structured_output is not None,
                           "error": (res.error_text or "")[:200]})
        self.ctx.log(f"[{self.phase}:{name}] {res.exit_status} turns={res.n_calls} tools={res.n_tool_calls} "
                     f"cost=${res.cost:.3f} ({time.time()-t0:.0f}s)")
        return res

    def step_structured(self, name: str, prompt: str, *, tools: str, max_turns: int, schema: dict,
                        extra_args: tuple = ()) -> runner.SessionResult:
        """`step()` for schema steps: if the turn cap ended the step before the structured answer
        was produced (a read-only exploration that never stopped), ask once more with NO tools --
        the model must answer from what it already read."""
        res = self.step(name, prompt, tools=tools, max_turns=max_turns, schema=schema, extra_args=extra_args)
        if res.structured_output is None:
            self.ctx.log(f"[{self.phase}:{name}] no structured answer (exit={res.exit_status}) -- "
                         f"asking again with tools off")
            res = self.step(f"{name}_answer",
                            "You have used your tool budget for this step. Using ONLY what you have already read "
                            "and reasoned in this conversation, produce the required structured answer now.",
                            tools=NO_TOOLS, max_turns=2, schema=schema)
        return res

    def qf(self, name: str, *, max_turns: int = 1):
        """A `prompt -> text` callable (tools off, in this conversation) for the stock devices."""
        counter = {"n": 0}

        def _q(prompt: str, **_kw) -> str:
            counter["n"] += 1
            r = self.step(f"{name}{counter['n']}", prompt, tools=NO_TOOLS, max_turns=max_turns)
            return r.submission or ""
        _q.supports_output_limit = False
        return _q

    def save(self, path: Path, **extra) -> None:
        path.write_text(json.dumps({"phase": self.phase, "session_id": self.session_id, "steps": self.steps,
                                    "cost": self.cost, "n_calls": self.n_calls, "n_tool_calls": self.n_tool_calls,
                                    "messages": self.messages, **extra}, indent=1, default=str))


def strip_submit_protocol(text: str) -> str:
    """Remove the bash-only submit protocol lines from a stock prompt."""
    import re
    return re.sub(r"^.*COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.*$",
                  "        (Claude Code runtime: when this step's work is done, stop and write your final message.)",
                  text, flags=re.M)


RUNTIME_NOTE = """

## Runtime note (Claude Code)
You are running as Claude Code; the repository is at {repo_path} (your working directory). Any
instruction above that prescribes a THOUGHT section plus exactly one command per response, or
that ends a phase with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` or a patch file, does not apply:
work with your tools, and when the step's work is done, STOP and write your final message. Keep
scratch files under /tmp, never inside the repository. Do not commit, do not run `git checkout`,
`reset`, `stash` or `restore` on repository files. Do not ask questions; nobody will answer.
"""
