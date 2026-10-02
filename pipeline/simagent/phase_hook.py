from __future__ import annotations

import re
from typing import Optional

# Phase tagging uses the vendored `plan_monitor` (simagent/plan_monitor).
from simagent.plan_monitor.monitor import StatefulPhaseMonitor  # noqa: E402
from simagent.plan_monitor.phases import ActionEvent  # noqa: E402

# --- Regexes for extracting command / thought / observation from messages ----
_THOUGHT_RE = re.compile(r"THOUGHT:\s*(.*?)(?=```|\Z)", re.DOTALL)
_BASH_RE = re.compile(r"```(?:bash|mswea_bash_command)?\s*\n(.*?)\n```", re.DOTALL)
_OUTPUT_RE = re.compile(r"<output>\s*(.*?)\s*</output>", re.DOTALL)


def extract_command(assistant_message: dict) -> str:

    extra = assistant_message.get("extra") or {}
    if "actions" in extra:
        actions = extra.get("actions") or []
        return "\n".join(a["command"] for a in actions if isinstance(a, dict) and a.get("command"))
    return "\n".join(m.strip() for m in _BASH_RE.findall(assistant_message.get("content") or ""))


def extract_thought(assistant_message: dict) -> str:
    content = assistant_message.get("content") or ""
    match = _THOUGHT_RE.search(content)
    return match.group(1).strip() if match else content.strip()


def extract_observation(observation_messages: list[dict]) -> str:
    parts = []
    for msg in observation_messages:
        content = msg.get("content", "") or ""
        match = _OUTPUT_RE.search(content)
        parts.append(match.group(1).strip() if match else content.strip())
    return "\n".join(p for p in parts if p)


class StepPhaseTagger:
    """Drives a ``StatefulPhaseMonitor`` step-by-step and records the phases."""

    def __init__(self, enable_rules: bool = False, rules_config: Optional[str] = None):
        # `rules_config` only reaches the monitor when it is actually set: the
        # the vendored StatefulPhaseMonitor takes (parser, enable_rules)
        # and rejects the kwarg outright, and no caller in this tree passes one.
        kwargs = {"enable_rules": enable_rules}
        if rules_config is not None:
            kwargs["rules_config"] = rules_config
        self.monitor = StatefulPhaseMonitor(**kwargs)
        self.records: list[dict] = []

    def tag(self, step_index: int, command: str, thought: str = "", observation: str = "") -> dict:
        """Classify one step; returns and stores the per-step record.

        ``phases`` holds the fine-grained role(s) emitted by this step (a
        compound command may contribute more than one), and ``current_phase`` is
        the monitor's phase after the step.
        """
        history_before = len(self.monitor.get_phase_history())
        if command.strip():
            self.monitor.on_step(
                ActionEvent(step_index=step_index, command=command),
                thought=thought,
                observation=observation,
            )
        phases = self.monitor.get_phase_history()[history_before:]
        current = self.monitor.get_current_phase()
        record = {
            "step": step_index,
            "command": command,
            "phases": phases,
            "current_phase": str(current) if current is not None else None,
        }
        self.records.append(record)
        return record


