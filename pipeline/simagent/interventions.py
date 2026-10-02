
from __future__ import annotations

import copy
import json
import re
import types
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from simagent.phase_hook import (
    StepPhaseTagger,
    extract_command,
    extract_observation,
    extract_thought,
)

# A compliance checker takes the step's reasoning text, the bash command it ran and the
# observation, and returns ``(complied, feedback)``. ``feedback`` explains what was
# missing and is surfaced to the agent on the next retry.
ComplianceChecker = Callable[[str, str, str], "tuple[bool, str]"]

DEFAULT_MAX_RETRIES = 3

DEFAULT_RETRY_PREFIX = """\
[REASONING INTERVENTION -- retry {attempt}/{max_retries}]
Your previous response did NOT do what this step requires. Specifically, it still needs to:
{feedback}.
This step is NOT for more exploration. In your very next response you must produce the
required reasoning AND then the single command described below.

{base}
"""


_INPUT_TERMS = re.compile(
    r"\b(input|argument|arg|call(?:ing|s)?|when\s+\w+|returns?|output|value|"
    r"for\s+example|e\.g\.|case|=)",
    re.IGNORECASE,
)
_SIM_TERMS = re.compile(
    r"\b(simulat|trace|step\s+through|step[-\s]by[-\s]step|execut|evaluat|"
    r"goes?\s+through|walk\s+through|line\s+\d+|the\s+code\s+(?:then|will|does))",
    re.IGNORECASE,
)
_MIN_REASONING_CHARS = 120

_UNCHANGED_TERMS = re.compile(  # patch: reasons that other/unchanged inputs are not broken
    r"\b(unchanged|still\s+(work|pass|return|hold)|does\s+not\s+(break|affect|change|regress)|"
    r"other\s+(case|input|value)|existing\s+(test|behaviou?r|case)|regress|backward|"
    r"preserv|not\s+affected|leaves?\s+\w+\s+intact|no\s+side\s+effect)",
    re.IGNORECASE,
)
_ORACLE_TERMS = re.compile(  # newtest: tests generalize from the issue / fail-old-pass-new
    r"\b(issue|report|description|fails?\s+(on|without|before|with)\s+the\s+(old|unpatched|original|buggy)|"
    r"pass(es)?\s+(on|with|after)\s+the\s+(fix|patch|change)|generaliz|not\s+just|beyond\s+the|"
    r"all\s+(the\s+)?(case|behaviou?r|example)|every\s+(case|behaviou?r|example))",
    re.IGNORECASE,
)


def _check_criteria(text: str, criteria: "list[tuple]", need: int, too_short: str) -> "tuple[bool, str]":
    """K-of-N heuristic check. ``criteria`` is a list of ``(regex, missing_message)``.

    Passes if the reasoning is substantial AND at least ``need`` of the ``criteria`` match.
    Using K-of-N (rather than all-of-N) tolerates phrasing variance, which avoids the
    over-strict false-negative exhaustion seen with the original all-or-nothing checkers.
    """
    text = (text or "").strip()
    if len(text) < _MIN_REASONING_CHARS:
        return False, too_short
    missing = [msg for rx, msg in criteria if not rx.search(text)]
    if (len(criteria) - len(missing)) < need:
        return False, "; ".join(missing)
    return True, ""


# =============================================================================
# Patch intervention (target phase: P)
# =============================================================================
DEFAULT_PATCH_REMINDER_TEMPLATE = """\
[REASONING INTERVENTION -- pre-fix/patch phase reasoning]

>>> WHAT TO DO IN YOUR NEXT STEP (do BOTH, in this order, in your very next response):

  STEP A -- REASON FIRST (in your thinking/THOUGHT, before any command):
    1. ROOT CAUSE & FIX PLAN: name the ROOT CAUSE (not a symptom) and the exact edit --
       which file / function / line. Prefer a minimal, direct edit to the source.
    2. SIMULATE ON SEVERAL INPUTS: simulate the PATCHED code on (a) the bug's input, AND
       (b) the other / edge cases the issue mentions, AND (c) at least one input that
       already worked and must stay UNCHANGED. The grading tests are HIDDEN, so the fix
       must be general, not tailored to one example.
    3. COMPARE vs EXPECTED for each: state the patched result for each input above and
       confirm it matches the expected output -- the bug cases now pass AND the
       previously-working case is not broken (no regression).

  STEP B -- THEN ACT: in the SAME response, issue ONE bash command that edits the source
    file(s) to apply exactly the fix you reasoned about. (Next turn: re-run the
    reproduction to confirm it now passes -- green.)

Do not spend this step on more exploration -- your next command must be the source edit
itself, preceded by the analysis above.
"""

_FIX_TERMS = re.compile(
    r"\b(fix|patch|change|modify|edit|replace|update|correct|insert|remove|"
    r"function|method|line\s+\d+|file)\b",
    re.IGNORECASE,
)
_COMPARE_TERMS = re.compile(
    r"\b(expected|match(?:es)?|compare|correct\s+(?:result|output|value)|"
    r"equal|produces?|yields?|now\s+returns?|gives?|result\s+is)\b",
    re.IGNORECASE,
)


def default_patch_compliance_checker(thought: str, command: str, observation: str) -> "tuple[bool, str]":
    """Patch: root-cause fix plan + simulate patched code + compare vs expected + reason
    that other/unchanged inputs are not broken (no regression). Requires >=3 of 4."""
    return _check_criteria(
        thought,
        [
            (_FIX_TERMS, "name the root cause + the exact edit (file/function/line)"),
            (_SIM_TERMS, "simulate the patched code on concrete input(s)"),
            (_COMPARE_TERMS, "compare each patched result against the expected output"),
            (_UNCHANGED_TERMS, "check a previously-working input stays unbroken (no regression)"),
        ],
        need=3,
        too_short="the reasoning is too short to contain a real fix analysis",
    )


# =============================================================================
# New-test intervention (target phase: V_newly_generated_test)
# =============================================================================
DEFAULT_NEWTEST_REMINDER_TEMPLATE = """\
[REASONING INTERVENTION -- pre-new-test phase reasoning]

NOTE: the real grading tests are HIDDEN and are NOT in this repo. A test that merely passes
locally proves nothing -- your tests must GENERALIZE from the issue, not from your one
reproduction. (If you are only re-running your reproduction to confirm the fix, that is
validation, not new-test design -- just do it; this analysis is for writing NEW tests.)

>>> WHAT TO DO IN YOUR NEXT STEP (do BOTH, in this order, in your very next response):

  STEP A -- REASON FIRST (in your thinking/THOUGHT, before any command):
    1. CASES FROM THE ISSUE: enumerate every behaviour/example the issue describes (not just
       the one you reproduced) and, for each, the concrete input -> expected output.
    2. CORNER CASES: add edge / boundary / special-case inputs (empty, zero, negative,
       single-element, maximum, type extremes) with their expected outputs.
    3. MEANINGFUL ORACLE: each test must FAIL on the old (unpatched) code and PASS on the
       patched code -- otherwise it is a tautology that proves nothing.

  STEP B -- THEN ACT: in the SAME response, issue ONE bash command that creates and/or runs
    the new test(s) asserting exactly the input->output behaviour you just reasoned about.

Do not spend this step on more exploration -- your next command must be the new test
itself, preceded by the analysis above.
"""

_EDGE_TERMS = re.compile(
    r"\b(edge\s*case|corner\s*case|boundary|special\s*case|empty|zero|negative|"
    r"single|None|null|maximum|minimum|overflow|extreme|degenerate|invalid)\b",
    re.IGNORECASE,
)


def default_newtest_compliance_checker(thought: str, command: str, observation: str) -> "tuple[bool, str]":
    """New-test: concrete inputs + expected outputs + corner cases + generalize-from-issue
    (cases drawn from the issue / fail-on-old-pass-on-new). Requires >=3 of 4."""
    return _check_criteria(
        thought,
        [
            (_INPUT_TERMS, "enumerate concrete valid inputs to test"),
            (_COMPARE_TERMS, "state the expected output for those inputs"),
            (_EDGE_TERMS, "reason about corner / edge cases"),
            (_ORACLE_TERMS, "derive cases from the issue; tests fail on old, pass on patched"),
        ],
        need=3,
        too_short="the reasoning is too short to contain a real test-design analysis",
    )


# =============================================================================
# Intervention spec
# =============================================================================
@dataclass
class Intervention:
    """One phase-triggered reasoning intervention.

    Attributes:
        name: short identifier; used as the log tag and the per-step metadata key
            ``f"{name}_intervention"`` stamped onto the committed assistant message.
        target_phase: the fine-grained phase whose first appearance triggers the
            intervention (e.g. ``"L_reproduce"`` or ``"P"``).
        reminder_template: the user-message injected before the agent re-takes the step.
        compliance_checker: ``(reasoning, command, observation) -> (complied, feedback)``.
        max_retries: max number of redo attempts while the agent stays non-compliant.
        intervene_each_transition: if ``True`` intervene on every fresh entry into the
            target phase; if ``False`` (default) only on the first one in the run.
    """

    name: str
    target_phase: str
    reminder_template: str
    compliance_checker: ComplianceChecker
    max_retries: int = DEFAULT_MAX_RETRIES
    intervene_each_transition: bool = False
    _fired: bool = field(default=False, init=False, repr=False)

    def matches(self, record: dict, prev_phase: Optional[str]) -> bool:
        entered = self.target_phase in record["phases"] or record["current_phase"] == self.target_phase
        was_in = prev_phase == self.target_phase
        fresh = entered and not was_in
        if not fresh:
            return False
        if not self.intervene_each_transition and self._fired:
            return False
        return True


def make_patch_intervention(
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    compliance_checker: Optional[ComplianceChecker] = None,
    reminder_template: str = DEFAULT_PATCH_REMINDER_TEMPLATE,
    intervene_each_transition: bool = False,
) -> Intervention:
    return Intervention(
        name="patch",
        target_phase="P",
        reminder_template=reminder_template,
        compliance_checker=compliance_checker or default_patch_compliance_checker,
        max_retries=max_retries,
        intervene_each_transition=intervene_each_transition,
    )


def make_newtest_intervention(
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    compliance_checker: Optional[ComplianceChecker] = None,
    reminder_template: str = DEFAULT_NEWTEST_REMINDER_TEMPLATE,
    intervene_each_transition: bool = False,
) -> Intervention:
    return Intervention(
        name="newtest",
        target_phase="V_newly_generated_test",
        reminder_template=reminder_template,
        compliance_checker=compliance_checker or default_newtest_compliance_checker,
        max_retries=max_retries,
        intervene_each_transition=intervene_each_transition,
    )


# =============================================================================
# The hook
# =============================================================================
class PhaseInterventionHook:
    """State + behaviour driving phase-triggered reasoning interventions.

    Holds a :class:`phase_hook.StepPhaseTagger` (so the phase timeline is recorded the
    same way as the plain phase hook) plus a list of :class:`Intervention` specs, and
    drives the rollback / inject / redo loop whenever a step is a fresh transition into a
    target phase.
    """

    def __init__(
        self,
        agent,
        interventions: list[Intervention],
        *,
        enable_rules: bool = False,
        rules_config: Optional[str] = None,
        output_path: Optional[Path] = None,
        on_step: Optional[Callable[[dict], None]] = None,
        enable_env_rollback: bool = True,
        verbose: bool = True,
    ):
        self.agent = agent
        self.interventions = list(interventions)
        self.tagger = StepPhaseTagger(enable_rules=enable_rules, rules_config=rules_config)
        self.output_path = Path(output_path) if output_path is not None else None
        self.on_step = on_step
        self.verbose = verbose
        # When True, each rollback also rewinds the execution environment's git working
        # tree (tracked + untracked) so re-taken steps run against the pre-step-t state
        # rather than on top of the rolled-back step's filesystem side effects.
        self._rollback_env = enable_env_rollback

        # One entry per redo attempt (incl. discarded ones), tagged with the intervention.
        self.intervention_log: list[dict] = []
        # One entry per TRIGGER: what action fired which intervention at which step.
        self.trigger_log: list[dict] = []
        self._original_step = agent.step
        self._in_intervention = False  # guard against recursive triggering during redo
        self._prev_phase: Optional[str] = None  # committed monitor phase before this step

    # -- helpers ---------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # -- environment rollback --------------------------------------------------------
    def _env_execute(self, command: str) -> Optional[dict]:
        """Run a shell command in the agent's execution environment, if it has one.

        Returns the env's result dict, or ``None`` if the env exposes no ``execute``
        (e.g. mock envs in tests) or the call raises. Env bookkeeping must never crash
        the run, so all failures are swallowed.
        """
        env = getattr(self.agent, "env", None)
        execute = getattr(env, "execute", None)
        if execute is None:
            return None
        try:
            return execute({"command": command})
        except Exception as e:  # pragma: no cover - defensive
            self._log(f"    [env-rollback] execute failed: {e}")
            return None

    def _snapshot_env(self) -> Optional[str]:
        """Capture the repo working tree (tracked + untracked) as a container-side patch.

        Returns a token (the patch path inside the env) for use with :meth:`_restore_env`,
        or ``None`` if rollback is disabled, the env is not a git checkout, or anything
        fails. HEAD is never moved, so the final grading diff is unaffected.
        """
        if not self._rollback_env:
            return None
        token = f"/tmp/.phase_snap_{uuid.uuid4().hex[:12]}.patch"
        script = (
            'REPO=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0; '
            '[ -z "$REPO" ] && exit 0; cd "$REPO" || exit 0; '
            'git add -A -N >/dev/null 2>&1 || true; '          # intent-to-add untracked files
            f'git diff --binary HEAD > {token} 2>/dev/null || true; '
            'git reset -q >/dev/null 2>&1 || true; '           # undo intent-to-add; tree unchanged
            'echo __SNAP_OK__'
        )
        out = self._env_execute(script)
        if not out or "__SNAP_OK__" not in (out.get("output") or ""):
            return None
        return token

    def _restore_env(self, token: Optional[str]) -> None:
        """Rewind the repo working tree to the state captured by :meth:`_snapshot_env`.

        Hard-resets tracked files to HEAD, deletes files created after the snapshot, then
        re-applies the snapshot patch. HEAD stays at the base commit throughout.
        """
        if not token:
            return
        script = (
            'REPO=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0; '
            '[ -z "$REPO" ] && exit 0; cd "$REPO" || exit 0; '
            'git reset -q --hard HEAD >/dev/null 2>&1 || true; '   # drop tracked modifications
            'git clean -qfd >/dev/null 2>&1 || true; '             # drop files created after snapshot
            f'if [ -s {token} ]; then git apply --whitespace=nowarn {token} >/dev/null 2>&1 '
            f'|| git apply --whitespace=nowarn --3way {token} >/dev/null 2>&1 || true; fi'
        )
        self._env_execute(script)

    def _cleanup_snapshot(self, token: Optional[str]) -> None:
        """Delete a snapshot patch file once it is no longer needed."""
        if token:
            self._env_execute(f"rm -f {token}")

    @staticmethod
    def _reasoning_text(assistant: dict) -> str:
        """Gather all reasoning text from an assistant message.

        Reasoning models put their chain-of-thought in ``content`` and/or a separate
        ``reasoning`` / ``reasoning_content`` field; we consider all of them so the
        compliance check sees the model's actual reasoning, not just the surface text.
        """
        parts = [extract_thought(assistant)]
        for key in ("reasoning", "reasoning_content"):
            val = assistant.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        return "\n".join(p for p in parts if p)

    def _tag_step(self, n_before: int) -> "tuple[Optional[dict], Optional[dict], str]":
        """Tag the messages the agent appended after index ``n_before``.

        Returns ``(record, assistant_message, observation_text)``; the first two are
        ``None`` if the step produced no assistant message.
        """
        agent = self.agent
        new_messages = agent.messages[n_before:]
        assistant = next((m for m in new_messages if m.get("role") == "assistant"), None)
        if assistant is None:
            return None, None, ""
        observations = [m for m in new_messages if m is not assistant]
        observation_text = extract_observation(observations)
        record = self.tagger.tag(
            step_index=agent.n_calls,
            command=extract_command(assistant),
            thought=extract_thought(assistant),
            observation=observation_text,
        )
        extra = assistant.setdefault("extra", {})
        extra["phase"] = record["phases"][-1] if record["phases"] else None
        extra["phases_in_step"] = record["phases"]
        return record, assistant, observation_text

    def _select_intervention(self, record: dict) -> Optional[Intervention]:
        """Return the first intervention whose target phase this step freshly entered."""
        for intervention in self.interventions:
            if intervention.matches(record, self._prev_phase):
                return intervention
        return None

    @staticmethod
    def _build_reminder(intervention: Intervention, attempt: int, feedback: str) -> str:
        if attempt == 0 or not feedback:
            return intervention.reminder_template
        return DEFAULT_RETRY_PREFIX.format(
            attempt=attempt,
            max_retries=intervention.max_retries,
            feedback=feedback,
            base=intervention.reminder_template,
        )

    def _finalize(self, record: dict, assistant: Optional[dict]) -> None:
        """Persist the committed step's record and fire the ``on_step`` callback."""
        self._prev_phase = record["current_phase"]
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        if self.on_step is not None:
            self.on_step(record)

    # -- the hooked step -------------------------------------------------------------
    def hooked_step(self) -> Any:
        agent = self.agent
        n_before = len(agent.messages)
        n_calls_before = agent.n_calls
        monitor_snapshot = copy.deepcopy(self.tagger.monitor)
        records_len_before = len(self.tagger.records)
        env_snapshot = self._snapshot_env()

        result = self._original_step()
        record, assistant, _ = self._tag_step(n_before)

        if record is None or self._in_intervention:
            self._cleanup_snapshot(env_snapshot)
            if record is not None:
                self._finalize(record, assistant)
            return result

        intervention = self._select_intervention(record)
        if intervention is None:
            self._cleanup_snapshot(env_snapshot)
            self._finalize(record, assistant)
            return result

        # Record what action triggered which intervention at which step (the step-t
        # action that is about to be rolled back).
        trigger = {
            "intervention": intervention.name,
            "target_phase": intervention.target_phase,
            "step": record["step"],
            "trigger_command": record["command"],
            "trigger_phases": record["phases"],
            "current_phase": record["current_phase"],
        }
        self.trigger_log.append(trigger)
        self._log(
            f"  [intervention:{intervention.name}] {intervention.target_phase} triggered at "
            f"step {record['step']} by action: {(record['command'] or '').splitlines()[0][:70]!r} "
            f"-> rolling back and injecting reminder"
        )
        return self._run_intervention(
            intervention, n_before, n_calls_before, monitor_snapshot, records_len_before,
            record, assistant, env_snapshot,
        )

    def _run_intervention(
        self,
        intervention: Intervention,
        n_before: int,
        n_calls_before: int,
        monitor_snapshot,
        records_len_before: int,
        record: dict,
        assistant: Optional[dict],
        env_snapshot: Optional[str] = None,
    ) -> Any:
        agent = self.agent
        self._in_intervention = True
        intervention._fired = True
        result = None
        feedback = ""
        complied = False
        attempt = 0
        try:
            while attempt < intervention.max_retries:
                # (1) Roll the trajectory back to the end of step t-1.
                del agent.messages[n_before:]
                agent.n_calls = n_calls_before
                self.tagger.monitor = copy.deepcopy(monitor_snapshot)
                del self.tagger.records[records_len_before:]
                # ...and rewind the execution environment to match, so the redo runs
                # against the pre-step-t filesystem rather than step t's side effects.
                self._restore_env(env_snapshot)

                # (2) Inject the reasoning reminder, then re-take step t.
                reminder = self._build_reminder(intervention, attempt, feedback)
                agent.add_messages(agent.model.format_message(role="user", content=reminder))
                redo_n_before = len(agent.messages)
                result = self._original_step()
                record, assistant, observation = self._tag_step(redo_n_before)
                attempt += 1

                # (3) Check whether the agent reasoned as instructed before acting.
                if record is None:
                    feedback = "the step produced no action"
                    complied = False
                else:
                    reasoning = self._reasoning_text(assistant)
                    complied, feedback = intervention.compliance_checker(
                        reasoning, record["command"], observation
                    )
                self.intervention_log.append(
                    {
                        "intervention": intervention.name,
                        "target_phase": intervention.target_phase,
                        "step": record["step"] if record else n_calls_before + 1,
                        "attempt": attempt,
                        "complied": complied,
                        "feedback": feedback,
                        "command": record["command"] if record else "",
                        "phases": record["phases"] if record else [],
                    }
                )
                self._log(
                    f"    [{intervention.name} attempt {attempt}/{intervention.max_retries}] "
                    f"complied={complied}" + (f" ({feedback})" if not complied else "")
                )
                if complied:
                    break

            if not complied:
                self._log(
                    f"  [intervention:{intervention.name}] step {record['step'] if record else '?'} "
                    f"still non-compliant after {intervention.max_retries} retries; "
                    f"committing last attempt"
                )

            if record is not None:
                extra = assistant.setdefault("extra", {}) if assistant else {}
                extra[f"{intervention.name}_intervention"] = {
                    "complied": complied,
                    "attempts": attempt,
                    "max_retries": intervention.max_retries,
                    "final_feedback": feedback,
                }
                self._finalize(record, assistant)
            return result
        finally:
            self._in_intervention = False
            self._cleanup_snapshot(env_snapshot)


# =============================================================================
# Installers
# =============================================================================
def install_interventions(
    agent,
    interventions: list[Intervention],
    *,
    enable_rules: bool = False,
    rules_config: Optional[str] = None,
    output_path: Optional[Path] = None,
    on_step: Optional[Callable[[dict], None]] = None,
    enable_env_rollback: bool = True,
    verbose: bool = True,
) -> PhaseInterventionHook:
    """Patch ``agent.step`` to tag phases and run the given phase-triggered interventions.

    Like :func:`phase_hook.install_phase_hook`, this also annotates each assistant message
    in-place with its ``phase`` / ``phases_in_step`` and wraps ``agent.serialize`` to embed
    ``phase_timeline`` and the intervention log in the saved trajectory.

    Returns the :class:`PhaseInterventionHook` (``.tagger.records`` is the committed phase
    log; ``.intervention_log`` lists every redo attempt across all interventions).
    """
    hook = PhaseInterventionHook(
        agent,
        interventions,
        enable_rules=enable_rules,
        rules_config=rules_config,
        output_path=output_path,
        on_step=on_step,
        enable_env_rollback=enable_env_rollback,
        verbose=verbose,
    )
    agent.phase_records = hook.tagger.records
    agent.intervention_log = hook.intervention_log
    agent.intervention_triggers = hook.trigger_log
    agent.step = types.MethodType(lambda self: hook.hooked_step(), agent)

    # Embed the phase timeline + intervention/trigger logs in the saved trajectory.
    original_serialize = agent.serialize.__func__

    def patched_serialize(self, *extra_dicts):
        data = original_serialize(self, *extra_dicts)
        data["phase_timeline"] = hook.tagger.records
        data["interventions"] = hook.intervention_log
        data["intervention_triggers"] = hook.trigger_log
        # Per-intervention views (back-compat: reproduce keeps its own top-level key).
        for iv in hook.interventions:
            data[f"{iv.name}_interventions"] = [
                e for e in hook.intervention_log if e["intervention"] == iv.name
            ]
        return data

    agent.serialize = types.MethodType(patched_serialize, agent)
    return hook


def install_patch_intervention(
    agent,
    *,
    enable_rules: bool = False,
    rules_config: Optional[str] = None,
    output_path: Optional[Path] = None,
    on_step: Optional[Callable[[dict], None]] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    compliance_checker: Optional[ComplianceChecker] = None,
    reminder_template: str = DEFAULT_PATCH_REMINDER_TEMPLATE,
    intervene_each_transition: bool = False,
    enable_env_rollback: bool = True,
    verbose: bool = True,
) -> PhaseInterventionHook:
    """Install only the ``P`` (fix/patch) reasoning intervention."""
    return install_interventions(
        agent,
        [
            make_patch_intervention(
                max_retries=max_retries,
                compliance_checker=compliance_checker,
                reminder_template=reminder_template,
                intervene_each_transition=intervene_each_transition,
            )
        ],
        enable_rules=enable_rules,
        rules_config=rules_config,
        output_path=output_path,
        on_step=on_step,
        enable_env_rollback=enable_env_rollback,
        verbose=verbose,
    )


def install_newtest_intervention(
    agent,
    *,
    enable_rules: bool = False,
    rules_config: Optional[str] = None,
    output_path: Optional[Path] = None,
    on_step: Optional[Callable[[dict], None]] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    compliance_checker: Optional[ComplianceChecker] = None,
    reminder_template: str = DEFAULT_NEWTEST_REMINDER_TEMPLATE,
    intervene_each_transition: bool = False,
    enable_env_rollback: bool = True,
    verbose: bool = True,
) -> PhaseInterventionHook:
    """Install only the ``V_newly_generated_test`` (new-test design) reasoning intervention."""
    return install_interventions(
        agent,
        [
            make_newtest_intervention(
                max_retries=max_retries,
                compliance_checker=compliance_checker,
                reminder_template=reminder_template,
                intervene_each_transition=intervene_each_transition,
            )
        ],
        enable_rules=enable_rules,
        rules_config=rules_config,
        output_path=output_path,
        on_step=on_step,
        enable_env_rollback=enable_env_rollback,
        verbose=verbose,
    )
