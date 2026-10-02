"""ccpipe -- SWE-bench Pro pipeline on top of Claude Code headless sessions.

Claude Code runs INSIDE the task container (binary bind-mounted read-only); the
container sits on an internal docker network whose only egress is an allowlist
CONNECT proxy sidecar (simagent.claude_code.net), so the repo can reach nothing but the
Anthropic API. The Python orchestrator on the host owns container lifecycle,
prompting, patch extraction, trajectories and predictions.
"""
