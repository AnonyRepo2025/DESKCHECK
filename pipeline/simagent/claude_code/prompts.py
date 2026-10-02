"""Prompt texts for the Claude Code sessions.

BASELINE_* mirrors the SWE-bench Pro mini-swe-agent template section by section (overview,
boundaries, recommended workflow, environment) with the parts that only exist because
mini-swe-agent is a bash-only loop (THOUGHT+command protocol, the
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT submission dance) replaced by the Claude Code
equivalents: native file tools, and "stop when done -- the harness diffs the repo".
"""

BASELINE_SYSTEM = (
    "You are a helpful assistant that can interact with a computer shell to solve programming tasks."
)

BASELINE_INSTANCE = """\
<pr_description>
Consider the following PR description:
{task}
</pr_description>

<instructions>
# Task Instructions

## Overview

You're a software engineer working in a checked-out repository.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in {repo_path} in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.

## Important Boundaries

- MODIFY: Regular source code files in {repo_path} (this is the working directory)
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, package.json, go.mod unless the change itself requires a new dependency), packaging or setup scripts
- Do NOT commit. Do NOT create patch files. The harness collects `git diff` of {repo_path} when you finish.
- Put any reproduction script, probe, or note you create under /tmp (never inside the repository); anything left inside the repository is treated as part of your change.

## Recommended Workflow

1. Analyze the codebase by finding and reading relevant files
2. Create a script (under /tmp) to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust, and run the existing tests that cover the files you changed

## Environment Details

- You have a full Linux shell environment (Bash tool) plus file tools (Read, Edit, Write, Grep, Glob)
- Always use non-interactive flags (-y, -f) for commands; avoid interactive tools like vi, nano, or anything requiring user input
- There is no network access; do not try to install packages from the internet
- You can also create new tools or scripts (under /tmp) to help you with the task

## Finishing

When you've completed your work, make sure every intended change is saved in the source files under {repo_path}, then stop and reply with a brief summary of what you changed and why. Do not ask questions; there is nobody to answer them.
</instructions>
"""


def baseline_prompt(task: str, repo_path: str = "/app") -> str:
    return BASELINE_INSTANCE.format(task=task, repo_path=repo_path)
