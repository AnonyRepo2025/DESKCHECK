# SimAgent pipeline

SimAgent is a repair pipeline for SWE-bench Pro. Its phases ask the model to simulate how the code runs on concrete inputs: tracing a wrong value back to where it is built, mentally running the patched code against each requirement, and deriving expected outputs from the specification. Deterministic checks sit between those steps. The prompt cores are listed in [`../prompts.md`](../prompts.md).

```
issue + requirements + interface
   |
   1 LOCALIZE   explore -> anchor prediction* -> call-chain tracer / path pruning -> execution-path simulation*
   |            -> frames that must change (PRODUCER check, concrete inputs)
   2 REPAIR     fix -> simulation  -> refine
   3 AUDIT      requirement extraction* -> per-point simulated audit* -> refine
   4 VALIDATE   input - expected table* (+ corner cases) -> tests -> fix
   * = tool-free reasoning step (the model cannot read, run or edit anything and answers by execution reasoning)
```

The same pipeline runs on two agent runtimes:

| runtime | models in the paper | entry point | launcher |
|---|---|---|---|
| mini-swe-agent 2.4.1 (bash-only agent, litellm) | MiniMax-M3, GPT-5.6 Luna (via OpenRouter) | `python -m simagent.plainrepair` | `scripts/run_mini.sh` |
| Claude Code CLI (`claude -p`, one conversation per call) | Claude Haiku 4.5 | `python -m simagent.claude_code.flow_run` | `scripts/run_claude_code.sh` |

## Layout

```
simagent/
  pipeline.py        phase driver: explore, localize, repair, spec audit, audit fix, validate, regression guard
  plainrepair.py     mini-swe-agent entry: plain repair agent + simulation loop (trace/contract lenses)
  localization.py    localization sub-agent helpers (anchor prediction, site parsing, repo search)
  graph_localize.py  call-graph path pruning + execution-path simulation over traced chains
  subagent.py        tool-free sub-agent queries, language profiles (Python/Go/JS/TS), call-chain tracers
  interventions.py   phase-conditioned reminders (e.g. new-test compliance during validate)
  phase_hook.py      step-level phase tagging (uses the vendored plan_monitor)
  validation.py      execution profile, import smoke, probe runner
  evidence.py        test-output evidence parsing
  production_guard.py rejects production code conditioned on mock / test-runner identity
  usage.py           token / cost accounting
  grade.py           SWE-bench Pro grading of a run directory (both runtimes)
  grade_watch.py     optional incremental grader
  plan_monitor/      vendored phase monitor (command parser, action-role map, phase rules)
  claude_code/       Claude Code runtime: flow_run (driver), flow (resumable sessions), runner (claude CLI),
                     env/net (task container on an internal network, allowlist proxy to the API),
                     phases/{localize,repair,audit,validate,repro_localize}, validate_cc, traj, proxy/
configs/             mini-swe-agent configs (M3, Luna), Claude Code base environment, M3 price registry
data/                the 731 SWE-bench Pro instance records of the reported runs + batch partition
patches/             two-hunk patch to mini-swe-agent 2.4.1
scripts/             run_mini.sh, run_claude_code.sh, grade.sh, select_instances.py, collect_preds.py
```

## Setup

Requirements are Linux, Docker (the task images are pulled from Docker Hub, `jefzda/sweap-images`) and Python 3.12.

```bash
pip install -r requirements.txt

# mini-swe-agent 2.4.1 plus two small fixes (the startup-command call signature, and a
# ContainerDied exit when the task container disappears mid-run)
git clone --branch v2.4.1 https://github.com/SWE-agent/mini-swe-agent.git
git -C mini-swe-agent apply ../patches/mini-swe-agent-2.4.1.patch    # run from this directory
pip install -e mini-swe-agent

# official SWE-bench Pro evaluation harness (grading only)
git clone https://github.com/scaleapi/SWE-bench_Pro-os.git
export SWEBENCH_PRO_DIR=$PWD/SWE-bench_Pro-os

gunzip -k data/swebench_pro_731.jsonl.gz
```

The two runtimes need different credentials:
- **mini-swe-agent:** set `OPENROUTER_API_KEY` in the environment or in mini-swe-agent's `.env` file.
- **Claude Code:** install the `claude` CLI on the host; its binary is bind-mounted read-only into each task container. For subscription auth, run `claude setup-token` and either export `CLAUDE_CODE_OAUTH_TOKEN` or save the token to `~/.config/ccpipe/oauth_token` (mode 600). Alternatively, set `CCPIPE_AUTH=apikey` together with `ANTHROPIC_API_KEY`.

## Running

Pick instances with `scripts/select_instances.py`. It accepts `--batch py_batch1`, `--lang go`, `--ids ...` or `--all`, and the batch names are in `data/batches.json`.

```bash
python scripts/select_instances.py --batch py_batch1 > py1.jsonl

# mini-swe-agent runtime (MiniMax-M3 or GPT-5.6 Luna)
PAR=8 scripts/run_mini.sh luna py1.jsonl runs/luna_py1          # or: m3, or a path to a yaml
scripts/grade.sh py1.jsonl runs/luna_py1

# Claude Code runtime (Haiku 4.5): SimAgent arm and the reproduction-test baseline arm
PAR=8 MODEL=haiku scripts/run_claude_code.sh py1.jsonl runs/haiku_py1
scripts/grade.sh py1.jsonl runs/haiku_py1/reason
scripts/grade.sh py1.jsonl runs/haiku_py1/repro
```

Both launchers keep a streaming queue of `PAR` instances and resume: an instance that already has a finished record is skipped.

Each instance directory holds:
- `pipeline.json`: the per-phase record, gate decisions and cost.
- `model_patch.diff`: the final patch.
- the per-phase trajectories, plus a combined `<iid>.traj.json` in mini-swe-agent-1.1 format.

`<run>/preds.json` gathers every instance's patch for grading. The grader counts an instance as resolved when all of its FAIL_TO_PASS and PASS_TO_PASS tests pass, and writes `<run>/eval/{results.json,RESULTS.md}`.


