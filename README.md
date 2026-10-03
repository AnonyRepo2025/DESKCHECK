
# DeskCheck: Replication Package

This repository holds the code and data for the paper's empirical study of code reasoning in
programming agents (RQ1–RQ3), and for DeskCheck (the `simagent` repair pipeline). DeskCheck guides an
agent to reason about how code executes, and the package evaluates it on SWE-bench Pro (RQ4–RQ5).

We also released the trajectories and patches of the proposed agentic framework on SWE-bench-pro in this [link](https://drive.google.com/file/d/17WxoWDYlPlEKKbGn-Qo4qbXAlFimzmZC/view?usp=sharing). Before you extract the archive, make sure your drive has at least 7 to 8 GB of free space.

| Directory | Paper section | Contents |
|---|---|---|
| [`Empirical_Study/`](Empirical_Study/README.md) | §2, RQ1–RQ3 | code reasoning taxonomy, aggregated results, figure scripts |
| [`Overall_Performance/`](Overall_Performance/README.md) | RQ4 | baseline vs. pipeline resolution, Venn and bar plots |
| [`Per_Stage_Evaluation/`](Per_Stage_Evaluation/README.md) | RQ5 | per-stage (localize / refine / audit / validate) data and Sankey plots |
| [`pipeline/`](pipeline/README.md) | DeskCheck | the repair pipeline on two agent runtimes (mini-swe-agent, Claude Code), grading |
| [`prompts.md`](prompts.md) | | the cores of reasoning prompts |
| [`taxonomy-standalone.pdf`](taxonomy-standalone.pdf) | | code reasoning taxonomy |
| [`Issue-protonmail-ac23d1ef.md`](Issue-protonmail-ac23d1ef.md) | | worked example of one SWE-bench Pro issue |
---

## 1. System requirements

| | Requirement |
|---|---|
| OS | **Linux x86-64** (tested on Ubuntu, kernel 6.8) |
| Python | ≥ 3.10 (**3.12 recommended and tested**) |
| Docker | Docker Engine ≥ 24 that your user can run without `sudo` (`docker ps` works) |
| Disk | ~2 GB for the repository (the RQ1 `results.zip` unpacks to 1.1 GB), plus 1–5 GB per task image, so plan for a few hundred GB for a full 731-instance run |
| Network | Docker Hub (`jefzda/sweap-images`), GitHub, and the model API |
| Other | `git`, `bash`, `xargs`; for the Claude Code runtime, the `claude` CLI |

## 2. Get the code and create a Python environment

```bash
git clone https://github.com/AnonyRepo2025/DESKCHECK.git
cd DESKCHECK
```

Pick **one** of the following.

**conda**
```bash
conda create -y -n deskcheck python=3.12
conda activate deskcheck
```

**venv**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Then install every Python dependency the repository's scripts use:

```bash
pip install -r requirements.txt      # analysis deps + pipeline/requirements.txt
```

[`requirements.txt`](requirements.txt) covers matplotlib, numpy, pandas, openpyxl, jinja2, pyyaml,
networkx, bashlex, litellm and pytest. You can install these optional extras when needed:
`pip install datasets` lets the pipeline and RQ1 detectors fetch instances from HuggingFace when no
local file is given, and `pip install pygraphviz` renders phase-graph PDFs.

> **If you only want the figures and tables (RQ1–RQ5), you can stop here** and go to
> [§6](#6-reproducing-the-paper-results).

Run all of the remaining steps from `pipeline/`:

```bash
cd pipeline
```

## 3. Install the patched mini-swe-agent 2.4.1 (required by the pipeline)

Both runtimes import `minisweagent`. They use its Docker environment, its SWE-bench helpers, and, for
the mini runtime, its agent loop. The patch has two hunks. One fixes the startup-command call
signature. The other ends a run cleanly when its task container disappears (`ContainerDied`).

```bash
git clone --branch v2.4.1 https://github.com/SWE-agent/mini-swe-agent.git
git -C mini-swe-agent apply ../patches/mini-swe-agent-2.4.1.patch
pip install -e mini-swe-agent
python -c "import minisweagent; print(minisweagent.__version__)"     # -> 2.4.1
```

## 4. Install the SWE-bench Pro evaluation harness (grading only)

```bash
git clone https://github.com/scaleapi/SWE-bench_Pro-os.git
pip install "pandas>=1.5" "tqdm>=4.64" "docker>=6.0"   # what the harness needs for local-docker grading
export SWEBENCH_PRO_DIR=$PWD/SWE-bench_Pro-os           # add to your shell rc; grade.sh reads it
```

The harness's own `requirements.txt` also lists `modal`, `datasets` and `huggingface_hub`. You only
need those for cloud evaluation on Modal. `scripts/grade.sh` always passes `--use_local_docker`.

## 5. Docker, instance data and credentials

**Instance data.** Unpack the 731 SWE-bench Pro records of the reported runs:

```bash
gunzip -k data/swebench_pro_731.jsonl.gz      # -> data/swebench_pro_731.jsonl (git-ignored)
```

**Docker.** Check that `docker ps` works without `sudo`. Task images are pulled on demand from
`jefzda/sweap-images`. To pre-pull one image (the tag is the `dockerhub_tag` field of a record):

```bash
docker pull jefzda/sweap-images:<dockerhub_tag>
```

Each runtime needs its own credentials.

### 5a. mini-swe-agent runtime (MiniMax-M3, GPT-5.6 Luna via OpenRouter)

```bash
export OPENROUTER_API_KEY=sk-or-...
# or store it in mini-swe-agent's global config (~/.config/mini-swe-agent/.env):
#   OPENROUTER_API_KEY=sk-or-...
```

The configs are `configs/swebench_pro_{m3,luna}.yaml`. The M3 price entry for litellm's cost tracking
is in `configs/model_registry.json`.

### 5b. Claude Code runtime (Claude Haiku 4.5)

1. Install the Claude Code CLI on the host. Native installer:
   `curl -fsSL https://claude.ai/install.sh | bash`, which puts it in `~/.local/bin/claude`. Check
   with `claude --version`. The pipeline resolves the binary with `which claude` (falling back to
   `~/.local/bin/claude`) and bind-mounts it read-only into every task container. You do **not**
   install anything inside the images.
2. Authenticate. Pick one:
   - **Subscription (default, `CCPIPE_AUTH=oauth`).** Run `claude setup-token`, then either export
     `CLAUDE_CODE_OAUTH_TOKEN=<token>`, or save the token to a file:
     ```bash
     mkdir -p ~/.config/ccpipe && printf '%s' '<token>' > ~/.config/ccpipe/oauth_token && chmod 600 ~/.config/ccpipe/oauth_token
     ```
   - **API key.** `export CCPIPE_AUTH=apikey ANTHROPIC_API_KEY=sk-ant-...`
3. Networking is set up for you. The first run creates an internal Docker network
   (`ccpipe_internal`) and an allowlist proxy sidecar (`ccpipe_proxy`, built from `python:3.12-slim`).
   The proxy lets task containers reach only `api.anthropic.com`. Both are reused across runs.
4. *(Only for Alpine-based task images, e.g. teleport.)* The glibc `claude` binary cannot run on
   musl. Download the `linux-x64-musl` build of the **same version** as your host CLI to
   `~/.local/share/claude/musl/<version>`, or point `CCPIPE_CLAUDE_MUSL` at it.

## 6. Reproducing the paper results

These steps use only the data shipped in the repository, so they need no Docker and no API keys. Run
them from the repository root with the environment from §2 activated. Each script rewrites the
corresponding committed figure. Every plotting script selects matplotlib's `Agg` backend itself, so
the scripts also run on a headless server.

```bash
# RQ1: prevalence of code-reasoning patterns (needs the bundled results, ~1.1 GB unpacked)
(cd Empirical_Study && unzip -q results.zip)                  # -> Empirical_Study/results/
(cd Empirical_Study/RQ1 && python scripts/collect_code_reasoning_patterns.py)

# RQ2 / RQ3: correlation with resolution, influencing factors
(cd Empirical_Study && python RQ2/scripts/plot_code_reasoning_resolution.py \
                    && python RQ2/scripts/plot_code_reasoning_resolution.py --by-scaffold \
                    && python RQ3/scripts/plot_difficulty_stacked.py \
                    && python RQ3/scripts/plot_phase_code_reasoning.py)

# RQ4: overall effectiveness
(cd Overall_Performance && python overall_resolution.py && python venn_diagrams.py && python bar_resolution.py)

# RQ5: per-stage contribution
(cd Per_Stage_Evaluation/refine   && python refinement_resolution.py)
(cd Per_Stage_Evaluation/audit    && python audit_chain.py)
(cd Per_Stage_Evaluation/validate && python postvalidation_resolution.py)
python -c "import pandas as pd; print(list(pd.read_excel('Per_Stage_Evaluation/localization/localization_patch_sites.xlsx', sheet_name=None)))"
```

To rerun the RQ1 detectors from the raw trajectories (6.5 GB download, ~31 GB unpacked), see
[`Empirical_Study/RQ1/README.md`](Empirical_Study/RQ1/README.md). Point `TRAJ_ROOT` at the unpacked
`empirical_study_trajectory/` folder, run `scripts/prepare_empirical_inputs.py` to build
`RQ1/data/inputs/`, then run `scripts/run_all.sh`.

## 7. Running the pipeline

<!-- Run everything below from `pipeline/` with the environment activated. Start with the offline unit
tests. They need no Docker, no network and no model calls:

```bash
tests/run_all.sh          # pytest (145 tests) + two plain-assert check scripts
``` -->

### example

The example is a Python instance from Open Library:

```bash
python scripts/select_instances.py \
  --ids instance_internetarchive__openlibrary-e8084193a895d8ee81200f49093389a3887479ce-ve8c8d62a2b60610a3c4631f5f23ed866bada9818 \
  > example.jsonl

# mini-swe-agent runtime (MiniMax-M3 or GPT-5.6 Luna)
PAR=1 scripts/run_mini.sh m3 example.jsonl runs/example_m3          # or: luna
scripts/grade.sh example.jsonl runs/example_m3

# Claude Code runtime (Claude Haiku 4.5), DeskCheck arm only
# (drop ARMS=reason to also run the reproduction-test baseline arm)
ARMS=reason PAR=1 MODEL=haiku scripts/run_claude_code.sh example.jsonl runs/example_haiku
scripts/grade.sh example.jsonl runs/example_haiku/reason
```


Progress is written to `runs/<run>/driver.log`, and each instance's log is in
`runs/<run>/logs/<iid>.log`. When the run finishes:

- `runs/<run>/<iid>/pipeline.json` holds the per-phase record, gate decisions and cost.
- `runs/<run>/<iid>/model_patch.diff` is the final patch.
- `runs/<run>/preds.json` collects all patches. `grade.sh` reads it.
- `runs/<run>/eval/RESULTS.md` and `results.json` are written by `grade.sh`. An instance counts as
  resolved when all of its FAIL_TO_PASS and PASS_TO_PASS tests pass.

<!-- ### Batches and full runs

```bash
python scripts/select_instances.py --batch py_batch1 > py1.jsonl     # batch names: data/batches.json
python scripts/select_instances.py --lang go         > go.jsonl
python scripts/select_instances.py --all             > all.jsonl     # all 731
PAR=8 scripts/run_mini.sh m3 py1.jsonl runs/m3_py1
```

Both launchers run a streaming queue of `PAR` instances. They also resume: re-running the same
command skips instances that already have a finished `pipeline.json`. The main knobs are environment
variables:

- `run_mini.sh`: `PAR`, `PYTHON`, `INSTANCE_COST_CAP` (USD, default 3.0), `EXPLORE_STEPS`,
  `REPAIR_STEPS`.
- `run_claude_code.sh`: `PAR`, `PYTHON`, `MODEL` (default `haiku`), `ARMS` (default
  `reason repro`), `REPAIR_STEPS`, `CCPIPE_AUTH`.

See [`pipeline/README.md`](pipeline/README.md) for the phase design and module layout. -->

