# Replication Package for the Empirical Study

This directory contains the data and scripts needed to replicate the results in Section 2,
"Code Reasoning in Programming Agents".

| RQ | Question | What we provide |
|---|---|---|
| [RQ1](RQ1/README.md) | How prevalent is code reasoning in programming agents? | Scripts that detect code reasoning behaviors in agent trajectories |
| [RQ2](RQ2/README.md) | Does code reasoning correlate with resolution? | Scripts that analyze the correlation between code reasoning and resolution rates |
| [RQ3](RQ3/README.md) | What factors influence code reasoning? | Scripts that analyze the impact of issue difficulty and of the different bug-fixing phases |

## Layout

```
RQ1/        scripts/      detection pipeline (run_all.sh is the entry point)
RQ2/        scripts/  data/  figures/
RQ3/        scripts/  data/  figures/   (data/phase holds per-instance phase results)
results/    per-set JSON outputs of the analysis
```

`results/` contains two kinds of files:
- `<scaffold>_<benchmark>_<model>_allreasoning.json`: RQ1 detection results for each trajectory set.
- `phase_code_reasoning_*.json`: per-phase code reasoning results for RQ3.

## Quick start

Each RQ folder has its own README with the exact commands. The RQ2 and RQ3 figures can be
regenerated from the aggregated data in each folder's `data/` without rerunning RQ1:

```
python3 RQ2/scripts/plot_code_reasoning_resolution.py
python3 RQ3/scripts/plot_difficulty_stacked.py
python3 RQ3/scripts/plot_phase_code_reasoning.py
```

Rerunning RQ1 from scratch requires the raw agent trajectories (see `TRAJ_ROOT` in
[RQ1/README.md](RQ1/README.md)).
