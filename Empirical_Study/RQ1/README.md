# RQ1: Prevalence of Code Reasoning in Programming Agents.

## Overall code-reasoning patterns
First unzip the results.zip.
```
python3 scripts/collect_code_reasoning_patterns.py   # reads ../results/*_allreasoning.json
```
Prints, summed over the 14 sets, the number and share of trajectories with each code reasoning family
(output_and_exceptions, variable_state, call_dependency, conditional, loop) and its sub-categories, with hit
counts. Use `--in-dir` to point at `results/execution_annotation/` after `run_all.sh`. Counts are raw
detections (trajectories with >=1 hit), not the `has_code_reasoning` decision.


## Generate your own reports (Optional)
We also provide scripts to generate code reasoning behavior reports.
The raw trajectories used in the study are available [here](https://drive.google.com/file/d/1IssdqC35GnPdMBzRrYQdPtMGo5Yz4wZ0/view?usp=sharing)
(6.5 GB zip; about 31 GB once unzipped, so check your free disk space first). Point `TRAJ_ROOT` at the unzipped
`empirical_study_trajectory/` folder.

Produces the `<scaffold>_<benchmark>_<model>_allreasoning.json` results:
```
scripts/run_all.sh                                   # all 14 sets -> results/execution_annotation/
ONLY="mini_sbv_gpt-5-2-high sonar_sbv_claude-opus-4-5" scripts/run_all.sh   # a subset
```

Environment variables:
- `TRAJ_ROOT`: directory holding the trajectory sets.
- `OUT`: output directory (default `results/execution_annotation`).
- `PARALLEL`: concurrent sets (default 4).
- `ONLY`: space-separated prefixes to run.

## Detection settings
Every set is run with `--no-top-level-only --min-score 1`: the detector scans both visible response
text and thinking/CoT, and a turn counts as code reasoning with at least one scored hit.

| script | trajectory format | sets |
|---|---|---|
| `find_code_tokens_in_prose.py` | mini-SWE-agent / SWE-agent chat trajectories | `mini_*` |
| `find_code_tokens_in_prose_icat.py` | iCAT-Agent `.traj` | `icat_*` |
| `find_code_tokens_in_prose_multi.py --format {openhands,sonar,claudecode,experepair}` | OpenHands JSONL, Sonar JSON, Claude Code session JSONL, ExpeRepair timelines | `openhands_*`, `sonar_*`, `claudecode_*`, `experepair_*` |

`find_execution_reasoning.py` is the shared pattern library that all three import.