# RQ4. Effectiveness of DeskCheck

We released the scripts to reproduced the results presented in RQ4.
## Overall resolution rate: baseline vs. pipeline (SWE-bench Pro, 731 instances)
```
python3 overall_resolution.py              # overall rate per model
```

## Venn diagrams
```
python3 venn_diagrams.py [--out-dir DIR]
```
Writes area-proportional baseline-vs-pipeline Venns (matplotlib only) as `venn_{minimax,luna,haiku}.png`
into `venn/` by default.
Orange = baseline, blue = pipeline; left number = baseline-only, middle = both, right = pipeline-only.

## Bar plots by difficulty and language

```
python3 bar_resolution.py [--out-dir DIR] [--y-max 300]
```
Writes `{MiniMax-M3,GPT-5.6-Luna,ccpipe-Haiku}_resolution_by_{difficulty,language}.png` into `bars/`:
resolved count per group (green = baseline, blue = pipeline), labelled with the rate within that
group.
