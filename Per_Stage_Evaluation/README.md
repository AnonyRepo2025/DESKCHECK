# RQ5: Contribution of Guided Code Reasoning to Each Stage of DeskCheck

Per-stage analysis of the pipeline on SWE-bench Pro (731 instances per model) for three models:
MiniMax-M3 (`minimax_m3`), GPT-5.6 Luna (`gpt56_luna`) and Claude Haiku 4.5 on Claude Code
(`claude_haiku`). Each stage folder is self-contained (data + script + figures; matplotlib only)
and has its own README.

## Structure

```
Per_Stage_Evaluation/
├── README.md
├── localization/                         localization stage
│   └── localization_patch_sites.xlsx     per-instance ground-truth vs. predicted edit sites, with
│                                         method- and file-level recall / precision / F1, for the
│                                         localization output and the final patch. Tabs:
│                                         pipeline-{MiniMax,gpt-5.6,haiku},
│                                         baseline-{MiniMax,gpt-5.6,haiku},
│                                         swe-doctor-{gpt-5.6,minimax}
├── refine/                               repair stage: refinement
│   ├── README.md
│   ├── refinement_resolution.py          Sankey: refined / not refined -> PASS / FAIL
│   ├── data/{minimax_m3,gpt56_luna,claude_haiku}.csv
│   └── figures/<model>/
│       ├── refinement_resolution.png             all instances
│       └── refinement_resolution_rescue.png      baseline-failed instances only
├── audit/                                audit stage
│   ├── README.md
│   ├── audit_chain.py                    Sankey: violation -> code edit -> PASS / FAIL
│   ├── data/{minimax_m3,gpt56_luna,claude_haiku}.csv
│   └── figures/<model>/
│       ├── audit_chain.png
│       └── audit_chain_rescue.png
└── validate/                             validation stage
    ├── README.md
    ├── postvalidation_resolution.py      Sankey: post-validation edit / none -> PASS / FAIL
    ├── data/{minimax_m3,gpt56_luna,claude_haiku}.csv
    └── figures/<model>/
        ├── postvalidation_resolution.png
        └── postvalidation_resolution_rescue.png
```

## Reproducing the figures

```
cd refine   && python3 refinement_resolution.py
cd audit    && python3 audit_chain.py
cd validate && python3 postvalidation_resolution.py
```

Each script writes its two figures per model into `figures/<model>/` (`--out-dir DIR` or
`--out MODEL=DIR` to write elsewhere) and prints the flow counts.

Each data file has one row per instance: `instance_id`, `language`, `difficulty`, the baseline and
pipeline outcomes (`PASS`/`FAIL`), and the stage flag (`refinement`, `audit_violation` +
`audit_change`, or `post_validation_edit`; `Y`/`Y*` = set, blank = not set).

The `*_rescue` figures cover only the instances the baseline agent failed. On those, a PASS means
the pipeline rescued the instance.
