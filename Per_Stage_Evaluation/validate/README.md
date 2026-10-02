# Post-validation edit vs. resolution

Sankey diagrams that follow each SWE-bench Pro instance (731 per model) from whether a gate after
the spec audit (validate / regression fix) edited its patch to the pipeline's final resolution.
The check mark means edited and the cross means not edited.

```
python3 postvalidation_resolution.py [--out-dir DIR] [--out MODEL=DIR ...] [--models ...]
```

For each model (`minimax_m3`, `gpt56_luna`, `claude_haiku`) this writes two figures into
`figures/<model>/` by default:

- `postvalidation_resolution.png`: all instances, edited / not edited -> PASS / FAIL
- `postvalidation_resolution_rescue.png`: only the instances the baseline agent failed, so PASS
  means the pipeline rescued the instance

Use `--out MODEL=DIR` to write one model's figures into a given directory. The script also prints
the flow counts and pass rates per group. The only dependency is matplotlib.

Data: `data/<model>.csv`, one row per instance:

| column | values |
|---|---|
| `instance_id` | SWE-bench Pro instance |
| `language`, `difficulty` | python / go / js / ts; easy / medium / difficult |
| `baseline`, `pipeline` | PASS / FAIL |
| `post_validation_edit` | `Y` = a post-audit gate edited the patch, `Y*` = same, verified by hand; blank = no edit (`Y` and `Y*` are counted together) |
