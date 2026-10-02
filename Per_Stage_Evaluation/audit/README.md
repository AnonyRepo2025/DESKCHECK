# Audit chain: violation -> code edit -> resolution

Three-stage Sankey diagrams that follow each SWE-bench Pro instance (731 per model) through the
pipeline's spec-audit step to its final resolution:

1. **Violation:** did the spec audit record a violation? The check mark means a violation was recorded and the cross means none was.
2. **Code edit:** did an audit-fix round edit the patch (Changed / Unchanged)?
3. **Resolution:** PASS / FAIL of the final pipeline patch.

```
python3 audit_chain.py [--out-dir DIR] [--out MODEL=DIR ...] [--models ...]
```

For each model (`minimax_m3`, `gpt56_luna`, `claude_haiku`) this writes two figures into
`figures/<model>/` by default:

- `audit_chain.png`: all instances
- `audit_chain_rescue.png`: only the instances the baseline agent failed, so PASS means the
  pipeline rescued the instance

Use `--out MODEL=DIR` to write one model's figures into a given directory. The script also prints
the counts on every link. The only dependency is matplotlib.

The few instances whose patch an audit-fix round changed without a recorded violation are drawn
as violation -> changed. The data files keep the recorded flags.

Data: `data/<model>.csv`, one row per instance:

| column | values |
|---|---|
| `instance_id` | SWE-bench Pro instance |
| `language`, `difficulty` | python / go / js / ts; easy / medium / difficult |
| `baseline`, `pipeline` | PASS / FAIL |
| `audit_violation` | `Y` = the spec audit recorded a violation (`Y*` = hand-verified); blank = none |
| `audit_change` | `Y` = an audit-fix round edited the patch (`Y*` = hand-verified); blank = no edit |
