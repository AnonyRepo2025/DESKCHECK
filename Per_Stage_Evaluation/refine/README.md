# Refinement vs. resolution

Sankey diagrams that follow each SWE-bench Pro instance (731 per model) from whether the pipeline's
refinement step changed its patch (check = refined, cross = not refined) to the pipeline's final
resolution.

```
python3 refinement_resolution.py [--out-dir DIR] [--out MODEL=DIR ...] [--models ...]
```

For each model (`minimax_m3`, `gpt56_luna`, `claude_haiku`) this writes two figures into
`figures/<model>/` by default:

- `refinement_resolution.png`: all instances, refined / not refined -> PASS / FAIL
- `refinement_resolution_rescue.png`: only the instances the baseline agent failed, so Pass means
  the pipeline rescued the instance
