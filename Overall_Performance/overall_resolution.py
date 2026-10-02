#!/usr/bin/env python3
"""Print baseline vs. pipeline resolution rates on SWE-bench Pro (731 instances) per model.

Data: data/<model>.csv, one row per instance with columns
instance_id, language, difficulty, baseline (PASS/FAIL), pipeline (PASS/FAIL).
Usage: python3 overall_resolution.py [--breakdown]
"""
import csv
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
MODELS = [
    ("MiniMax M3", "minimax_m3.csv"),
    ("GPT-5.6 Luna", "gpt56_luna.csv"),
    ("Claude Haiku (Claude Code)", "claude_haiku.csv"),
]


def load(name):
    with open(DATA / name, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        assert r["baseline"] in ("PASS", "FAIL") and r["pipeline"] in ("PASS", "FAIL"), r
    assert len({r["instance_id"] for r in rows}) == len(rows), f"duplicate ids in {name}"
    return rows


def rate(rows, arm):
    k = sum(r[arm] == "PASS" for r in rows)
    return f"{k:>3}/{len(rows):<3} {100 * k / len(rows):5.1f}%"


def table(title, groups):
    print(f"  {title:<14} {'baseline':>15} {'pipeline':>15} {'delta':>6}")
    for label, rows in groups:
        d = sum(r["pipeline"] == "PASS" for r in rows) - sum(r["baseline"] == "PASS" for r in rows)
        print(f"  {label:<14} {rate(rows, 'baseline'):>15} {rate(rows, 'pipeline'):>15} {d:>+6}")


def main():
    breakdown = "--breakdown" in sys.argv
    for model, fname in MODELS:
        rows = load(fname)
        print(f"== {model}  ({len(rows)} instances)")
        table("", [("overall", rows)])
        both = sum(r["baseline"] == r["pipeline"] == "PASS" for r in rows)
        p_only = sum(r["baseline"] == "FAIL" and r["pipeline"] == "PASS" for r in rows)
        b_only = sum(r["baseline"] == "PASS" and r["pipeline"] == "FAIL" for r in rows)
        print(f"  both {both}, pipeline-only {p_only}, baseline-only {b_only}, "
              f"neither {len(rows) - both - p_only - b_only}")
        if breakdown:
            for key in ("language", "difficulty"):
                vals = sorted({r[key] for r in rows})
                table(key, [(v, [r for r in rows if r[key] == v]) for v in vals])
        print()


if __name__ == "__main__":
    main()
