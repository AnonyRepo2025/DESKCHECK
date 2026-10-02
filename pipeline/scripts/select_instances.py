#!/usr/bin/env python3
"""Write a dataset subset (jsonl) for the run scripts.

Usage:
    python scripts/select_instances.py --batch go_batch1 > go_batch1.jsonl
    python scripts/select_instances.py --lang python > python.jsonl
    python scripts/select_instances.py --ids ID [ID ...] > some.jsonl
    python scripts/select_instances.py --all > all.jsonl

Reads data/swebench_pro_731.jsonl.gz (the exact instance records of the reported runs) and the
batch partition in data/batches.json.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--batch", help="batch name from data/batches.json (e.g. py_batch1, go_batch3, ts_batch1)")
    g.add_argument("--lang", choices=("python", "go", "js", "ts"))
    g.add_argument("--ids", nargs="+")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--dataset", type=Path, default=DATA / "swebench_pro_731.jsonl.gz")
    args = ap.parse_args()
    opener = gzip.open if args.dataset.suffix == ".gz" else open
    rows = [json.loads(l) for l in opener(args.dataset, "rt") if l.strip()]
    if args.batch:
        batches = json.loads((DATA / "batches.json").read_text())
        if args.batch not in batches:
            sys.exit(f"unknown batch {args.batch}; known: {', '.join(sorted(batches))}")
        keep = set(batches[args.batch])
        rows = [r for r in rows if r["instance_id"] in keep]
    elif args.lang:
        rows = [r for r in rows if r.get("repo_language") == args.lang]
    elif args.ids:
        keep = set(args.ids)
        rows = [r for r in rows if r["instance_id"] in keep]
        missing = keep - {r["instance_id"] for r in rows}
        if missing:
            print(f"warning: {len(missing)} ids not in the dataset: {sorted(missing)[:3]}...", file=sys.stderr)
    for r in rows:
        sys.stdout.write(json.dumps(r) + "\n")
    print(f"{len(rows)} instances", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
