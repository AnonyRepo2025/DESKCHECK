#!/usr/bin/env python3
"""Rebuild <run_dir>/preds.json from every <run_dir>/<iid>/model_patch.diff.

Usage: python scripts/collect_preds.py RUN_DIR [--label NAME]
"""
import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--label", default="simagent")
    args = ap.parse_args()
    preds = {}
    for diff in sorted(args.run.glob("*/model_patch.diff")):
        iid = diff.parent.name
        preds[iid] = {"model_name_or_path": args.label, "instance_id": iid, "model_patch": diff.read_text()}
    (args.run / "preds.json").write_text(json.dumps(preds, indent=1))
    print(f"{args.run}: {len(preds)} predictions "
          f"({sum(1 for p in preds.values() if p['model_patch'].strip())} non-empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
