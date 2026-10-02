#!/usr/bin/env python3
"""Incremental grader: grade each instance as soon as its pipeline run lands.

Polls RUN_DIR for finished instances (pipeline.json present) that have no eval output yet,
grades each with the official Pro harness (one harness process per instance, up to
--parallel at a time), appends a line to RUN_DIR/eval/incremental.jsonl and rewrites
RUN_DIR/eval/RESULTS.md after every grade. Exits when RUN_DIR/driver.log says ALL DONE and
nothing is pending.

Usage: python -m simagent.grade_watch --dataset test.jsonl --run RUN_DIR [--prefix a4] [--parallel 2]
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

PRO = Path(os.environ.get("SWEBENCH_PRO_DIR", "SWE-bench_Pro-os")).resolve()  # official SWE-bench Pro harness checkout
PY = sys.executable


def grade_one(row: dict, run: Path, prefix: str) -> dict:
    iid = row["instance_id"]
    eval_dir = run / "eval"
    patch = (run / iid / "model_patch.diff").read_text() if (run / iid / "model_patch.diff").exists() else ""
    rec = {"instance_id": iid, "difficulty": row.get("difficulty"), "repo": row.get("repo"),
           "patch_chars": len(patch), "graded_at": time.strftime("%Y-%m-%d %H:%M")}
    if not patch.strip():
        rec.update(resolved=False, detail="empty patch")
        return rec
    pdir = eval_dir / "_patches"
    pdir.mkdir(parents=True, exist_ok=True)
    pfile = pdir / f"{iid}.json"
    pfile.write_text(json.dumps([{"instance_id": iid, "patch": patch, "prefix": prefix}]))
    cmd = [PY, str(PRO / "swe_bench_pro_eval.py"), "--raw_sample_path", str(run / "eval" / "dataset.jsonl"),
           "--patch_path", str(pfile), "--output_dir", str(eval_dir), "--dockerhub_username", "jefzda",
           "--scripts_dir", str(PRO / "run_scripts"), "--use_local_docker", "--num_workers", "1"]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=PRO, capture_output=True, text=True)
    rec["harness_rc"] = p.returncode
    rec["harness_secs"] = round(time.time() - t0)
    out_file = eval_dir / iid / f"{prefix}_output.json"
    if not out_file.exists():
        rec.update(resolved=False, detail="no_output", stderr=p.stderr[-500:])
        return rec
    try:
        output = json.loads(out_file.read_text())
        passed = {t["name"] for t in output["tests"] if t["status"] == "PASSED"}
        f2p = set(ast.literal_eval(row["fail_to_pass"]))
        p2p = set(ast.literal_eval(row["pass_to_pass"]))
        rec.update(resolved=bool(f2p) and (f2p | p2p) <= passed,
                   detail=f"f2p {len(f2p & passed)}/{len(f2p)} p2p_regressed {len(p2p - passed)}/{len(p2p)}")
    except Exception as e:  # noqa: BLE001
        rec.update(resolved=False, detail=f"parse_error:{e}")
    return rec


def write_results(run: Path, rows_by_id: dict) -> None:
    inc = run / "eval" / "incremental.jsonl"
    recs = {}
    if inc.exists():
        for l in inc.open():
            if l.strip():
                r = json.loads(l)
                recs[r["instance_id"]] = r
    by_diff = defaultdict(lambda: [0, 0])
    for r in recs.values():
        by_diff[r.get("difficulty") or "?"][0] += int(bool(r.get("resolved")))
        by_diff[r.get("difficulty") or "?"][1] += 1
    n_ok = sum(1 for r in recs.values() if r.get("resolved"))
    total = len(rows_by_id)
    lines = [f"# {run.name}: {n_ok}/{len(recs)} resolved so far ({len(recs)}/{total} graded, "
             f"updated {time.strftime('%Y-%m-%d %H:%M')})", "",
             "| difficulty | resolved | graded | cohort |", "|---|---|---|---|"]
    cohort = defaultdict(int)
    for r in rows_by_id.values():
        cohort[r.get("difficulty") or "?"] += 1
    for d in sorted(cohort):
        a, b = by_diff.get(d, [0, 0])
        lines.append(f"| {d} | {a} | {b} | {cohort[d]} |")
    lines += ["", "| instance | difficulty | resolved | detail |", "|---|---|---|---|"]
    for r in sorted(recs.values(), key=lambda x: x["graded_at"]):
        lines.append(f"| {r['instance_id'][9:52]} | {r.get('difficulty')} | {'yes' if r.get('resolved') else 'no'} | {r.get('detail')} |")
    (run / "eval" / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (run / "eval" / "results.json").write_text(json.dumps(
        {k: {"resolved": v.get("resolved"), "detail": v.get("detail"), "difficulty": v.get("difficulty")} for k, v in recs.items()},
        indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--prefix", default="a4")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args(argv)
    rows = {json.loads(l)["instance_id"]: json.loads(l) for l in args.dataset.open() if l.strip()}
    eval_dir = args.run / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "dataset.jsonl").write_text(args.dataset.read_text())
    inc = eval_dir / "incremental.jsonl"
    log = (eval_dir / "grade_watch.log").open("a")

    def _log(m):
        log.write(f"{time.strftime('%H:%M:%S')} {m}\n"); log.flush()

    _log(f"watching {args.run} ({len(rows)} instances, parallel={args.parallel})")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        while True:
            graded = set()
            if inc.exists():
                graded = {json.loads(l)["instance_id"] for l in inc.open() if l.strip()}
            pending = [iid for iid in rows if iid not in graded and (args.run / iid / "pipeline.json").exists()]
            if pending:
                _log(f"grading {len(pending)}: {[p[9:40] for p in pending]}")
                for rec in ex.map(lambda i: grade_one(rows[i], args.run, args.prefix), pending):
                    with inc.open("a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    write_results(args.run, rows)
                    _log(f"{rec['instance_id'][9:40]} resolved={rec.get('resolved')} {rec.get('detail')} ({rec.get('harness_secs')}s)")
            drv = args.run / "driver.log"
            all_done = drv.exists() and "ALL DONE" in drv.read_text()
            if all_done and not pending and len(graded) >= sum(1 for iid in rows if (args.run / iid / "pipeline.json").exists()):
                _log("driver ALL DONE and nothing pending -- exiting")
                write_results(args.run, rows)
                return 0
            time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
