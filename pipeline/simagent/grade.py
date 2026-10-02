#!/usr/bin/env python3
"""Grade a pipeline run directory (either runtime) with the official SWE-bench Pro local-docker harness.

Reads RUN_DIR/preds.json (mini-swe-agent shape), writes RUN_DIR/eval/{results.json,RESULTS.md}.
Same resolve rule as the Luna baseline grader: every fail_to_pass AND pass_to_pass test PASSED.

Usage: python -m simagent.grade --dataset test.jsonl --run RUN_DIR [--num-workers 8] [--redo] [--prefix cc]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PRO = Path(os.environ.get("SWEBENCH_PRO_DIR", "SWE-bench_Pro-os")).resolve()  # official SWE-bench Pro harness checkout


_TESTS_RAN_RE = re.compile(r"^Tests:\s+.*\b(\d+)\s+(?:passed|failed|total)", re.M)


def _log_ran_tests(inst_eval_dir: Path, prefix: str) -> bool:
    """Did the runner actually execute tests, whatever the parser managed to extract?"""
    for name in (f"{prefix}_stdout.log", f"{prefix}_stderr.log"):
        fp = inst_eval_dir / name
        try:
            text = fp.read_text(errors="replace") if fp.exists() else ""
        except OSError:
            continue
        if _TESTS_RAN_RE.search(text) or re.search(r"^Tests:\s+\d+\s+total", text, re.M):
            return True
    return False


def resolved(row: dict, eval_dir: Path, prefix: str) -> tuple[bool, str]:
    out_file = eval_dir / row["instance_id"] / f"{prefix}_output.json"
    if not out_file.exists():
        return False, "no_output"
    try:
        output = json.loads(out_file.read_text())
        passed = {t["name"] for t in output["tests"] if t["status"] == "PASSED"}
        f2p = set(ast.literal_eval(row["fail_to_pass"]))
        p2p = set(ast.literal_eval(row["pass_to_pass"]))
        f2p_ok = len(f2p & passed)
        p2p_bad = len(p2p - passed)
        ok = bool(f2p) and (f2p | p2p) <= passed
        detail = f"f2p {f2p_ok}/{len(f2p)} p2p_regressed {p2p_bad}/{len(p2p)}"
        # The Pro jest parser only walks blocks headed by `PASS`, so a suite with ANY failure emits
        # NO rows at all and every test scores ABSENT: js_batch2 element-web-b007ea81 reads 0/7 while
        # jest printed "1 failed, 32 passed" (true 6/7). The verdict is unaffected -- an unparsed
        # suite is genuinely unresolved -- but the partial counts are, so say so rather than silently
        # recording 0/N.
        if not output["tests"] and _log_ran_tests(out_file.parent, prefix):
            detail += " [PARSER DROPPED ALL ROWS: the runner log reports executed tests; partial counts unreliable]"
        return ok, detail
    except Exception as e:  # noqa: BLE001
        return False, f"parse_error:{e}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--prefix", default="cc")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--only-with-patch", action="store_true", help="skip instances whose patch is empty")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in args.dataset.open() if l.strip()]
    preds = json.loads((args.run / "preds.json").read_text()) if (args.run / "preds.json").exists() else {}
    rows = [r for r in rows if r["instance_id"] in preds]
    eval_dir = args.run / "eval"
    eval_dir.mkdir(exist_ok=True)
    patches = [{"instance_id": r["instance_id"],
                "patch": preds[r["instance_id"]].get("model_patch", "") or "",
                "prefix": args.prefix} for r in rows]
    if args.only_with_patch:
        patches = [p for p in patches if p["patch"].strip()]
    (eval_dir / "patches.json").write_text(json.dumps(patches, indent=1))
    cmd = [sys.executable, str(PRO / "swe_bench_pro_eval.py"),
           "--raw_sample_path", str(args.dataset),
           "--patch_path", str(eval_dir / "patches.json"), "--output_dir", str(eval_dir),
           "--dockerhub_username", "jefzda",
           "--scripts_dir", str(PRO / "run_scripts"), "--num_workers", str(args.num_workers),
           "--use_local_docker"]
    if args.redo:
        cmd.append("--redo")
    print("[grade]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PRO, check=False)

    res, by_diff = {}, defaultdict(lambda: [0, 0])
    for r in rows:
        ok, detail = resolved(r, eval_dir, args.prefix)
        res[r["instance_id"]] = {"resolved": ok, "detail": detail, "difficulty": r.get("difficulty"),
                                 "lang": r.get("repo_language")}
        by_diff[r.get("difficulty", "?")][0] += int(ok)
        by_diff[r.get("difficulty", "?")][1] += 1
    (eval_dir / "results.json").write_text(json.dumps(res, indent=1))
    n_ok = sum(v["resolved"] for v in res.values())
    lines = [f"# {args.run.name}: {n_ok}/{len(res)} resolved", "", "| difficulty | resolved | n |", "|---|---|---|"]
    lines += [f"| {d} | {a} | {b} |" for d, (a, b) in sorted(by_diff.items())]
    (eval_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
