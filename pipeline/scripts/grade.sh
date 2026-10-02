#!/bin/bash
# grade.sh <dataset.jsonl> <run_dir> [num_workers]
#
# Grade <run_dir>/preds.json with the official SWE-bench Pro harness (local docker).
# Resolved = every FAIL_TO_PASS and PASS_TO_PASS test passes.
# Writes <run_dir>/eval/{results.json, RESULTS.md}.
# Env: SWEBENCH_PRO_DIR (checkout of scaleapi/SWE-bench_Pro-os), PYTHON (python).
set -eu
if [ $# -lt 2 ]; then sed -n '2,8p' "$0"; exit 2; fi
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DS=$(realpath "$1"); RUN=$(realpath "$2")
cd "$ROOT" && "${PYTHON:-python}" -m simagent.grade --dataset "$DS" --run "$RUN" --num-workers "${3:-6}" --prefix simagent
