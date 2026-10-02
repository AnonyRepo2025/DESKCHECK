#!/bin/bash
# run_claude_code.sh <dataset.jsonl> <out_dir> [instance_id ...]
#
# SimAgent on the Claude Code runtime (the Claude Haiku 4.5 runs): the four-call flow
# localize -> repair -> audit -> validate, one `claude -p` conversation per call.
# ARMS selects what runs (default "reason repro"):
#   reason  SimAgent (code-reasoning / execution-simulation steps)          -> <out_dir>/reason
#   repro   reproduction-test baseline on the same runtime and budgets      -> <out_dir>/repro
# Instances of both arms are interleaved in one streaming queue of PAR slots. Completed runs
# (pipeline.json with status Completed) are skipped, so re-running resumes; a second sweep retries
# anything that did not complete.
#
# Env: MODEL (haiku), PAR (8), PYTHON (python), ARMS ("reason repro"), REPAIR_STEPS (200),
#      CCPIPE_AUTH (oauth | apikey), see README for authentication.
set -u
if [ $# -lt 2 ]; then sed -n '2,16p' "$0"; exit 2; fi
DS=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export ROOT DS OUT PY=${PYTHON:-python} MODEL=${MODEL:-haiku} REPAIR_STEPS=${REPAIR_STEPS:-200}
export CCPIPE_AUTH=${CCPIPE_AUTH:-oauth}
ARMS=${ARMS:-reason repro}
for ARM in $ARMS; do mkdir -p "$OUT/$ARM/logs"; done

run_one() {  # "<arm> <iid>"
  ARM=$1; IID=$2; D=$OUT/$ARM
  if grep -q '"status": "Completed"' "$D/$IID/pipeline.json" 2>/dev/null; then return; fi
  echo "$(date '+%m-%d %H:%M') start $ARM $IID"
  (cd "$ROOT" && "$PY" -u -m simagent.claude_code.flow_run --dataset "$DS" --out "$D" --model "$MODEL" \
     --arm "$ARM" "$IID" > "$D/logs/$IID.log" 2>&1)
  echo "$(date '+%m-%d %H:%M') done $ARM $IID rc=$?"
}
export -f run_one

ids() {
  if [ $# -gt 0 ]; then printf '%s\n' "$@"
  else "$PY" -c "import json,sys; [print(json.loads(l)['instance_id']) for l in open(sys.argv[1]) if l.strip()]" "$DS"; fi
}
for pass in 1 2; do
  ids "$@" | while read -r iid; do for ARM in $ARMS; do echo "$ARM $iid"; done; done \
    | xargs -P "${PAR:-8}" -I{} bash -c 'run_one {}' | tee -a "$OUT/driver.log"
done
for ARM in $ARMS; do "$PY" "$ROOT/scripts/collect_preds.py" "$OUT/$ARM" --label "simagent-cc-$ARM-$MODEL"; done
echo "$(date '+%m-%d %H:%M') ALL DONE" >> "$OUT/driver.log"
