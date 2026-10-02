#!/bin/bash
# run_mini.sh <m3|luna|CONFIG.yaml> <dataset.jsonl> <out_dir> [instance_id ...]
#
# SimAgent on the mini-swe-agent runtime (the MiniMax-M3 and GPT-5.6-Luna runs).
# One pipeline process per instance, PAR at a time (a streaming work queue: a slot is refilled
# as soon as an instance finishes). Without instance ids, every row of the dataset is run.
# Instances whose <out_dir>/<iid>/pipeline.json already exists are skipped, so re-running resumes.
#
# Output: <out_dir>/<iid>/{pipeline.json, model_patch.diff, <iid>.traj.json, per-phase trajs},
#         <out_dir>/preds.json (all instances), <out_dir>/logs/<iid>.log
#
# Env: PAR (8), PYTHON (python), INSTANCE_COST_CAP (3.0 USD), EXPLORE_STEPS (35), REPAIR_STEPS (200).
set -u
if [ $# -lt 3 ]; then sed -n '2,13p' "$0"; exit 2; fi
MODEL_KEY=$1; DS=$(realpath "$2"); OUT=$(realpath -m "$3"); shift 3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
case $MODEL_KEY in
  m3)   CFG=$ROOT/configs/swebench_pro_m3.yaml ;;
  luna) CFG=$ROOT/configs/swebench_pro_luna.yaml ;;
  *)    CFG=$(realpath "$MODEL_KEY") ;;
esac
export ROOT DS OUT CFG PY=${PYTHON:-python}
export INSTANCE_COST_CAP=${INSTANCE_COST_CAP:-3.0} EXPLORE_STEPS=${EXPLORE_STEPS:-35} REPAIR_STEPS=${REPAIR_STEPS:-200}
mkdir -p "$OUT/logs" "$OUT/_work"

run_one() {
  IID=$1
  if [ -f "$OUT/$IID/pipeline.json" ]; then echo "$(date '+%m-%d %H:%M') skip $IID (done)"; return; fi
  echo "$(date '+%m-%d %H:%M') start $IID"
  (cd "$ROOT" && INSTANCES_JSONL="$DS" MODEL_CONFIG="$CFG" \
     "$PY" -u -m simagent.plainrepair "$IID" --out "$OUT/_work/$IID" > "$OUT/logs/$IID.log" 2>&1)
  rc=$?
  if [ -d "$OUT/_work/$IID/$IID" ]; then rm -rf "$OUT/$IID"; mv "$OUT/_work/$IID/$IID" "$OUT/$IID"; fi
  rm -rf "$OUT/_work/$IID"
  echo "$(date '+%m-%d %H:%M') done $IID rc=$rc"
}
export -f run_one

{ if [ $# -gt 0 ]; then printf '%s\n' "$@"
  else "$PY" -c "import json,sys; [print(json.loads(l)['instance_id']) for l in open(sys.argv[1]) if l.strip()]" "$DS"; fi
} | xargs -P "${PAR:-8}" -I{} bash -c 'run_one {}' | tee -a "$OUT/driver.log"

"$PY" "$ROOT/scripts/collect_preds.py" "$OUT" --label "simagent-$(basename "$CFG" .yaml)"
echo "$(date '+%m-%d %H:%M') ALL DONE" >> "$OUT/driver.log"
