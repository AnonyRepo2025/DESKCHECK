#!/usr/bin/env bash
# Run code-reasoning detection (--no-top-level-only, --min-score 1) over all 14
# trajectory sets and write $OUT/<prefix>_allreasoning.{json,report.txt}.
#
# Script per format:
#   mini-SWE-agent / SWE-agent chat trajs -> find_code_tokens_in_prose.py
#   iCAT-Agent .traj                      -> find_code_tokens_in_prose_icat.py
#   OpenHands JSONL / Sonar JSON          -> find_code_tokens_in_prose_multi.py
#   Claude Code (Pro) / ExpeRepair (Lite) -> find_code_tokens_in_prose_multi.py
#
# Env: TRAJ_ROOT (trajectory sets), OUT (output dir), PARALLEL (default 4),
#      ONLY (space-separated prefixes to run, e.g. ONLY="mini_sbv_gpt-5-2-high").
set -uo pipefail
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
RQ1="$(dirname "$SCRIPTS")"
DATA="${TRAJ_ROOT:-empirical_study_trajectory}"
IN="$RQ1/data/inputs"
OUT="${OUT:-$RQ1/results/execution_annotation}"
GOLD="$RQ1/data/gold_patch_file_analysis.csv"
PARALLEL="${PARALLEL:-4}"
ONLY="${ONLY:-}"
mkdir -p "$OUT"
cd "$SCRIPTS"

COMMON="--no-top-level-only --min-score 1"

jobs_file=$(mktemp)
job() {  # prefix, command...
  local prefix=$1; shift
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $prefix "* ]]; then return; fi
  printf '%s\t%s\n' "$prefix" "$* --out $OUT/${prefix}_allreasoning.json" >> "$jobs_file"
}

# --- mini-SWE-agent, SWE-bench Verified (per_instance_details.json; official difficulty)
for m in claude-4-5-opus-high gpt-5-2-high minimax-2-5-high; do
  job mini_sbv_$m python3 find_code_tokens_in_prose.py \
    --trajs-dir $DATA/minisweagent_${m}_verified/trajs \
    --resolutions $IN/minisweagent_${m}_verified.resolutions.json $COMMON
done
# --- SWE-agent-style .traj, SWE-bench Pro (eval_results.json; gold-patch file-count difficulty)
for m in gpt-5 claude-opus-4; do
  job mini_sbp_$m python3 find_code_tokens_in_prose.py \
    --trajs-dir $DATA/minisweagent_${m}_pro/traj \
    --resolutions $IN/minisweagent_${m}_pro.resolutions.json \
    --difficulty-dataset none --difficulty-json $IN/pro_difficulty.json $COMMON
done
# --- iCAT-Agent
for m in gpt-5.4 sonnet-4.5; do
  job icat_sbp_$m python3 find_code_tokens_in_prose_icat.py \
    --trajs-dir $DATA/icatAgent_${m}_pro \
    --resolutions $IN/icatAgent_${m}_pro.resolutions.csv \
    --difficulty-file-counts $GOLD $COMMON
done
job icat_sbv_minimax-2.5 python3 find_code_tokens_in_prose_icat.py \
  --trajs-dir $DATA/icatAgent_minimax-2.5_verified/trajs \
  --resolutions $IN/icatAgent_minimax-2.5_verified.resolutions.csv $COMMON
# --- OpenHands
for m in GPT-5.5 claude-opus-4-5 MiniMax-M2.5; do
  job openhands_sbv_$(echo $m | tr 'A-Z' 'a-z') python3 find_code_tokens_in_prose_multi.py --format openhands \
    --trajs-dir $DATA/openhands_${m}_verified/run \
    --resolutions $IN/openhands_${m}_verified.resolutions.json $COMMON
done
# --- Sonar Foundation Agent
job sonar_sbv_claude-opus-4-5 python3 find_code_tokens_in_prose_multi.py --format sonar \
  --trajs-dir $DATA/sonar-foundation-agent_claude-opus-4-5_verified \
  --resolutions $IN/sonar-foundation-agent_claude-opus-4-5_verified.resolutions.json $COMMON
# --- Claude Code scaffold, SWE-bench Pro (resolution.csv; gold-patch file-count difficulty)
job claudecode_sbp_gpt-5.4 python3 find_code_tokens_in_prose_multi.py --format claudecode \
  --trajs-dir $DATA/claudecode_gpt-5.4_pro/trajs \
  --resolutions $IN/claudecode_gpt-5.4_pro.resolutions.json \
  --difficulty-dataset none --difficulty-json $IN/pro_difficulty.json $COMMON
# --- ExpeRepair timelines, SWE-bench Lite (leaderboard results.json; Verified difficulty for the 93 overlapping ids)
job experepair_sbl_claude-sonnet-4 python3 find_code_tokens_in_prose_multi.py --format experepair \
  --trajs-dir $DATA/experepair_timelines_claude-sonnet-4_lite \
  --resolutions $IN/experepair_timelines_claude-sonnet-4_lite.resolutions.json \
  --difficulty-dataset none --difficulty-json $IN/lite_difficulty.json $COMMON

run_one() {
  local prefix=$1; shift
  local t0=$(date +%s)
  if "$@" > "$OUT/${prefix}_allreasoning.report.txt" 2> "$OUT/${prefix}_allreasoning.stderr.txt"; then
    rm -f "$OUT/${prefix}_allreasoning.stderr.txt"
    echo "OK   $prefix ($(( $(date +%s) - t0 ))s)"
  else
    echo "FAIL $prefix ($(( $(date +%s) - t0 ))s) -- see $OUT/${prefix}_allreasoning.stderr.txt"
  fi
}

while IFS=$'\t' read -r prefix cmd; do
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
  echo "START $prefix"
  run_one "$prefix" $cmd &
done < "$jobs_file"
wait
rm -f "$jobs_file"
echo "Done. Outputs under $OUT"
