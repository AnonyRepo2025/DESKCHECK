#!/bin/bash
# Offline unit tests (no docker, no model calls). Run from anywhere: tests/run_all.sh
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd); PY=${PYTHON:-python}; rc=0
cd "$ROOT" && "$PY" -m pytest -q tests || rc=1
for t in test_audit_probe_and_trace test_claude_code_js_port; do   # plain-assert check scripts
  PYTHONPATH="$ROOT" "$PY" "tests/$t.py" | tail -1 || rc=1
done
exit $rc
