#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
L=.weco/session-stage/eval.log; : > "$L"
NI26_MAX_GB=6 OMP_NUM_THREADS=6 scripts/run_capped.sh "$L" .venv/bin/python .weco/session-stage/evaluate.py
grep -E "g1_new_ba|ERROR" "$L"
