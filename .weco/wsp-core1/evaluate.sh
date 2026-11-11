#!/bin/bash
# One capped local job at a time (scripts/run_capped.sh); prints the metric lines for Weco.
cd "$(dirname "$0")/../.." || exit 1
L=.weco/wsp-core1/eval.log; : > "$L"
NI26_MAX_GB=8 OMP_NUM_THREADS=6 scripts/run_capped.sh "$L" .venv/bin/python .weco/wsp-core1/evaluate.py
grep -E "g1_chain_ba|standalone_g1_ba|ERROR" "$L"
