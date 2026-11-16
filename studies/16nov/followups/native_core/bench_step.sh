#!/usr/bin/env bash
set -euo pipefail
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/16nov/followups/native_core/bench_infer.py
"$PY" -B -I studies/16nov/followups/native_core/bench_train.py
