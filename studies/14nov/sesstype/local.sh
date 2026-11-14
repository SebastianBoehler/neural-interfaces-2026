#!/bin/bash
# s236 on the Mac: one shufcombo job (order, tag, env assignment), the same command as launch.sh; skips existing outputs.
# Usage: studies/14nov/sesstype/local.sh within0 em_lam0.25 EM_LAM=0.25   (run two at a time at most)
set -euo pipefail
o=$1 tag=$2; shift 2; sc=${o%[0-9]*} ps=${o##*[a-z]}; out=runs/chainout/s236_${o}_${tag}.npz
[ -f "$out" ] && { echo "exists $out"; exit 0; }
mkdir -p runs/sesstype/log
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NJ=4 PYTHONWARNINGS=ignore SAVE_OUT=$out SCHEME=$sc PSEED=$ps STRICT=1 GATE_P0=1 \
  OCU_DIR=runs/ocushuf/$o FT_V=gC5 REC_PFX=ttt7 T6_DIR=runs/ttt/probs7 T17_DIR=runs/ttt/probs18 FT_TAG=ttt18${o}L "$@" \
  .venv/bin/python studies/13nov/shufcombo.py > runs/sesstype/log/s236_${o}_${tag}.txt 2>&1
