#!/bin/bash
# s236: per-window outputs of v6f with one stage constant moved (shufcombo knobs EM_LAM / TS_LAM / BTS), six orders,
# as Galvani CPU jobs (KNOBS=" " skips them). Run on Galvani from the repo root. NOOP=1 adds the shipped-value no-op control on within 0.
set -euo pipefail
ORDERS=${ORDERS:-"within0 within1 full0 full1 runs0 runs1"}
KNOBS=${KNOBS:-"EM_LAM=0.25 EM_LAM=1.0 TS_LAM=0.075 TS_LAM=0.3 BTS=2 BTS=8"}
run() {  # order, tag, env assignments
  local o=$1 tag=$2; shift 2; local sc=${o%[0-9]*} ps=${o##*[a-z]}
  local base="SCHEME=$sc PSEED=$ps STRICT=1 GATE_P0=1 OCU_DIR=runs/ocushuf/$o FT_V=gC5 REC_PFX=ttt7 T6_DIR=runs/ttt/probs7 T17_DIR=runs/ttt/probs18 FT_TAG=ttt18${o}L"
  PART=cpu-galvani scripts/cloud/galv_cpu.sh s236$o$tag 01:00:00 \
    "env SAVE_OUT=runs/chainout/s236_${o}_${tag}.npz $base $* python studies/13nov/shufcombo.py > runs/diag/cpu/s236_${o}_${tag}.txt 2>&1"
}
for o in $ORDERS; do for kv in $KNOBS; do k=${kv%=*}; v=${kv#*=}; run $o "$(echo $k | tr A-Z a-z)$v" "$kv"; done; done
# EYECOPY=1: eye-view weight 2 / 3 as 1 / 2 extra copies of the eye view (eyecopy.py; EYEW scaling is a no-op for w > 0)
[ -n "${EYECOPY:-}" ] && for o in $ORDERS; do for k in 1 2; do run $o eyex$((k + 1)) EXTRA_VIEW=runs/sesstype/eyecopy_k$k.npz; done; done
[ -n "${NOOP:-}" ] && run within0 noop EM_LAM=0.5 TS_LAM=0.15 BTS=4
true
