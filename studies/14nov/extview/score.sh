#!/bin/bash
# s233 step 3: one view file through the chain on one order, env exactly as runs/s215_within0_off.log's run.
# Usage (repo root): studies/14nov/extview/score.sh NAME ORDER   e.g. score.sh mivma within0  -> runs/views_ext/chain_NAME_ORDER.log
set -euo pipefail
NAME=$1; O=$2; S=${O%[0-9]}; P=${O#$S}
test -f runs/views_ext/$NAME.npz
EXTRA_VIEW=runs/views_ext/$NAME.npz SCHEME=$S PSEED=$P STRICT=1 GATE_P0=1 OCU_DIR=runs/ocushuf/$O FT_V=gC5 REC_PFX=ttt7 \
  T6_DIR=runs/ttt/probs7 T17_DIR=runs/ttt/probs18 FT_TAG=ttt18${O}L .venv/bin/python studies/13nov/shufcombo.py > runs/views_ext/chain_${NAME}_$O.log 2>&1
test -f runs/shufcombo_${O}_ocu_gC5_all_ge_hb_ts_gp0_ttt18${O}L_xv$NAME.npz
