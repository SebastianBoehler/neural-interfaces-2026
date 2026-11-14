#!/bin/bash
# s221 scoring: shufcombo per order and arm (new people, package gate, eye net re-streamed), the ftrel arms of tag
# ttt18<order>RP and the s15 rows of RF and KF as references; then summary.py. Run through scripts/run_capped.sh.
# Usage (repo root): studies/14nov/ftrelabel/score.sh   -> runs/s221/<order>_<tag>_<arm>.log, runs/s221_summary.txt
set -uo pipefail
mkdir -p runs/s221
for o in ${ORDERS:-within0 within1 within2 within3 full0 full1}; do
  S=${o%?}; P=${o: -1}
  for ta in RP:s15 RP:r15 RP:e15 RP:b15 RP:q15 RF:s15 KF:s15; do
    t=${ta%%:*}; v=${ta##*:}; log=runs/s221/${o}_${t}_${v}.log
    [ -s "$log" ] && grep -q "both shuffled" "$log" && continue
    SCHEME=$S PSEED=$P OCU_DIR=runs/ocushuf/$o GATE_P0=1 REC_PFX=ttt7 T6_DIR=runs/ttt/probs7 T17_DIR=runs/ttt/probs18 \
      FT_TAG=ttt18$o$t FT_V=$v .venv/bin/python studies/13nov/shufcombo.py > "$log" 2>&1 || echo "FAILED $log"
  done
done
.venv/bin/python studies/14nov/ftrelabel/summary.py | tee runs/s221_summary.txt
