#!/bin/bash
# s229 scoring: as score.sh with STRICT=1, tag ttt18<order>LP (v5.1b labels), arms s15 (in-run control) and r15; then
# summary229.py. Run through scripts/run_capped.sh from the repo root. -> runs/s229/<order>_<arm>.log, runs/s229_summary.txt
set -uo pipefail
mkdir -p runs/s229
for o in ${ORDERS:-within0 within1 within2 within3 within4 within5 within6 within7 within8 within9 within10 within11 full0 full1 full2 full3 runs0 runs1 runs2 runs3}; do
  S=$(echo $o | sed 's/[0-9]*$//'); P=${o#$S}
  for v in s15 r15; do
    log=runs/s229/${o}_${v}.log; [ -s "$log" ] && grep -q "both shuffled" "$log" && continue
    STRICT=1 SCHEME=$S PSEED=$P OCU_DIR=runs/ocushuf/$o GATE_P0=1 REC_PFX=ttt7 T6_DIR=runs/ttt/probs7 T17_DIR=runs/ttt/probs18 \
      FT_TAG=ttt18${o}LP FT_V=$v .venv/bin/python studies/13nov/shufcombo.py > "$log" 2>&1 || echo "FAILED $log"
  done
done
.venv/bin/python studies/14nov/ftrelabel/summary229.py | tee runs/s229_summary.txt
