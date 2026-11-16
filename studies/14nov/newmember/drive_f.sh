#!/usr/bin/env bash
# s289: queue every refit arm x order through run_capped (one job at a time). Usage: studies/14nov/newmember/drive_f.sh
cd "$(dirname "$0")/../../.." || exit 1
ORDERS="within0 within1 within2 within3 full0 full1 runs0 runs1"
arms=("1.0|")
for f in core1 hilb shallow teacher sc_fb sc_fbl sc_sinc; do for x in 0.5 1.5; do arms+=("0.5|$f=$x"); done; done
for a in "${arms[@]}"; do w=${a%%|*}; rf=${a##*|}; nm="nmA_w$w"; [ -n "$rf" ] && nm="${nm}_r$(echo "$rf" | sed 's/=/x/')"
  for o in $ORDERS; do [ -f runs/chainout/nm_${nm}_$o.npz ] && continue
    ARM=A NEWSLOT=eegn NEWW=$w REFIT=$rf NM_NAME=$nm scripts/run_capped.sh runs/newmember/s289_${nm}_$o.log .venv/bin/python studies/14nov/newmember/run_w.py $o; done; done
echo DONE > runs/newmember/s289_done.txt
