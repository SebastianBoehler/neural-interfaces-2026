#!/bin/bash
# s235 upper-bound sweeps: one Leonardo CPU job per (order, arm group); each job runs shufcombo once per arm, sequentially.
# Run from the Leonardo repo root (/leonardo_work/AIFAC_P02_1636/ni26). Usage:
#   studies/14nov/sweeps/submit.sh WAVE ORDERS...      WAVE: check | w1 | <name> with ARMS="K=v,K=v;K=v ..." in the env
# Arms are ';'-separated, each a ','-separated list of knob assignments ('-' = no knobs, the reference); '@O@' in an arm or
# in PRE (a command run first in every job, e.g. building an EXTRA_VIEW file) is replaced by the order.
# Output: runs/shufcombo_<order>_..._ttt18<order>LF<knob suffix>.npz; logs runs/diag/cpu/s235_<wave>_<order>_<g>-<job>.out
set -euo pipefail
WAVE=$1; shift
ACC=${ACC:-AIFAC_P02_1636}
case $WAVE in
  check) GROUPS_=("-;EM_LAM=0.5,TS_LAM=0.15,BTS=4") ;;
  w1) GROUPS_=("EYEW=0.5;EYEW=1.5;EYEW=2" "REST_W=0.5;REST_W=2;REST_W=3" "EM_LAM=0.3;EM_LAM=0.7;EM_LAM=1.0"
               "TS_LAM=0.1;TS_LAM=0.2;TS_LAM=0.3" "BTS=2;BTS=8") ;;
  *) IFS='|' read -r -a GROUPS_ <<< "${ARMS:?ARMS='a;b|c;d' groups for a custom wave}" ;;
esac
for o in "$@"; do
  s=${o%%[0-9]*}; p=${o#$s}
  BASE="SCHEME=$s PSEED=$p STRICT=1 GATE_P0=1 OCU_DIR=runs/diag/ocushuf/$o RS_DIR=runs/diag/restshuf FT_V=s15 REC_PFX=ttt7 T6_DIR=runs/diag/ttt/probs7 T17_DIR=runs/diag/ttt/probs18 FT_TAG=ttt18${o}LF"
  g=0
  for grp in "${GROUPS_[@]}"; do
    CMD="${PRE:+${PRE//@O@/$o} || exit 1; }"
    IFS=';' read -r -a arms <<< "$grp"
    for a in "${arms[@]}"; do
      kv=$([ "$a" = "-" ] && echo "" || echo "${a//,/ }"); kv=${kv//@O@/$o}
      CMD+="echo ARM $o [$a]; env $BASE $kv python studies/13nov/shufcombo.py || exit 1; "
    done
    scripts/cloud/leo_cpu.sh "s235_${WAVE}_${o}_$g" "$ACC" 00:40:00 "${CMD%; }"
    g=$((g + 1))
  done
done
