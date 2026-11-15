#!/bin/bash
# s281 step 1: base posteriors of fb_coral and fb_am (p 0.5) for the 8 chain orders, DRAW 0 (the chain's base files ttt18<order>_<fam>_base-all_loso-s<k>.npz).
# 96 tasks = 8 orders x (2 families x 6 seeds); task T -> order T/12, family/seed T%12 (ttt7: task = family + 2*seed). FAMS=fb is the bit-equal control (tasks 0-11 of one order).
T=$1; ORDERS=(full0 full1 runs0 runs1 within0 within1 within2 within3); O=${ORDERS[$((T/12))]}
export SCHEME=${O%%[0-9]*} PSEED=${O##*[a-z]} FAMS=${FAMS:-fb_coral,fb_am} BASE_ONLY=1 PROTOS=all_loso
[ -z "${DRAW:-}" ] || { echo "DRAW must be unset (chain bases are draw 0)"; exit 3; }
echo "order $O SCHEME=$SCHEME PSEED=$PSEED task $((T%12))"
exec bash studies/15nov/s281/task.sh $((T%12))
