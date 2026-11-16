#!/bin/bash
# §281 returning check: eight pinned orders × two members × six seeds.
# The task driver writes ttt18<order>_<family>_base-seen_session-s<seed>.npz.
# A plain-fb control may use FAMS=fb with the first six tasks of any order.
set -euo pipefail
[[ $# -eq 1 && $1 =~ ^(0|[1-9][0-9]*)$ ]] || { echo "expected one task index from 0 to 95" >&2; exit 2; }
T=$1
(( T < 96 )) || { echo "task index exceeds 95" >&2; exit 2; }
[[ -z ${DRAW:-} ]] || { echo "DRAW must be unset or empty (chain bases use draw 0)" >&2; exit 3; }
FAMS=${FAMS:-fb_coral,fb_am}
[[ $FAMS == fb_coral,fb_am || $FAMS == fb ]] || { echo "unsupported returning families: $FAMS" >&2; exit 4; }
if [[ $FAMS == fb ]] && (( T % 12 >= 6 )); then
    echo "plain-fb control uses the first six tasks per order" >&2; exit 4
fi
ORDERS=(full0 full1 runs0 runs1 within0 within1 within2 within3)
O=${ORDERS[$((T / 12))]}
export SCHEME=${O%%[0-9]*} PSEED=${O##*[a-z]} FAMS BASE_ONLY=1 PROTOS=seen_session
echo "order $O SCHEME=$SCHEME PSEED=$PSEED task $((T % 12))"
exec bash studies/15nov/s281/task.sh $((T % 12))
