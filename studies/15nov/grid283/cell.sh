#!/bin/bash
# s283/s286/s287 grid: array over 4 cells x (len(FAMS) families x 3 seeds); task T -> cell T/(3n) (within0d1, within0d2, within1d1, within1d2), ttt7 task T%(3n) = family + n*seed. FAMS = comma list of ttt5.ARM names.
T=$1; N=$(echo $FAMS | tr ',' '\n' | wc -l | tr -d ' '); P=$((3*N)); C=$((T/P)); export SCHEME=within PSEED=$((C/2)) DRAW=$((C%2+1)) BASE_ONLY=1 PROTOS=all_loso
echo "grid cell $C FAMS=$FAMS ($N families)"
exec bash studies/15nov/grid283/task.sh $((T%P))
