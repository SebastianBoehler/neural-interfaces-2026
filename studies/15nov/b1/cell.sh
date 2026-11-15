#!/bin/bash
# s279 B1: one array of 24 tasks = 4 cells x (2 families x 3 seeds); task T -> cell T/6 (within0d1, within0d2, within1d1, within1d2), family/seed T%6 (ttt7: task = family + 2*seed)
T=$1; C=$((T/6)); export SCHEME=within PSEED=$((C/2)) DRAW=$((C%2+1)) FAMS=${FAMS:-fb_am,eegn_am} BASE_ONLY=1 PROTOS=all_loso
exec bash studies/15nov/b1/task.sh $((T%6))
