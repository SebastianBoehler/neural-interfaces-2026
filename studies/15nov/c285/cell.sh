#!/bin/bash
# s285: one array of 36 tasks = 4 cells x (3 families x 3 seeds); task T -> cell T/9 (within0d1, within0d2, within1d1, within1d2), family/seed T%9 (ttt7: task = family + 3*seed)
T=$1; C=$((T/9)); export SCHEME=within PSEED=$((C/2)) DRAW=$((C%2+1)) FAMS=${FAMS:-fb_coral,fb_dann,fb_swad} BASE_ONLY=1 PROTOS=all_loso
exec bash studies/15nov/c285/task.sh $((T%9))
