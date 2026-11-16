#!/bin/bash
# s290 part 2: array of 12 tasks = 4 cells x 3 seeds for one family (env FAMS); task T -> cell T/3 (within0d1, within0d2, within1d1, within1d2), seed T%3
T=$1; C=$((T/3)); export SCHEME=within PSEED=$((C/2)) DRAW=$((C%2+1)) BASE_ONLY=1 PROTOS=all_loso
echo "s290 cell FAMS=$FAMS"; exec bash studies/15nov/s290/task.sh $((T%3))
