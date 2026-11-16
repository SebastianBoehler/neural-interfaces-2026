#!/bin/bash
# s288: array of 12 tasks = 4 cells x 3 seeds; task T -> cell T/3 (within0d1, within0d2, within1d1, within1d2), seed T%3 (one family: task = seed)
T=$1; C=$((T/3)); export SCHEME=within PSEED=$((C/2)) DRAW=$((C%2+1)) FAMS=tsm BASE_ONLY=1 PROTOS=all_loso
exec bash studies/15nov/tsm288/task.sh $((T%3))
