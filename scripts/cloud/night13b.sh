#!/bin/bash
# Night of 13 Nov, wave 2 (ledger s137 replication): TTT9 on fresh seeds 6-11, held until wave 1's TTT9 arrays end.
# Usage (Leonardo login node, $WORK/ni26): scripts/cloud/night13b.sh T9T_JOB T9S_JOB
set -euo pipefail
W=scripts/cloud; R="python -I studies/13nov/ttt_reduce_score.py"
sc() { DEP="--dependency=afterany:$2" $W/leo_cpu.sh "$1" AIFAC_P02_1636 03:00:00 "$3 > runs/night13/$1.txt 2>&1 || { echo SCORER FAILED >> runs/night13/$1.txt; exit 1; }"; }
A=$(DEP="--dependency=afterany:$1" $W/leo_array.sh ttt9t2 AIFAC_P02_1675 01:00:00 "LAB=true python -I studies/13nov/ttt9.py" "24-47")
B=$(DEP="--dependency=afterany:$2" $W/leo_array.sh ttt9s2 AIFAC_P02_1675 01:00:00 "LAB=stack python -I studies/13nov/ttt9.py" "24-47")
echo "wave 2 arrays $A $B"
sc ttt9_true_s6-11 $A "PFX=ttt9true VARS=cum10,fixA FORMS=all6,delta1,delta2 SEEDS=6-11 $R runs/diag/ttt/probs9_true"
sc ttt9_stack_s6-11 $B "PFX=ttt9stack VARS=cum10,fixA FORMS=all6,delta1,delta2 SEEDS=6-11 $R runs/diag/ttt/probs9_stack"
