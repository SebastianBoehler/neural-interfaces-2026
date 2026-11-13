#!/bin/bash
# Night of 13 Nov, wave 3 (ledger s137 round 3): labels from round 2's fine-tuned stack, then fine-tuning on them.
# Usage (Leonardo login node, $WORK/ni26): scripts/cloud/night13c.sh T9S_JOB
set -euo pipefail
W=scripts/cloud; R="python -I studies/13nov/ttt_reduce_score.py"
L=$(DEP="--dependency=afterok:$1" $W/leo_cpu.sh labels3 AIFAC_P02_1636 01:00:00 "LABEL_OUT=chain_stack2 FT_DIR=runs/diag/ttt/probs9_stack FT_PFX=ttt9stack FT_SEEDS=0-5 OCU_SEEDS=0,1,2,3,4,5,6,7,8 python -I studies/13nov/make_labels.py > runs/night13/labels3.txt 2>&1 || { echo FAILED >> runs/night13/labels3.txt; exit 1; }")
A=$(DEP="--dependency=afterok:$L" $W/leo_array.sh ttt9r3 AIFAC_P02_1675 01:00:00 "LAB=stack2 python -I studies/13nov/ttt9.py" "0-23")
S=$(DEP="--dependency=afterany:$A" $W/leo_cpu.sh ttt9_stack2 AIFAC_P02_1636 03:00:00 "PFX=ttt9stack2 VARS=cum10,fixA FORMS=all6,delta1,delta2 SEEDS=0-5 $R runs/diag/ttt/probs9_stack2 > runs/night13/ttt9_stack2.txt 2>&1 || { echo SCORER FAILED >> runs/night13/ttt9_stack2.txt; exit 1; }")
echo "labels $L, array $A, scoring $S"
