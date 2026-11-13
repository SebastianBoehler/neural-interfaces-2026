#!/bin/bash
# Night of 13 Nov, repair (ledger s135/s137): the fresh-seed sinc tasks (seeds >= 6) died on a missing pretrained trunk
# and showed COMPLETED. Rerun only those tasks with the trunk fix (ttt5.make: trunk seed = seed mod 6), then the
# scorings that failed loudly on the missing files. Usage (Leonardo login node): scripts/cloud/night13d.sh
set -euo pipefail
W=scripts/cloud; R="python -I studies/13nov/ttt_reduce_score.py"; SI="26,30,34,38,42,46,50,54,58,62,66,70"; SJ="26,30,34,38,42,46"
sc() { DEP="--dependency=afterany:$2" $W/leo_cpu.sh "$1" AIFAC_P02_1636 03:00:00 "$3 > runs/night13/$1.txt 2>&1 || { echo SCORER FAILED >> runs/night13/$1.txt; exit 1; }"; }
A=$($W/leo_array.sh ttt8fs AIFAC_P02_1675 01:15:00 "python -I studies/13nov/ttt8.py" "$SI")
B=$($W/leo_array.sh ttt9t2s AIFAC_P02_1675 01:00:00 "LAB=true python -I studies/13nov/ttt9.py" "$SJ")
C=$($W/leo_array.sh ttt9s2s AIFAC_P02_1675 01:00:00 "LAB=stack python -I studies/13nov/ttt9.py" "$SJ")
echo "sinc reruns: ttt8 $A, ttt9 true $B, ttt9 stack $C"
V="cum10,cum20,cum10e5,cum5e5,cum10lr,cum10c2"
sc ttt8_s6-11 $A "PFX=ttt8 VARS=$V FORMS=all6,delta1,delta2 SEEDS=6-11 $R runs/diag/ttt/probs8"
sc ttt8_s12-17 $A "PFX=ttt8 VARS=$V FORMS=all6,delta1,delta2 SEEDS=12-17 $R runs/diag/ttt/probs8"
sc ttt9_true_s6-11 $B "PFX=ttt9true VARS=cum10,fixA FORMS=all6,delta1,delta2 SEEDS=6-11 $R runs/diag/ttt/probs9_true"
sc ttt9_stack_s6-11 $C "PFX=ttt9stack VARS=cum10,fixA FORMS=all6,delta1,delta2 SEEDS=6-11 $R runs/diag/ttt/probs9_stack"
CP="rsync -a --ignore-existing runs/diag/ocunet/probs/ runs/ocunet/probs/ &&"
sc stackf $A "$CP FT_DIR=runs/diag/ttt/probs8 FT_PFX=ttt8 FT_SEEDS=6-11 OCU_SEEDS=3,4,5,6,7,8 STACK_TAG=_fresh SUBSETS=FT,OCU,FT+OCU,FT+REST+OCU+MIX python -I studies/13nov/stack.py"
sc pkg_fresh $A "$CP FT_DIR=runs/diag/ttt/probs8 FT_PFX=ttt8 FT_SEEDS=6-11 OCU_SEEDS=3,4,5,6,7,8 python -I studies/13nov/pkg.py"
