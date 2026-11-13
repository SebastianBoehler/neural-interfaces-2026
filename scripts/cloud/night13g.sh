#!/bin/bash
# 13 Nov 03:04 UTC deep queue (ledger s145): replication sets B and C, two-family eye-net, CONFHALF labels.
set -euo pipefail
W=scripts/cloud; R="python -I studies/13nov/ttt_reduce_score.py"
sc() { DEP="--dependency=afterany:$2" $W/leo_cpu.sh "$1" AIFAC_P02_1636 04:00:00 "$3 > runs/night13/$1.txt 2>&1 || { echo SCORER FAILED >> runs/night13/$1.txt; exit 1; }"; }
F=$($W/leo_array.sh ttt8bc AIFAC_P02_1675 01:30:00 "python -I studies/13nov/ttt8.py" "72-119")
O=$($W/leo_array.sh ocubc AIFAC_P02_1675 01:00:00 "python -I studies/12nov/ocunet.py" "37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71")
M=$($W/leo_array.sh ocumix AIFAC_P02_1636 01:00:00 "python -I studies/12nov/ocunet.py" "6,8,10,12,14,16,18,20,22")
Q=$($W/leo_array.sh restbc AIFAC_P02_1675 01:00:00 "python -I studies/11nov/rest4/rest4.py" "58,64,70,76,82,88,94,100,106,112,118,124")
C=$($W/leo_array.sh confhalf AIFAC_P02_1636 01:00:00 "LAB=confhalf python -I studies/13nov/ttt9.py" "0-47")
echo "ttt8 $F | ocu $O | ocu mix $M | rest $Q | confhalf $C"
CP="rsync -a --ignore-existing runs/diag/ocunet/probs/ runs/ocunet/probs/ && rsync -a --ignore-existing runs/diag/rest4/probs/ runs/rest4/probs/ &&"
sc pkg_setB $F:$O:$Q "$CP FT_DIR=runs/diag/ttt/probs8 FT_PFX=ttt8 FT_SEEDS=18-23 OCU_SEEDS=18,19,20,21,22,23,24,25,26 REST_SEEDS=9,10,11,12,13,14 python -I studies/13nov/pkg.py"
sc pkg_setC $F:$O:$Q "$CP FT_DIR=runs/diag/ttt/probs8 FT_PFX=ttt8 FT_SEEDS=24-29 OCU_SEEDS=27,28,29,30,31,32,33,34,35 REST_SEEDS=15,16,17,18,19,20 python -I studies/13nov/pkg.py"
for S in 0-5 6-11; do sc confhalf_s$S $C "PFX=ttt9confhalf VARS=cum10,fixA FORMS=all6,delta1,delta2 SEEDS=$S $R runs/diag/ttt/probs9_confhalf" >/dev/null; done
