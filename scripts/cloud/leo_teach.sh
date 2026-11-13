#!/bin/bash
# Teacher retraining (ledger s148): EEGNeX and IFNet trunks, 12 seeds each, the m7r recipe (30 epochs, rotation 0.05),
# held until the corpus jobs end; then the head export (scripts/export_heads.py) into runs/teach/heads_TAG.npz.
# Usage (Leonardo login node, $WORK/ni26): scripts/cloud/leo_teach.sh TAG CORPORA DEPJOBS(colon-separated)
# pretrain.py runs without -I: it imports its neighbours (dg, deep, features) from experiments/.
set -euo pipefail
TAG=$1; CORP=$2; DEPJ=$3; W=scripts/cloud; mkdir -p runs/teach
E=$(DEP="--dependency=afterany:$DEPJ" $W/leo_array.sh pt-$TAG-nx AIFAC_P02_1675 02:30:00 "python experiments/pretrain.py eegnex ea $TAG $CORP --epochs 30 --rot 0.05 --seed" "0-11")
I=$(DEP="--dependency=afterany:$DEPJ" $W/leo_array.sh pt-$TAG-if AIFAC_P02_1675 02:30:00 "python experiments/pretrain.py ifnet ea $TAG $CORP --epochs 30 --rot 0.05 --seed" "0-11")
H=$(DEP="--dependency=afterany:$E:$I" $W/leo_cpu.sh heads-$TAG AIFAC_P02_1636 01:00:00 "python -I scripts/export_heads.py runs/teach/heads_$TAG.npz runs/pretrain/eegnex_ea-$TAG-s*.pt runs/pretrain/ifnet_ea-$TAG-s*.pt")
echo "teacher $TAG: eegnex $E, ifnet $I, heads $H"
