#!/bin/bash
# Teacher retraining (ledger s148): EEGNeX and IFNet trunks, 12 seeds each, the m7r recipe (30 epochs, rotation 0.05),
# held until the corpus jobs end; then the head export (scripts/export_heads.py) into runs/teach/heads_TAG.npz.
# Usage (Leonardo login node, $WORK/ni26): scripts/cloud/leo_teach.sh TAG CORPORA DEPJOBS(colon-separated)
# pretrain.py runs without -I: it imports its neighbours (dg, deep, features) from experiments/.
set -euo pipefail
TAG=$1; CORP=$2; DEPJ=$3; W=scripts/cloud; mkdir -p runs/teach
# pre-flight (13 Nov: the first release started before Schirrmeister existed and all 48 tasks failed on it): a tiny
# serial job checks that every corpus has windows; the arrays start only if it succeeds
PF=$(sbatch --parsable --dependency=afterany:$DEPJ --account=AIFAC_P02_1675 -p lrd_all_serial --cpus-per-task=1 --mem=1G --time=00:05:00 -J ni26-pf-$TAG -o runs/teach/preflight-$TAG-%j.out \
  --wrap "for c in ${CORP//,/ }; do ls data/\$c/windows_*.npz >/dev/null 2>&1 || { echo MISSING \$c; exit 1; }; done; echo ALL CORPORA PRESENT")
DEPJ=$PF; DEPK=afterok
E=$(DEP="--dependency=$DEPK:$DEPJ" $W/leo_array.sh pt-$TAG-nx AIFAC_P02_1675 02:30:00 "python experiments/pretrain.py eegnex ea $TAG $CORP --epochs 30 --rot 0.05 --seed" "0-11")
I=$(DEP="--dependency=$DEPK:$DEPJ" $W/leo_array.sh pt-$TAG-if AIFAC_P02_1675 02:30:00 "python experiments/pretrain.py ifnet ea $TAG $CORP --epochs 30 --rot 0.05 --seed" "0-11")
H=$(DEP="--dependency=afterany:$E:$I" $W/leo_cpu.sh heads-$TAG AIFAC_P02_1636 01:00:00 "python -I scripts/export_heads.py runs/teach/heads_$TAG.npz runs/pretrain/eegnex_ea-$TAG-s*.pt runs/pretrain/ifnet_ea-$TAG-s*.pt")
echo "teacher $TAG: preflight $PF, eegnex $E, ifnet $I, heads $H"
