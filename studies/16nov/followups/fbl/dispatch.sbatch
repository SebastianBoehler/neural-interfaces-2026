#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-299-dispatch
#SBATCH --output=logs/299-dispatch-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes299_20261116/fbl
sha256sum -c chain-src.sha256
sha256sum -c stage-src.sha256
sha256sum -c fbl-src.sha256
sha256sum -c fbl-dispatch.sha256
sha256sum -c fbl-registration.sha256
for id in 3114134 3114135 3113408; do
 state=$(sacct -X -n -P -j "$id" --format=State,ExitCode | head -1)
 [[ "$state" == 'COMPLETED|0:0' ]] || { echo "parent $id not verified complete: $state"; exit 1; }
done
grep -qx OK logs/replay-299-3114135.out
grep -q '^### 299\.' runs/lane1_inputs/fbl-registration.md
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/16nov/followups/fbl/smoke_check.py
count=$(squeue -h -u "$USER" -r | wc -l); running=$(squeue -h -u "$USER" -r -t RUNNING | wc -l)
(( count + 31 <= 272 && running + 4 <= 32 )) || { echo "unsafe queue capacity $count/$running"; exit 1; }
[[ ! -e runs/lane1_inputs/fbl299.intent ]] || { echo 'dispatch already attempted'; exit 1; }
( set -o noclobber; printf 'dispatcher %s\n' "$SLURM_JOB_ID" > runs/lane1_inputs/fbl299.intent )
wave=$(sbatch --parsable --array=0-7,9-23,25-31%4 studies/16nov/followups/fbl/job.sbatch)
printf 'wave %s\n' "$wave" | tee -a runs/lane1_inputs/fbl299.intent
score=$(sbatch --parsable --dependency="afterok:$wave" studies/16nov/followups/fbl/score.sbatch)
printf 'score %s\n' "$score" | tee -a runs/lane1_inputs/fbl299.intent
