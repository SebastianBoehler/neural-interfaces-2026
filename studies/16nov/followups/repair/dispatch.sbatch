#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-302-dispatch
#SBATCH --output=logs/302-dispatch-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes302_20261116/repair
sha256sum -c chain-src.sha256
sha256sum -c stage-src.sha256
sha256sum -c followup-src.sha256
sha256sum -c repair-src.sha256
sha256sum -c repair-v3-src.sha256
sha256sum -c repair-score.sha256
sha256sum -c repair-dispatch.sha256
sha256sum -c repair-registration.sha256
sha256sum -c runs/lane1_inputs/e05-copy.sha256 >/dev/null
for id in 3114261 3114227 3113408; do
 state=$(sacct -X -n -P -j "$id" --format=State,ExitCode | head -1)
 [[ "$state" == 'COMPLETED|0:0' ]] || { echo "parent $id not verified complete: $state"; exit 1; }
done
grep -qx OK logs/replay-302-3114227.out
grep -q '^### 302\.' runs/lane1_inputs/repair-registration.md
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/16nov/followups/repair/smoke_check.py
count=$(squeue -h -u "$USER" -r | wc -l); running=$(squeue -h -u "$USER" -r -t RUNNING | wc -l)
(( count + 38 <= 272 && running + 2 <= 32 )) || { echo "unsafe queue capacity $count/$running"; exit 1; }
[[ ! -e runs/lane1_inputs/repair302.intent ]] || { echo 'dispatch already attempted'; exit 1; }
( set -o noclobber; printf 'dispatcher %s\n' "$SLURM_JOB_ID" > runs/lane1_inputs/repair302.intent )
wave=$(sbatch --parsable --array=0-7,9-15,17-31,33-39%2 studies/16nov/followups/repair/job_v3.sbatch)
printf 'wave %s\n' "$wave" | tee -a runs/lane1_inputs/repair302.intent
score=$(sbatch --parsable --dependency="afterok:$wave" studies/16nov/followups/repair/score.sbatch)
printf 'score %s\n' "$score" | tee -a runs/lane1_inputs/repair302.intent
