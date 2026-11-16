#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --job-name=ni26-296-dispatch
#SBATCH --output=logs/296-dispatch-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes296_20261116/gated
sha256sum -c gated-src.sha256
sha256sum -c gated-score.sha256
sha256sum -c gated-dispatch.sha256
for id in 3113678 3113679 3113408; do
  state=$(sacct -X -n -P -j "$id" --format=State,ExitCode | head -1)
  [[ "$state" == 'COMPLETED|0:0' ]] || { echo "parent $id not verified complete: $state"; exit 1; }
done
grep -qx OK logs/replay-296-3113679.out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/16nov/followups/gated/smoke_check.py
count=$(squeue -h -u "$USER" -r | wc -l); running=$(squeue -h -u "$USER" -r -t RUNNING | wc -l)
(( count + 31 <= 272 && running + 4 <= 32 )) || { echo "unsafe queue capacity $count/$running"; exit 1; }
[[ ! -e runs/lane1_inputs/followup296.intent ]] || { echo 'dispatch already attempted'; exit 1; }
( set -o noclobber; printf 'dispatcher %s\n' "$SLURM_JOB_ID" > runs/lane1_inputs/followup296.intent )
wave=$(sbatch --parsable --array=1-23,25-31%4 studies/16nov/followups/gated/job.sbatch)
printf 'wave %s\n' "$wave" | tee -a runs/lane1_inputs/followup296.intent
score=$(sbatch --parsable --dependency="afterok:$wave" studies/16nov/followups/gated/score.sbatch)
printf 'score %s\n' "$score" | tee -a runs/lane1_inputs/followup296.intent
