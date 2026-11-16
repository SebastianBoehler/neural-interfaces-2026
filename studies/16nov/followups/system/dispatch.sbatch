#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --job-name=ni26-294-dispatch
#SBATCH --output=logs/294-dispatch-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_lanes294_20261116/system
sha256sum -c followup-src.sha256
sha256sum -c followup-score.sha256
sha256sum -c wave-src.sha256
sha256sum -c dispatch-src.sha256
for id in 3113522 3113523 3113408; do
  state=$(sacct -X -n -P -j "$id" --format=State,ExitCode | head -1)
  [[ "$state" == 'COMPLETED|0:0' ]] || { echo "parent $id is not verified complete: $state"; exit 1; }
done
grep -qx OK logs/replay-294-3113523.out
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/16nov/followups/system/smoke_check.py
count=$(squeue -h -u "$USER" -r | wc -l)
running=$(squeue -h -u "$USER" -r -t RUNNING | wc -l)
(( count + 125 <= 272 && running + 8 <= 32 )) || { echo "queue capacity unsafe: count=$count running=$running"; exit 1; }
[[ ! -e runs/lane1_inputs/followup294.intent ]] || { echo 'dispatch already attempted'; exit 1; }
( set -o noclobber; printf 'dispatcher %s\n' "$SLURM_JOB_ID" > runs/lane1_inputs/followup294.intent )
wave=$(sbatch --parsable --array=1-95,97-119,121-127%8 studies/16nov/followups/system/wave.sbatch)
printf 'wave %s\n' "$wave" | tee -a runs/lane1_inputs/followup294.intent
score=$(sbatch --parsable --dependency="afterok:$wave" studies/16nov/followups/system/score.sbatch)
printf 'score %s\n' "$score" | tee -a runs/lane1_inputs/followup294.intent
