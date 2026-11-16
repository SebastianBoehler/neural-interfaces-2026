#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-304-release
#SBATCH --output=logs/304-release-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s304_coral_chain_20261116
sha256sum -c chain304-src.sha256
sha256sum -c stage-src.sha256
sha256sum -c coral304-chain-src.sha256
sha256sum -c runs/coral304_inputs/registration.sha256
for id in "${1:?replay}" "${2:?smoke}"; do
 [[ "$(sacct -X -n -P -j "$id" -o State,ExitCode | head -1)" == 'COMPLETED|0:0' ]] || { echo "failed parent $id"; exit 1; }
done
grep -qx OK "logs/304-replay-$1.out"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/16nov/followups/coral_repair/chain/smoke_check.py
count=$(squeue -h -u "$USER" -r | wc -l); running=$(squeue -h -u "$USER" -r -t RUNNING | wc -l)
(( count + 70 <= 272 && running + 4 <= 32 )) || { echo "unsafe queue capacity $count/$running"; exit 1; }
[[ ! -e runs/coral304_inputs/wave.intent ]] || { echo 'wave already attempted'; exit 1; }
( set -o noclobber; printf 'release %s\n' "$SLURM_JOB_ID" > runs/coral304_inputs/wave.intent )
wave=$(sbatch --parsable --array=1-39,41-63,65-71%4 studies/16nov/followups/coral_repair/chain/job.sbatch)
printf 'wave %s\n' "$wave" | tee -a runs/coral304_inputs/wave.intent
score=$(sbatch --parsable --dependency="afterok:$wave" studies/16nov/followups/coral_repair/chain/score.sbatch)
printf 'score %s\n' "$score" | tee -a runs/coral304_inputs/wave.intent
