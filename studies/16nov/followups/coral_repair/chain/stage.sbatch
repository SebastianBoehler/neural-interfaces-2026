#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-304-stage
#SBATCH --output=/mnt/lustre/work/gehler/gfh527/ni26_s304_coral_chain_20261116/logs/304-stage-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s304_coral_chain_20261116
sha256sum -c chain304-src.sha256
sha256sum -c stage-src.sha256
sha256sum -c coral304-chain-src.sha256
sha256sum -c runs/coral304_inputs/registration.sha256
grep -q '^### 304\.' runs/coral304_inputs/registration.md
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NI26_WS="$PWD"
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/16nov/followups/coral_repair/chain/stage.py "${1:?producer workspace required}"
[[ ! -e runs/coral304_inputs/dag.intent ]] || { echo 'chain dispatch already attempted'; exit 1; }
( set -o noclobber; printf 'stage %s\n' "$SLURM_JOB_ID" > runs/coral304_inputs/dag.intent )
replay=$(sbatch --parsable --account=gehler --partition=cpu-galvani --cpus-per-task=1 --mem=8G --time=00:10:00 --job-name=ni26-304-replay --output=logs/304-replay-%j.out --wrap="export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=; $PY -B -I replay/check.py")
printf 'replay %s\n' "$replay" | tee -a runs/coral304_inputs/dag.intent
smoke=$(sbatch --parsable --time=00:25:00 studies/16nov/followups/coral_repair/chain/job.sbatch smoke)
printf 'smoke %s\n' "$smoke" | tee -a runs/coral304_inputs/dag.intent
post=$(sbatch --parsable --dependency="afterok:$replay:$smoke" studies/16nov/followups/coral_repair/chain/post.sbatch "$replay" "$smoke")
printf 'post %s\n' "$post" | tee -a runs/coral304_inputs/dag.intent
