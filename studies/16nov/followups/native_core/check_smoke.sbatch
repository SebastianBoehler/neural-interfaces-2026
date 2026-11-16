#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-s301-smokecheck
#SBATCH --output=logs/smokecheck-%j.out
#SBATCH --error=logs/smokecheck-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export LANE5_MANIFEST_SHA="$(sha256sum producer.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export CUDA_VISIBLE_DEVICES=''
unset PYTHONPATH
sha256sum -c producer.sha256
srun --cpu-bind=cores /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/native_core/check_smoke.py "${1:?prepare or fit}"
