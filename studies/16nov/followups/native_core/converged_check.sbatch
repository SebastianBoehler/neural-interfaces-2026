#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-s305-check
#SBATCH --output=logs/converged-check-%j.out
#SBATCH --error=logs/converged-check-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export LANE5_MANIFEST_SHA="$(sha256sum converged.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export CUDA_VISIBLE_DEVICES='' S305_PRODUCER=/mnt/lustre/work/gehler/gfh527/ni26_s305_converged_v2_20261116 S305_SMOKE_GATE="$PWD/runs/s305_gates/P01.json"
unset PYTHONPATH
sha256sum -c producer.sha256 >/dev/null
sha256sum -c converged.sha256
sha256sum -c assemble.sha256
srun --cpu-bind=cores /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/native_core/converged_check.py
