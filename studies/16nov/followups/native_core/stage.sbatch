#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-s301-stage
#SBATCH --output=logs/stage-%j.out
#SBATCH --error=logs/stage-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export LANE5_MANIFEST_SHA="$(sha256sum chain.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export CUDA_VISIBLE_DEVICES=''
export S301_PRODUCER_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_producer_v3_20261116
: "${S301_PRODUCER_SHA:?producer lock required}"
unset PYTHONPATH
sha256sum -c s295-source.sha256 >/dev/null
sha256sum -c chain.sha256
srun --cpu-bind=cores /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/native_core/stage.py
