#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-s305-stage-recovery
#SBATCH --output=logs/converged-stage-recovery-%j.out
#SBATCH --error=logs/converged-stage-recovery-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export LANE5_MANIFEST_SHA="$(sha256sum chain.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export CUDA_VISIBLE_DEVICES='' S301_PRODUCER_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_s305_assembler_recover_v1_20261116
: "${S301_PRODUCER_SHA:?converged source lock required}"
unset PYTHONPATH
sha256sum -c s295-source.sha256 >/dev/null
sha256sum -c chain.sha256 >/dev/null
srun --cpu-bind=cores /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I replay/check.py
srun --cpu-bind=cores /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/native_core/converged_stage.py
