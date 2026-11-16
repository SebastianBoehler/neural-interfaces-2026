#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=ni26-s305-featureaudit
#SBATCH --output=logs/feature-audit-%j.out
#SBATCH --error=logs/feature-audit-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export LANE5_MANIFEST_SHA="$(sha256sum feature_audit.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export NI26_ROOT="$PWD" CUDA_VISIBLE_DEVICES=''
export S301_INPUT_PRODUCER=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_producer_v3_20261116
unset PYTHONPATH
sha256sum -c source.sha256 >/dev/null
sha256sum -c producer.sha256 >/dev/null
sha256sum -c feature_audit.sha256
(cd "$S301_INPUT_PRODUCER" && sha256sum -c producer.sha256 >/dev/null)
srun --cpu-bind=cores /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/native_core/audit_all_features.py
