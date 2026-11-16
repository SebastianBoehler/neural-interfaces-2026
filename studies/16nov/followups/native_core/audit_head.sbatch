#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=ni26-s301-audit
#SBATCH --output=logs/audit-%j.out
#SBATCH --error=logs/audit-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export LANE5_MANIFEST_SHA="$(sha256sum audit.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export NI26_ROOT="$PWD" CUDA_VISIBLE_DEVICES=''
export NATIVE_CORE_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_full_v3_20261116/runs/s301_full_build/libnative_core.so
export NATIVE_TRAIN_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_trainref_v3_20261116/runs/s301_trainref/libtrainref.so
export S301_ALL_REFERENCES=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_reference_all_20261116/runs/s301_reference_all
unset PYTHONPATH
sha256sum -c source.sha256 >/dev/null
sha256sum -c producer.sha256 >/dev/null
sha256sum -c audit.sha256
(cd /mnt/lustre/work/gehler/gfh527/ni26_s301_native_full_v3_20261116 && sha256sum -c runs/s301_full_build/library.sha256)
srun --cpu-bind=cores /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/native_core/audit_head.py
