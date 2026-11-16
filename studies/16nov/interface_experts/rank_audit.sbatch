#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-306-rank
#SBATCH --output=logs/rank-%j.out
#SBATCH --error=logs/rank-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export AUDIT_ROOT="$PWD" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export AUDIT_DATA=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_producer_v3_20261116/data/proteus_windows.npz
export AUDIT_PERM=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_producer_v3_20261116/runs/shuffle_perms.npz
: "${AUDIT_MANIFEST_SHA:?}"
test "$(sha256sum audit.sha256 | cut -d' ' -f1)" = "$AUDIT_MANIFEST_SHA"
sha256sum -c audit.sha256 >/dev/null
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I rank_audit.py
sha256sum -c audit.sha256 >/dev/null
