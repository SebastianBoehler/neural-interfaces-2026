#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-s305-assemble
#SBATCH --output=logs/converged-assemble-%j.out
#SBATCH --error=logs/converged-assemble-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export LANE5_MANIFEST_SHA="$(sha256sum converged.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export NI26_ROOT="$PWD" CUDA_VISIBLE_DEVICES=''
export S301_PARITY_REPORT=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_parity_v3_20261116/runs/s301_parity/report.json
export S301_ALL_REFERENCES=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_reference_all_20261116/runs/s301_reference_all
unset PYTHONPATH
sha256sum -c producer.sha256 >/dev/null
sha256sum -c converged.sha256
sha256sum -c assemble.sha256
test -f runs/s305_gates/P01.json
srun --cpu-bind=cores /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/native_core/converged_assemble_checked.py
