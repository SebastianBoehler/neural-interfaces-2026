#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:18:00
#SBATCH --job-name=ni26-concurrency368
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
root=/mnt/lustre/work/gehler/gfh527
bench="$root/ni26_native366_exact_ea_v1_20261117"
sha256sum -c source.sha256
(cd "$bench" && sha256sum --quiet -c source.sha256 && sha256sum --quiet -c stress.sha256 && sha256sum --quiet -c inputs.sha256)
python3 measure.py
sha256sum -c source.sha256
