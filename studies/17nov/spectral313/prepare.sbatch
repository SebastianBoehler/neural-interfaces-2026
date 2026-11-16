#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --job-name=ni26-s313-features
#SBATCH --output=logs/prepare-%j.out
#SBATCH --error=logs/prepare-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source studies/17nov/spectral313/runtime.sh
sha256sum -c source.sha256 >/dev/null
test "$(sha256sum source.sha256 | cut -d' ' -f1)" = "$S313_SOURCE_SHA"
srun --cpu-bind=cores "$PYTHON" -B -I studies/17nov/spectral313/prepare.py
sha256sum -c source.sha256 >/dev/null
