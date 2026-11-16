#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-s311-blas-build
#SBATCH --output=logs/build-%j.out
#SBATCH --error=logs/build-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source studies/16nov/followups/head_kernel/blas_runtime.sh
sha256sum -c source.sha256 >/dev/null
test "$(sha256sum source.sha256 | cut -d' ' -f1)" = "$S311_BLAS_SOURCE_SHA"
export S311_BLAS_BUILD="$PWD/runs/build"
srun --cpu-bind=cores bash studies/16nov/followups/head_kernel/blas_build.sh
sha256sum "$S311_BLAS_LIB" > "$S311_BLAS_BUILD/library.sha256"
