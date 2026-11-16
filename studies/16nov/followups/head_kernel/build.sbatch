#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-s311-build
#SBATCH --output=logs/build-%j.out
#SBATCH --error=logs/build-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source studies/16nov/followups/head_kernel/runtime.sh
sha256sum -c source.sha256 >/dev/null
test "$(sha256sum source.sha256 | cut -d' ' -f1)" = "$S311_SOURCE_SHA"
export S311_BUILD_DIR="$PWD/runs/build"
srun --cpu-bind=cores bash studies/16nov/followups/head_kernel/build.sh
sha256sum "$S311_BUILD_DIR/libheadkernels.so" > "$S311_BUILD_DIR/library.sha256"
