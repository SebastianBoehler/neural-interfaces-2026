#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-s313-whole-score
#SBATCH --output=logs/whole-score-%j.out
#SBATCH --error=logs/whole-score-%j.err
set -euo pipefail
cd "${S313_COLLECT_WS:?}"
source "$S313_FITS/studies/17nov/spectral313/fit_runtime.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export S313_OUT="$S313_FITS/runs/s313" S313_MANIFEST="$S313_FITS/source.sha256"
export S313_COLLECT_MANIFEST="$PWD/source.sha256" S313_COLLECT_OUT="$PWD/runs/s313_collect"
export S313_REFERENCE_CHAIN="$S313_CHAIN"
sha256sum -c source.sha256 >/dev/null
test "$(sha256sum source.sha256 | cut -d' ' -f1)" = "$S313_COLLECT_SHA"
( cd "$S313_CHAIN" && sha256sum -c s295-source.sha256 >/dev/null && sha256sum -c chain.sha256 >/dev/null )
test "$(sha256sum "$S313_CHAIN/chain.sha256" | cut -d' ' -f1)" = "$S313_CHAIN_SHA"
srun --cpu-bind=cores "$PYTHON" -B -I studies/17nov/spectral313/score_whole.py
