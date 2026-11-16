#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-s313-stage
#SBATCH --output=logs/stage-%j.out
#SBATCH --error=logs/stage-%j.err
set -euo pipefail
cd "${S313_CHAIN:?}"
export LANE5_MANIFEST_SHA="$(sha256sum chain.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export CUDA_VISIBLE_DEVICES='' S313_OUT="$S313_FITS/runs/s313" S313_MANIFEST="$S313_FITS/source.sha256"
export S313_COLLECT_MANIFEST="$S313_COLLECT_WS/source.sha256" S313_COLLECT_OUT="$S313_COLLECT_WS/runs/s313_collect"
unset PYTHONPATH
sha256sum -c s295-source.sha256 >/dev/null
sha256sum -c chain.sha256 >/dev/null
test "$(sha256sum chain.sha256 | cut -d' ' -f1)" = "$S313_CHAIN_SHA"
test "$(sha256sum "$S313_COLLECT_MANIFEST" | cut -d' ' -f1)" = "$S313_COLLECT_SHA"
P=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
srun --cpu-bind=cores "$P" -B -I replay/check.py
srun --cpu-bind=cores "$P" -B -I studies/17nov/spectral313/stage_chain.py
