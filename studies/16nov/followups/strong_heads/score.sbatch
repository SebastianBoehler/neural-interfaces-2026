#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-s308-score
#SBATCH --output=logs/s308-score-%j.out
#SBATCH --error=logs/s308-score-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export LANE5_MANIFEST_SHA="$(sha256sum chain.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export CUDA_VISIBLE_DEVICES='' S295_MANIFEST="$PWD/runs/s301_native/manifest.json"
export S295_MANIFEST_SHA="$(sha256sum "$S295_MANIFEST" | cut -d' ' -f1)"
unset PYTHONPATH
sha256sum -c s295-source.sha256 >/dev/null
sha256sum -c chain.sha256 >/dev/null
test "$(sha256sum chain.sha256 | cut -d' ' -f1)" = "$S308_CHAIN_SHA"
srun --cpu-bind=cores /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/strong_heads/score_308.py
