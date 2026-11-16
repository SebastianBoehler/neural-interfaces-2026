#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-s310-score
#SBATCH --output=logs/s310-score-%j.out
#SBATCH --error=logs/s310-score-%j.err
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s310_entropy_sign_20261116/chain_v2
export LANE5_MANIFEST_SHA="$(sha256sum chain.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export CUDA_VISIBLE_DEVICES='' S295_MANIFEST="$PWD/runs/s301_native/manifest.json" S295_MANIFEST_SHA="$(sha256sum runs/s301_native/manifest.json | cut -d' ' -f1)"
for f in s295-source.sha256 chain.sha256 s310-chain.sha256; do sha256sum -c "$f" >/dev/null; done
srun --cpu-bind=cores /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/entropy_sign/sign_score.py
