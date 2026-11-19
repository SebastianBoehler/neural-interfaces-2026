#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH -t 01:20:00
#SBATCH -J ni26-laneC-reve
#SBATCH -o reve-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum --quiet -c laneC.sha256
root=$(cd ../neural-interfaces-2026 && pwd -P)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 MNE_DONTWRITE_HOME=true
blob="$HOME/.cache/huggingface/hub/models--brain-bzh--reve-base/snapshots/dc2a075c287bb2f6c04ee5875bd79535a0f7dba6/model.safetensors"
test -s "$HOME/mne_data/reve_positions.json"
exec "$root/.venv/bin/python" -u train.py "$1" "${SLURM_ARRAY_TASK_ID:-0}" --root "$root" --work "$PWD" --checkpoint "$blob"
