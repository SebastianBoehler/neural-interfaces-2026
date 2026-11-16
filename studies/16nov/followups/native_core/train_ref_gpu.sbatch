#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-s301-trainrefgpu
#SBATCH --output=logs/trainref-gpu-%j.out
#SBATCH --error=logs/trainref-gpu-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export LANE5_MANIFEST_SHA="$(sha256sum trainref.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export NATIVE_TRAIN_LIB="$PWD/runs/s301_trainref/libtrainref.so" NATIVE_CORE_TEST_DEVICES=0,1
export LD_LIBRARY_PATH="/mnt/lustre/work/gehler/gfh527/flightrl-puffer5.pmbvox/toolchain/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset PYTHONPATH
sha256sum -c trainref.sha256
sha256sum -c runs/s301_trainref/library.sha256
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/native_core/smoke_trainref.py
