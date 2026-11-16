#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-s301-benchsmoke
#SBATCH --output=logs/benchmark-smoke-%j.out
#SBATCH --error=logs/benchmark-smoke-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export LANE5_MANIFEST_SHA="$(sha256sum benchmark.sha256 | cut -d' ' -f1)"
source studies/16nov/lane5/cluster_env.sh
export NI26_ROOT="$PWD" NATIVE_CORE_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_full_v3_20261116/runs/s301_full_build/libnative_core.so
export NATIVE_TRAIN_LIB=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_trainref_v3_20261116/runs/s301_trainref/libtrainref.so
export S301_BENCH_REFERENCE=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_reference_20261116/runs/s301_reference/P01_within0.npz
export S301_BENCH_PRODUCER=/mnt/lustre/work/gehler/gfh527/ni26_s301_native_producer_v3_20261116
export S301_BENCH_CACHE="$PWD/runs/s301_native_inputs/P01.npz" S301_BENCH_SMOKE="$PWD/runs/s301_benchmark/smoke.json"
unset PYTHONPATH
sha256sum -c source.sha256 >/dev/null
sha256sum -c benchmark.sha256
(cd "$S301_BENCH_PRODUCER" && sha256sum -c producer.sha256 >/dev/null && test -f runs/s301_gates/prepare_P01.json)
(cd /mnt/lustre/work/gehler/gfh527/ni26_s301_native_full_v3_20261116 && sha256sum -c full.sha256 >/dev/null && sha256sum -c runs/s301_full_build/library.sha256)
(cd /mnt/lustre/work/gehler/gfh527/ni26_s301_native_trainref_v3_20261116 && sha256sum -c trainref.sha256 >/dev/null && sha256sum -c runs/s301_trainref/library.sha256)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun --cpu-bind=cores /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I studies/16nov/followups/native_core/bench_smoke.py
