#!/usr/bin/env bash
# Used by graph.py with explicit resource requests and an immutable release path.
set -euo pipefail
program=${1:?relative Python program}
shift
cd "${CAP250_RELEASE:?}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NI26_TORCH_THREADS=${SLURM_CPUS_PER_TASK:-1}
export CS_OFFICIAL=$CAP250_RELEASE/2026-competition/tracks/bci_decoding
export PYTHONPATH=$CS_OFFICIAL:$CAP250_RELEASE/2026-competition${PYTHONPATH:+:$PYTHONPATH}
if [[ -n ${CAP250_IMAGE:-} ]]; then
  container=apptainer
  if [[ ${CAP250_SITE:-galvani} == leonardo ]]; then
    container=singularity
  fi
  options=(--bind "$CAP250_RELEASE,$CAP250_RUN,$CAP250_ASSETS")
  [[ -z ${CAP250_HARD_SOURCE:-} ]] || options+=(--bind "$CAP250_HARD_SOURCE")
  [[ -z ${CAP250_CB_STAGEA:-} ]] || options+=(--bind "$CAP250_CB_STAGEA")
  [[ -z ${CAP250_CB_CHECKPOINT:-} ]] || options+=(--bind "$(dirname "$CAP250_CB_CHECKPOINT")")
  [[ -z ${CAP250_DATA:-} ]] || options+=(--bind "$(dirname "$CAP250_DATA")")
  [[ -z ${SLURM_JOB_GPUS:-} ]] || options+=(--nv)
  exec "$container" exec "${options[@]}" "$CAP250_IMAGE" python -B "$program" "$@"
fi
if [[ ${CAP250_SITE:-galvani} == leonardo ]]; then
  module load python/3.11.7
fi
exec "${CAP250_PYTHON:?set CAP250_IMAGE or CAP250_PYTHON}" -B "$program" "$@"
