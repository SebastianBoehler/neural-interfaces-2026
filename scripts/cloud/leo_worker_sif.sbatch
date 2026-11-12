#!/bin/bash
#SBATCH --job-name=ni26-sif
#SBATCH --account=AIFAC_P02_1675
#SBATCH --partition=lrd_all_serial
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --output=runs/diag/leo_worker_sif-%j.out
# Builds the worker-image replica (scripts/cloud/worker_image.def, as Galvani's runs/cloud/ni26_worker.sif) with
# Singularity on a serial node (internet). Usage: sbatch scripts/cloud/leo_worker_sif.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p runs/cloud
export SINGULARITY_CACHEDIR="$PWD/runs/cloud/sing_cache" SINGULARITY_TMPDIR="$PWD/runs/cloud/sing_tmp"; mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
singularity --version
singularity build --fakeroot runs/cloud/ni26_worker.sif scripts/cloud/worker_image.def || singularity build runs/cloud/ni26_worker.sif scripts/cloud/worker_image.def
singularity exec runs/cloud/ni26_worker.sif cat /compet/pip_freeze.txt | grep -iE "^(torch|braindecode|numpy|scipy|benchopt|neural|exca|mne)"; echo SIF_OK
