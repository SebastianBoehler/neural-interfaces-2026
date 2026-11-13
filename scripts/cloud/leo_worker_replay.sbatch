#!/bin/bash
#SBATCH --job-name=ni26-wrkleo
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=runs/diag/leo_worker_replay-%j.out
# Worker-image replay on a Leonardo A100: the Codabench worker replica built on Galvani (runs/cloud/ni26_worker.sif,
# copied with its md5; Leonardo cannot build it, no fakeroot) runs scripts/replay_package.py on each given ZIP, as
# scripts/cloud/worker_replay.sbatch does on Galvani. Order as given (put v5.1b first and last for a warm timing pair).
# SMOKE=1: image check only (md5, pip freeze, CUDA), no replay. Nothing is uploaded.
# Usage: sbatch scripts/cloud/leo_worker_replay.sbatch runs/sealed/final_zips/v51b_uploaded.zip ZIP ... v51b_uploaded.zip
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; unset PYTHONPATH
SIF=runs/cloud/ni26_worker.sif
if [ ! -f "$SIF" ]; then  # uploaded in chunks (split -b, parallel streams); joined here, not on the login node
  cat runs/cloud/sifparts/part_* > "$SIF.tmp" && mv "$SIF.tmp" "$SIF"
fi
[ "$(md5sum "$SIF" | cut -d' ' -f1)" = "$(cat "$SIF.md5")" ] || { echo "IMAGE MD5 MISMATCH"; exit 2; }
nvidia-smi --query-gpu=name,driver_version --format=csv; hostname; date -u
RUN=(singularity exec --nv --bind "$PWD" --env "BENCHOPT_DATA_HOME=$PWD/data" "$SIF")
"${RUN[@]}" cat /compet/pip_freeze.txt | grep -iE "^(torch|braindecode|numpy|scipy|benchopt|neural|exca|mne)"
"${RUN[@]}" python -c 'import torch; assert torch.cuda.is_available(), "no CUDA in the image"; print("cuda", torch.__version__, torch.cuda.get_device_name(0))'
[ "${SMOKE:-0}" = 1 ] && { echo SMOKE_OK; exit 0; }
[ $# -ge 1 ] || { echo "no ZIP given"; exit 2; }
i=0
for Z in "$@"; do
  i=$((i + 1)); OUT="${Z%.zip}_wrkleo_${SLURM_JOB_ID}_$i"
  echo "== $Z sha256 $(sha256sum "$Z" | cut -c1-12)"
  "${RUN[@]}" bash -c "cd '$PWD' && python scripts/replay_package.py '$Z' '$OUT' cuda"
  python3 -c "import json; r=json.load(open('$OUT/replay.json')); print('$Z', r['files_unchanged'], r['scores']['balanced_accuracy_cell_mean'], r['zip_sha256'][:12], r['ingestion_wall_seconds'], 's')"
done
date -u; echo WORKER_LEO_DONE
