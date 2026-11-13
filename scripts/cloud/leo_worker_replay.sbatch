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
# Every ZIP must be listed in its folder's SHA256SUMS and match it (the exact uploaded or staged files only).
# V6TIME="P5 P6 [--gates-on];...": afterwards, studies/13nov/v6_time.py inside the image for each spec (the FT path, which
# replays never reach because their sessions are recognised at window 10).
# A CUDA error fails the job and names the node; resubmit with sbatch --exclude=NODE. SMOKE=1: image check only.
# Usage: [V6TIME=...] sbatch [--exclude=NODE] scripts/cloud/leo_worker_replay.sbatch v51b_uploaded.zip ZIP ... v51b_uploaded.zip
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
[ $# -ge 1 ] || [ -n "${V6TIME:-}" ] || { echo "nothing to run"; exit 2; }
for Z in "$@"; do  # all hashes first, so a stale or truncated ZIP fails before any GPU time is spent
  want=$(awk -v b="$(basename "$Z")" '{n=split($2, a, "/"); if (a[n] == b) print $1}' "$(dirname "$Z")/SHA256SUMS")
  [ -n "$want" ] || { echo "$Z is not in SHA256SUMS"; exit 2; }
  [ "$(sha256sum "$Z" | cut -d' ' -f1)" = "$want" ] || { echo "$Z does not match SHA256SUMS"; exit 2; }
done
cuda_fail() { grep -qi "CUDA error\|cudaError\|uncorrectable ECC" "$1" && { echo "CUDA ERROR on $(hostname): resubmit with --exclude=$(hostname)"; exit 3; }; return 0; }
i=0
for Z in "$@"; do
  i=$((i + 1)); OUT="${Z%.zip}_wrkleo_${SLURM_JOB_ID}_$i"
  echo "== $Z sha256 $(sha256sum "$Z" | cut -c1-12)"
  "${RUN[@]}" bash -c "cd '$PWD' && python scripts/replay_package.py '$Z' '$OUT' cuda" 2>&1 | tee "runs/diag/wrkleo_$SLURM_JOB_ID.log" || { cuda_fail "runs/diag/wrkleo_$SLURM_JOB_ID.log"; exit 1; }
  python3 -c "import json; r=json.load(open('$OUT/replay.json')); print('$Z', r['files_unchanged'], r['scores']['balanced_accuracy_cell_mean'], r['zip_sha256'][:12], r['ingestion_wall_seconds'], 's')"
done
IFS=';' read -ra SPECS <<< "${V6TIME:-}"
for spec in "${SPECS[@]}"; do
  echo "== v6_time $spec"
  "${RUN[@]}" bash -c "cd '$PWD' && python -I studies/13nov/v6_time.py cuda $spec" 2>&1 | tee "runs/diag/wrkleo_$SLURM_JOB_ID.log" || { cuda_fail "runs/diag/wrkleo_$SLURM_JOB_ID.log"; exit 1; }
done
date -u; echo WORKER_LEO_DONE
