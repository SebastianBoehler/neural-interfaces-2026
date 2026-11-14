#!/usr/bin/env bash
#SBATCH --job-name=ni26-worker
#SBATCH --partition=a100-galvani
#SBATCH --account=gehler
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
# Usage: sbatch scripts/cloud/worker_replay.sbatch PKG_DIR|ZIP [PKG_DIR|ZIP ...]  (ZIPs: listed in their folder's SHA256SUMS)
# Official ingestion + scoring replay of each package inside an Apptainer
# replica of the Codabench worker image (scripts/cloud/worker_image.def),
# CUDA visible as on the worker. Builds the image once (cached as
# runs/cloud/ni26_worker.sif). Writes PKG_worker_JOBID_<position>/replay.json and the
# image's pip freeze. Nothing is uploaded.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
SIF=runs/cloud/ni26_worker.sif
if [ ! -f "$SIF" ]; then
  export APPTAINER_CACHEDIR="$PWD/runs/cloud/apptainer_cache" APPTAINER_TMPDIR="${TMPDIR:-/tmp}"
  apptainer build --fakeroot "$SIF" scripts/cloud/worker_image.def \
    || apptainer build "$SIF" scripts/cloud/worker_image.def
fi
apptainer exec "$SIF" cat /compet/pip_freeze.txt | grep -iE "^(torch|braindecode|numpy|scipy|benchopt|neural|exca|mne)"
for PKG in "$@"; do  # a ZIP must be listed in its folder's SHA256SUMS and match it, checked before any GPU time
  case "$PKG" in *.zip)
    want=$(awk -v b="$(basename "$PKG")" '{n=split($2, a, "/"); if (a[n] == b) print $1}' "$(dirname "$PKG")/SHA256SUMS")
    [ -n "$want" ] || { echo "$PKG is not in SHA256SUMS"; exit 2; }
    [ "$(sha256sum "$PKG" | cut -d' ' -f1)" = "$want" ] || { echo "$PKG does not match SHA256SUMS"; exit 2; } ;;
  esac
done
i=0
for PKG in "$@"; do
  PKG="${PKG%/}"; i=$((i + 1))
  OUT="${PKG%.zip}_worker_${SLURM_JOB_ID}_$i"  # one folder per position: a ZIP given twice (v5.1b first and last) gets two
  echo "== $PKG $( [ -f "$PKG" ] && sha256sum "$PKG" | cut -c1-12 )"
  apptainer exec --nv --bind "$PWD" --env BENCHOPT_DATA_HOME="$PWD/data" "$SIF" \
    bash -c "cd '$PWD' && python -c 'import torch; print(\"cuda\", torch.cuda.is_available(), torch.__version__)' && python scripts/replay_package.py '$PKG' '$OUT' cuda"
  python3 -c "import json; r=json.load(open('$OUT/replay.json')); print('$PKG', r['files_unchanged'], r['scores'])"
done
