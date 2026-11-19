#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:08:00
#SBATCH --job-name=ni26-c20e-compute
#SBATCH --output=c20e-compute-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_cross411b_20261118
other=$base/ni26_friday_frozen_rtx_nine_20261118
run=$base/ni26_friday_c20e_native_v4_20261119
actors=$base/ni26_friday_cap250_raw_model_v3_20261118
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --check --status source.sha256
(cd "$run" && echo '3d56e3f24611b220fa017de8a49728c4ddc619eb65194c0150ee8b165bb9e436  source.sha256' | sha256sum --check --status
 sha256sum --check --status source.sha256)
(cd "$base" && sha256sum --check --status "$run/parents.sha256")
(cd "$prior" && sha256sum --check --status inputs.sha256 && sha256sum --check --status members.sha256)
(cd "$actors" && sha256sum --check --status artifact.sha256)
grep -Fx OK "$prior/replay-proof.txt"
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
  --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,MPLCONFIGDIR=/tmp,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/v4:/prior \
  --bind "$work:/work:rw,$run:/v4:ro,$prior:/prior:ro,$prior/cap:/package:ro,$other:/rtx_other:ro,$actors:/actors:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B compute_repeat.py
