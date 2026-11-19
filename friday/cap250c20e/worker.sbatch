#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:13:00
#SBATCH --job-name=ni26-c20e
#SBATCH --output=c20e-%A_%a.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_cross411b_20261118
pair=$base/ni26_friday_frozen_rtx_20261118
other=$base/ni26_friday_frozen_rtx_nine_20261118
actors=$base/ni26_friday_cap250_raw_model_v3_20261118
known=$base/ni26_native366_exact_ea_normal_v1_20261117/reference_normal.json
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --check --status source.sha256
(cd "$base" && sha256sum --check --status "$work/parents.sha256")
(cd "$prior" && sha256sum --check --status inputs.sha256 && sha256sum --check --status members.sha256)
(cd "$actors" && sha256sum --check --status artifact.sha256)
grep -Fx OK "$prior/replay-proof.txt"
grep -Fx OK "$pair/replay.log"
mode=${1:?}
if [[ $mode == smoke ]]; then
    id=5
else
    [[ $mode == produce && -s smoke.json ]]
    id=${SLURM_ARRAY_TASK_ID:?}
fi
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
  --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/prior \
  --bind "$work:/work:rw,$prior:/prior:ro,$prior/cap:/package:ro,$actors:/actors:ro,$pair:/rtx_pair:ro,$other:/rtx_other:ro,$known:/known.json:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$repo/data/proteus_windows.npz:/input.npz:ro" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B worker.py "$mode" "$id"
