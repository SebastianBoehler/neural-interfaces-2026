#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:14:00
#SBATCH --job-name=ni26-c24
#SBATCH --output=c24-%A_%a.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_cross411b_20261118
pair=$base/ni26_friday_frozen_rtx_20261118
other=$base/ni26_friday_frozen_rtx_nine_20261118
owner=$base/ni26_friday_c20e_native_v4_20261119
known=$base/ni26_native366_exact_ea_normal_v1_20261117/reference_normal.json
rtx_known=$base/ni26_friday_c5r_rtx_v1_20261119/smoke.json
native=$base/ni26_friday_c21e_native_v2_20261119
shared=$base/ni26_friday_c20s_native_v1_20261119
matched=$base/ni26_friday_c21s_native_v1_20261119
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --check --status source.sha256
(cd "$matched" && echo 'd9a626eb1322fdda0d42d787142887389742be78437cdff5521dc015792ad647  source.sha256' | sha256sum --check --status
 sha256sum --check --status source.sha256)
(cd "$native" && echo '61feef26593423464c1e3d8935c0ef43ce01d737df84fa6c73035d0b3b663841  source.sha256' | sha256sum --check --status
 sha256sum --check --status source.sha256)
(cd "$shared" && echo 'f52e414494585fd2d6926e050dd2ac0462ebedc6c8f9981d16e19e6d6d02a3cb  mirror.py' | sha256sum --check --status)
(cd "$owner" && echo '3d56e3f24611b220fa017de8a49728c4ddc619eb65194c0150ee8b165bb9e436  source.sha256' | sha256sum --check --status
 sha256sum --check --status source.sha256)
(cd "$base" && sha256sum --check --status "$owner/parents.sha256")
(cd "$prior" && sha256sum --check --status inputs.sha256 && sha256sum --check --status members.sha256)
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
  --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,MPLCONFIGDIR=/tmp,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/matched:/native:/shared:/v4:/prior \
  --bind "$work:/work:rw,$matched:/matched:ro,$native:/native:ro,$shared:/shared:ro,$owner:/v4:ro,$prior:/prior:ro,$prior/cap:/package:ro,$pair:/rtx_pair:ro,$other:/rtx_other:ro,$known:/known.json:ro,$rtx_known:/known_rtx.json:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$repo/data/proteus_windows.npz:/input.npz:ro" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B native24.py "$mode" "$id"
