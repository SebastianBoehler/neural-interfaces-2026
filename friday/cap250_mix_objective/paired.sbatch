#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-c31b-source
#SBATCH --output=c31b-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_cross411b_20261118
owner=$base/ni26_friday_c20e_native_v4_20261119
shared=$base/ni26_friday_c20s_native_v1_20261119
stock=$base/ni26_s318_coral_fbl_20261117/runs/sealed/final_zips/v66_cp1_cap250.zip
order=$base/ni26_s318_coral_fbl_20261117/runs/shuffle_perms.npz
stress=$base/ni26_friday_c25pc_postcap_v2_20261119
natural=$base/ni26_friday_c31a_source_mix_v1_20261119
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --quiet -c source.sha256
(cd "$owner" && printf '%s\n' '3d56e3f24611b220fa017de8a49728c4ddc619eb65194c0150ee8b165bb9e436  source.sha256' | sha256sum --quiet -c -
 sha256sum --quiet -c source.sha256)
(cd "$shared" && printf '%s\n' 'f52e414494585fd2d6926e050dd2ac0462ebedc6c8f9981d16e19e6d6d02a3cb  mirror.py' | sha256sum --quiet -c -)
(cd "$prior" && sha256sum --quiet -c inputs.sha256 && sha256sum --quiet -c members.sha256)
grep -Fx OK "$prior/replay-proof.txt"
(cd "$stress" && printf '%s\n' '1e13bd34324921ee219996cebf2b3fc9911864bca8b2d261f230afbd1502f7a0  source.sha256' | sha256sum --quiet -c -
 sha256sum --quiet -c source.sha256)
(cd "$natural" && printf '%s\n' '5990a7fb7b8b01efc75830410dfc0e22d770d6af528621656848d5cbdb28b720  source.sha256' | sha256sum --quiet -c -
 sha256sum --quiet -c source.sha256)
mkdir -p "$work/.mpl-cache"
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
  --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,MPLCONFIGDIR=/work/.mpl-cache,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/shared:/v4:/prior,TMPDIR=/work \
  --bind "$work:/work:rw,$stock:/stock.zip:ro,$shared:/shared:ro,$owner:/v4:ro,$prior:/prior:ro,$repo/2026-competition:/official:ro,$repo/data/proteus_windows.npz:/input.npz:ro,$order:/orders.npz:ro,$stress/stress.json:/stress.json:ro,$natural/c31a-3135643_4294967294.out:/natural.log:ro" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B paired_frame.py
