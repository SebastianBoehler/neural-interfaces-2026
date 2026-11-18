#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-fat-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 12
#SBATCH -J ni26-c5ad
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_cross411b_20261118
c5=$base/ni26_friday_c5_v1_20261118
normal=$base/ni26_native366_exact_ea_normal_v1_20261117
qualify=$base/ni26_friday_c5a_v2_20261118
work=$PWD
sha256sum -c --status source.sha256
echo "d69d56a452f80992f23f99e96747831381817a14be796f87976c2d347e10591d  $qualify/source.sha256" | sha256sum -c --status
(cd "$qualify" && sha256sum -c --status source.sha256)
(cd "$c5" && sha256sum -c --status source.sha256)
(cd "$prior" && sha256sum -c --status "$qualify/parents.sha256" && sha256sum -c --status members.sha256 && sha256sum -c --status inputs.sha256)
grep -Fx OK "$qualify/replay.log"
for cut in stock1 stock2; do
    apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
        --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,MPLCONFIGDIR=/tmp,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/qualify:/c5:/prior \
        --bind "$work:/work:rw,$qualify:/qualify:ro,$prior:/prior:ro,$prior/cap:/package:ro,$c5:/c5:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$repo/data/proteus_windows.npz:/input.npz:ro,$normal/reference_normal.json:/normal.json:ro" \
        --pwd /work "$repo/runs/cloud/ni26_worker.sif" python diagnose.py "$cut.json" > "$cut.log" 2>&1
done
python3 compare.py
