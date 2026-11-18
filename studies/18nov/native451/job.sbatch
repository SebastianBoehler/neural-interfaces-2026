#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 15
#SBATCH -J ni26-native451
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
archive=$base/ni26_qualify385_v1_20261118
normal=$base/ni26_native366_exact_ea_normal_v1_20261117
native=$base/ni26_native366_exact_ea_v1_20261117/profile_package
id=${SLURM_ARRAY_TASK_ID:-5}
suite=${1:?}
sha256sum --quiet -c sources.sha256
(cd "$base" && sha256sum --quiet -c "$OLDPWD/anchors.sha256")
(cd "$normal" && sha256sum --quiet -c inputs.sha256)
(cd "$archive" && sha256sum --quiet -c members.sha256 && sha256sum --quiet -c sources.sha256)
if [[ "$suite" == smoke ]]; then
    (cd "$archive" && sha256sum --quiet -c inputs.sha256)
    for file in "$native"/*.pt; do
        [[ ${file##*/} == native.pt ]] || cmp "$file" "$archive/cap/${file##*/}"
    done
    echo "9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed  $archive/v66_cp1_cap250.zip" | sha256sum -c -
fi
arms=(native python)
if (( id % 2 )); then arms=(python native); fi
for arm in "${arms[@]}"; do
    package=$archive/cap
    if [[ "$arm" == native ]]; then package=$native; fi
    apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs         --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1         --bind "$PWD:/work:rw,$package:/package:ro,$archive:/archive:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro"         --pwd /work "$repo/runs/cloud/ni26_worker.sif" python worker.py "$arm" "$id" "$suite" >"$suite-$arm-$id.log" 2>&1
done
