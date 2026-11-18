#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-fat-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 15
#SBATCH -J ni26-c5a
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_cross411b_20261118
c5=$base/ni26_friday_c5_v1_20261118
normal=$base/ni26_native366_exact_ea_normal_v1_20261117
work=$PWD
sha256sum -c --status source.sha256
(cd "$c5" && sha256sum -c --status source.sha256)
(cd "$prior" && sha256sum -c --status "$work/parents.sha256" && sha256sum -c --status members.sha256)
mode=$1
if [[ $mode == smoke ]]; then
    index=5
    (cd "$prior" && sha256sum -c --status inputs.sha256)
    echo "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327  $repo/data/proteus_windows.npz" | sha256sum -c --status
    (cd "$base/ni26_s318_coral_fbl_20261117" && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$repo/.venv/bin/python" replay/check.py) > replay.log
else
    [[ $mode == produce ]]
    index=$SLURM_ARRAY_TASK_ID
fi
grep -Fx OK replay.log
run() {
    local arm="$1" prefix="$2"
    apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
        --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,MPLCONFIGDIR=/tmp,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/c5:/prior \
        --bind "$work:/work:rw,$prior:/prior:ro,$prior/cap:/package:ro,$c5:/c5:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$repo/data/proteus_windows.npz:/input.npz:ro,$normal/reference_normal.json:/normal.json:ro" \
        --pwd /work "$repo/runs/cloud/ni26_worker.sif" python qualify.py "$mode" "$index" "$arm" "$prefix-$arm.json" > "$prefix-$arm.log" 2>&1
}
if [[ $mode == smoke ]]; then
    run stock smoke
    run deferred smoke
    python3 verify_pair.py smoke-stock.json smoke-deferred.json
else
    run stock "run-$index-ab"
    run deferred "run-$index-ab"
    python3 verify_pair.py "run-$index-ab-stock.json" "run-$index-ab-deferred.json"
    run deferred "run-$index-ba"
    run stock "run-$index-ba"
    python3 verify_pair.py "run-$index-ba-stock.json" "run-$index-ba-deferred.json"
fi
