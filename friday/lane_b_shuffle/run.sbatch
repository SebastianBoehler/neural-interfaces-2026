#!/bin/bash
#SBATCH --account=gehler
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=00:35:00
#SBATCH --job-name=ni26-lane-b
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
niv=$base/ni26_niv_v1_20261119
sha256sum --quiet -c source.sha256
grep -Fxq OK replay.log
public=$repo/data/proteus_windows.npz
external=$repo/data/scherer2015/windows.npz
runs=$niv/scherer_rows.tsv.gz
mode=${1:?}
if [[ $mode == collect ]]; then
    export PYTHONPATH="$repo/2026-competition:$PWD"
    exec "$repo/.venv/bin/python" -B collect.py "$external" "$runs" out/ external.json
fi
run_python() {
    apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
    --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,TMPDIR=/work \
    --bind "$PWD:/work:rw,$repo/2026-competition:/official:ro,$external:/external.npz:ro,$public:/public.npz:ro,$runs:/rows.tsv.gz:ro" \
    --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B "$@"
}
case "$mode" in
    smoke|timing) run_python smoke.py "$mode" /work/zips /external.npz /rows.tsv.gz /public.npz cuda:0 ;;
    external)
        first=$(( ${SLURM_ARRAY_TASK_ID:?} * 8 ))
        for ((i=first; i<first+8; i++)); do
            run_python external.py /work/zips /external.npz /rows.tsv.gz "/work/out/$i.npz" "$i"
        done
        ;;
    *) exit 2 ;;
esac
