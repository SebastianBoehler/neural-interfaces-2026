#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-ear-u
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_cross411b_20261118
owner=$base/ni26_aux431_20261118
work=$PWD
sha256sum -c --status source.sha256
(cd "$prior" && sha256sum -c --status "$work/parents.sha256" && sha256sum -c --status members.sha256)
sha256sum -c --status inputs.sha256
grep -Fx OK "$owner/runs/aux431/replay.txt"
visits=('P04|2' 'P04|3' 'P10|3' 'P10|2')
case ${1:?smoke or run} in
    smoke) id=0;;
    run) id=${SLURM_ARRAY_TASK_ID:?}; test -s visit0/route.npz;;
    *) exit 2;;
esac
mkdir "visit$id"
apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
    --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,MPLCONFIGDIR=/tmp/mpl,PYTHONPATH=/official/tracks/bci_decoding \
    --bind "$base:/assets:ro,$work:/work:rw,$prior/cap:/package:ro,$repo/2026-competition:/official:ro" \
    --pwd "/work/visit$id" "$repo/runs/cloud/ni26_worker.sif" python -B /work/worker.py "${visits[$id]}"
