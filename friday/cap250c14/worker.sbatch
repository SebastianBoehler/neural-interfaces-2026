#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-fat-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:18:00
#SBATCH --job-name=ni26-c14
#SBATCH --output=c14-%A_%a.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_cross411b_20261118
old=$base/ni26_qualify385_v1_20261118
actor=$base/ni26_friday_c13_worker_v3_20261118/actors/seed0.pt
work=$PWD
mode=${1:?smoke or produce}
sha256sum --check --status source.sha256
(cd "$prior" && sha256sum --check --status "$work/parents.sha256")
case "$mode" in
    smoke)
        (cd "$prior" && sha256sum --check --status inputs.sha256 && sha256sum --check --status members.sha256)
        echo "9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed  $old/v66_cp1_cap250.zip" | sha256sum --check --status
        id=5
        ;;
    produce)
        test -s smoke.sha256
        id=${SLURM_ARRAY_TASK_ID:?}
        ;;
    *)
        exit 2
        ;;
esac
grep -Fx OK "$prior/replay-proof.txt"
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
    --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=/prior \
    --bind "$work:/work:rw,$prior:/prior:ro,$prior/cap:/package:ro,$actor:/actor.pt:ro,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro" \
    --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B worker.py "$mode" "$id"
