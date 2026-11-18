#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 20
#SBATCH -J ni26-qual408
set -eu
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
qual=$base/ni26_qualify385_v1_20261118
prior=$base/ni26_runtime387_v1_20261118
rep=${SLURM_ARRAY_TASK_ID:--1}
sha256sum -c source.sha256 >/dev/null
sha256sum -c members.sha256 >/dev/null
(cd "$qual" && sha256sum -c members.sha256 >/dev/null)
mode=time
arms=(stock pruned)
case "$rep" in
 -1) mode=smoke; (cd "$prior" && sha256sum -c inputs.sha256 >/dev/null);;
 1) arms=(pruned stock);;
 2) mode=parity;;
esac
for arm in "${arms[@]}"; do
    package=$qual/cap
    if [[ $arm == pruned ]]; then package=$PWD/pruned; fi
    /usr/bin/time -f %e -o "$arm-$rep.seconds" apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
      --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1 \
      --bind "$PWD:/work:rw,$package:/package:ro,$repo/2026-competition:/official:ro,$repo/data/schirrmeister2017/windows_1-2.npz:/external.npz:ro,$repo/data/proteus_windows.npz:/input.npz:ro,$prior/cap_0.json:/prior.json:ro" \
      --pwd /work "$repo/runs/cloud/ni26_worker.sif" python worker.py "$arm" "$mode" "$rep" >"$arm-$rep.log" 2>&1
done
python3 compare.py "$rep"
