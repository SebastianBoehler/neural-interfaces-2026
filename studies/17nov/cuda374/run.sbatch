#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:12:00
#SBATCH --job-name=ni26-fuse374
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
base=/mnt/lustre/work/gehler/gfh527
repo="$base/neural-interfaces-2026"
release="$base/ni26_native366_exact_release_v1_20261117"
qualified="$base/ni26_native366_exact_ea_v1_20261117"
phase="${1:?phase required}"
sha256sum --quiet -c source.sha256
sha256sum --quiet -c inputs.sha256
(cd "$release" && sha256sum --quiet -c package.sha256)
(cd "$qualified" && sha256sum --quiet -c stress.sha256)
if [[ "$phase" == smoke ]]; then
    LD_LIBRARY_PATH="$base/flightrl-puffer5.pmbvox/toolchain/lib" python3 build.py
    sha256sum candidate/libni26_runtime.so > candidate.sha256
    pairs=(0)
elif [[ "$phase" == timing ]]; then
    sha256sum --quiet -c candidate.sha256
    [[ -s smoke.json ]]
    pairs=(0 1)
else
    exit 2
fi
for mode in ordinary stress; do
    for pair in "${pairs[@]}"; do
        arms=(stock candidate)
        [[ "$pair" == 0 ]] || arms=(candidate stock)
        for arm in "${arms[@]}"; do
            if [[ "$arm" == stock ]]; then
                pkg="$release/package"
                [[ "$mode" == ordinary ]] || pkg="$qualified/stress_native"
            else
                pkg="$PWD/candidate"
                [[ "$mode" == ordinary ]] || pkg="$PWD/candidate_stress"
            fi
            apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
                --env OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1 \
                --bind "$PWD:/work:rw,$pkg:/package:ro,$repo/2026-competition:/official:ro,$repo/data/proteus_windows.npz:/input.npz:ro" \
                --pwd /work "$repo/runs/cloud/ni26_worker.sif" python run.py "$phase" "$arm" "$mode" "$pair"
        done
    done
done
python3 collect.py "$phase"
sha256sum --quiet -c source.sha256
