#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-native353-worker
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
sha256sum -c inputs.sha256 >/dev/null
base=/mnt/lustre/work/gehler/gfh527
repo="$base/neural-interfaces-2026"
reference="$base/ni26_native353_qualify_v1_20261117/package"
build="$base/ni26_native353_package_v3_20261117"
fixture="$base/ni26_native353_runtime_source_v2_20261117/golden.pt"
image="$repo/runs/cloud/ni26_worker.sif"
(cd "$build" && sha256sum -c package.sha256 >/dev/null)
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > gpu.txt
for arm in reference candidate; do
    package="$reference"
    if [[ "$arm" == candidate ]]; then package="$build/package"; fi
    apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
        --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1 \
        --bind "$PWD:/work:rw,$repo/2026-competition:/official:ro,$package:/package:ro,$fixture:/fixture.pt:ro" \
        --pwd /work "$image" python worker_runtime.py /official/tracks/bci_decoding /package /fixture.pt "/work/$arm.json" \
        > "$arm.log" 2>&1
done
diff <(jq -c '.labels' reference.json) <(jq -c '.labels' candidate.json)
(cd "$build" && sha256sum -c package.sha256 >/dev/null)
sha256sum -c source.sha256 >/dev/null
sha256sum -c inputs.sha256 >/dev/null
echo NATIVE_A100_ADAPTER_PREFIX_PASS
