#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-eye-cal-native
#SBATCH --output=eye-cal-native-%A_%a.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_js321_20261117_returning_pair_cpu1
inner=$base/ni26_friday_returning_native_inner_v2_20261119
observer=$base/ni26_friday_returning_core2_native_v3_20261119
fit=$base/ni26_friday_return_eye_cal_fit_v1_20261119
head=$base/ni26_friday_return_eye_cal_head_v1_20261119
original=$base/ni26_friday_return_eye_native_v1_20261119
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --quiet -c source.sha256
(cd "$base" && sha256sum --quiet -c "$work/native_inputs.sha256")
(cd "$observer" && sha256sum --quiet -c source.sha256)
(cd "$prior" && sha256sum --quiet -c source.sha256)
(cd "$inner" && sha256sum --quiet -c source.sha256)
(cd "$fit" && sha256sum --quiet -c fit_source.sha256)
(cd "$head" && sha256sum --quiet -c source.sha256)
(cd "$original" && sha256sum --quiet -c source.sha256)
grep -Fx OK "$base/ni26_aux431_20261118/runs/aux431/replay.txt"
mode=${1:?smoke or visit}
slot=${SLURM_ARRAY_TASK_ID:?}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
case "$mode" in
    smoke) slot=$((slot * 2)); options=(--smoke) ;;
    visit)
        "$repo/.venv/bin/python" -B verify_calibration.py smoke
        options=()
        ;;
    *) exit 2 ;;
esac
out=$work/$mode$slot
mkdir "$out"
apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
  --env "OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,MPLCONFIGDIR=/tmp/mpl,NI26_WS=$repo,PYTHONPATH=$original:$observer:$inner:$prior:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding" \
  --bind "$base:$base:ro,$base:/assets:ro,$work:/work:ro,$out:/out:rw" \
  --pwd /out "$repo/runs/cloud/ni26_worker.sif" python -B /work/native_calibration.py "$slot" "${options[@]}"
