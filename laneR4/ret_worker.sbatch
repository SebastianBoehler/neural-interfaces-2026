#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=08:00:00
#SBATCH --job-name=ni26-R4-ret
#SBATCH --output=logs/R4ret-%x-%A_%a.out
# Usage (release dir): RET_VAR=R1 sbatch --export=ALL --array=0-15 ret_worker.sbatch [smoke]; task = order*2 + person (cells P|2 and P|3 of that person in sequence); arms hardcb + $RET_VAR
# out dir out_$RET_VAR; packages pkg_ret/{P04,P10}_{hardcb,$RET_VAR}
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_js321_20261117_returning_pair_cpu1
inner=$base/ni26_friday_returning_native_inner_v2_20261119
work=${SLURM_SUBMIT_DIR:?}
cd "$work"; var=${RET_VAR:?}; arms=hardcb+$var; mkdir -p out_$var logs
sha256sum --quiet -c source.sha256
(cd "$inner" && sha256sum --quiet -c "$work/parents.sha256")
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
t=${SLURM_ARRAY_TASK_ID:?}; order=${orders[$((t/2))]}; who=$([ $((t%2)) = 0 ] && echo P04 || echo P10)
fail=0
for day in 2 3; do
  visit="$who|$day"; opts=(); out="$work/out_$var/${order}_${who}_${day}"
  if [ "${1:-}" = smoke ]; then opts=(--smoke); out="$out.smoke"; fi
  rm -rf "$out"; mkdir "$out"
  apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
    --env "RET_ARMS=$arms,RET_N=${RET_N:-},OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,MPLCONFIGDIR=/tmp/mpl,PYTHONPATH=/work:$prior:$inner:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding" \
    --bind "$base:$base:ro,$base:/assets:ro,$work:/work:ro,$out:/out:rw" \
    --pwd /out "$repo/runs/cloud/ni26_worker.sif" python -B /work/ret_worker.py "$order" "$visit" "${opts[@]}" || fail=1
done
exit $fail
