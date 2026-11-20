#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=08:00:00
#SBATCH --job-name=ni26-D3-ret
#SBATCH --output=logs/D3ret-%x-%A_%a.out
# Usage (release ret/ dir): RET_ARMS=hardcb+D3f+D3g sbatch --export=ALL --array=0-3 ret_worker.sbatch [smoke]; task = order*2 + person, orders (full0 full1) (cells P|2, P|3 in sequence)
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_js321_20261117_returning_pair_cpu1
inner=$base/ni26_friday_returning_native_inner_v2_20261119
work=${SLURM_SUBMIT_DIR:?}
cd "$work"; arms=${RET_ARMS:-hardcb+D3f+D3g}; mkdir -p out logs
sha256sum --quiet -c source.sha256
(cd "$inner" && sha256sum --quiet -c "$work/parents.sha256")
orders=(full0 full1)
t=${SLURM_ARRAY_TASK_ID:?}; order=${orders[$((t/2))]}; who=$([ $((t%2)) = 0 ] && echo P04 || echo P10)
fail=0
for day in 2 3; do
  visit="$who|$day"; opts=(); out="$work/out/${order}_${who}_${day}"
  if [ "${1:-}" = smoke ]; then opts=(--smoke); out="$out.smoke"; fi
  rm -rf "$out"; mkdir "$out"
  apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
    --env "RET_ARMS=$arms,RET_N=${RET_N:-},OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,MPLCONFIGDIR=/tmp/mpl,PYTHONPATH=/work:$prior:$inner:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding" \
    --bind "$base:$base:ro,$base:/assets:ro,$work:/work:ro,$out:/out:rw" \
    --pwd /out "$repo/runs/cloud/ni26_worker.sif" python -B /work/ret_worker.py "$order" "$visit" "${opts[@]}" || fail=1
done
exit $fail
