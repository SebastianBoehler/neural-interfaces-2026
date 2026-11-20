#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=06:00:00
#SBATCH --job-name=ni26-R4-ret
#SBATCH --output=logs/R4ret-%x-%A_%a.out
# Usage (release dir): RET_VAR=R1 sbatch --export=ALL --array=0-31 ret_worker.sbatch [smoke]; task = order*4 + visit; arms hardcb + $RET_VAR
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
visits=('P04|2' 'P04|3' 'P10|2' 'P10|3')
t=${SLURM_ARRAY_TASK_ID:?}; order=${orders[$((t/4))]}; visit=${visits[$((t%4))]}
opts=(); out="$work/out_$var/${order}_${visit/|/_}"
if [ "${1:-}" = smoke ]; then opts=(--smoke); out="$out.smoke"; fi
rm -rf "$out"; mkdir "$out"
apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
  --env "RET_ARMS=$arms,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,MPLCONFIGDIR=/tmp/mpl,PYTHONPATH=/work:$prior:$inner:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding" \
  --bind "$base:$base:ro,$base:/assets:ro,$work:/work:ro,$out:/out:rw" \
  --pwd /out "$repo/runs/cloud/ni26_worker.sif" python -B /work/ret_worker.py "$order" "$visit" "${opts[@]}"
