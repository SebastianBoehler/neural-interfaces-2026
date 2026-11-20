#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=04:00:00
#SBATCH --job-name=ni26-laneT-ret
#SBATCH --output=logs/laneTret-%A_%a.out
# Usage (in the lane T ret dir): RET_VAR=lite5 sbatch --export=ALL --array=0-31 ttt_worker.sbatch [smoke]; task = order*4 + visit; arms hardcb + ft$RET_VAR
# out dir out_$RET_VAR (RET_VAR unset: out, arms hardcb+ftlight+ftheavy)
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_js321_20261117_returning_pair_cpu1
inner=$base/ni26_friday_returning_native_inner_v2_20261119
work=${SLURM_SUBMIT_DIR:?}
cd "$work"; var=${RET_VAR:-}; sfx=${var:+_$var}; if [ -n "$var" ]; then arms=hardcb+ft$var; else arms=hardcb+ftlight+ftheavy; fi; mkdir -p out$sfx logs
sha256sum --quiet -c source.sha256
(cd "$inner" && sha256sum --quiet -c "$work/parents.sha256")
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
visits=('P04|2' 'P04|3' 'P10|2' 'P10|3')
t=${SLURM_ARRAY_TASK_ID:?}; order=${orders[$((t/4))]}; visit=${visits[$((t%4))]}
opts=(); out="$work/out$sfx/${order}_${visit/|/_}"
if [ "${1:-}" = smoke ]; then opts=(--smoke); out="$out.smoke"; fi
rm -rf "$out"; mkdir "$out"
apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
  --env "RET_ARMS=$arms,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,MPLCONFIGDIR=/tmp/mpl,PYTHONPATH=/work:$prior:$inner:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding" \
  --bind "$base:$base:ro,$base:/assets:ro,$work:/work:ro,$out:/out:rw" \
  --pwd /out "$repo/runs/cloud/ni26_worker.sif" python -B /work/ttt_worker.py "$order" "$visit" "${opts[@]}"
