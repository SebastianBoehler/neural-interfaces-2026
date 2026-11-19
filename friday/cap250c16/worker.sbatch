#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:12:00
#SBATCH --job-name=ni26-c16
#SBATCH --output=c16-%A_%a.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_js321_20261117_returning_pair_cpu1
inner=$base/ni26_friday_returning_native_inner_v2_20261119
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --check --status fit.sha256
(
    cd "$inner"
    sha256sum --check --status "$work/parents.sha256"
    sha256sum --check --status source.sha256
    sha256sum --check --status inputs.sha256
)
(cd "$prior" && sha256sum --check --status source.sha256)
grep -Fx OK "$base/ni26_aux431_20261118/runs/aux431/replay.txt"
visits=('P04|2' 'P04|3' 'P10|2' 'P10|3')
case "${1:?fit, smoke or visits}" in
    fit) out="$work/fit"; command=(/work/fit.py);;
    smoke)
        sha256sum --check --status replay.sha256
        out="$work/smoke"; command=(/work/replay_head.py 'P10|2' --smoke);;
    visits)
        sha256sum --check --status replay.sha256
        test -s "$work/smoke/result.json"
        id=${SLURM_ARRAY_TASK_ID:?}
        out="$work/visit$id"; command=(/work/replay_head.py "${visits[$id]}");;
    *) exit 2;;
esac
mkdir "$out"
apptainer exec --cleanenv --containall --no-home --no-mount hostfs \
  --env "OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,MPLCONFIGDIR=/tmp/mpl,PYTHONPATH=$inner:$prior:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding" \
  --bind "$base:$base:ro,$base:/assets:ro,$work:/work:ro,$out:/out:rw" \
  --pwd /out "$repo/runs/cloud/ni26_worker.sif" python -B "${command[@]}"
