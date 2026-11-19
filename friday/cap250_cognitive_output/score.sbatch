#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:07:00
#SBATCH --job-name=ni26-c27
#SBATCH --output=c27-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
camp=$base/ni26_friday_causal_rebuild_20261118
repo=$base/neural-interfaces-2026
held=$camp/f32c_v2
external=$base/ni26_friday_f32x_external_cpu_v2_20261119
prior=$base/ni26_cross411b_20261118
scope=${1:?}
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c --status source.sha256
grep -Fx OK "$prior/replay-proof.txt"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
case "$scope" in
  external)
    (cd "$external" && sha256sum -c --status source.sha256 && sha256sum -c --status scoring.sha256)
    export PYTHONPATH="$external:$base/ni26_friday_c21e_native_v2_20261119:$base/ni26_friday_c20e_native_v4_20261119:$prior:$repo/experiments:$repo/2026-competition"
    ;;
  held)
    (cd "$held" && sha256sum -c --status consumer.sha256 && sha256sum -c --status scoring.sha256)
    export PYTHONPATH="$held:$camp/fc20_outer_v1:$camp/within0_terminal_shared_v1:$camp/fc18_necessity_v2:$camp/fc19_csp_gate_v2"
    ;;
  *) exit 2 ;;
esac
mkdir "$scope"
cd "$scope"
exec "$repo/.venv/bin/python" -B ../gate.py "$scope"
