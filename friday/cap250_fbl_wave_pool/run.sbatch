#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-c28
#SBATCH --output=c28-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
camp=$base/ni26_friday_causal_rebuild_20261118
held=$base/ni26_friday_c25d_held_v1_20261119
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c --status source.sha256
(cd "$held" && sha256sum -c --status source.sha256 && sha256sum -c --status scoring.sha256)
grep -Fx OK "$base/ni26_cross411b_20261118/replay-proof.txt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$held:$camp/fc20_outer_v1:$camp/within0_terminal_shared_v1:$camp/fc18_necessity_v2:$camp/fc19_csp_gate_v2:$camp/f32c_v2"
exec "$repo/.venv/bin/python" -B pool28.py
