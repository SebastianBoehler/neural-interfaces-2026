#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:04:00
#SBATCH --job-name=ni26-rel355-whole
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${R355_WS:?}"
sha256sum -c source.sha256 >/dev/null
ROOT=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
(cd "$ROOT" && .venv/bin/python replay/check.py) > replay_guard.log
 grep -q 'OK' replay_guard.log
"$ROOT/.venv/bin/python" code/collect.py --root "$ROOT" --workspace "$R355_WS"
