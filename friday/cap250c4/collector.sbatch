#!/bin/bash
#SBATCH -A gehler
#SBATCH -p 2080-galvani,cpu-galvani
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 5
#SBATCH -J ni26-score-c4
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_qualify385_v1_20261118
shared=$base/ni26_friday_c3_v1_20261118
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$prior:$repo/experiments
sha256sum -c --status source.sha256
echo "8dd08259cbc95e8042544e3c893d7c53a354b9a851da97590520e16fb90cb92a  $shared/source.sha256" | sha256sum -c --status
(cd "$shared" && sha256sum -c --status source.sha256)
(cd "$prior" && sha256sum -c --status "$shared/parents.sha256" && sha256sum -c --status inputs.sha256)
grep -Fx OK replay.log
"$repo/.venv/bin/python" mass_report.py
"$repo/.venv/bin/python" "$shared/collect_external.py"
