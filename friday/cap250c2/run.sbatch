#!/bin/bash
#SBATCH -A gehler
#SBATCH -p 2080-galvani,cpu-galvani
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 10
#SBATCH -J ni26-friday-c2
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
parent=$base/ni26_friday_c1_v1_20261118
export PYTHONPATH=$parent:$base/ni26_wave379_v1_20261118:$base/ni26_fusion383_v1_20261118:$base/ni26_soft_csp361_v1_20261117
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=$base/neural-interfaces-2026/.venv/bin/python
sha256sum -c --status source.sha256
echo "1e0807cdd9010cf324eb398e2ea7f105d12d51a91cf5cc95e7d944e5e73371a3  $parent/source.sha256" | sha256sum -c --status
(cd "$parent" && sha256sum -c --status source.sha256)
if [[ $1 == smoke ]]; then (cd "$base/ni26_s318_coral_fbl_20261117" && "$py" replay/check.py) > replay.log; fi
grep -Fx OK replay.log
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
case "$1" in
  smoke) "$py" run_csp.py within0 smoke ;;
  produce) "$py" run_csp.py "${orders[$SLURM_ARRAY_TASK_ID]}" ;;
  score) "$py" collect_csp.py ;;
  *) exit 2 ;;
esac
