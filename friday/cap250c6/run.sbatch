#!/bin/bash
#SBATCH -A gehler
#SBATCH -p 2080-galvani,cpu-galvani
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 10
#SBATCH -J ni26-friday-c6
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
parent=$base/ni26_friday_c1_v1_20261118
export PYTHONPATH=$parent:$base/ni26_wave379_v1_20261118:$base/ni26_fusion383_v1_20261118:$base/ni26_soft_csp361_v1_20261117
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=$base/neural-interfaces-2026/.venv/bin/python
sha256sum -c --status source.sha256
echo "1e0807cdd9010cf324eb398e2ea7f105d12d51a91cf5cc95e7d944e5e73371a3  $parent/source.sha256" | sha256sum -c --status
(cd "$parent" && sha256sum -c --status source.sha256)
(cd "$base" && sha256sum -c --status "$parent/parents.sha256")
args=()
if [[ $1 == smoke ]]; then
    args=(smoke)
    (cd "$base/ni26_s318_coral_fbl_20261117" && "$py" replay/check.py) > replay.log
else
    [[ $1 == score ]]
fi
grep -Fx OK replay.log
"$py" produce_c6.py new "${args[@]}"
"$py" produce_c6.py returning "${args[@]}"
if [[ $1 == score ]]; then
    "$py" collect_c6.py
fi
