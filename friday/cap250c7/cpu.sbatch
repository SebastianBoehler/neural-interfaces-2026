#!/bin/bash
#SBATCH -A gehler
#SBATCH -p 2080-galvani,cpu-galvani
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 10
#SBATCH -J ni26-friday-c7-cpu
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
c1=$base/ni26_friday_c1_v1_20261118
c6=$base/ni26_friday_c6_v2_20261118
external=$base/ni26_qualify385_v1_20261118
export PYTHONPATH=$c6:$c1:$base/ni26_wave379_v1_20261118:$base/ni26_fusion383_v1_20261118:$base/ni26_soft_csp361_v1_20261117:$external:$repo/experiments
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=$repo/.venv/bin/python
sha256sum -c --status source.sha256
echo "1e0807cdd9010cf324eb398e2ea7f105d12d51a91cf5cc95e7d944e5e73371a3  $c1/source.sha256" | sha256sum -c --status
echo "6ec797665c15d8218b758af8f1a53f895afd3d58822f2dc7a00f5258a88f7ced  $c6/source.sha256" | sha256sum -c --status
(cd "$c1" && sha256sum -c --status source.sha256)
(cd "$c6" && sha256sum -c --status source.sha256)
if [[ $1 == smoke ]]; then
    (cd "$base/ni26_s318_coral_fbl_20261117" && "$py" replay/check.py) > replay.log
fi
grep -Fx OK replay.log
work=$PWD
case "$1" in
  smoke) "$py" fit.py smoke ;;
  fit)
    "$py" fit.py
    sha256sum coefficients.json > coefficients.sha256
    ;;
  score)
    sha256sum -c --status coefficients.sha256
    (cd "$external" && sha256sum -c --status "$work/parents.sha256" && sha256sum -c --status inputs.sha256)
    "$py" collect_c7.py
    ;;
  *) exit 2 ;;
esac
