#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-s314-score-retry
#SBATCH --output=logs/s314-score-retry-%j.out
#SBATCH --error=logs/s314-score-retry-%j.err
set -euo pipefail
collector=/mnt/lustre/work/gehler/gfh527/ni26_s314_time_patterns_20261117/collector_fix_v2
cd "$collector"
sha256sum -c collector.sha256 >/dev/null
cd /mnt/lustre/work/gehler/gfh527/ni26_s314_time_patterns_20261117/production
sha256sum -c s314-source.sha256 >/dev/null
test "$(sha256sum s314-source.sha256 | cut -d' ' -f1)" = baa34aa59fc34c0722dcf0c65a02b6d1fc1b70e93e32190b5ac7712dde49af0a
test ! -e evidence/next_wave_16nov/s314_results.json
export NI26_WS="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export S314_COLLECTOR_MANIFEST="$collector/collector.sha256"
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I "$collector/verify_streams.py"
"$PY" -B -I "$collector/score_fixed.py"
