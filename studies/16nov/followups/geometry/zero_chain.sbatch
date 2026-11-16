#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-s295-zero
#SBATCH --output=logs/zero-%j.out
#SBATCH --error=logs/zero-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export NI26_WS="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
unset PYTHONPATH
sha256sum -c s295-source.sha256
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I replay/check.py
"$PY" -B -I studies/16nov/followups/geometry/zero.py
export S295_MANIFEST="$PWD/runs/s295_geometry/zero_manifest.json"
S295_MANIFEST_SHA=$(sha256sum "$S295_MANIFEST" | cut -d' ' -f1)
export S295_MANIFEST_SHA
"$PY" -B -I studies/16nov/followups/geometry/chain.py geo_zero within0
pids=()
for base in B2_pool B2_s345 B2_s012 B2_s0 B2_s1 B2_s2; do
    "$PY" -B -I studies/16nov/followups/geometry/cp.py geo_zero within0 "$base" >"logs/zero-cp-${SLURM_JOB_ID}-${base}.out" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then echo 'a zero CP process failed' >&2; exit 1; fi
"$PY" -B -I studies/16nov/followups/geometry/verify_zero.py within0
