#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-s295-chain
#SBATCH --output=logs/geometry-%A_%a.out
#SBATCH --error=logs/geometry-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export NI26_WS="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
unset PYTHONPATH
sha256sum -c s295-source.sha256 >/dev/null
sha256sum -c s295-stage.sha256 >/dev/null
sha256sum -c s295-chain.sha256
export S295_MANIFEST="$PWD/runs/s295_geometry/manifest.json"
S295_MANIFEST_SHA=$(sha256sum "$S295_MANIFEST" | cut -d' ' -f1)
export S295_MANIFEST_SHA
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
if [[ ${1:-} == zero ]]; then
    arm=geo_zero
    order=${orders[$SLURM_ARRAY_TASK_ID]}
elif [[ ${1:-} == treatment ]]; then
    arms=(geo_s50 geo_d25 geo_d50 geo_d100)
    arm=${arms[$((SLURM_ARRAY_TASK_ID / 8))]}
    order=${orders[$((SLURM_ARRAY_TASK_ID % 8))]}
else
    exit 2
fi
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/16nov/followups/geometry/chain.py "$arm" "$order"
pids=()
for base in B2_pool B2_s345 B2_s012 B2_s0 B2_s1 B2_s2; do
    "$PY" -B -I studies/16nov/followups/geometry/cp.py "$arm" "$order" "$base" >"logs/cp-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}-${base}.out" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then echo "CP failed for $arm $order" >&2; exit 1; fi
if [[ $arm == geo_zero ]]; then
    "$PY" -B -I studies/16nov/followups/geometry/verify_zero.py "$order"
fi
