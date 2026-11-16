#!/usr/bin/env bash
set -euo pipefail
arm=${1:?arm} order=${2:?order}
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/16nov/followups/native_core/converged_chain.py "$arm" "$order"
pids=()
for base in B2_pool B2_s345 B2_s012 B2_s0 B2_s1 B2_s2; do
    "$PY" -B -I studies/16nov/followups/native_core/converged_cp.py "$arm" "$order" "$base" >"logs/cp-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}-${base}.out" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then echo "CP failed for $arm $order" >&2; exit 1; fi
if [[ $arm == Rzero ]]; then "$PY" -B -I studies/16nov/followups/native_core/verify_zero.py "$order"; fi
