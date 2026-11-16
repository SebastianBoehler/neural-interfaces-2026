#!/bin/bash
set -euo pipefail
if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then echo 'CUDA mask unexpectedly set in step' >&2; exit 1; fi
export NI26_WS="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
assign="logs/postcheck-affinity-${SLURM_JOB_ID}.json"
trace="logs/postcheck-trace-${SLURM_JOB_ID}.strace"
test ! -e "$assign"
test ! -e "$trace"
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I - "$assign" <<'PY'
import json, os, sys
allowed = sorted(os.sched_getaffinity(0))
if len(allowed) != int(os.environ['EXPECT_CPUS']):
    raise SystemExit('Slurm affinity does not match allocated CPUs')
with open(sys.argv[1], 'x') as f: json.dump(allowed, f)
print('postcheck assigned logical CPUs:', allowed, flush=True)
PY
strace -f -qq -e trace=open,openat,openat2,creat,open_by_handle_at,io_uring_setup,io_uring_enter,pidfd_getfd,recvmsg,sched_setaffinity -o "$trace" -- /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I compare_resource.py
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I resource_tracecheck.py "$trace" "$assign" --cpus "$EXPECT_CPUS" --json-assignment
sha256sum -c resource_unmasked.sha256
sha256sum -c resource_postcheck.sha256
