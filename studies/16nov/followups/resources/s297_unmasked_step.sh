#!/bin/bash
set -euo pipefail
if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then echo 'CUDA mask unexpectedly set in Slurm step' >&2; exit 1; fi
export NI26_WS="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I - <<'PY'
import os, subprocess
allowed = sorted(os.sched_getaffinity(0))
topology = subprocess.check_output(['lscpu', '-p=CPU,CORE,SOCKET'], text=True)
mapping = {}
for line in topology.splitlines():
    if not line.startswith('#'):
        cpu, core, socket = map(int, line.split(',')[:3])
        mapping[cpu] = (socket, core)
cores = {mapping[cpu] for cpu in allowed}
print('affinity_logical_cpus=', allowed, 'physical_cores=', sorted(cores), flush=True)
if len(allowed) != int(os.environ['EXPECT_CPUS']):
    raise SystemExit('Slurm step affinity does not match allocated logical CPUs')
PY
trace="logs/device-open-${SLURM_JOB_ID}.strace"
test ! -e "$trace"
case "${1:?}" in
  zero) cmd=(bash studies/16nov/followups/geometry/chain_full.sbatch zero) ;;
  compare) cmd=(/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I compare_resource.py) ;;
  *) exit 2 ;;
esac
strace -f -qq -e trace=open,openat,openat2,creat,open_by_handle_at,io_uring_setup,io_uring_enter,pidfd_getfd,recvmsg,sched_setaffinity -o "$trace" -- "${cmd[@]}"
if grep -E '/dev/(nvidia[^" ]*|dri/(renderD|card)[^" ]*|char/[0-9]+:[0-9]+)' "$trace"; then echo 'GPU-capable device access attempted' >&2; exit 1; fi
if grep -E '(open_by_handle_at|io_uring_setup|io_uring_enter|pidfd_getfd|recvmsg|sched_setaffinity)\(' "$trace"; then echo 'unverified indirect access or affinity change' >&2; exit 1; fi
printf 'DEVICE AND AFFINITY TRACE CLEAR: %s\n' "$trace"
