#!/bin/bash
# EPISODIC wave (ledger s220), one Galvani GPU job: the 1-epoch smoke first (the wave runs only if it passes), then the
# 12 tasks (0-5 arm E, 6-11 arm E0) in 4 parallel lanes on the one GPU. The user's job limit (36 running, shared with the
# CPU servers) is the bottleneck, not the GPU. Tasks skip finished outputs, so a requeue resumes.
# Usage (Galvani, repo root): sbatch ... studies/14nov/episodic/wave.sh   (see the ledger entry for the sbatch line)
set -uo pipefail
export PATH=$PWD/.venv/bin:$PATH NI26_WS=$PWD OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore
python -c 'import torch; assert torch.cuda.is_available()' || exit 1
L=runs/diag/episodic/logs; mkdir -p $L
python -I studies/14nov/episodic/episodic.py smoke > $L/smoke.out 2>&1 || { echo SMOKE FAILED; tail -30 $L/smoke.out; exit 1; }
tail -3 $L/smoke.out
for lane in 0 1 2 3; do
  ( for t in $lane $((lane + 4)) $((lane + 8)); do python -I studies/14nov/episodic/episodic.py $t > $L/task$t.out 2>&1; echo "task $t EXIT $?"; done ) &
done
wait
grep -c TOTAL $L/task*.out  # 14 per finished task (7 orders x 2 protocols)
