#!/bin/bash
# s290 part 1 on Galvani (run in the work dir ni26_s290; only sbatch): prep of the regional views, then a 10-task fold array afterok (task = held-out person, all eight arms).
# CPU partition cpu-galvani (as s263). Outputs runs/s290/<arm>_fold<I>.json (fixed paths; finished folds are overwritten identically). Collect: python -I studies/15nov/s290/collect.py
cd "$(dirname "$0")/../../.." || exit 1; mkdir -p runs/s290/logs
ENV='export PATH=$PWD/.venv/bin:$PATH OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONWARNINGS=ignore NI26_ROOT=$PWD; '
COM="-A gehler -p cpu-galvani --cpus-per-task=4 --mem=48G"
# SMOKE=1: prep, then folds 0-1 into runs/s290_smoke (afterok). SKIPPREP=1: the array only (prep already ran). Default: prep + array 0-9.
if [ -z "${SKIPPREP:-}" ]; then P=$(sbatch --parsable $COM --time=02:00:00 -J s290prep -o runs/s290/logs/prep-%j.out --wrap "$ENV python -I studies/15nov/s290/s290.py prep"); DEP="--dependency=afterok:$P"; else P=none; DEP=""; fi
if [ -n "${SMOKE:-}" ]; then
  F=$(sbatch --parsable $COM --time=01:00:00 --array=0-1 $DEP -J s290smoke -o runs/s290/logs/smoke-%A_%a.out --wrap "$ENV python -I studies/15nov/s290/s290.py fold \$SLURM_ARRAY_TASK_ID runs/s290_smoke"); echo prep $P smoke $F; exit 0
fi
F=$(sbatch --parsable $COM --time=04:00:00 --array=0-9%6 --requeue $DEP -J s290fold -o runs/s290/logs/fold-%A_%a.out --wrap "$ENV python -I studies/15nov/s290/s290.py fold \$SLURM_ARRAY_TASK_ID")
echo prep $P folds $F
