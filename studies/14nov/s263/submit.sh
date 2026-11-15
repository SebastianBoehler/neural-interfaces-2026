#!/bin/bash
# On Galvani in the work dir: prep job per variant, then a 10-task fold array depending on it (afterok).
cd "$(dirname "$0")/../../.." || exit 1; mkdir -p runs/s263/logs
for V in C A B D Z; do
  case $V in C) M=160G;; A|B) M=90G;; *) M=60G;; esac
  ENV='export PATH=$PWD/.venv/bin:$PATH OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONWARNINGS=ignore; '
  P=$(sbatch --parsable -A gehler -p cpu-galvani --cpus-per-task=8 --mem=$M --time=04:00:00 -J s263p$V -o runs/s263/logs/prep$V-%j.out --wrap "$ENV python -I studies/14nov/s263/s263.py prep $V")
  F=$(sbatch --parsable -A gehler -p cpu-galvani --cpus-per-task=8 --mem=$M --time=06:00:00 --array=0-9 --requeue --dependency=afterok:$P -J s263f$V -o runs/s263/logs/fold$V-%A_%a.out --wrap "$ENV python -I studies/14nov/s263/s263.py fold $V \$SLURM_ARRAY_TASK_ID")
  echo $V prep $P folds $F
done
