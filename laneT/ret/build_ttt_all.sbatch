#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-laneT-retbuild
#SBATCH --output=logs/retbuild-%j.out
# usage (lane T ret dir): sbatch build_ttt_all.sbatch -> pkg_ttt/{P04,P10}_{ftlight,ftheavy,hardcb}; hardcb = byte copy of the final-stack fold package
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
src=$base/ni26_final_stack_ret_20261120/pkg_stack
cd "${SLURM_SUBMIT_DIR:?}"; mkdir -p pkg_ttt logs
for who in P04 P10; do
  [ -d pkg_ttt/${who}_hardcb ] || cp -rL $src/${who}_hardcb pkg_ttt/${who}_hardcb
  for f in light heavy; do
    [ -d pkg_ttt/${who}_ft$f ] || $base/neural-interfaces-2026/.venv/bin/python -B ../laneT/build_ttt.py $f pkg_ttt/${who}_ft$f $src/${who}_hardcb
  done
done
