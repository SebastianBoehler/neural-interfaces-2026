#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-laneP2-retbuild
#SBATCH --output=logs/retbuild-%j.out
# usage (lane P2 ret dir): sbatch build_fold.sbatch -> pkg_ttt/{P04,P10}_{hardcb,ftlite5,ftds_l5,ftds_all}; hardcb = byte copy of the final-stack fold package,
# ftlite5 = build_ttt.py lite5 on it, ftds_* = build_p2.py on ftlite5 (fails loudly if the fold package lacks the dispersion member or lite5)
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
src=$base/ni26_final_stack_ret_20261120/pkg_stack
rel=$base/ni26_laneP2_20261120
PY=$base/neural-interfaces-2026/.venv/bin/python
cd "${SLURM_SUBMIT_DIR:?}"; mkdir -p pkg_ttt logs
for who in P04 P10; do
  rm -rf pkg_ttt/${who}_*
  cp -rL $src/${who}_hardcb pkg_ttt/${who}_hardcb
  $PY -B $base/ni26_laneT_20261120/laneT/build_ttt.py lite5 pkg_ttt/${who}_ftlite5 $src/${who}_hardcb
  for f in ds_l5 ds_all dsu_l5 dsu_all; do $PY -B $rel/laneP2/build_p2.py $f pkg_ttt/${who}_ft$f pkg_ttt/${who}_ftlite5; done
done
ls pkg_ttt; sha256sum pkg_ttt/*/submission.py
