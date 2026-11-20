#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-R4-build
#SBATCH --output=logs/R4build-%j.out
# release ret/ dir (siblings ../laneR4, ../laneT): builds pkg_ret/{P04,P10}_$ARMS from the final-stack fold packages, and the NI-V / timing zips (zips/hardcb{,B}_$arm.zip)
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
src=$base/ni26_final_stack_ret_20261120/pkg_stack
py=$base/neural-interfaces-2026/.venv/bin/python
cd "${SLURM_SUBMIT_DIR:?}"; mkdir -p pkg_ret zips logs
for a in ${R4_ARMS:?}; do
  for who in P04 P10; do [ -d pkg_ret/${who}_$a ] || $py -B ../laneR4/build_keep.py $a pkg_ret/${who}_$a $src/${who}_hardcb; done
  [ -e zips/final_stack_hardcb_$a.zip ] || $py -B ../laneR4/build_keep.py $a zips/final_stack_hardcb_$a.zip $base/ni26_laneT_20261120/time/final_stack_hardcb.zip
  [ -e zips/hardcbB_$a.zip ] || $py -B ../laneR4/build_keep.py $a zips/hardcbB_$a.zip $base/ni26_niv_v1_20261119/zips/hardcbB.zip
done
for who in P04 P10; do [ -d pkg_ret/${who}_hardcb ] || cp -rL $src/${who}_hardcb pkg_ret/${who}_hardcb; done
