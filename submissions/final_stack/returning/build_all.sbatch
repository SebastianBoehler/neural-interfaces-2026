#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-stackA-build
#SBATCH --output=logs/stackA-build-%j.out
# usage (release dir): sbatch build_all.sbatch      -> pkg_stack/{P04,P10}_{hard,cb,hardcb}
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
cd "${SLURM_SUBMIT_DIR:?}"; mkdir -p pkg_stack logs
for who in P04 P10; do for arm in hard cb hardcb; do
  [ -d pkg_stack/${who}_$arm ] || STACK_HARD_DIR=$PWD $base/neural-interfaces-2026/.venv/bin/python -B build_fold.py $who $arm pkg_stack/${who}_$arm \
    $base/ni26_js321_20261117_returning_bundle2/${who}_s23/base $base/ni26_friday_return_c20_actors_v2_20261119 $PWD/blobs_returning
done; done
