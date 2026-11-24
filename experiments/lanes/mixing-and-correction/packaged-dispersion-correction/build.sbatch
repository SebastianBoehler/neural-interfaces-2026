#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-laneP2-build
#SBATCH --output=logs/build-%j.out
# usage (release dir): sbatch laneP2/build.sbatch -> zips/{ds_l5,ds_all,fbl2chk,hardcbB_ds_l5,hardcbB_ds_all}.zip + SHA256SUMS
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
niv=/mnt/lustre/work/gehler/gfh527/ni26_niv_v1_20261119/zips
rm -f zips/ds_l5.zip zips/ds_all.zip zips/fbl2chk.zip zips/hardcbB_ds_l5.zip zips/hardcbB_ds_all.zip
$PY -B laneP2/build_p2.py ds_l5 zips/ds_l5.zip zips/final_stack_hardcb_tttlite5.zip
$PY -B laneP2/build_p2.py ds_all zips/ds_all.zip zips/final_stack_hardcb_tttlite5.zip
BUILD_P2_ANY_FT=1 $PY -B laneP2/build_p2.py ds_all zips/fbl2chk.zip zips/final_stack_hardcb.zip
$PY -B laneP2/build_p2.py ds_l5 zips/hardcbB_ds_l5.zip $niv/hardcbB_tttlite5.zip
$PY -B laneP2/build_p2.py ds_all zips/hardcbB_ds_all.zip $niv/hardcbB_tttlite5.zip
sha256sum zips/*.zip $niv/hardcbB_tttlite5.zip | tee zips/SHA256SUMS
