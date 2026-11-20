#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-laneP2-buildu
#SBATCH --output=logs/buildu-%j.out
# usage (release dir): sbatch laneP2/build_u.sbatch -> zips/{dsu_l5,dsu_all,hardcbB_dsu_l5,hardcbB_dsu_all}.zip (correction only on unrecognised sessions) + SHA256SUMS_u
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
niv=/mnt/lustre/work/gehler/gfh527/ni26_niv_v1_20261119/zips
rm -f zips/dsu_l5.zip zips/dsu_all.zip zips/hardcbB_dsu_l5.zip zips/hardcbB_dsu_all.zip
$PY -B laneP2/build_p2.py dsu_l5 zips/dsu_l5.zip zips/final_stack_hardcb_tttlite5.zip
$PY -B laneP2/build_p2.py dsu_all zips/dsu_all.zip zips/final_stack_hardcb_tttlite5.zip
$PY -B laneP2/build_p2.py dsu_l5 zips/hardcbB_dsu_l5.zip $niv/hardcbB_tttlite5.zip
$PY -B laneP2/build_p2.py dsu_all zips/hardcbB_dsu_all.zip $niv/hardcbB_tttlite5.zip
sha256sum zips/dsu_*.zip zips/hardcbB_dsu_*.zip | tee zips/SHA256SUMS_u
