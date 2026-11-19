#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-laneE-build
#SBATCH --output=logs/build-%A.out
# Builds E0 Ea Eb Ec Ed Ed_g2 on the cluster (the zips are ~70 MB each; the code and members are small) into zips/, copies the
# five scored arms to the NI-V release dir as laneE_<arm>.zip and pins them in its inputs.sha256.
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527; work=${SLURM_SUBMIT_DIR:?}; R=$base/ni26_niv_v1_20261119
cd "$work"; rm -f zips/E*.zip
PY=$base/neural-interfaces-2026/.venv/bin/python
sha256sum --quiet -c code.sha256
stock=$R/zips/v66_cp1_cap250.zip
$PY -I code/lane_e/make_arms.py $stock members zips --guard 150
$PY -I code/lane_e/make_arms.py $stock members zips --guard 2 --arms Ed --suffix _g2
for a in E0 Ea Eb Ec Ed; do cp zips/$a.zip $R/zips/laneE_$a.zip; (cd $R && sha256sum zips/laneE_$a.zip >> inputs.sha256); done
ln -sf $stock zips/v66_cp1_cap250.zip
sha256sum zips/*.zip > zips.sha256; cat zips.sha256
