#!/bin/bash
# Runs ON the Leonardo login node (submits Slurm jobs only): bash leo_submit.sh ACCOUNT   from the Lane E release dir.
set -euo pipefail
ACC=${1:?}; cd "$(dirname "$0")/../.."
ln -sfn "${LEO_REPO:-/leonardo_work/AIFAC_P02_1636/ni26}/2026-competition" code/2026-competition; ln -sfn "$PWD/data" code/data
sed -i -e 's#"/official/tracks/bci_decoding"#os.environ["NIV_OFFICIAL"] + "/tracks/bci_decoding"#' -e 's#"/external.npz"#os.environ["NIV_EXTERNAL"]#' \
  -e 's#"/rows.tsv.gz"#os.environ["NIV_ROWS"]#' niv_leo/worker.py
test "$(grep -c 'NIV_' niv_leo/worker.py)" -ge 3
sha256sum zips/*.zip > zips.sha256; sha256sum data/* niv_leo/*.py > inputs.sha256
echo "timing A100 $(sbatch --parsable --account=$ACC leo_timing.sbatch)"
declare -A J
for a in cap250:v66_cp1_cap250 E0:E0 Ea:Ea Eb:Eb Ec:Ec Ed:Ed; do
  J[${a%%:*}]=$(sbatch --parsable --account=$ACC --array=0-35 leo_score.sbatch "${a##*:}.zip" "${a%%:*}"); echo "${a%%:*} ${J[${a%%:*}]}"
done
for a in Ea Eb Ec Ed; do
  echo "collector $a $(sbatch --parsable --account=$ACC --dependency=afterany:${J[$a]}:${J[cap250]}:${J[E0]} leo_collect.sbatch $a cap250 E0)"
done
echo "null collector E0 $(sbatch --parsable --account=$ACC --dependency=afterany:${J[E0]}:${J[cap250]} leo_collect.sbatch E0 cap250 '')"
