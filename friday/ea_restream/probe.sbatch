#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:04:00
#SBATCH --job-name=ni26-dea-preflight
#SBATCH --output=dea-%A_%a.out
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
REPO=$BASE/neural-interfaces-2026
cd "$BASE/ni26_friday_ea_restream_20261118"
sha256sum --quiet -c source.sha256
sha256sum --quiet -c archive.sha256
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export BENCHOPT_DATA_HOME="$REPO/data"
families=(shallow_ea fbcnet_ea_tf)
revisions=(82120152ace8731cb262c86c8f00b765f6b81418 c625ecc736a881836a25f85a741ea9147bf698bf)
mode=${1:?expected inputs or reconstruct}
if [[ "$mode" == inputs ]]; then
    cells=(0 1)
elif [[ "$mode" == reconstruct ]]; then
    cells=("${SLURM_ARRAY_TASK_ID:?}")
else
    exit 2
fi
for id in "${cells[@]}"; do
    export PYTHONPATH="$PWD/${revisions[$id]}/experiments:$REPO/2026-competition:$REPO/2026-competition/tracks/bci_decoding"
    "$REPO/.venv/bin/python" -B probe.py "$REPO" "${families[$id]}" "$mode"
done
