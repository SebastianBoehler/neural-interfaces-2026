#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --job-name=rt-niv-kappa40

set -euo pipefail
W=/leonardo_work/AIFAC_P02_1636
R4=$W/ni26_laneR4_20261120
RUN=$W/ni26/cap250/runs/redteam-21nov
CODE=$RUN/src/scripts/redteam/niv_dose.py
source "$W/ni26/.venv/bin/activate"
export PYTHONPATH=$R4/stubs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NIV_VISITS=2
k=${SLURM_ARRAY_TASK_ID:-0}
pkg=$RUN/niv/kappa40_k$k.zip
mkdir -p "$RUN/niv"
case ${1:?smoke or full} in
  smoke)
    out=$RUN/niv/smoke.npz
    arms=full:s71008
    ;;
  full)
    out=$RUN/niv/ret-$k.npz
    arms=$(python -c "print(','.join(f'{s}:s{i}' for s in ('full', 'within') for i in range(71008, 71024)))")
    ;;
  *) exit 2 ;;
esac
if [[ ! -f $pkg || $1 == smoke || $k != 0 ]]; then
  python -B "$CODE" build "$R4/enrolled/hardcbB_k$k.zip" "$pkg"
fi
cd "$R4"
python -B "$R4/niv_leo/worker.py" "$pkg" "$out" "$k" "$arms"
if [[ $1 == smoke ]]; then
  python -B "$CODE" smoke "$R4/out/hardcbB_k$k/ret.npz" "$out"
fi
