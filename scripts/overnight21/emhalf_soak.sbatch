#!/bin/bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%j.log

set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$run/src/2026-competition:$run/src/cleanslate/harness"
source "$WORK/ni26/.venv312/bin/activate"

mode=${1:?smoke, measure or worker}
cutoff=${2:?Unix time for measurement cutoff}
source_commit=$(cat "$run/source.rev")
case $mode in
    smoke)
        # The historical v3 regression is the repo preflight. Only its OK marker is surfaced.
        export NI26_REPLAY_DATA="$WORK/ni26/runs/replay_data"
        preflight=$(mktemp)
        trap 'rm -f "$preflight"' EXIT
        if ! python -B "$WORK/ni26/replay/check.py" >"$preflight" ||
           ! grep -x OK "$preflight"; then
            echo "Historical v3 regression failed" >&2
            exit 1
        fi
        arm=smoke
        ;;
    measure)
        # Independent arms continue if a peer fails. One rank per node is set by sbatch.
        exec srun --kill-on-bad-exit=0 --output="$run/logs/arm-%j-%t.log" \
            bash "$run/src/scripts/overnight21/emhalf_soak.sbatch" worker "$cutoff"
        ;;
    worker)
        arms=(em_half_a em_half_b cap250 nine25_fc)
        arm=${arms[${SLURM_PROCID:?}]}
        ;;
    *) exit 2 ;;
esac
python -Bu "$run/src/submissions/final_stack/emhalf_soak.py" "$arm" "$run/zips" \
    "$WORK/ni26/cap250/assets/data/proteus_windows.npz" "$run/out/$arm-$SLURM_JOB_ID.json" \
    --cutoff "$cutoff" --source-commit "$source_commit"
