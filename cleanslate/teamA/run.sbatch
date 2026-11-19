#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36G
#SBATCH --time=01:30:00
set -euo pipefail
mode=$1
arch=$2
root=$3
stage=$4
corpora=$5
orders=${6:-71000,71001}
base=/mnt/lustre/work/gehler/gfh527
repo=${TEAM_A_REPO:-$base/neural-interfaces-2026}
data=${TEAM_A_DATA:-$repo/data/proteus_windows.npz}
python=${TEAM_A_PYTHON:-$repo/.venv/bin/python}
if [[ -n "${TEAM_A_PYTHON_MODULE:-}" ]]; then
    module load "$TEAM_A_PYTHON_MODULE"
fi
code=$root/cleanslate/teamA
harness=$root/cleanslate/harness
export PYTHONPATH=$repo/2026-competition
export CS_OFFICIAL=$repo/2026-competition/tracks/bci_decoding
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$stage/matplotlib-${SLURM_JOB_ID}"
flags=(--arch "$arch" --assets "$root/assets" --data "$data" --external "$corpora" --external-dir "$root/external")
all=P01,P02,P03,P04,P05,P06,P07,P08,P09,P10
printf '%s %s %s\n' "$mode" "$arch" "$corpora"
case "$mode" in
    score)
        "$python" -B -u "$harness/score.py" "$stage" --data "$data"
        ;;
    niv_stream)
        external="$stage/niv/scherer.npz"
        held=$("$python" -c 'import numpy as np, sys; z=np.load(sys.argv[1]); print(sorted(set(z["subject"]))[int(sys.argv[2])].split("/")[-1])' \
            "$external" "$SLURM_ARRAY_TASK_ID")
        "$python" -B -u "$harness/stream.py" --data "$external" --pkg "$stage/production" --held "$held" \
            --orders "$orders" --out "$stage/niv/pred_new_$held.npz"
        ;;
    niv_score)
        "$python" -B -u "$harness/score.py" "$stage/niv" --data "$stage/niv/scherer.npz"
        ;;
    niv_prepare)
        mkdir -p "$stage/niv"
        "$python" -B -u "$code/prepare_niv.py" "${TEAM_A_SCHERER:-$repo/data/scherer2015/windows.npz}" "$stage/niv/scherer.npz"
        ;;
    smoke)
        "$python" -B -u "$code/train.py" "${flags[@]}" --people "$all" --first-session-only P04,P10 --batch 6 --smoke --out "$stage/smoke"
        "$python" -B -u "$code/check.py" "$stage/smoke" "$data"
        ;;
    external)
        "$python" -B -u "$code/train.py" "${flags[@]}" --people "$all" --external-fit --out "$stage/external"
        ;;
    fold|replay)
        row=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$stage/jobs.tsv")
        IFS='|' read -r role held people first sessions <<< "${row//$'\t'/|}"
        out="$stage/${role}_${held}"
        if [[ "$mode" == fold ]]; then
            "$python" -B -u "$code/train.py" "${flags[@]}" --people "$people" --first-session-only "$first" \
                --transfer "$stage/external/weights.pt" --out "$out"
        fi
        "$python" -B -u "$harness/stream.py" --data "$data" --pkg "$out" --held "$held" --sessions "$sessions" \
            --orders "$orders" --out "$stage/pred_${role}_${held}.npz"
        ;;
    production)
        "$python" -B -u "$code/train.py" "${flags[@]}" --people "$all" --transfer "$stage/external/weights.pt" \
            --out "$stage/production"
        ;;
    *)
        printf 'Invalid stage %s\n' "$mode" >&2
        exit 2
        ;;
esac
