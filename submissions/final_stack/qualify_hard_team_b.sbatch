#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail

stage=${1:?missing stage}
arm=${2:?missing dose}
sha=${3:?missing archive SHA}
run=${SLURM_SUBMIT_DIR:?}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$run/src/2026-competition"
cd "$run/src"
zips="$run/zips"
if [[ $stage == pair || $stage == zero ]]; then
    source "$WORK/ni26/.venv312/bin/activate"
    py=(python -B)
else
    compat="$run/cuda-compat/usr/local/cuda-13.0/compat"
    test -r "$compat/libcuda.so.1"
    py=(singularity exec --nv --bind "$WORK" "$WORK/ni26/runs/cloud/ni26_worker.sif"
        env "LD_LIBRARY_PATH=$compat:/.singularity.d/libs" python -B)
fi
if [[ $stage == pair ]]; then
    "${py[@]}" friday/qualify_package.py pair "$zips/final_stack_hard_safe_st_team_b_${arm}_tbgraph.zip" \
        "$run/pair-$arm-$SLURM_JOB_ID" --ref-zip "$zips/v66_cp1_cap250.zip" --expect-sha "$sha"
    exit
fi

flags=()
if [[ $stage == terminal ]]; then
    task=${SLURM_ARRAY_TASK_ID:?missing terminal task}
    order=recording
    if ((task >= 2)); then order=71000; fi
    flags=(--part "$((task % 2))" --order "$order")
fi
"${py[@]}" submissions/final_stack/check_hard_team_b.py "$stage" "$arm" "$zips" \
    data/proteus_windows.npz "$run/$stage-$arm-$SLURM_JOB_ID.json" --sha "$sha" "${flags[@]}"
