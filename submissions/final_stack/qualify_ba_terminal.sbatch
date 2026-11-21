#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
set -euo pipefail
host=${1:?Leonardo or Galvani}; stage=${2:?}; sha=${3:?}
run=${SLURM_SUBMIT_DIR:?}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$run/src/2026-competition" NI26_MAX_MB=450
case $host in
    leonardo)
        export NI26_EXPECT_GPU=A100 NI26_QUAL_GPU=A100
        if [[ $stage == pair || $stage == zero ]]; then
            py=("$WORK/ni26/.venv312/bin/python" -B)
        else
            compat="$run/cuda-compat/usr/local/cuda-13.0/compat"
            test -r "$compat/libcuda.so.1"
            py=(singularity exec --nv --bind "$WORK" "$WORK/ni26/runs/cloud/ni26_worker.sif"
                env "LD_LIBRARY_PATH=$compat:/.singularity.d/libs" python -B)
        fi
        ;;
    galvani)
        export NI26_EXPECT_GPU='2080 Ti' NI26_QUAL_GPU='2080 Ti'
        repo=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
        export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
        export APPTAINERENV_CUDA_VISIBLE_DEVICES=${APPTAINERENV_CUDA_VISIBLE_DEVICES%%,*}
        py=(apptainer exec --nv --bind "$run" --bind "${repo%/*}" "$repo/runs/cloud/ni26_worker.sif" python -B)
        ;;
    *) exit 2 ;;
esac
cd "$run/src"
if [[ $stage == pair ]]; then
    "${py[@]}" friday/qualify_package.py pair "$run/zips/final_stack_hard_safe_st_team_b_nine25_ba_terminal.zip" \
        "$run/out/pair-$SLURM_JOB_ID" --ref-zip "$run/zips/v66_cp1_cap250.zip" --expect-sha "$sha"
    exit
fi
flags=()
if [[ $stage == terminal ]]; then
    task=${SLURM_ARRAY_TASK_ID:?}
    order=recording
    if ((task >= 2)); then order=71000; fi
    flags=(--part "$((task % 2))" --order "$order")
fi
"${py[@]}" submissions/final_stack/check_ba_terminal.py "$stage" "$run/zips" data/proteus_windows.npz \
    "$run/out/$stage-$SLURM_JOB_ID.json" --sha "$sha" "${flags[@]}"
