#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
stage=${1:?missing stage}
run=${SLURM_SUBMIT_DIR:?}
package=()
if [[ -n ${2:-} ]]; then package=(--team-b "$2"); fi
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$run/src/2026-competition"
py=(python -B)
if [[ ${NI26_CB_WORKER:-0} == 1 ]]; then
    # R535 supports CUDA 13.0 with NVIDIA's process-local compatibility libraries.
    compat="$run/cuda-compat/usr/local/cuda-13.0/compat"
    if [[ $stage == smoke && ! -r "$compat/libcuda.so.1" ]]; then
        deb=cuda-compat-13-0_580.196.07-1_amd64.deb
        (cd "$run"; printf '%s  %s\n' d2d3262512046b23215ff197fe79a6ca0d44299ed0e665275ac05bb28b79e4e3 "$deb" | sha256sum -c)
        mkdir -p "$run/cuda-compat"
        ar p "$run/$deb" data.tar.xz | tar -xJf - -C "$run/cuda-compat"
    fi
    test -r "$compat/libcuda.so.1"
    py=(singularity exec --nv --bind "$WORK" "$WORK/ni26/runs/cloud/ni26_worker.sif"
        env "LD_LIBRARY_PATH=$compat:/.singularity.d/libs" python -B)
else
    source "$WORK/ni26/.venv312/bin/activate"
fi
cd "$run/src"
check=submissions/final_stack/check_cb_speed.py
data=data/proteus_windows.npz
if [[ $stage == smoke ]]; then
    "${py[@]}" "$check" smoke "$run/zips" "$data" "$run/smoke-$SLURM_JOB_ID.json" "${package[@]}"
elif [[ $stage == bench ]]; then
    "${py[@]}" "$check" bench "$run/zips" "$data" "$run/bench-$SLURM_JOB_ID.json" "${package[@]}"
elif [[ $stage == options-smoke || $stage == options ]]; then
    flags=()
    if [[ $stage == options-smoke ]]; then flags+=(--smoke); fi
    "${py[@]}" submissions/final_stack/cb_speed_options.py "$run/zips" "$data" \
        "$run/$stage-$SLURM_JOB_ID.json" "${flags[@]}"
elif [[ $stage == terminal ]]; then
    task=${SLURM_ARRAY_TASK_ID:?expected task 0-3}
    if ((task > 3)); then exit 2; fi
    order=recording
    if ((task / 2)); then order=71000; fi
    "${py[@]}" "$check" terminal "$run/zips" "$data" "$run/terminal-$SLURM_ARRAY_JOB_ID-$task.json" \
        --order "$order" --part "$((task % 2))" 2 "${package[@]}"
else
    exit 2
fi
