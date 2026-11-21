#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=18G
#SBATCH --time=01:30:00
set -euo pipefail
mode=${1:?}
root=/mnt/lustre/work/gehler/gfh527
repo=$root/neural-interfaces-2026
run=${SLURM_SUBMIT_DIR:?}
cd "$run"
task=${SLURM_ARRAY_TASK_ID:-0}
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${APPTAINERENV_CUDA_VISIBLE_DEVICES%%,*}
export APPTAINERENV_NIV_VISITS=1,2
sha=()
bind="$run:/work:rw,$repo/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$run/scherer_rows.tsv.gz:/rows.tsv.gz:ro,$run/zips/ba_terminal_niv.zip:/candidate.zip:ro"
case $mode in
    smoke) who=5; arm=full:0; out=out/ba/smoke.npz ;;
    new)
        who=$((task / 4)); chunk=$((task % 4)); a=$((chunk * 2)); b=$((a + 1))
        arm="full:$a,full:$b,within:$a,within:$b"; out=out/ba/p${who}_c${chunk}.npz ;;
    smoke-ret|ret)
        who=$task; export APPTAINERENV_NIV_VISITS=2
        enrolled="$root/ni26_niv_finalA_20261121/zips/enrolled/hss_n25_k${who}.zip"
        bind+=",$enrolled:/enrolled.zip:ro"
        sha=("$(awk -v name="hss_n25_k${who}.zip" '$2 == name {print $1}' enrolled.sha256)")
        [[ ${#sha[0]} == 64 ]]
        out="out/ret_ba_k$who/p${who}_c0.npz"
        if [[ $mode == smoke-ret ]]; then
            arm=full:0; out="out/ret_ba_k$who/smoke.npz"
        else
            arm=$(python3 -c "print(','.join(f'{s}:{i}' for s in ('full','within') for i in range(8)))")
        fi
        ;;
    *) exit 2 ;;
esac
mkdir -p "$(dirname "$out")"
test ! -e "$out"
apptainer exec --nv --cleanenv --containall --no-home --no-mount hostfs \
    --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONDONTWRITEBYTECODE=1,TMPDIR=/work \
    --bind "$bind" --pwd /work "$repo/runs/cloud/ni26_worker.sif" \
    python -B ba_terminal_niv.py "/work/$out" "$who" "$arm" "${sha[@]}"
