#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=ni26-fc-rtx
# Galvani RTX 2080 Ti checks of the upload candidates, worker SIF (torch 2.14.1+cu130). Run from the release dir
# (/mnt/lustre/work/gehler/gfh527/ni26_final_confirm_20261121). Usage: sbatch confirm_gpu.sbatch MODE ARGS...
#   hard-smoke|hard-timing ARM          (check_hard_team_b.py; ARM nine15|nine25; sha from SHAS below)
#   hard-terminal ARM PART ORDER
#   cb-smoke|cb-bench [graph|none]      (check_cb_speed.py; graph = nine25 hardcb with both graphs, none = plain hardcb)
#   e2e ZIP_NAME[=SHA]...               (submissions/final_stack/e2e_timing.py: end-to-end ms/window of the named ZIPs, first = reference)
#   pair ZIP_NAME SHA                   (friday/qualify_package.py pair on the 2080 Ti: cap250,cand,cap250,cand)
set -euo pipefail
mode=${1:?}; shift
R=${SLURM_SUBMIT_DIR:?}; G=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
tag=${SLURM_JOB_ID}
declare -A SHA=([nine15]=9ad45b11de1bcc356c56a0b7ff9ae74eb0c471f64546cb6a18f716b2f956188b [nine25]=6e09953709b2338507deaac1ec5f6afa4a21eef52de7814f7c2ae77f42e86ee9)
nvidia-smi -L
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}; export APPTAINERENV_CUDA_VISIBLE_DEVICES=${APPTAINERENV_CUDA_VISIBLE_DEVICES%%,*}  # --exclusive lists all 8 GPUs; use the first (commas also break apptainer --env)
# no --cleanenv: it dropped SLURM_JOB_ID and CUDA_VISIBLE_DEVICES, so jobs shared one GPU and ran out of memory (21 Nov)
run() { apptainer exec --nv --bind "$R" --bind "$G" \
  --env "PYTHONDONTWRITEBYTECODE=1,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,SLURM_JOB_ID=$SLURM_JOB_ID,NI26_EXPECT_GPU=2080 Ti,NI26_QUAL_GPU=2080 Ti,NI26_MAX_MB=110,BENCHOPT_DATA_HOME=$R/src/data,PYTHONPATH=$R/src/2026-competition" \
  "$G/runs/cloud/ni26_worker.sif" bash -c "cd $R/src && python -B $*"; }
cd "$R"; (cd zips && sha256sum -c ../zips.sha256)
# GPU clocks and power during the run: a GPU that idles between small kernels can downclock (diagnosis of the e2e vs bench gap)
nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.max.sm,pstate,utilization.gpu,power.draw --format=csv -l 2 > "$R/out/smi-$tag.csv" &
trap 'kill %1 2>/dev/null || true' EXIT
case $mode in
  hard-smoke|hard-timing|hard-terminal)
    arm=${1:?}; st=${mode#hard-}; extra=""
    [[ $st == terminal ]] && extra="--part ${2:?} --order ${3:?}"
    run submissions/final_stack/check_hard_team_b.py $st $arm $R/zips data/proteus_windows.npz $R/out/hard-$st-$arm-$tag.json --sha ${SHA[$arm]} $extra ;;
  cb-smoke|cb-bench)
    st=${mode#cb-}; flag=""; [[ ${1:-none} == graph ]] && flag="--team-b graph"
    run submissions/final_stack/check_cb_speed.py $st $R/zips data/proteus_windows.npz $R/out/cb-$st-${1:-none}-$tag.json $flag ;;
  e2e)
    run submissions/final_stack/e2e_timing.py $R/zips data/proteus_windows.npz $R/out/e2e-$tag.json "$@" ${E2E_FLAGS:-} ;;
  profile)  # profile_team_b_overhead.py ZIP_NAME [flags]: sync-timed Team B sub-steps inside the Solver
    run submissions/final_stack/profile_team_b_overhead.py $R/zips data/proteus_windows.npz $R/out/prof-$tag.json "$@" ;;
  convbench)  # bench_team_b_conv.py ZIP_NAME: batch-64 feature forward cost under cudnn settings
    run submissions/final_stack/bench_team_b_conv.py $R/zips data/proteus_windows.npz $R/out/conv-$tag.json "$@" ;;
  ident)  # label_identity.py REF NEW [--device cuda|cpu] [--stream-set ...] [--part N --count M]: patched vs signed ZIP labels
    run submissions/final_stack/label_identity.py $R/zips data/proteus_windows.npz $R/out/ident-$tag.json "$@" ;;
  fusebench)  # bench_team_b_fused.py ZIP_NAME
    run submissions/final_stack/bench_team_b_fused.py $R/zips data/proteus_windows.npz $R/out/fuse-$tag.json "$@" ;;
  pair)
    zip=${1:?}; sha=${2:?}
    run friday/qualify_package.py pair $R/zips/$zip $R/out/pair-${zip%.zip}-$tag --ref-zip $R/zips/v66_cp1_cap250.zip --expect-sha $sha ;;
  *) echo "bad mode"; exit 2 ;;
esac
