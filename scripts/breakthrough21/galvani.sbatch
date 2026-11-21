#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00:35:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
work=/mnt/lustre/work/gehler/gfh527
repo=$work/neural-interfaces-2026
lead=final_stack_hard_safe_st_team_b_nine25_tbgraph_fc
lead_sha=7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07
cap_sha=9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${APPTAINERENV_CUDA_VISIBLE_DEVICES%%,*}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NI26_EXPECT_GPU='2080 Ti' NI26_QUAL_GPU='2080 Ti'
export NI26_MAX_MB=450 BENCHOPT_DATA_HOME="$run/src/data" PYTHONPATH="$run/src/2026-competition"
container() { apptainer exec --nv --bind "$work" "$repo/runs/cloud/ni26_worker.sif" python -B "$@"; }
cd "$run/src"
stage=${1:?}; zip=${2:?}; sha=${3:?}
case "$stage" in
    floor-smoke|floor-zero)
        container -c 'import sys
from argparse import Namespace
from pathlib import Path
sys.path.insert(0, "submissions/final_stack")
import check_ba_terminal as check
stage, zip, sha, run, job, check.PARENT, check.PARENT_SHA = sys.argv[1:]
check.CANDIDATE = zip + ".zip"
check.run(Namespace(stage=stage, zips=Path(run) / "zips", data=Path("data/proteus_windows.npz"),
                    out=Path(run) / "out" / f"floor-{stage}-{job}.json", sha=sha))
' "${stage#floor-}" "$zip" "$sha" "$run" "$SLURM_JOB_ID" "$lead.zip" "$lead_sha"
        ;;
    e2e)
        container submissions/final_stack/e2e_timing.py "$run/zips" data/proteus_windows.npz \
            "$run/out/e2e-$SLURM_JOB_ID.json" \
            "v66_cp1_cap250=$cap_sha" "$lead=$lead_sha" "$zip=$sha"
        ;;
    pair)
        container friday/qualify_package.py pair "$run/zips/$zip.zip" "$run/out/pair-$SLURM_JOB_ID" \
            --ref-zip "$run/zips/v66_cp1_cap250.zip" --expect-sha "$sha"
        ;;
    ident)
        part=${4:-0}; count=${5:-1}
        container submissions/final_stack/label_identity.py "$run/zips" data/proteus_windows.npz \
            "$run/out/identity-$SLURM_JOB_ID.json" "${BT_IDREF:-$lead=$lead_sha}" "$zip=$sha" \
            --part "$part" --count "$count" ${BT_IDFLAGS:-}
        ;;
    *) exit 2 ;;
esac
