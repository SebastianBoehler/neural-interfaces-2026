#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source "$run/env.sh"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$run/timing-src/2026-competition"
module load python/3.11.7
cd "$run/timing-src/submissions/final_stack"
arms=(
    v66_cp1_cap250=9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed
    final_stack_hard_safe_st_team_b_nine25_tbgraph_fc=7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07
    final_stack_hard_safe_st_team_b_nine25_ba_terminal_fc=1ef25b365c609354f275295f7e36fa9d1d71bbb02c8815ff4362ead078e6b23e
)
rank=${SLURM_PROCID:-0}
printf 'Timing repeat %s on %s\n' "$rank" "${SLURMD_NODENAME:?}"
"$CAP250_PYTHON" -B - "${1:?smoke or measure}" "$run/zips" "$CAP250_DATA" "$run/out/e2e-$SLURM_JOB_ID-$rank.json" "${arms[@]}" <<'PY'
import sys
from itertools import islice

import e2e_timing as timing

mode = sys.argv.pop(1)
gpu = timing.torch.cuda.get_device_name()
if "A100-SXM" not in gpu:
    raise RuntimeError(f"expected A100-SXM, got {gpu}")
if mode == "smoke":
    plans = timing.C.plans
    timing.C.plans = lambda *args: dict(islice(plans(*args).items(), 3))
    stream = timing.C.stream

    def check_graph(model, windows):
        path = "graph" if getattr(model, "team_b", None) is not None else None
        return stream(model, windows, team_path=path)

    timing.C.stream = check_graph
elif mode != "measure":
    raise ValueError(mode)
timing.main()
PY
