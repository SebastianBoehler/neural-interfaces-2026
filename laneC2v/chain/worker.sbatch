#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=01:00:00
#SBATCH --job-name=ni26-laneC2v
#SBATCH --output=laneC2v-%x-%j.out
# usage (cd to the lane C2v chain dir): sbatch worker.sbatch hardcb|ds_l5   (pinned scorer/feature modules from the lane S2 parent, as score_s2)
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
camp=$base/ni26_friday_causal_rebuild_20261118
repo=$base/neural-interfaces-2026
parent=$camp/within0_terminal_shared_v1
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --check --status source.sha256
echo 'ad7f624831d9e25ab636a7096a07a6e64999bdf47e7dcea3994e4210b84d8326  source.sha256' | (cd "$parent" && sha256sum --check --status)
(cd "$parent" && sha256sum --check --status source.sha256)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$work:$parent:$camp/fc18_necessity_v2:$camp/fc19_csp_gate_v2:$camp/fc20_outer_v1:$base/ni26_final_stack_20261120"
exec "$repo/.venv/bin/python" -B votes_c2v.py "${1:?}"
