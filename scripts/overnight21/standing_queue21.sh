#!/bin/bash
# Galvani standing queue 21 Nov (run from the release dir ni26_final_confirm_20261121, sbatch only).
# (1) Chained repeated Solver end-to-end timing (afterany chains, 4 links each) of the signed lead, nine15 and hardcb cbgraph on 2080 Ti and A100-PCIe.
# (2) Release E NI-V (orders 71024-71031) submitted by a cpu job after the release D collectors, once the 300-task cap has room.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
n25=final_stack_hard_safe_st_team_b_nine25_tbgraph=6e09953709b2338507deaac1ec5f6afa4a21eef52de7814f7c2ae77f42e86ee9
n15=final_stack_hard_safe_st_team_b_nine15_tbgraph=9ad45b11de1bcc356c56a0b7ff9ae74eb0c471f64546cb6a18f716b2f956188b
cap=v66_cp1_cap250; cbg=final_stack_hardcb_safe_st_cbgraph
A100=(-p a100-galvani --gres=gpu:a100:1)
chain() { # name n_links gpuargs... -- zips...
  local nm=$1 n=$2; shift 2; local g=(); while [ "$1" != -- ]; do g+=("$1"); shift; done; shift
  local prev=""; local ids=""
  for i in $(seq $n); do
    local dep=(); [ -n "$prev" ] && dep=(--dependency=afterany:$prev)
    prev=$(sbatch --parsable -o "logs/%x-%j.log" -J $nm "${dep[@]}" "${g[@]}" confirm_gpu.sbatch e2e "$@"); ids="$ids $prev"
  done; echo "$nm:$ids"
}
for k in 1 2 3 4; do chain s21-e2e-hard 4 -- $cap $n25 $n15; done
for k in 1 2; do chain s21-e2e-cb 4 -- $cap $cbg; done
for k in 1 2; do chain s21-e2e-A100h 4 "${A100[@]}" -- $cap $n25 $n15; done
chain s21-e2e-A100cb 4 "${A100[@]}" -- $cap $cbg
