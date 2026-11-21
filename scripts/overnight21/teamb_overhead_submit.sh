#!/bin/bash
# Queue for evidence/wave_20nov/teamb_overhead.md (run from the Galvani release dir ni26_final_confirm_20261121).
# Label identity of the channels_last patch (2080 Ti, all 14 sessions of shuffle 71000, 7 parts per ZIP; A100 in 4 parts),
# then end-to-end Solver timing repeats on the 2080 Ti (3x) and the A100-PCIe (2x) with arms alternating.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
n25=final_stack_hard_safe_st_team_b_nine25_tbgraph=6e09953709b2338507deaac1ec5f6afa4a21eef52de7814f7c2ae77f42e86ee9
n25c=final_stack_hard_safe_st_team_b_nine25_tbgraph_cl=9e3e43c7829114d2be8387b4a85d07b034e0a476fa17430bc8c3676567855541
n15=final_stack_hard_safe_st_team_b_nine15_tbgraph=9ad45b11de1bcc356c56a0b7ff9ae74eb0c471f64546cb6a18f716b2f956188b
n15c=final_stack_hard_safe_st_team_b_nine15_tbgraph_cl=147bb3aa94824a9a075a3397fa7f07868d05bbbd33388a57a655f69595575d75
cbn=final_stack_hardcb_safe_team_b_nine25_cbgraph_tbgraph=911a31af329c3f38f75ae8a0f1331e18bc921e6c36b002538839ba04a2384c4e
cbnc=final_stack_hardcb_safe_team_b_nine25_cbgraph_tbgraph_cl=8cac82d764c0eb81a01092a1dc26b0f9c68c41fd5cfa0b02f2e7752ea6ceac06
cap=v66_cp1_cap250
cbg=final_stack_hardcb_safe_st_cbgraph
A100=(-p a100-galvani --gres=gpu:a100:1)
S() { sbatch --parsable -o "logs/%x-%j.log" "$@"; }
for p in 0 1 2 3 4 5 6; do
  S -J tbo-id-n25 confirm_gpu.sbatch ident $n25 $n25c --part $p --count 7
  S -J tbo-id-n15 confirm_gpu.sbatch ident $n15 $n15c --part $p --count 7
  S -J tbo-id-cb confirm_gpu.sbatch ident $cbn $cbnc --part $p --count 7
done
for p in 0 1 2 3; do S -J tbo-id-a100 "${A100[@]}" confirm_gpu.sbatch ident $n25 $n25c --part $p --count 4; done
for r in 1 2 3; do
  S -J tbo-e2e-A confirm_gpu.sbatch e2e $cap $n25 $n25c $cbg
  S -J tbo-e2e-B confirm_gpu.sbatch e2e $cap $n15 $n15c $cbn $cbnc
done
for r in 1 2; do
  S -J tbo-e2e-A100A "${A100[@]}" confirm_gpu.sbatch e2e $cap $n25 $n25c $cbg
  S -J tbo-e2e-A100B "${A100[@]}" confirm_gpu.sbatch e2e $cap $n15 $n15c $cbn $cbnc
done
