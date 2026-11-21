#!/bin/bash
# Queue for evidence/wave_20nov/teamb_overhead.md (run from the Galvani release dir ni26_final_confirm_20261121).
# Label identity of the fused-conv patch (v2) (2080 Ti, all 14 sessions of shuffle 71000, 7 parts per ZIP; A100 in 4 parts),
# then end-to-end Solver timing repeats on the 2080 Ti (3x) and the A100-PCIe (2x) with arms alternating.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
n25=final_stack_hard_safe_st_team_b_nine25_tbgraph=6e09953709b2338507deaac1ec5f6afa4a21eef52de7814f7c2ae77f42e86ee9
n25c=final_stack_hard_safe_st_team_b_nine25_tbgraph_fc=7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07
n15=final_stack_hard_safe_st_team_b_nine15_tbgraph=9ad45b11de1bcc356c56a0b7ff9ae74eb0c471f64546cb6a18f716b2f956188b
n15c=final_stack_hard_safe_st_team_b_nine15_tbgraph_fc=6e6bbbb841db4ff1ea5e83e4bad75a090ab0a30455fa5ca570063b5ba828d1d4
cbn=final_stack_hardcb_safe_team_b_nine25_cbgraph_tbgraph=911a31af329c3f38f75ae8a0f1331e18bc921e6c36b002538839ba04a2384c4e
cbnc=final_stack_hardcb_safe_team_b_nine25_cbgraph_tbgraph_fc=086c7600555efba5545c035e1ef3c4245ab036599b588400c34ac118753fcd73
cap=v66_cp1_cap250
cbg=final_stack_hardcb_safe_st_cbgraph

A100=(-p a100-galvani --gres=gpu:a100:1)
S() { sbatch --parsable -o "logs/%x-%j.log" "$@"; }
for p in 0 1 2 3 4 5 6; do
  S -J tbo2-id-n25 confirm_gpu.sbatch ident $n25 $n25c --part $p --count 7
  S -J tbo2-id-n15 confirm_gpu.sbatch ident $n15 $n15c --part $p --count 7
  S -J tbo2-id-cb confirm_gpu.sbatch ident $cbn $cbnc --part $p --count 7
done
for p in 0 1 2 3; do S -J tbo2-id-a100 "${A100[@]}" confirm_gpu.sbatch ident $n25 $n25c --part $p --count 4; done
for r in 1 2 3; do
  S -J tbo2-e2e-A confirm_gpu.sbatch e2e $cap $n25 $n25c $cbg
  S -J tbo2-e2e-B confirm_gpu.sbatch e2e $cap $n15 $n15c $cbn $cbnc
done
for r in 1 2; do
  S -J tbo2-e2e-A100A "${A100[@]}" confirm_gpu.sbatch e2e $cap $n25 $n25c $cbg
  S -J tbo2-e2e-A100B "${A100[@]}" confirm_gpu.sbatch e2e $cap $n15 $n15c $cbn $cbnc
done
S -J tbo2-id-cpu confirm_gpu.sbatch ident $n25 $n25c --device cpu --stream-set zero379
