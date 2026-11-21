# Final hard-only Team B · 21 November 2026

Both ZIPs start from `final_stack_hard_safe_st.zip` (`239c86bd18d0a7e9d0216dda1fb95d9b918bf6e933c0383322446aed37a83c1e`). The builder copies the mixer, encoder and CUDA graph from `final_stack_hardcb_safe_team_b_nine25_cbgraph_tbgraph.zip` (`911a31af329c3f38f75ae8a0f1331e18bc921e6c36b002538839ba04a2384c4e`). Every base entry except `submission.py` is byte-identical. The added `team_b.pt` is the signed nine-member bank (`40ccb568394fc8b9ee6f1e3d5a6f144fe01481abe75ad6fab45841193e7de15b`). The only difference between candidate source files is the blend weight. Source `abc700f0`. [Builder](../../submissions/final_stack/build_hard_team_b.py) · [Registration](../../docs/LEDGER.md#final-hard-only-team-b-archives--21-november)

| ZIP in `runs/sealed/final_zips/` | Bytes | SHA-256 |
|---|---:|---|
| `final_stack_hard_safe_st_team_b_nine15_tbgraph.zip` | 72,463,472 | `9ad45b11de1bcc356c56a0b7ff9ae74eb0c471f64546cb6a18f716b2f956188b` |
| `final_stack_hard_safe_st_team_b_nine25_tbgraph.zip` | 72,463,472 | `6e09953709b2338507deaac1ec5f6afa4a21eef52de7814f7c2ae77f42e86ee9` |

Both ZIPs passed CRC and inventory checks and rebuilt byte for byte. Local `replay/check.py` returned `OK`. The Mac CPU smoke matched zero-blend mixer, final scores and labels on the first eight windows of P01/1 and P04/2. Clean load restored hard and Team B state. An injected Team B exception and a rolled posterior failed `load_model` in both ZIPs. An injected live error marked each session dead and the guard let all eight predictions return valid labels.

The Team B load check compares two deterministic posteriors with signed-bank CPU references at absolute tolerance `0.001`. CUDA load also checks real graph capture and replay for batches of 1 and 64 and validates the batch-64 features. The package resets the Team B encoder after the checks. The live path uses the preserved dead-session guard inherited from hard_safe_st.

Leonardo Slurm run `cap250/runs/final-hard-teamb-21nov`. Terminal and member timing use the worker SIF on A100, with the same process-local CUDA 13.0 compatibility libraries as the donor qualification. The cap250 pairs use the native Leonardo A100 environment and pinned cap250 SHA `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed`. [Dispatch](DISPATCH_21NOV.md) · [Checker](../../submissions/final_stack/check_hard_team_b.py) · [Qualification](../../friday/qualify_package.py)

Public replay checks package behavior on all-data weights. Fresh fold evidence for hard-only nine15 was +0.77 pp held-new [+0.43, +1.09] and +0.47 pp returning. Nine25 was +0.97 pp [+0.31, +1.54] and +0.59 pp returning, using eight fresh orders and people-nested 80% intervals. [Lane TB-verify](laneTB_verify.md)
