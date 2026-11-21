# Final hard-only Team B · 21 November 2026

Both ZIPs start from `final_stack_hard_safe_st.zip` (`239c86bd18d0a7e9d0216dda1fb95d9b918bf6e933c0383322446aed37a83c1e`). The builder copies the mixer, encoder and CUDA graph from `final_stack_hardcb_safe_team_b_nine25_cbgraph_tbgraph.zip` (`911a31af329c3f38f75ae8a0f1331e18bc921e6c36b002538839ba04a2384c4e`). Every base entry except `submission.py` is byte-identical. The added `team_b.pt` is the signed nine-member bank (`40ccb568394fc8b9ee6f1e3d5a6f144fe01481abe75ad6fab45841193e7de15b`). The only difference between candidate source files is the blend weight. Source `abc700f0`. [Builder](../../submissions/final_stack/build_hard_team_b.py) · [Registration](../../docs/LEDGER.md#final-hard-only-team-b-archives--21-november)

| ZIP in `runs/sealed/final_zips/` | Bytes | SHA-256 |
|---|---:|---|
| `final_stack_hard_safe_st_team_b_nine15_tbgraph.zip` | 72,463,472 | `9ad45b11de1bcc356c56a0b7ff9ae74eb0c471f64546cb6a18f716b2f956188b` |
| `final_stack_hard_safe_st_team_b_nine25_tbgraph.zip` | 72,463,472 | `6e09953709b2338507deaac1ec5f6afa4a21eef52de7814f7c2ae77f42e86ee9` |

Both ZIPs passed CRC and inventory checks and rebuilt byte for byte. Local `replay/check.py` returned `OK`. The Mac CPU smoke matched zero-blend mixer, final scores and labels on the first eight windows of P01/1 and P04/2. Clean load restored hard and Team B state. An injected Team B exception and a rolled posterior failed `load_model` in both ZIPs. A separate Mac CPU check injected an error at the fourth live window after three successful v6 calls. The guard marked the session dead and returned all eight labels in each ZIP.

The Team B load check compares two deterministic posteriors with signed-bank CPU references at absolute tolerance `0.001`. CUDA load also checks real graph capture and replay for batches of 1 and 64 and validates the batch-64 features. The package resets the Team B encoder after the checks. The live path uses the preserved dead-session guard inherited from hard_safe_st.

Leonardo Slurm run `cap250/runs/final-hard-teamb-21nov`. Terminal and member timing use the worker SIF on A100, with the same process-local CUDA 13.0 compatibility libraries as the donor qualification. The cap250 pairs use the native Leonardo A100 environment and pinned cap250 SHA `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed`. [Dispatch](DISPATCH_21NOV.md) · [Checker](../../submissions/final_stack/check_hard_team_b.py) · [Qualification](../../friday/qualify_package.py)

Worker SIF smokes 65001147 and 65001152 passed on A100-SXM-64GB with torch 2.14.1+cu130. Both capture-size faults, the posterior exception and the rolled posterior raised the Team B self-test error. The 72-window P01/1 prefix matched eager labels in recording order and shuffle 71000 while exercising both graph shapes.

## Full public replay checks

| Check | nine15 | nine25 |
|---|---|---|
| CPU blend zero vs hard_safe_st, shuffle 71120 | Bitwise equal on 379 windows · 65001148 | Bitwise equal on 379 windows · 65001153 |
| A100 graph vs eager labels, recording order | 0 / 2,655 changed · 65001149 tasks 0–1 | 0 / 2,655 changed · 65001155 tasks 0–1 |
| A100 graph vs eager labels, shuffle 71000 | 0 / 2,655 changed · 65001149 tasks 2–3 | 0 / 2,655 changed · 65001155 tasks 2–3 |

CPU blend-zero compares the mixer scores, final scores and labels on P01/1 (190 windows) and P04/2 (189). Each A100 delivery order covered all 14 sessions through the actual Solver and `fresh_model`. The eager arm cleared the feature graph map after each reset. The production arm replayed both graph sizes at the expected counts, used Team B on every window, and matched eager recognition ages and refit counts. Template state stayed empty and the normal streams returned zero v6 errors.

## Official A100 warm-up qualification

`friday/qualify_package.py pair` passed for both ZIPs on Leonardo A100-SXM-64GB. Each job replayed cap250, candidate, cap250, candidate using pinned official warm-up ingestion and scoring. Each replay covered 755 windows and four sessions. Both repeats agreed on all predictive scores per arm, preserved extracted-file hashes and reported zero v6 errors. Cell-balanced accuracy was **0.9959229390681004** for both candidates and **0.994578853046595** for cap250 (+0.1344 pp).

| Dose | Job | cap250 ingestion wall, s | Candidate ingestion wall, s | Warm wall difference, s |
|---|---:|---:|---:|---:|
| nine15 | 65001151 | 273.9 / 126.9 | 159.8 / 145.9 | +19.0 |
| nine25 | 65001157 | 251.4 / 129.4 | 154.2 / 145.5 | +16.1 |

Warm wall difference uses the faster repeat for each arm, including ingestion overhead. The component timings below imply +16.3 s and +14.1 s over 755 windows. The first cap250 runs include cold setup.

## Additive runtime projection

Each component timing job used all 2,655 public windows in shuffle 71000 and warmed up on 100 windows. The Team B comparison counterbalanced eager and graph order by session. Both jobs measured absolute graph/eager posterior error **0.0**. The measured hard actor and Team B costs include session-start actor restore and graph capture at the public session lengths.

| Dose | A100 job | Hard actors, ms/window | Team B eager → graph, ms/window | Added, ms/window | Projection, s | Margin to 3,600 s |
|---|---:|---:|---:|---:|---:|---:|
| nine15 | 65001150 | 13.831 | 9.104 → 7.823 | 21.654 | **2,989.9** | 610.1 |
| nine25 | 65001156 | 11.303 | 8.263 → 7.406 | 18.709 | **2,930.7** | 669.3 |

Projection is `2,555 + 20,083 × added_ms / 1,000`. The ZIPs differ only in the mixer weight. The timings came from separate A100 nodes, with most of the difference in the identical CPU hard actors. These are A100 proxies. RTX PRO 4500 timing and private session lengths are unknown, and Team B's prefix refresh cost grows with session length.

Public replay checks package behavior on all-data weights. Fresh fold evidence for hard-only nine15 was +0.77 pp held-new [+0.43, +1.09] and +0.47 pp returning. Nine25 was +0.97 pp [+0.31, +1.54] and +0.59 pp returning, using eight fresh orders and people-nested 80% intervals. [Lane TB-verify](laneTB_verify.md)

All Slurm jobs completed by 03:04:44 CET on 21 November. The ZIPs are staged locally and on Leonardo. No Codabench upload was performed.

RTX 2080 Ti follow-up (21 Nov, Galvani): graphs engage and match eager, but Team B costs about 36 ms per window inside the Solver there, so the 2080 Ti projections are 3,520-3,657 s, not the A100 proxies above. [Confirmation record](final_hard_teamb_confirm.md)
