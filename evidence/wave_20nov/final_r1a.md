# Final R1a archives · 21 November 2026

Both candidates use `laneR4/build_keep.py R1a` against pinned self-test parents. On recognised streams R1a keeps the stock fine-tuning objective and schedules one refit at age 100 with five optimizer steps per fine-tuned family. CSP retires at recognition. If recognition precedes the first stock refit, the R1a code skips fine-tuned forward passes until age 100. If a stock refit already ran, its fine-tuned actors can continue forwarding after recognition. `replay/check.py` printed OK for all six original v3 references. [Registration](../../docs/LEDGER.md#final-r1a-zip-qualification--21-november) · [Lane R4 design](laneR4_keepadapt.md)

| Archive in `runs/sealed/final_zips/` | Bytes | SHA-256 |
|---|---:|---|
| `final_stack_hard_safe_st.zip` | 66,964,554 | `239c86bd18d0a7e9d0216dda1fb95d9b918bf6e933c0383322446aed37a83c1e` |
| `final_stack_hard_safe_st_r1a.zip` | 66,964,924 | `df31c7c9576fbbf8a0da3ff1959d77dc744bb0058052204f1401911c19b33322` |
| `final_stack_hardcb_safe_st.zip` | 94,905,529 | `47faa27b25f6fc1fcbcc1c74f2be4b8b2aeb2a1e02c6c957d342eb6c91b9c2ec` |
| `final_stack_hardcb_safe_st_r1a.zip` | 94,905,915 | `82cc75456c486529cbebe9b1b9ddb1621b8478a4ad69e81b6fa11a9a083c4861` |

The A100 smokes verified archive CRC, member inventory and the exact builder output. Only `submission.py` differs from each parent. Both ZIPs loaded through the self-test and matched every label on the 8-window P01/1 prefix with zero recognition and zero v6 errors. That prefix is a smoke. The full CPU held-new check uses the clean fold's weights and actor hashes with each archive's solver. The original archive's self-test uses references specific to its fitted actors.

## Qualification and tape checks

Leonardo run `/leonardo_work/AIFAC_P02_1636/ni26/cap250/runs/final-r1a-21nov`. Official replay code is pinned at `a66bc23d1a0e686b895c093a642b45c74970758b`. The real-window export is SHA-256 `2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327`. The official warm-up reference is cap250 SHA-256 `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed`. [Job IDs](DISPATCH_21NOV.md)

The tape checks use the frozen H1 A100 tapes on the clean folds (seeds 71000–71003) and the original Lane R4 tapes on their own row order. The original tapes use different fitted weights and delivery-aware EA banks for the comparisons. Exact-ZIP and fold-matched results are recorded separately. [H1 receipt](redteam_handoffs.md#h1--keep-adapting-after-recognition) · [Original Lane R4 tapes](laneR4_keepadapt.md)

| Check | hard + R1a | hardcb + R1a |
|---|---|---|
| Exact ZIP A100 smoke and self-test load | 64992707 passed | 64992710 passed |
| Held-new P01/1 CPU and self-test fault injection | 64993648, 190/190 equal and injection passed | 64993649, 190/190 equal and injection running |
| Exact ZIP returning effect and timings | 64992709 completed, 0/762 labels changed | 64992712 completed, 0/762 labels changed |
| Terminal score movement | 64994100 passed, 360/362 post-refit windows | 64994102 passed, 360/362 post-refit windows |
| Frozen tape label comparison | 64992919 complete, 64995390 awaiting remaining orders | 64992934 complete, 64995392 awaiting remaining orders |
| Official A100 cap250 comparison | 64992708 passed | 64992711 passed |

Both exact ZIPs recognised all four visits at age 10 and executed a single late refit at age 100. This was on P04/2, P04/3, P10/2 and P10/3 at seed 71000. The terminal-score trace bound before age 100 was 2e-6. After the refit, scores crossed that threshold in 360 of 362 windows for each ZIP. The largest absolute change was 0.05574 for hard and 0.05621 for hardcb. Paired timings warm both models on P04/2 before measurement. The new A100 mean increments were **3.07 s** for hard (range 3.00–3.21 s) and **3.18 s** for hardcb (2.88–3.48 s).

### Frozen tape label comparison

| Comparison | hard + R1a | hardcb + R1a |
|---|---:|---:|
| Exact ZIP against the first 16 original Lane R4 stream-orders | 0/16 exact, 230/3,048 label differences | 0/16 exact, 229/3,048 label differences |
| Exact ZIP against the 16 clean H1 stream-orders | 0/16 exact, 598/3,048 label differences | 0/16 exact, 597/3,048 label differences |
| Final solver with matched clean fold assets against H1 | 3/16 exact, 27/3,048 label differences | 16/16 exact, 0/3,048 label differences |

For both solver variants the matched-fold replay reproduced all 199 labels on P10/3 at seed 71001. The hard-only fold also matched P10/2 at seed 71003 and P10/3 at seed 71002. The exact ZIPs use the production fit. On the P04/2 smoke, the direct comparison already differed before recognition and before R1a could affect the score.

### Official A100 warm-up replay

`friday/qualify_package.py pair` ran the official ingestion and scoring programs on an NVIDIA A100-SXM-64GB, in order cap250, candidate, cap250, candidate. Each replay covered 755 windows across four sessions. All eight scores matched (cell-balanced accuracy 0.994578853046595), all submission file hashes were preserved, and v6 reported zero errors on every stream. Candidate logs show one refit and 20 optimizer steps per stream (five steps across each of four families). The repeat check compares the scores without the wall-clock duration field. The warm-up replay uses the pinned public phase's delivery. The layout check used `NI26_MAX_MB=100`, as in the prior hardcb qualification.

| Candidate | cap250 ingestion wall, s (two repeats) | Candidate ingestion wall, s (two repeats) |
|---|---:|---:|
| hard + R1a | 206.0 / 127.5 | 161.4 / 152.1 |
| hardcb + R1a | 206.6 / 128.7 | 172.9 / 170.1 |

## Additive sealed-time projection

Use 2,555 s cap250 sealed time + 20,083 windows × measured hard/hardcb cost + 46 assumed recognised sessions × paired Lane R4 R1a cost. Lane R4 measured +3.5 s per recognised session on A100 and +3.3 s on RTX 2080 Ti using hardcb with recognition at age 10. Recognition after a stock refit can change the cost because fine-tuned actors may forward before age 100. For hard, this projection assumes the same fine-tuning cost. The RTX PRO 4500 sealed worker remains unmeasured. [Per-window timing](final_stack.md#review-fix-2--additive-runtime-and-load-time-self-test-20-nov-sat-0000-0100-cet) · [Paired R1a timing](laneR4_keepadapt.md)

| Package | GPU proxy | Added window cost | Added session cost | Projected time | Margin to 3,600 s |
|---|---|---:|---:|---:|---:|
| hard + R1a | A100 | 11.8 ms | 3.5 s | 2,953.0 s | +647.0 s |
| hard + R1a | RTX 2080 Ti | 14.3 ms | 3.3 s | 2,994.0 s | +606.0 s |
| hardcb + R1a | A100 | 39.6 ms | 3.5 s | 3,511.3 s | +88.7 s |
| hardcb + R1a | RTX 2080 Ti | 47.1 ms | 3.3 s | 3,652.7 s | −52.7 s |

With the new exact-ZIP A100 measurements, the additive estimates are **2,933.2 s** for hard (margin **666.8 s**) and **3,496.6 s** for hardcb (margin **103.4 s**). The historical 2080 Ti timings remain the only basis for that GPU proxy.

Galvani rejected noninteractive SSH. A100 qualification uses Leonardo's native Python environment. No Codabench upload was performed.
