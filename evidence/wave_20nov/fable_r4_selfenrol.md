# Fable R4 Bet 1 · mid-stream self-enrolment

21 Nov, 09:19 CET. The redundancy check passed. Returning enrolment fits a supervised 256-PC head and a separate all-channel slow-waveform head. SessionEM sees frontal summary, eye features and ocular rates. TSEM uses only the leading tangent PCs. The oracle tests these two supervised heads at the terminal of the fused lead, ahead of any pseudo-label score. [Preregistration](../../docs/LEDGER.md#fable-r4-bet-1--mid-stream-self-enrolment-registered-21-nov-0919-cet-before-scoring)

The mechanism audit used `cleanslate/cap250/families/geometry.py::fit_personal`, `cleanslate/cap250/families/_id_own.py::Space.person`, `studies/17nov/js_readiness321/returning_context.py`, `friday/ea_returning/returning_stages.py` and the signed fused lead's `OwnStage`, `SessionEM` and `TSEM`.

## Oracle dispatch

Source `899ae8b5` builds on the signed fused lead and the real production fold assets. Local `.venv/bin/python replay/check.py` reproduced all six v3 reference scores and printed `OK`. The weighted slow-waveform LDA matched scikit-learn's native 0.7-shrunk LDA at unit weights within `1e-11`. Leonardo smokes 65014266 and 65014272 began at 09:23 CET. The four fresh orders 71700–71703 are reserved for the oracle.

The returning smoke matched 189 pinned labels and recognized P04 on delivery 12. Held-new P01, P07 and P09 matched their pinned lead labels but passed zero checkpoints at the 75% agreement threshold. P07 and P09 peaked at 63.5% and 58.1%. The preregistered smoke addendum tests the same T=100 head without the agreement guard as an oracle diagnostic. The candidate keeps the guard.

Leonardo smoke 65014625 passed on P07 (source `1fdec342`). It matched all 189 pinned lead decisions. The ungated T=100 head changed 24 decisions after delivery 100 and reproduced the full run on a truncated prefix. All four fits completed in at most 0.59 s. The guarded arm stayed inactive. No accuracy was computed. Oracle array 65014943 and dependent collector 65014944 run fresh orders 71700–71703.

## Decision at 09:53 CET

**Stop at the true-label oracle.** The guarded T=100 head scored 75.63% versus 76.17% on deliveries 101+, a change of **−0.55 pp [−1.50, +0.35]** against the +4 pp release bar. [Leonardo receipt](fable_r4_selfenrol_oracle.json)
- The four-checkpoint arm scored 74.09% versus 74.54% overall, **−0.45 pp [−0.98, +0.01]**. The separate ungated T=100 diagnostic fell **9.15 pp [−12.97, −5.59]**. Intervals resample people and orders. The four orders reuse ten people and cover 112 interface cells and 10,620 deliveries.
- The guard accepted 1/56, 2/56, 4/56 and 7/56 fits at 40, 70, 100 and 140. The guarded arm applied a head on 679 deliveries and changed 181 decisions. Every delivery through 40 matched the lead. Maximum fit time including leave-one-out was 0.715 s on Leonardo A100 workers.
- All 20 tasks in 65014943 completed. Collector 65014944 hit a NumPy integer JSON serialization error. Source `2362452c` fixed serialization and retry 65015403 completed on the same tapes. A local calculation using the official metric independently matched both whole-stream scores across all 112 cells. Receipt SHA-256 `e91a9958de9a7fda22da199e8636c6e645f50199492bfa2798ecf8e80661866f`.
- The oracle stop rule withheld pseudo-label scoring, full returning confirmation, ZIP construction and end-to-end timing. The returning evidence is the P04 pinned smoke only.
