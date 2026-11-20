# Dispatch log · 21 Nov

- 00:20 Fable clean room (~/Documents/GitHub/ni26-cleanroom, b801a93): reviewed, approaches A-D deliberately NOT dispatched.
  A (adaptive population decoder, est. N 0.50-0.60) and B (personal experts) are weaker forms of mechanisms cap250 already
  has (cap250 held-new 73.7 % exceeds the clean room's own same-session supervised probe ceiling 0.72); B's ranking test is
  covered by redteam H1-H5 and laneRP. C (Graz cue-order prior) is void: organisers removed chronology from sealed delivery
  "to avoid tricks ... solving the track without using EEG" (13 Nov, memory sealed-order-not-chronological; FAQ L109-113).
  D (REVE probe) est. 0.36-0.50, closed earlier (foundation models). Hourly tick: do not dispatch these.

- Lane TB · real production-20261120-32a94ab7 fold packages, orders 71120–71123, Team B R1 encoder blend at 15% and 25%, with a fixed three-member trim. Registered in the ledger before scoring. Leonardo Slurm; Galvani batch SSH rejected authentication, so R1 weights will be refitted if the original artifacts remain inaccessible. Results due Sat 09:00 CET. [Lane card](laneTB_blend.md)
- 00:30 Reversed per user: Fable A+B dispatched to GPT build thread mcp:da9a9c35-9ca8-4053-b4f2-4f26cc6eb0f8 (clean room repo, results by 07:30, shuffled + chronological delivery). C still not built. 07:53 cron feeds RESULTS.md plus de-blinded main-system data points back to Fable (design only) -> PROPOSALS_2.md; hourly tick dispatches each proposal to GPT.

- RT-1 · dispatched 20 Nov 22:58–23:00 CET for the Sat 08:30 CET handoff. Leonardo H1/R0z and H5 first (64988017–64988060), then full R0z, H2 donor swap, H3 visit 1, and hardcb_safe reference (64988092–64988102). H4 reused Lane RP 64986260. Registered before scoring at `050198b1`. Slurm run `cap250/runs/redteam-21nov`; exact IDs and dependencies in its `jobs.tsv`. [Handoff](redteam_handoffs.md)

- Lane TB jobs · Leonardo `cap250/runs/laneTB-21nov`. Real-data fit smoke 64988026 completed. Fit array 64988029, P01 live smoke 64988278, score streams 64988281, collector 64988283, conditional ZIP and CPU blend-zero job 64988385. Scoring waits on the live smoke.

- RT-1 · 20 Nov 23:26 CET: kappa40 cleared its PROTEUS comparison. Registered Scherer NI-V B-ret confirmation (`9ff53b5a`), submitted Leonardo smoke 64989686 and dependent array 64989695 (nine people). [Handoff](redteam_handoffs.md)

- Lane TB retry · all 12 fits completed. Live smoke 64988278 matched pinned labels, then failed bitwise posteriors at the first GPU fine-tuning update (max 4.034e-7). No fresh orders scored. Revised smoke 64990355 → streams 64990356 → collector 64990357 → conditional ZIP/CPU zero 64990358, all afterok.

- Lane TB second retry · A100 repeated hardcb_safe pre-FT posteriors differed by 3.291e-9 (identical pinned labels). The zero arm now checks bitwise identity inside each live mixer call and retains the 2e-6 cross-run bound plus exact labels. New chain 64991339 (smoke) → 64991340 (streams) → 64991341 (collect) → 64991342 (conditional ZIP and CPU zero).

- RT-1 · 21 Nov 00:00 CET: all H1–H5 checks and all nine kappa40 NI-V confirmations completed. [Results and intervals](redteam_handoffs.md)

- Lane TB update 21 Nov 00:05 CET · 64991339 passed all pinned labels and same-call mixer identity. Its unchanged hardcb_safe control had max posterior Δ 0.021349 and zero label changes, so GPU cross-run posterior drift is logged. New smoke 64992565 → streams 64992566 with a baseline repeat on 71120 → collector 64992567 → conditional ZIP and CPU zero 64992568.
