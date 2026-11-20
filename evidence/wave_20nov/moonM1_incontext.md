# M1 · Learned in-session adaptation

Registered 20 November 2026 at 07:45 CET before any M1 held score. This card is the M1 ledger entry for the clean-slate program. The implementation scope is `cleanslate/m1/`.

## Question and fixed comparison

Can an unlabeled set of earlier EEG embeddings improve classification of the current window? Team C's preliminary P07/P09 gains (73.0/68.8 → 81.5/77.3) are person-level accuracies from its own order 0. M1 uses their C1 E6 training recipe and A2h optimizer (8 steps, learning rate 0.0003, prefix BN). The same six outer-fold member weights feed every arm. A0 is source BN plus causal Euclidean alignment (EA). M1 adds a learned residual to A0 using only earlier embeddings, with a limit of 190 earlier windows. Blank is the identical M1 model with the context prefix always empty, exactly reproducing A0. A1 (prefix BN) and A2h (prefix BN plus gradient TTT) are matched controls. This is a complete alternative decoder, evaluated without cap250.

Use nested leave-one-person-out training. For outer held person H, the C1 recipe fits the other nine. For each inner episode person E, fit a C1 encoder on the eight excluding H and E. Create eight independently shuffled full-session orders of embeddings for each inner episode person. The embedding of each window uses EA up to that window and fixed source BN. One set cross-attention block conditions the current token on the earlier tokens using per-member embedding cosine similarity, with class logits as features. This representation keeps the attention inputs comparable across independently trained encoders. It has no recording positions, labels, identities or interface fields at inference. Each target episode samples a full shuffled stream and uniformly samples query age from 0 to min(190, session length − 1). The residual at age 0 is identically zero.

Mix PROTEUS episodes 1:1 with external episodes. Cycle foreign corpora uniformly in expectation, then sample a person uniformly inside the selected corpus. Use all seven Lane H corpora and all three B6 exports (62 ds007221 and 19 NETBCI people). Sample one session per person and two independently shuffled deliveries for each external encoding bank. Encode external episodes with the same outer encoder and train one classifier per corpus over the shared attention features. These heads use their own exported class maps. They are absent at inference. Keep AdamW fixed at 0.001, weight decay 0.01, 1,200 updates of 16 episodes (8 target, 8 external), with seeds fixed in source. No held score selects a checkpoint. Archive fold rosters, provenance and corpus coverage.

## Evaluation and decision

Score 10 held-new people, every public session, full-session shuffle orders 71000 and 71001 from `harness/data.py`. Use the harness's real one-window `predict` path, official cell-mean balanced accuracy, and paired 80% and 95% bootstrap intervals resampling people with the corresponding two orders inside each sampled person. Preserve official cell weights in the point and resampled estimates. Require complete coverage and pinned-order equality, exact blank/A0 labels, exact first-window context/blank output, reset and prefix-only checks, and an active context residual. Measure synchronized wall time per window on the same A100, including forwarding and copying session state. The criterion for M1 is approximately 100 ms/window or less.

Predictions in percentage points (80% ranges): blank BA 60 [53, 67]; M1 minus blank +2.5 [−0.5, +5.5]; A2h minus A1 +3.0 [−0.5, +6.5]; M1 minus A2h −1.0 [−5.0, +3.0]. At 22:00 CET Friday 20 November, stop this form if the complete context gain is below +2 pp. A paired 80% upper bound below +2 pp refutes a gain of that size for this form. Incomplete folds at cutoff leave the gain unverified, and any interval spanning +2 pp leaves the effect size unresolved. A point gain at least +2 pp with a positive paired 80% lower bound supports follow-up. This held-new test supplies no returning estimate or sealed-score projection.

Slurm only, with an unscored smoke before each path scales. Use Leonardo account AIFAC_P02_1675 and Galvani A100 for overflow. No submission upload.

## Run receipts

Leonardo release `/leonardo_work/AIFAC_P02_1675/ni26_m1_20261120`. Team C trainer from `b05e04e0` (SHA-256 `27765da37ff835f76ea7ea85a04ed81b55b28df54d9e2a568ec0c3a21459a67e`), Team C solver SHA-256 `035032be3b44b7930ccdb2e274bbaad17a764cb8bb225dc4918ae76e5dbffa76`. The trainer's external input option is disabled for all C1 fits. PROTEUS SHA-256 `2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327`. Official scorer code at the submodule pin `a66bc23d`, online rules at upstream `8447e181`.

- Encoder smoke `64916659` tested the P01 outer roster and P01/P02 nested roster. External smoke `64916945` staged the two real ds007221g deliveries with the expected own-head label map. Both passed.
- Full encoder array `64917233`. External staging array `64917235` passed all ten corpora. Corpus-person counts in the registered order were 109, 52, 54, 87, 14, 29, 29, 14, 48, 19 (455 corpus-person entries total).
- Paired solver smoke `64917236` trained the attention model and failed on the package path before prediction. Slurm cancelled the dependent collector `64917237`. Corrected the path and retried with `64917400` and dependent collector `64917402`.
- Corrected paired solver smoke `64917400` and collector smoke `64917402` passed. The encoder array completed all 55 fits. Held-new array `64917544` completed all ten folds. Collector `64917545` completed at 08:17:30 CET on 20 November 2026. All jobs ran on Leonardo A100s.
- M1 source `05d3e4e1`. All arms shared the same per-fold encoder checkpoints and delivery rows. The collector verified 50 prediction files (ten people × five arms), and all ten folds passed the first-window, zero-context, prefix and reset checks. Raw receipts and predictions are under `full/` in the release directory. `full/report.json` SHA-256 `d70b165546d5a48f2979642771b8bea227618670d6178b010babdf06e8d772bb`.

## Held-new result

Official balanced accuracy (%) over 28 equally weighted person/session/interface cells per order. Timing includes session copying and synchronized one-window predictions after a warm-up. Intervals resample people, then orders within resampled people (10,000 paired draws).

| Arm | 71000 BA | 71001 BA | Mean BA | 80% interval | 95% interval | A100 ms/window |
|---|---:|---:|---:|---|---|---:|
| M1 · true context | 62.10 | 61.44 | **61.77** | [57.75, 67.24] | [55.78, 69.74] | 4.78 |
| M1 · blank (exact C1 A0 labels) | 59.23 | 59.64 | **59.44** | [54.83, 65.53] | [52.53, 68.23] | 3.70 |
| C1 A0 · source BN | 59.23 | 59.64 | **59.44** | [54.83, 65.53] | [52.53, 68.23] | 3.00 |
| C1 A1 · prefix BN | 62.56 | 63.30 | **62.93** | [58.19, 69.45] | [55.95, 72.71] | 7.39 |
| C1 A2h · prefix BN + gradient TTT | 61.02 | 62.02 | **61.52** | [56.50, 68.99] | [54.26, 72.81] | 104.53 |

| Paired contrast | Δ pp | 80% interval | 95% interval |
|---|---:|---|---|
| M1 context − blank | +2.34 | [0.78, 3.82] | [0.10, 4.72] |
| M1 context − C1 A2h | +0.25 | [-3.00, 2.91] | [-4.61, 4.25] |
| C1 A2h − C1 A1 | -1.41 | [-3.85, 1.40] | [-5.14, 2.74] |

| Held person | Blank BA | M1 context BA | C1 A1 BA | C1 A2h BA |
|---|---:|---:|---:|---:|
| P04 | 43.19 | 47.99 | 47.16 | 47.12 |
| P10 | 57.20 | 58.18 | 57.61 | 49.64 |
| P07 | 61.73 | 64.47 | 70.02 | 70.03 |
| P09 | 58.76 | 71.83 | 72.25 | 69.07 |

The +2.34 pp point gain passed the +2 pp decision threshold and both paired intervals have positive lower bounds. The 80% interval spans +2 pp, so the size of the gain relative to that threshold remains uncertain. The paired M1–A2h interval spans zero. M1 was 21.9× faster than A2h on these A100 runs. A1 had the highest BA and was 1.16 pp above M1. The ten public people and two shuffle orders limit generalization.
