# Strategy review (30 Oct): where we stand, what can still move

All scores are cell-mean balanced accuracy in %, rebalanced. Selection =
mean(`dev_session`, `dev_loso_all`); `dev_loso` reported.

## 1. Measurement first: how much of our recent progress is real

| Package | seeds 0-2 | seeds 3-5 | **seeds 0-5 pooled** |
|---|---:|---:|---:|
| R10 | 78.58 | 77.69 | **78.30** |
| R14 (+ rotation) | 79.31 | 77.92 | **78.46** |
| R15 (+ rotation pretraining) | 79.59 | 78.23 | **78.68** |

- The same recipe scores 1.4 points apart on two seed sets. Three-seed
  comparisons overstate small gains; pooled over six seeds, each of R14 and
  R15 adds about +0.2.
- A bootstrap over participants gives a standard deviation of 2.6 points for
  the `dev_loso_all` score. The sealed composition moves every team, but it
  makes our own estimate uncertain by ±3.
- **Rule from now on:** decide on six seeds pooled, and look at the paired
  difference per participant. A three-seed gain below 0.5 is noise.

## 2. Where the errors are (R15, `dev_loso_all`, unseen participants)

- Recall: motor imagery 66, subtraction 78, word generation 74. Motor imagery
  is the weakest class (MI → word generation 18%, MI → subtraction 16%).
- Per-cell scores range from 59 (P03 BrainHero) to 89 (P08 Graz). P10's
  motor imagery recall is 26 / 38.
- P04 (our weakest, about 49 on all_loso) has clean channels and low
  amplitude. Its failure is not data quality. Others have bad channels
  (P06 C6 ×6.9, FC4 ×6.1; P02 CP5 ×0.18), and these sessions still score 75–83.
- Headroom: the within-session ceiling is about 82 against 72 on unseen
  participants. About 10–17 points sit in participant-specific boundaries
  that no label-free method has reached here (R4, R8, R12).

## 3. Levers checked today

| Lever | Result |
|---|---|
| Best-of-5 portfolio from mixing knobs (α, W, v2 weight, EA/raw subsets) | +0.2 to +0.4 in sample: small |
| More seeds per family (3 → 6) | no gain beyond seed luck |
| Sealed-feedback adaptation | bounded by the same small portfolio spread |
| Rotation-pretrained FBCNet (R16, seeds 0-2) | +0.2 to +0.3 in R15 (noise level); seeds 3-5 running |
| Data quality / bad channels as P04's cause | no: P04 is clean |

## 4. The solution space, by source of new information

Past step changes came only from **new information**: session alignment,
the < 4 Hz signal, heterogeneous input streams, and the balanced runs. Each
added 2–7 points. Everything since R9 has added ≤ 1.

| Source | Status | Remaining value |
|---|---|---|
| Better invariance (DG augmentation, pretraining) | R14/R15/R16 work | +0.2 per step, saturating |
| Test-time adaptation (pseudo-labels, entropy, T-TIME, AdaBN, TTA) | closed | – |
| Output constraints (rebalancing, exact counts, order prior) | mined | +0.4 even with oracle boundaries |
| Personal models for seen participants | identification fails across sessions (P04) | ≈ 0 net |
| External same-task data | Scherer (excluded) gave +0.6; Shin A+B has a block confound | small |
| Rest-anchored unified pretraining (task vs rest inside every external session) | untested | uncertain, 1–2 days |
| Self-supervised test-time training (masked reconstruction) | untested | low prior (relatives closed) |

## 5. Honest outlook

- **Expected sealed score:** about 69–70 ± 3 (unseen ≈ 67–68 × 0.57,
  seen ≈ 72 × 0.43).
- **Field:** unknown. On the warm-up, 69 entries score ≥ 0.70; they almost
  certainly trained on the test participants. Among plausible honest
  entries, 1 is at 0.65+, 6 at 0.60–0.65 and 20 at 0.55–0.60; our honest
  replay is 0.536. The warm-up is 6/8 P04 cells, so this is a weak signal,
  and it does not show us ahead.
- **Top-three probability:** our estimate is about 15–25%. No measurement
  available to us can establish 70%. Reaching that would need a step change
  of several points, and no lever in sight has shown one.

## 6. Plan to the deadline (21 Nov)

1. Sealed slots: upload r9g on 5 Nov (proves the real worker). Hold the
   other slots and upload the best packages later, the last close to 21 Nov.
2. Candidate quality: decide on six-seed pooled results only. R16 is a
   candidate if it holds at six seeds.
3. One high-variance bet with a stop rule: rest-anchored unified pretraining.
   Use within-session task vs rest contrasts from PhysioNet (MI vs rest),
   Shin 2017B (subtraction vs rest), Shin 2018WG (word generation vs
   baseline) and Schirrmeister (MI vs rest) in one 4-way head, then
   fine-tune its MI/SUB/WG outputs on PROTEUS. Stop if the single family
   does not gain ≥ 1 point on six seeds.
4. Keep every package audit-ready: committed config, clean build, worker
   replica, documented pretraining recipe.
