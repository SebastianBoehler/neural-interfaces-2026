# Red-team strategy review (Opus 5.5) · Thursday 19 Nov 2026

> **Correction 19 Nov (upstream rules 8447e181):** the sealed budget is 3,600 s on one RTX PRO 4500 (the raised-limit / 4,800 s reading is superseded), and only the team's best sealed score counts (the "best of last five" rule is gone). See OFFICIAL_CONTRACT.md.


Scope: read-only review of main after the Friday merge (`a7db5818`). No job, no commit, no upload. Labels: **F** = fact
(measured; file:line given), **I** = inference, **P** = proposal. Line numbers refer to this checkout.

## 0. Verdict in five lines

1. The project optimizes a **held-new-person** instrument (10 public people, about 76 %). The sealed score (83 %) is a
   mixture where **returning people** (public people's new sessions) probably score much higher. That half of the
   metric was tuned on two people and still enters the deployed mix at a **1/16 weight**. This is the largest
   under-exploited lever I can find (I).
2. Dev deltas below about 0.5 pp do not predict sealed deltas. Every increment since 13 Nov sits in that range, so
   the 22-hour campaign could not tell its own wins from noise. The cause is the instrument, not a shortage of ideas (F/I).
3. The final ranking uses each team's **best** sealed score. The last upload therefore has no downside and should
   buy **variance with positive mean**, not certainty. A re-upload of a +0.2 pp candidate is close to worthless (F/I).
4. Two step changes have a ceiling ≥ +1.5 pp: (S1) returning-person exploitation at full strength, and (S2) a
   leaders'-recipe whole alternative (pretrained high-capacity network on all public data plus simple TTA). Only S1
   can realistically be tested and shipped by Friday 18:00 CET (I/P).
5. Odds as I see them (I): top 10 (≥ 0.8351, +0.52 pp) about 35 %; top 3 (≥ 0.8470, +1.72 pp) about 10 %. Top 3
   is reachable only through S1.

## 1. Diagnosis

### 1a. What the top of the board likely does differently

- F, board (mirror updated 17 Nov 15:44 UTC, neural-interfaces26.github.io/leaderboard.html#bci). #1 shangeth 0.859
  (16 Nov), #2 PNPL 0.848 (14 Nov), #3/#4 ETRI_EnSeption 0.847/0.846 (11–12 Nov), #10 0.834. 263 entries. No
  write-ups. The page says the final ranking uses each team's best eligible sealed score and that the top 3 are
  replayed for audit (±2σ).
- F: on 11 Nov seohee scored 0.821 in **739 s**, sungjunwang 0.830 in 1,222 s, we 0.813 in 3,732 s
  (docs/SEALED_UPLOADS.md:177-182). The #3 score dates from 11–12 Nov (DECISION_18NOV.md:19; mirror).
- I: leaders do not win by compute or stack depth. The cheapest fitting explanation is a strong **supervised network
  trained on all public people** (implicitly memorizing returning people) plus per-session alignment and light
  label-free adaptation. "EnSeption" suggests an Inception-style ensemble; PNPL is a pretraining lab (guesses from names).
- I: our system is selected on leave-one-person-out generalization, which rewards dropping person-specific fit, the
  very thing the returning half of the sealed set pays for. Size of that half: **P10 recalls MI 26 % unseen but scores
  84 % once session 1 is in training** (docs/archive/STRATEGY_REVIEW.md:62-64); R10 dev_session (seen people) 84.48 vs
  dev_loso 70.43 (docs/archive/OPTIMIZATION_R10.md:274); a true-label personal decoder recovered 72.4 % of
  population-chain mistakes on hard windows (docs/LEDGER.md:291).
- F: the routed personal evidence enters the cap250 tail at "real 1/16 weight, gain 3" (docs/LEDGER.md:712), plus the
  own-space stage at β = 0.10 (docs/SEALED_UPLOADS.md:325). Both returning uploads were tuned on P04/P10 and were
  "P04-driven" (docs/SEALED_UPLOADS.md:234,326). They still moved sealed by +0.267 and +0.361 (:305,:431). That is
  more than P04 alone can explain (:432-435). I: the returning lever transfers. It was dosed for a two-person
  instrument.
- Literature (for S2): [T-TIME](https://arxiv.org/pdf/2412.07228), [online MI TTA](https://arxiv.org/pdf/2311.18520),
  [NeuroTTT on CBraMod/LaBraM](https://arxiv.org/pdf/2509.26301v2), [backprop-free TTA](https://arxiv.org/html/2601.07556),
  [EDAPT](https://arxiv.deeppaper.ai/papers/2508.10474v1/similar). Record: CBraMod +2 dl / +1.5 ds as a member, rejected
  on seen_session −2.2; LaBraM no gain; REVE/EEGPT open; multi-dataset pretraining untested; GPU unused
  (docs/archive/STRATEGY_REVIEW.md:49,53-54); REVE later only as a distillation source (docs/LEDGER.md:848).
  **No foundation-model system was ever scored as a complete system.**

### 1b. Why dev (~76 %) differs from sealed (~83 %)

I (ranked by my belief):
1. **Mixture shift.** Sealed is about 57 % new / 43 % returning (docs/SEALED_UPLOADS.md:40,432; cohort text withdrawn,
   unverified). 0.57·76 + 0.43·x = 83 gives x ≈ 92 % on returning cells, plausible against 84–85 % seen-session above.
2. Minor: 9-person dev fits vs 10-person sealed fits; sealed sessions possibly longer (unknown); sealed shuffle scope
   (session, interface or run) not stated (OFFICIAL_CONTRACT.md:36-41).

### 1c. How reliable dev deltas are (every upload)

| Upload | Dev Δ predicted | Sealed Δ | Ratio | Driver |
|---|---|---|---|---|
| 2 TSEM | +3.7–3.9 | +4.02 (+3.57 rescored) | 0.94–1.07 | new + returning |
| 3 route3geb | +0.75 honest | +0.43 (+0.55 rescored) | 0.57–0.73 | returning, 2 people |
| 4 v4.1 routing | +0.58 | +0.267 | 0.46 | P04-driven |
| 5 v5.1b own-space | +0.89 | +0.361 | 0.41 | P04 +3.51, P10 +0.07 |
| 6 cap250 | +1.11 weighted (+1.38 new); corrected +0.479 | +1.011 | 0.91 / 2.1 | new-person fine-tuning |

Sources: SEALED_UPLOADS.md:124,153,199-200,305,326,431; DECISION_18NOV.md:3,34-35,56.

- F: 5 of 5 sealed signs matched dev. Magnitudes range from 0.4× to 2×.
- I: large new-person steps transfer ~1:1; P04/P10-measured returning gains ~0.4× (two people, 20/80 guess); the
  corrected instrument **under**-predicted cap250 2×. Noise floor ≈ ±0.5 pp sealed. No 19 Nov alternate clears it
  (hard +0.3–0.4, FBL2 +0.4–0.5, F-C22 +0.5–0.8 public, ≈0 on Scherer; STATE.md:11-32).

### 1d. Proposed validation instrument (P)

- **Returning instrument, 9 people instead of 2.** Scherer has two visits per person (DECISION_18NOV.md:36). Enrol on
  labelled visit 1 (personal heads, own-space, TSEM prior, gallery); stream visit 2 shuffled through the real package;
  add P04/P10. Never run: "galleries were never tested externally" (SEALED_UPLOADS.md:274); no Scherer enrolment.
- **New-person instrument.** Use 19 people (10 PROTEUS LOPO + 9 Scherer) in one fully re-streamed harness, where
  every member, including Transformer and Shallow EA, is recomputed in delivery order (scoring_audit_18nov.md:9-38).
  Report the 57/43 mixture with the returning term from the 9-person instrument, not the 20/80 two-person guess.
- Impostor control: gate FAR on all 19 held-new people at the new dose (Stieger 0.42 %; SEALED_UPLOADS.md:98).

## 2. Constraints audit

| Constraint | Class | Evidence | Comment |
|---|---|---|---|
| No state crosses sessions; label-free; one window in, label out | BINDING | OFFICIAL_CONTRACT.md:42-47,68-70 | Keep. |
| Metric = per-cell BA mean; preserve the model-facing tensor | BINDING | OFFICIAL_CONTRACT.md:51-56 | Keep. |
| Runtime 3,600 s A100, now RTX with a higher limit (4,800 s reported) | BINDING (value unverified) | OFFICIAL_CONTRACT.md:71-78; DECISION_18NOV.md:64 | cap250 2,555 s; RTX proxy 1.97× v5.1b. Headroom about 1,000–2,200 s. |
| 1 upload left; ranking uses best score | BINDING | DECISION_18NOV.md:84; mirror | Makes the last upload a free option. |
| Bit-exact parity, C++/CUDA worker | AGREED (self-imposed) | STATE.md:9,22 | Costs effort and adds nothing to accuracy. RTX already breaks parity (906/960; DECISION_18NOV.md:68). Drop as a gate. |
| Every change scored inside the full stack, never alone | AGREED (AGENTS.md) | | Right for parts. It also turns every whole alternative into "a member of cap250", which locks the local optimum. |
| Weight new people at 57 %; returning at 20/80 P04/P10 | INHERITED | SEALED_UPLOADS.md:40,432; AGENTS.md | Based on withdrawn cohort text, and two people decide the 43 %. |
| "Interface detector not worth building" | INHERITED (R13, oracle alignment only) | OPTIMIZATION_R10.md:257-262 | Closed only at alignment scope, not for count/prior per cell. Low priority. |
| "Output-level adaptation / ensembles exhausted" | INHERITED (Oct, chronological order) | STRATEGY_REVIEW.md:44-46 | Closed before the shuffle and before RTX; scope differs. |
| Foundation models not adopted | INHERITED (R8, member scope) | STRATEGY_REVIEW.md:49 | Never scored as a whole system. |
| Preregistration and decide only on intervals | AGREED | AGENTS.md | Sound, but with ±0.5 pp noise and 0.0–0.3 pp effects every run returns "undecided". |

## 3. Is every stage adapted to shuffled delivery?

- F: the ZIP streams causally; the **dev replay** kept Transformer and Shallow EA in recording order
  (scoring_audit_18nov.md:9-31). Corrected within0/full0 re-stream them, full0 on one order (DECISION_18NOV.md:56).
- F: stage estimators are exchangeable without forgetting (running-mean recentring; SessionEM/TSEM ramp t/(t+n0);
  submissions/lean_v66/submission.py:117,513,573). That suits shuffled delivery.
- I (unchecked risk): several fitted constants predate the shuffle notice (organisers 13 Nov) or were fit on
  recording-order replays: GEB early boost (7 Nov), routing thresholds (11 Nov), em/tsem λ and n0, and
  CountFilter (fit 14 Nov "ten-person fit"; check which order `lean/tools/fit_cf.py` used). Under shuffling every
  prefix is class-balanced in expectation from window 1, so priors and boosts tuned on blocked runs are
  mis-set. Cheap test: refit these constants on shuffled re-streamed banks (P, item 4 below).
- I: the 250-window cap and cadence (t0 20, every 10; CAP250_REBUILD.md:19-20) were A100 runtime choices, not
  re-optimized for accuracy under shuffling or the higher RTX limit.
- I: the sealed shuffle scope is unknown; if blocked per run or interface, order carries information dev never shows.
  Untestable without an upload, so do not build for it.

## 4. Step changes with ceiling ≥ +1.5 pp

**S1 · Returning people at full strength (recommended; the only one shippable by Friday).**
- Mechanism: for sessions the gate recognizes, raise the personal evidence from 1/16 toward a dominant share, or a
  hard switch after recognition plus k windows. The evidence is personal heads trained on **all** that person's
  public sessions and both interfaces, a multi-seed FBCNet and tangent LR. Keep the population chain until recognition.
- Ceiling: at a 43 % returning share and +3.5–8 pp per returning cell, the gain is **+1.5–3.4 pp** sealed (I). The
  inputs are 26 → 84 % MI recall for seen P10, 70 → 84 % dev_loso vs dev_session, and 72 % personal recovery (§1a).
- Fastest decisive test (P), about 6–10 h:
  1. Run a weight sweep {1/16 (stock), 1/4, 1/2, switch} on the Scherer two-visit enrolment (9 people, two fresh
     orders) and on P04/P10, scored through the real package tail.
  2. Rerun the same arms on the 10 held-new PROTEUS people to show zero routing and an identical score.
  3. Advance if the 9-person returning gain is ≥ +2 pp with a positive lower bound, the held-new score is unchanged,
     and FAR stays ≤ 0.5 %.
- Risks: personal-head builders on Scherer montage/crop; misrouted new people lose their cell; gallery recognition of
  8 single-session public people untested. Odds by Friday 18:00: ~30 % qualifying test, ~20 % shipped package.

**S2 · Leaders'-recipe whole alternative (explore; probably too late for Friday).**
- Mechanism: one high-capacity pretrained network (CBraMod, REVE or EEG-Inception ensemble), fine-tuned on **all**
  10 public people plus task-matched external data (Shin 2017 MI and arithmetic, Scherer). Run it with per-session EA
  and T-TIME/IM test-time adaptation plus count balancing, and score it **whole**, not as a terminal member.
- Ceiling: unknown on new people, large on returning ones (training on them is the point); CBraMod +2 dl as a member
  (STRATEGY_REVIEW.md:49). Standalone→system is 59 → 76 % (RAW_MODEL.md:8), so S2 still needs the live adaptation.
- Fastest test: CBraMod fine-tune per LOPO fold + all-public (~4 GPU-h), scored whole on both instruments. Odds by
  Friday ≤ 10 %; keep as research or as the personal model inside S1's routed tail.

**Not a step change:** stacking the undecided small positives (hard, FBL2, F-C22, JS, spectral kernel). The ceiling is
about +0.5–1.0 pp public, and Scherer gave ≈ 0 to −0.6 (STATE.md:28; DECISION_18NOV.md:39). The early-window warm-up
costs about 6 pts on 16 % of each session (STRATEGY_REVIEW.md:65-66), so its ceiling is about 1 pp. The selection
oracle (+2.3 pp, DECISION_18NOV.md:51) is a ceiling no causal selector has approached.

## 5. Process: what prevents a jump

- F: 821 commits, +81.7k/−13.7k lines since 18 Nov 13:30 (`git log --since`); LEDGER.md 3,186 very long lines.
- I: **resolution mismatch.** 0.0–0.3 pp effects under ±0.5 pp noise: nearly every preregistered run ends
  "undecided", the queue refills with neighbours, and AGENTS.md forbids closing on noise: open threads, no decisions.
- I: **anchoring.** Every idea must enter the cap250 mix, often as a terminal blend
  (scoring_audit_18nov.md:77-90). A better whole decoder diluted at 1/90 (SOLUTION_SPACE_18NOV.md:30) cannot show a
  gain, so the local optimum defends itself.
- I: **effort in exactness, not accuracy.** Bit-exact parity, C++/CUDA, SHA receipts and RTX tolerance checks
  (STATE.md:9,22,28) matter for the audit of the top 3, which we are not in. RTX already changes 154 of 4,260 labels
  (RETROSPECTIVE_18NOV.md:37).
- I: **wrong half of the metric.** Much of 18–19 Nov went to held-new waveform decoders scoring 38–48 % standalone
  (RAW_MODEL.md:7-17). The returning half, about 43 % of sealed, got a two-person instrument.
- I: **reused cohorts** (scoring_audit_18nov.md:61-76): the same 10 + 9 people have informed hundreds of choices.

## 6. Ranked plan (Thursday afternoon → Friday 18:00 CET)

1. **S1 test (now).** Build the Scherer visit-1 enrolment and run the returning weight sweep on 9 + 2 people and the
   held-new zero-routing check. Preregister thresholds as in §4. Delegate to one executor; Galvani.
2. **Freeze the fallback.** cap250's score is kept automatically. Stop parity, native and RTX-exactness lanes now.
3. **Build the upload candidate in parallel.** cap250 + the S1 dose that passes (or 1/4 as a prior if the test
   lands late) + FBL2. FBL2 is the alternate with the most repeated public positives and an RTX-timed package
   (STATE.md:11). Smoke it on RTX (runtime ≤ 1.3× cap250) and run `replay/check.py`.
4. **If idle GPU remains:** refit CountFilter, GEB, em/tsem constants on re-streamed shuffled banks (0.1–0.3 pp).
5. **S2 as a background research lane** (CBraMod whole system on all public data). It does not block the upload.
6. **Friday 18:00 decision rule (P):** upload the S1 candidate if its 9-person returning gain is ≥ +2 pp and held-new
   is unchanged. Otherwise upload the highest-mean stacked candidate. Never re-upload cap250.

## 7. Recommendation for the single remaining upload

The last upload is a free option: the best score counts, and a failed run does not use quota (SEALED_UPLOADS.md:64).
Choose by **P(sealed > 0.8351)** and **P(> 0.8470)**, not by the narrowest interval.

1. First choice: **cap250 + S1 returning dose + FBL2**, if S1 passes on the 9-person Scherer enrolment. Expected
   +0.8 to +2.0 pp; P(top 10) about 50 %, P(top 3) about 15–20 %.
2. Otherwise: **cap250 + FBL2 + hard** (the best stack of repeated positives). Expected +0.2 to +0.6 pp; P(top 10)
   about 25 %, P(top 3) under 3 %.

The phase ends Saturday 23:59 CET; uploading Saturday buys a day for S1. That timing is the user's call.
