# Revisit (Opus) · clean-slate sources as cap250's mixer input (registered Fri 20 Nov 2026 ~23:35 CET, before any chain arm was scored)

Question (user, 20 Nov night): did the clean-slate systems (6–15 pp below cap250 held-new) lose because they lacked cap250's
unsupervised session adaptation (rebalance → SessionEM → TSEM → count → CSP), not because their features or members were
worse? Members were only ever entered as diluted slots (lane Z: a slot gain reaches the mix at 0.10–0.18). Never tested: a
clean-slate family as the **main posterior source** (the mix replaced) or as a **50/50 partner** of cap250's mix, through
cap250's real stages.

## Instrument
- Lane Z's corrected replay chain (`laneZ/chain/final_z.py` machinery, bitwise guards), stock system, FT on. New hook in
  `revisit/chain/final_r.py`: after cap250's mixer, `m = (1 − w)·mix + w·source`; w = 1 replaces the mix (`rep`), w = 0.5
  (`h50`), w = 0.25 (`h25`). Everything downstream is cap250's: rebalance, SessionEM and TSEM (own GE/tangent features with
  the posterior as prior), count filter, CSP/LDA pseudo-label refits.
- Not exercised (stated): the GEB gate, FT banks and the SessionEM/TSEM shadow stay the frozen incumbent ones, so with w = 1
  cap250 still reaches the output through the gate and shadow (a leak in cap250's favour for `rep`). Recognition is not
  replayed (held-new only).
- Orders: full0 (= `shuffle_perms full|0|all_loso`, verified equal to the chain delivery) and within0 (= `within|0|all_loso`,
  verified). Ten held-new people, official cell-mean BA, paired people-resampled 80 % intervals (10k draws), pooled over both
  orders and per order. Dev only.
- Sources, each causal, streamed under the chain's own delivery order, LOPO (person never in its package):
  - `tcA1` Team C C1 E6 CNN ensemble (shallow/EEGNet/FBNet × 2 seeds), causal EA + prefix BN, no TTT.
  - `tcA2` same + 4-step gradient TTT (Team C's selected form; 65.25 held-new standalone, best whole clean-slate system).
  - `tbE` Team B R1 encoder ensemble (9 encoders, causal EA, supcon-trained; `enc` arm 65.3 standalone).
  - `tcb` 0.5 tcA2 + 0.5 tbE.
  - Not used: CBraMod bank (52.7 alone on full0, too weak to be a main source), Team A MANAS (59.2; no full0/within0 stream).
- Code: `revisit/stream_tc.py`, `revisit/stream_tb.py`, `revisit/stream.sh` (Galvani GPU streams), `revisit/build_src.py`,
  `revisit/chain/{final_r.py,worker.sbatch}`, `revisit/score_r.py`. Release `/mnt/lustre/work/gehler/gfh527/ni26_revisit_20261121`.

## Seen before registering (stated)
Mix level only, full0 only, Team C only (`revisit/offline.py`): cap250 mix 69.05, terminal 73.74; tcA1 alone 63.03, tcA2
65.18; 50/50 with the mix tcA1 68.12, tcA2 70.29; 75/25 tcA1 69.08, tcA2 70.38. No chain (terminal) value, nothing on
within0, nothing for Team B. CBraMod alone 52.7 (full0) / 52.7 (within0).

## Predictions (terminal BA minus stock|base, pp, pooled over full0 and within0; 80 % intervals)
| arm | rep | h50 | h25 |
|---|---|---|---|
| tcA1 | −5.5 [−8, −3] | −0.8 [−2.0, +0.4] | −0.2 [−0.8, +0.4] |
| tcA2 | −3.5 [−5.5, −1.5] | +0.6 [−0.6, +1.8] | +0.6 [−0.3, +1.5] |
| tbE | −3.5 [−6, −1] | +0.3 [−1.0, +1.5] | +0.3 [−0.5, +1.1] |
| tcb | −2.0 [−4, 0] | +0.8 [−0.5, +2.0] | +0.6 [−0.3, +1.5] |
Stage gain (terminal − mix tap) for `rep` arms: 70–100 % of cap250's own (+4.7 full0, +8.4 within0).
Hypothesis H ("the stages explain the gap") predicts `rep` ≥ −1.0 for the best source.

## Rules (fixed now)
- H refuted if every source's pooled `rep` upper 80 % bound is below −1.0; supported if the best `rep` point ≥ −1.0 with
  lower bound > −2.0; else undecided.
- Candidate (user bar): pooled point ≥ +1.0 held-new, paired 80 % lower bound > 0, positive on both orders, P04 and P10
  reported (neither decides), returning ≥ −0.3. 12 arms on ten people is selection: a passing arm is a lead that needs
  fresh orders (fold packages) and returning before any adoption.
- Returning P04/P10 cannot be measured in this chain (recognition not replayed). If an arm passes held-new, the returning
  measurement is the next required step and is not claimed here. Within ±0.5: "undecided", not closed.
- Runtime is part of the result: tcA2 TTT costs ~57 ms/window A100 (~1,142 s per sealed run) on top of cap250's 2,555 s,
  Team B ~45 ms/window; tcA1 ~11 ms (2080 Ti). A gain that needs TTT is not deployable as is.

## Jobs (Galvani) and checks
- Stream smokes: Team C full0 P01 3144946 (re-stream vs stored C1 eval: A1 bitwise, A2 max 2.4e-7, 0 label changes; first
  try 3144926 died on node cn116, excluded); Team B P01 3144947 (first try 3144927 failed on an attribute of the older R1
  package, assertion fixed). Full streams: Team C within0 3144951 (10 tasks), Team B full0+within0 3144961 (10 tasks).
- Chain smoke 3144943 FAILED on its own guard (my first hook skipped lane Z's corrected-slot patching, stock|base ≠
  release); rewritten to wrap lane Z's `run_chain`; smoke 3144950 passed (stock|base = release bitwise, w = 0 arm = stock
  bitwise, chain permutation = source delivery). Complete runs 3144978 (full0), 3144979 (within0) passed every guard.
- Scoring checks (local `revisit/score_r.py`): each `rep` arm's mix tap equals its source exactly (max |Δ| 0.0); cap250's
  own mix tap is untouched in every arm; stock|base terminal equals lane Z's prof bank bitwise. Sources: src_full0 sha
  ab7db4ee…, src_within0 2e61e5bb…; full table `revisit_opus_score.json`.

## Result (scored Fri 20 Nov ~23:55 CET; terminal BA minus cap250, pp, pooled full0+within0, paired 80 % resampling people and orders)
Stage map, full0 / within0 (BA %):
| arm | mix | rebal | EM | TSEM | terminal | stage gain (terminal − mix) | Δ terminal [80 %] | full0 ; within0 | P04 | P10 |
|---|---|---|---|---|---|---|---|---|---|---|
| cap250 (stock) | 69.0 / 69.9 | 72.6 / 73.8 | 73.4 / 76.3 | 73.9 / 77.6 | 73.7 / 78.3 | +4.7 / +8.4 | — | — | — | — |
| tbE rep | 64.9 / 65.9 | 65.3 / 66.7 | 68.5 / 70.5 | 70.2 / 72.0 | 70.3 / 72.8 | +5.4 / +6.9 | −4.46 [−7.12, −1.93] | −3.42 ; −5.50 | +2.9 | −7.7 |
| tcA1 rep | 63.0 / 62.0 | 62.6 / 63.4 | 64.7 / 65.4 | 65.1 / 66.5 | 65.5 / 67.0 | +2.5 / +5.0 | −9.80 [−14.34, −4.67] | −8.24 ; −11.36 | −5.7 | −24.7 |
| tcA2 rep | 65.2 / 65.5 | 64.6 / 65.7 | 64.5 / 66.6 | 65.2 / 67.7 | 65.8 / 68.4 | +0.6 / +2.9 | −8.95 [−14.04, −3.29] | −7.96 ; −9.94 | −3.2 | −26.2 |
| tcb rep | 68.3 / 68.6 | 68.2 / 70.0 | 70.4 / 71.6 | 70.7 / 73.3 | 70.3 / 74.3 | +2.0 / +5.7 | −3.72 [−7.15, −0.27] | −3.39 ; −4.05 | +2.1 | −14.4 |
| tbE h50 | 70.3 / 70.8 | 71.4 / 71.8 | 74.0 / 75.4 | 74.1 / 77.1 | 74.0 / 77.7 | | −0.17 [−1.86, +1.43] | +0.29 ; −0.63 | +4.5 | −2.9 |
| tbE h25 | 71.1 / 71.6 | 72.9 / 74.3 | 74.5 / 77.0 | 74.7 / 78.2 | 74.3 / 78.5 | | +0.36 [−0.67, +1.32] | +0.51 ; +0.20 | +2.6 | −1.5 |
| tcA1 h50 / h25 | 68.1 / 67.8 · 69.1 / 69.4 | | | | 72.2 / 74.3 · 73.4 / 76.7 | | −2.83 [−5.30, −0.08] · −0.96 [−2.09, +0.15] | | −1.0 · −0.3 | −10.8 · −4.0 |
| tcA2 h50 / h25 | 70.3 / 70.0 · 70.4 / 69.8 | 70.7 / 71.9 · 72.6 / 73.5 | | | 71.7 / 74.2 · 73.5 / 77.0 | | −3.09 [−6.27, +0.21] · −0.80 [−2.11, +0.60] | | +0.6 · +0.7 | −13.7 · −5.0 |
| tcb h50 | 70.9 / 71.2 | 72.7 / 73.3 | 73.7 / 75.0 | 74.3 / 76.8 | 74.7 / 77.6 | | +0.08 [−1.96, +1.99] | +0.93 ; −0.77 | +4.0 | −5.2 |
| **tcb h25** | 70.8 / 70.9 | 73.6 / 74.3 | 74.6 / 76.7 | 74.6 / 77.8 | 75.1 / 78.5 | | **+0.73 [−0.38, +1.74]** | +1.34 [+0.75, +2.06] ; +0.13 [−1.14, +1.40] | +2.2 | −1.7 |
(P04/P10 = pooled per-person delta; people with a positive delta: tcb h25 7/10, tbE h25 5/10.)

Against the registered predictions: rep arms tbE −4.5 (pred −3.5 [−6, −1], inside) and tcb −3.7 (pred −2.0 [−4, 0], inside);
tcA1 −9.8 and tcA2 −9.0, both far below their intervals (pred −5.5 / −3.5). Blends: tbE and tcb h25/h50 inside their
intervals; tcA2 h50 −3.1 (pred +0.6 [−0.6, +1.8]) refuted; tcA1 h50 −2.8 below its interval.

Rules applied:
- **H (the stages explain the gap): not supported** (best `rep` point −3.72 < −1.0). By the letter it is undecided, not
  refuted: tcb's upper bound is −0.27, not below −1.0. My reading: the stages are not what the clean-slate systems lacked.
- **Candidate bar (≥ +1.0 held-new, lower bound > 0, both orders positive): no arm passes.** Best tcb h25 +0.73
  [−0.38, +1.74] (positive on both orders but within0 only +0.13); tbE h25 +0.36 [−0.67, +1.32]. Both are undecided-positive
  inside the ±0.5 noise and were picked from 12 arms on the same ten people. Returning was not measured (not replayable here).

## Reading (what changed in understanding)
1. **For a non-adapted source, cap250's stages transfer almost fully.** Team B's encoder ensemble gains +5.4 / +6.9 from
   the stages, against +4.7 / +8.4 for cap250's own mix (115 % / 82 %). Its terminal deficit (−3.4 / −5.5) is about its
   mix-level deficit (−4.1 / −4.0). So Team B lost on member quality: 65 % from one 9-encoder family against 69 % from
   cap250's 13-slot mix. Having no adaptation was not the cause.
2. **TTT outputs and cap250's stages do not stack. They conflict.** Team C's TTT posteriors gain only +0.6 / +2.9 from the
   stages, and its no-TTT A1 only +2.5 / +5.0. P10 collapses (−24 to −26 pp as the source, −11 to −14 at h50). Team C's
   EA + prefix-BN + Sinkhorn/IM self-training already moves the session class balance, and rebalance plus SessionEM then
   push it a second time (tcA2 h50: mix 70.3 → rebal 70.7 → EM 70.5, where cap250 goes 69.0 → 72.6 → 73.4). An adapted
   source and an adaptation stack must not both act on the same posterior.
3. Mix-level gains do not predict terminal gains when the source is already adapted. tcA2 h50 is +1.2 at the mix and
   −3.1 at the terminal. The lane Z transmission factor of 0.8–1.2 holds only for sources the stages have not already
   adapted (tbE, tcb h25).
4. P04 gains in almost every blend (+2 to +4.5) and P10 loses in every one. That fits lane Y: P04 needs a different class
   map, and P10 is near ceiling under cap250's stages.

## Decision and next tests
- Not an upload candidate. No arm meets the bar. The positive arms need Team B (about 45 ms/window) or Team C TTT (about
  57 ms/window A100) on top of cap250's 2,555 s, and the runtime fit is unmeasured. Returning is unmeasured, and every
  blend loses on P10, the stand-in for the returning-heavy sealed mix. hardcb stays the upload default.
- Open, not closed (undecided-positive, |Δ| < 0.8): a non-adapted, non-redundant ensemble (Team B style) as a 15–25 %
  partner of cap250's mix. The next decisive form, which was not run: tbE at w 0.15/0.25 on the hardcb system, on new
  orders through real fold packages, plus returning P04/P10, plus a runtime card for a trimmed Team B (enc only, 3–5
  encoders).
- Closed at its tested scope: an adapted source (TTT) as cap250's mixer input or partner, both as `rep` and at h50.


## Coverage audit (scientific-discovery, references/coverage-audit.md; ledger and wave_20nov cards, read 20 Nov ~23:20)
Scope codes: S standalone · D diluted slot / terminal insertion in cap250 · W whole replacement of the mixer input through
cap250's stages · T tuning of cap250's own mixer or stages.
| family | scope measured | best measured | W before tonight? |
|---|---|---|---|
| Foundation models (REVE, CBraMod, MANAS; ten FM variants) | S (MANAS 59.2, CBraMod 50.4 held-new) · D (1/16 slot, 0.2 of raw slot) | slots −0.23..+0.26; CBraMod 0.2 raw slot +0.13 (kept in hardcb) | no |
| Clean-slate A (MANAS B6) | S | −14.95 held-new vs cap250 | no |
| Clean-slate B (encoders + session vMF EM) | S with its own session model | ≈65.3 (−8.4) | no |
| Clean-slate C (CNN ensemble + TTT) | S with its own adaptation | A2 ≈65 (−8.6) | no |
| TTT inside cap250 (lane T, lite5) | T/D (FT-bank objective) | +0.07 vs hardcb | no |
| Waveform/raw decoders F1–F11, F29–F37 | S, D (0.2 terminal logits, slot replays) | all full-stack points below stock | no |
| More people (B5) / external corpora (B6) | S; W-like for ONE family (raw FBC slot swapped) | +4.6 standalone family → −0.4 / +0.05 full system | partial (one slot) |
| F-C25 hard / hardcb | slot swap (W-like at slot level) | +0.32 / +0.39 | slot only |
| Virtual subjects (V) | S; raw-slot only in chain | refuted at the tested dose | slot only |
| Alignment ideas (Y #5 oracle rotation) | S / oracle, no chain | +3.4..+5.1 oracle on a weak member | no |
| Moon M1 in-context / M2 experts | S | M1 +2.3 vs blank standalone; M2 stopped at stage 0 | no |
| Mixer re-weighting (R1, lane Z levers) | T | best +0.27 undecided; learned weights ≤ 0 | n/a |
| Stage retuning (S2), EM coupling (G1), post-mixer corrections (W), vote calibration (C2v) | T | S2 +0.21 single move (nested ≈0); G1 −5.9; W ≤ +0.06; C2v 0 | n/a |
| Returning keep-adapt (R4), recognition/personal (Q, G) | T / D | R4 undecided-negative in-lab, +0.4..+3.7 NETBCI; G refuted in four forms | n/a |
Gap: no new family was ever the main posterior source of cap250's stages (W); every family with a strong standalone result
(Team B/C ensembles, TTT, B5 data, FMs) entered as a slot. Chosen restriction (not a hard bound): "members enter through
the 13-slot mixer". Tonight's arms test exactly that restriction. Proposals not run tonight: W with an FM fine-tuned with
B5/B6 data; W with Y #5 rotation estimated label-free.
