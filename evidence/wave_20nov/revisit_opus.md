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

## Jobs (Galvani)
Smokes: Team C stream full0 P01 (zero-change vs C1 eval) 3144946; Team B stream P01 3144947; chain full0 smoke (w = 0 arm
bitwise equal to stock) 3144943.

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
