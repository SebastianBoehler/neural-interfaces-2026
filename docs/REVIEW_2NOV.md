# Review, 2 Nov 2026 (evening): learnings, solution space, next levers

Four independent reads fed this review: our own docs and evidence; the official
repo, website, Codabench API and dataset papers; the 2024–2026 ML literature; and an
interdisciplinary sweep (neurophysiology, public datasets, other fields). Agent
findings are marked "reported" where we have not checked them ourselves.

## 1. Where we stand

Lead package: **cand_nxif**. Its dev numbers:
- sel Δ +2.91 over R19e (dla 78.01, ds ≈ 89.6);
- it beats m7r4 on three disjoint pretraining-seed sets (s05, s611, s1223);
- it passes harness, official A100 replay and worker replica.

It has not run on the real worker. Fallback: cand_m7r4.

What moved the metric, in order:

| Lever | Gain |
|---|---|
| Causal recentering | +7 dl |
| <4 Hz slow display-locked signal | +6.8 dl |
| Heterogeneous ensemble | +6.8 dl |
| Rebalancing | +2.3 dl / +6.8 ds |
| Gaze + event EM | +2.7 dla |
| External-MI teacher | +2.9 sel |

**The common thread is information the 10 PROTEUS participants cannot supply on their own:**
- session statistics, through recentering;
- the known class balance, through rebalancing;
- the screen layout, through gaze;
- hundreds of external brains, through the teacher.

Everything that only re-learns PROTEUS from PROTEUS has plateaued since R10.

## 2. Risks the dev numbers hide

- **P10 dominates selection.**
  - ds is P10 only, and P10 carries about 65 % of sel.
  - cand_nxif per participant (dla): P10 +8.3, P02 +4.8, P08 +3.2, P05 +1.6, P07 +1.6, P01 −1.1, P03 −1.1, P06 −2.1.
  - Without P10 the mean gain is +0.99, positive for 4 of 7.
  - The teacher is a real but smaller win for a typical unseen person than sel suggests.
  - P09, the warm-up test participant, never used for selection, drops 5.9 in all_loso.
- **ds is at the within-session ceiling** (≈ 89.6 against 88.9 cited), so it no longer discriminates.
- **Audit.** Organisers rerun the training pipeline from committed config and require ±2σ.
  - The teacher (pretraining 24 nets plus export) has no committed one-command recipe yet.
  - External corpora and compute must be declared.
- **Sealed task still bound to 5 Oct placeholder programs** on the public Codabench API.
  - Re-check on 4–5 Nov. A failed run costs no quota, but its logs are hidden.
  - Worker image digest unchanged (`neural-compet:v3`, pushed 12 Oct).

## 3. Rules that change strategy (upstream #64, today)

- **Sealed phase:** 5 Nov 00:01 CET → 21 Nov 23:59 CET. One upload per day, 5 per participant per track. Logs hidden.
- **Force_Best:** the leaderboard keeps the team's best sealed score, so a weaker upload never lowers it. Uploads 2–5 are free options, and the right policy is to maximise the expected *maximum* over decorrelated arms, not to repeat the single best.
- **Quota:** a failed run does not count; a cancelled run does. The daily reset is probably UTC midnight (01:00 CET), unverified.
- **External data:** allowed when the original terms permit competition use; declare corpora and compute. Pretrained weights keep their licences.
- **Test-time adaptation:** causal, label-free and within a session; unchanged.

## 4. Searched vs allowed solution space

Allowed after the code freeze: anything expressible as `weights.npz` (member weights, rebalancing and EM parameters) or a stateless TorchScript member (aligned window → logits), trained offline on any licence-clean public data.

| Region of the allowed space | Status |
|---|---|
| Re-weighting existing members, EM/rebalancing parameters | saturated (plateaus, closed grids) |
| New PROTEUS-only members (≈ 30 architectures, DG tricks, foundation-model fine-tunes) | saturated; fine-tuned members now negative |
| Teacher: architecture, recipe, C/w, seeds, recipe mixes | saturated today (waves 5–7) |
| **Teacher: training data (which brains, which tasks)** | **barely explored**: the same 7 MI corpora since R10 |
| Teacher: second decorrelated teacher as its own member (late fusion) | untested |
| Teacher on the raw (unaligned) stream | untested |
| In-graph test-time augmentation inside the teacher | untested for the teacher (closed for members) |
| Physiology-informed views (beta lateralisation over fronto-temporal pairs, aperiodic normalisation) | partly closed (band power, alpha lateralisation, EMG); beta fronto-temporal pairs untested |
| Session-stateful TTA (T-TIME etc.) | allowed by the rules, impossible after our freeze |

## 5. What the research adds

- **No public healthy-EEG corpus has all three tasks.**
  - Only Scherer 2015 (excluded by user decision) has MI + SUB + WORD.
  - Rathee 2021 MEG has them, but in 306 MEG sensors (CC0, reported); projecting it to EEG is 2–3 days and uncertain.
- **SUB vs WORD is the intrinsically hard pair**, even in MEG: 68 % leave-one-subject-out vs 80–85 % for hand vs either (Youssofzadeh 2023, reported).
  - Our largest confusion is MI→WG (18 %), then SUB↔WG (14–15 %).
- **Population-level sign prior.** PROTEUS MI uses the *non-dominant* hand, which is right-hemisphere ERD in right-handers. WORD and SUB are left-lateralised in beta (lateralisation index ≈ 0.26–0.30).
- **New licence-clean data for the teacher's weakest side**, the cognitive classes. All of this is agent-reported, so verify before use:
  - ds007554 CMx7-MM: 30 people, 32 EEG, arithmetic + MI + n-back + rest, CC0;
  - Zyma 2019 (PhysioNet eegmat): 36 people, serial subtraction vs rest, ODC-By;
  - Ajra 2026 (CC0, MI + arithmetic), flagged in our docs.
- **Literature levers that fit the freeze:**
  - learned per-window TTA (BFT, arXiv 2601.07556): +1.3–1.5 LOSO MI, reported; plain averaging ≈ 0;
  - segment-and-reconstruction augmentation (arXiv 2405.14994): ≈ +2 cross-subject with EA, reported.
- **Foundation models:** the 2026 benchmarks show no reliable margin over compact CNNs in LOSO MI, consistent with our CBraMod/LaBraM/REVE results. Deprioritised.

## 6. Ranked next levers (all data-file arms, fit the freeze)

1. **Bigger, more diverse teacher data** (needs your licence decisions). Add arithmetic-vs-rest corpora (ds007554, Zyma 2019; Ajra 2026 if the same or separate) as new heads, and more MI corpora if licence-clean. This is the one lever with a track record of step changes; it targets SUB, where external evidence is thinnest. About 1 day per corpus (export + pretrain + screen).
2. **Late fusion: a second teacher as its own member.** Each teacher keeps its own logistic map, and they mix as separate members. Concatenating features (m7r4 + nx+if) hurt; separate maps are untested. Cheap: eval code plus an exporter that writes two Jit members.
3. **Segment-and-reconstruction augmentation in teacher pretraining.** About 20 lines in pretrain.py; one wave.
4. **Raw-stream teacher.** Pretrain with `align raw` and feed the unaligned stream. Decorrelated from the EA teacher, so it pairs naturally with lever 2.
5. **In-graph TTA for the teacher** (time shifts, amplitude jitter, averaged inside teacher.pt). Cheap; low prior.
6. **Robustness check before staging any arm:** report sel without P10 and the worst-participant change.

**Strategy, independent of new arms:**
- Upload cand_nxif first on 5 Nov. That is the first real-worker check of the new teacher, and it should be our best expected score.
- Keep later slots for decorrelated arms: R19e-g, cand_m7r4, and anything from levers 1–4 that passes.
- Commit an audit recipe for cand_nxif now.

## 7. First-principles reframing (2 Nov, late evening)

**The underlying problem.** A plug-and-play BCI: three mental commands for any new person on
any day, with no labelled calibration. Unconstrained, the field solves it with four ingredients:
1. a short labelled calibration;
2. co-adaptive learning, where the decoder adapts to the user online and feedback provides
   implicit labels;
3. multimodal sensing (EEG + eye tracking / fNIRS / EMG);
4. population priors from very large multi-task pretraining.

**What the challenge removes, and our proxy for each:**

| Ingredient | Our proxy |
|---|---|
| Calibration labels | the known 1/3 class balance → rebalancing |
| Co-adaptation | label-free in-session EM |
| Eye tracking | gaze from the frontal EEG |
| Multi-task population prior | the MI teacher (only one of three tasks has public brains at scale) |

**Our evidence says the score is bought by label sources whose errors are independent of the
ensemble.** R10 round 4 simulation, product of experts at exponent 0.5: an independent
per-window teacher of accuracy 0.50 / 0.55 / 0.60 is worth dla +1.0 / +2.1 / +2.9. The
EEGNeX + IFNet teacher has 53.7 % alone (dla), with P(right | R19e wrong) ≈ 0.45–0.50, and
gives +2.9. Raising a teacher's standalone cross-subject accuracy while keeping it
independent is the main dial. Everything trained on PROTEUS shares PROTEUS's errors,
which is why new members, recipes and fusions are now flat or negative.

**Narrowing back to what is reachable:**
- **Stateless, fits the freeze:**
  - A *three-task* population prior. Arithmetic (and, if any exist, verbal) corpora give
    the teacher brains for SUB, its weakest side. This is the one direction that raises the
    teacher's standalone accuracy on the classes it cannot see now. It needs licence
    decisions.
  - In-domain self-supervised pretraining on unlabelled public PROTEUS EEG (rest runs,
    inter-trial periods, the tail after each window; never P04/P09). It adapts the trunk to this
    amplifier, montage and population without labels. Untested.
  - A hierarchical teacher: an "MI vs cognitive" head (hand vs either is 80–85 % even across
    subjects in MEG), separate from the harder SUB vs WG.
- **Needs the code freeze lifted:** cross-view co-training inside each session.
  - The gaze view's per-session posterior labels the *EEG* teacher's map (a per-session
    re-fit of the teacher's logistic map, or the teacher head log-probs as a second EM
    view), so each new person's mapping is learned from errors independent of it.
  - This is the textbook co-adaptive move, and the per-person conditional shift
    (P10's reversed theta, P05's reversed laterality) is exactly what it targets.
  - Since warm-up closed on 1 Nov, new code cannot be proven on the real worker. But the
    sealed rules now state that a failed run uses no quota, and our worker-image replica has
    matched the real worker on every package so far (1024447, 1026185). A crash would cost a
    day, not a slot.
- **Strategy (Force_Best keeps the best score):**
  - Treat the five sealed uploads as an adaptive experiment. The first upload's score is
    visible, so later uploads can move teacher weight in whichever direction the sealed set
    rewards. A weaker arm costs only a slot.
