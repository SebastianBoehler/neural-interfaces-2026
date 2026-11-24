# First-principles design review · Thursday 19 Nov 2026, 13:00 CET

Author: research sub-agent (Fable). Read-only review; no jobs, no edits besides this file. Other reports in
this folder were not read. "Old ledger" and "old state" mean `git show a5843492:docs/LEDGER.md` and
`git show a5843492:docs/STATE.md`; line numbers refer to those blobs.

## 0. Verdict in five lines

1. The repo already contains almost every mechanism family a 0.85–0.87 system needs. The gap is not a missing
   family. It is that the one lever with the largest measured ceiling (an independent, people-scalable base
   decoder fed by external people) was last built on 1–2 Nov and has since been run only in weak forms.
2. In-session label-free adaptation is measured to be label-noise-bound in this feature set. More variants of
   it (the bulk of the 60+ experiments since 18 Nov) cannot produce +1.7 pp.
3. Leaderboard ranks 6–16 sit within 0.7 pp (0.830–0.837). That is a plateau many independent teams reach.
   The three entries above it (0.846–0.859) most plausibly have one extra component worth 1–2.5 pp.
4. Proposal: one step-change build (a montage-agnostic pretrained backbone trained on pooled task-matched
   corpora, shipped as frozen members in both the new-person and returning paths), one dependent amplifier
   (agreement-gated refits, small), and one regime fix (tune for full shuffle only). I found evidence for only
   one step-change candidate, not three; section 4 says what was rejected and why.
5. The last upload is a free option: the board keeps our best score (docs/SEALED_UPLOADS.md:18-20). Spend it
   on the highest-ceiling candidate that passes worker conformance, not on a +0.3 pp polish.

## 1. What the task physically offers (first principles)

**Facts.**
- Montage: 41 scalp channels from a 64-channel actiCAP, reference M1, no EOG or EMG channel, no Fp, O or T7/T8
  sites (evidence/baseline/data_audit.json; NEMAR record https://nemar.org/dataset/nm000290).
- Public data: 10 people, 14 sessions, 2,655 windows; eight people have one session, P04 and P10 three
  (https://github.com/nemarDatasets/nm000290). Six cued runs per session: one acquisition (sham feedback) and
  two online (real feedback) per interface.
- Sealed set: about 21k windows (docs/COMPETITION.md:88); assumed 106 sessions, 46 of them returning
  (old ledger :1345); "57 % unseen participants plus 43 % later sessions" (docs/COMPETITION.md:84-85).
- Our error profile on new people: MI 81.0, SUB 75.2, WORD 73.5; P04 54 %, the other nine 75–90 %
  (old state :459; old ledger :1356-1359).

**Inference: error budget.** With returning sessions near 90 % (P10 routed 91–93 %, P04 61–70 %, old ledger
:6997-7010) and a sealed score of 0.830, sealed new people sit near 78 %. New people therefore carry about
12.5 of our 17 points of error (0.57 × 22), returning people about 4.5. A score of 0.86 needs roughly +5 pp
on new people at unchanged returning accuracy, or +3 pp new and +3 pp returning.

**Signal sources, ranked by what is physically person-independent.**
1. Band power and covariance over sensorimotor and parietal cortex (MI ERD at C4/C3; arithmetic at parietal
   and frontal-midline sites). Present: core1, 69 % alone (old state :355-357).
2. Ocular and slow frontal potentials. Feedback bars and game lanes are class-specific, so gaze is too; the
   corneo-retinal dipole has the same sign in everyone. Present: SessionEM and eye nets, 51 % alone
   (old state :365; old ledger :7464).
3. Covert-speech activity for WORD (left inferior frontal; glossokinetic slow potentials). No dedicated
   member; the montage lacks temporal sites, so the ceiling is modest.
4. Session structure: balanced classes per run and cell. Present, see section 3.

**Physical upper bound.** A fully personal system trained with a session's own true labels (four runs) reaches
84.4 % with fine-tuned nets and 87.1 % fused with the chain, against 79.6 % for the chain on the same windows
(old ledger :6997-7002). So about 87 % is the supervised within-session ceiling of our feature set for new
people. A label-free system at 82–83 % on new people is plausible; 87 % is not. With returning people at
90–93 %, that bounds a realistic best system near 0.86–0.87, which is where rank 1 sits (0.859).

## 2. What a 0.85–0.87 method most plausibly does

No public write-up exists. The leaderboard has no method column (https://neural-interfaces26.github.io/leaderboard.html#bci).
Two public participant repositories were found; neither is competitive (one reports 57.7 % leave-one-person-out
with a spectral model and causal alignment, against our 76 %: https://github.com/dpny518/neural-interfaces-2026-track02).
What follows is inference from the task, the board and the literature.

- **Plateau recipe (0.830–0.837, about ten teams including us):** covariance/tangent-space and compact
  convolutional members, per-session Euclidean or Riemannian alignment, session batch-norm statistics, class
  prior correction, ensembles. This matches the online test-time adaptation literature for motor imagery:
  alignment plus entropy and marginal-balance objectives (T-TIME, https://arxiv.org/abs/2412.07228;
  Wimpff et al., https://arxiv.org/abs/2311.18520; SPDIM, https://arxiv.org/abs/2411.07249).
- **Break-out candidates (0.846–0.859).** Team names suggest deep-learning groups running ensembles of 26–29
  models (ETRI "SED29", "SDE26") and a lab known for pretrained neural models (PNPL). The most likely extra
  component is a high-capacity backbone pretrained on many people and fine-tuned on all public data, because:
  (a) it adds a member whose errors are independent of the plateau recipe;
  (b) trained on all 14 public sessions it memorises the ten public people, which lifts the 43 % returning
  cells with no recognition step and no routing errors;
  (c) montage-agnostic models (REVE, 92 corpora, 25,000 people, 4D electrode positions:
  https://arxiv.org/abs/2510.21585) can pool external task-matched people, the one resource that scales.
- **Less likely:** better label-free adaptation alone. Our own measurements (section 3) say adaptation on
  noisy chain labels saturates; other teams face the same limit.
- **Possible but not recommended:** recovering recording order or run membership from slow EEG drift to revive
  cue-order and run-count priors. Our run recovery reached purity 0.5 under full shuffle (old ledger :5193),
  the organisers removed chronology on purpose (old state :332-338), and top-three entries are audited.

## 3. Comparison with this repo

| Mechanism | Status here | Evidence | Reading |
|---|---|---|---|
| Spectral/Riemannian core, causal recentring | Present, strong | core1 69 % alone (old state :355) | Fine |
| Compact nets, session batch norm, Euclidean alignment | Present | about 65 nets (old state :358-359) | Fine; nets flatten from 7 to 9 people (old ledger :149) |
| Class-prior rebalancing | Present, large | worth 5.75 J (old state :364) | Fine |
| Session EM on class templates | Present | TSEM, SessionEM (old state :365-366) | Saturated |
| Pseudo-label fine-tuning in the session | Present | +0.4 to +0.7 new (old ledger :41) | Bound by label purity: chain labels are 74–78 % pure; a personal model needs about 88 % purity at 100 windows for +1 (old ledger :7153-7163, :7467). Only a more accurate chain raises purity |
| Transductive balanced clustering | Tried, fails | purity 48 % vs chain 79 % (old ledger :7488-7492) | Classes do not form clusters in this space; closed on evidence |
| Label-free class-mean alignment | Tried, fails | −2.6 to −5.8 fused (old ledger :1556-1557) | The +24 oracle was the labels |
| Run-level count filter | **Present but mis-tuned for full shuffle** | within +0.54, runs +0.27, full +0.03 / −0.02, 12.5 % false latches under full (old ledger :5997-6003; lean/tools/fit_cf.py:43) | Inert if sealed delivery is shuffled across the session |
| Dev order mix | **Mis-weighted** | 8 pinned orders = within0–3, full0–1, runs0–1 (old ledger :8045) | Six of eight orders are run-contiguous; the chain was selected mostly on a regime the sealed set probably does not use |
| Historical comparisons | **Partly invalid** | two alignment members used recording-order caches under shuffled scoring (evidence/next_wave_16nov/scoring_audit_18nov.md:11-18) | Member weights chosen on those comparisons are suspect |
| Returning people: recognition and personal head | Present | gate 86.7 % accept at 1.9 % false accept (old state :368-371); routed P10 91–93 % | Validated on two people; sealed transfer of returning-path gains was 0.46× and 0.19× (old state :420; old ledger :1345-1347), new-person gains 1:1 (BRIEF.md:36-37) |
| External people as training data | **Weak form** | teacher trunks on 7 MI corpora, frozen, logistic head (old state :360-361); Scherer rows: members +1–3 alone, chain 0, core1 −1 (old ledger :54) | Fixed-montage nets needed channel mapping; no pooled montage-agnostic training |
| Foundation models | **Weak form** | frozen probes 36–44 % (old ledger :336); REVE fine-tuned on nine PROTEUS people only: 55.8 % alone, correlation 0.36–0.38 with our nets, pretraining worth +7.3 alone and +1.07 [+0.26, +1.85] in the mix (old ledger :8229-8238) | Never pooled with external task data, never in the chain, never on returning people |
| Foundation-model deployability | **Wrong rule** | rejected because one forward is 7× the four small nets (old ledger :8240) | Absolute cost is 1.446 ms × 21k windows = 30 s per model per sealed run, of 3,600 s |
| Independent 70 % member | Absent | estimated at +3.1 on new people (old state :436; old ledger :1878) | The single quantified route to +1.7 pp sealed |
| More people | Absent | core1 +8 points per doubling, +7.3 after stages (old ledger :149; old state :433) | Largest measured ceiling; no ten more PROTEUS people exist, so it must come from external people |
| Meta-learned or in-context session decoder | Weak form | F1/F2 prototypes, 38–44 % (friday/RETROSPECTIVE_18NOV.md:9) | 14 sessions cannot train it; not a Friday item |

**Why the recent search stayed flat (inference).** Since 18 Nov the ledger lists 37 raw-waveform lanes on one
small decoder (41–48 % alone) and about 30 edits inside cap250 (docs/LEDGER.md:548-1458). The retrospective
says it itself: the search "was narrow in where they entered the system and who tested them"
(friday/RETROSPECTIVE_18NOV.md:29). Every lane perturbs a chain whose constants were co-tuned on ten people,
so extra members are rejected "even when strong alone" (old state :449). None attacks the two large ceilings
(people, independence).

## 4. Proposals

### D1. Pooled pretrained backbone as an independent member in both paths (the step-change build)

**Mechanism.** Fine-tune REVE in two stages. Stage A, shared across folds: supervised training on pooled
external people with one head over motor imagery, subtraction/arithmetic, word generation and rest, loss
masked per corpus. Corpora already exported or fetched here: Scherer 2015 (WORD, SUB, hand; 9 people × 2
days), Shin 2017 A and B (29 people, MI and arithmetic), Shin 2018 word generation (26 people), PhysioNet MI,
Dreyer 2023 (old ledger :2377, :2449-2456, :2539). REVE takes electrode coordinates, so no channel
interpolation is needed. Stage B: fine-tune on PROTEUS with causal per-session alignment of the input, as in
§280. Three to five seeds. At test the members are frozen: no refits, about 30 s per model.

**Why +1.7 pp is plausible.** (i) Old ledger :1878 puts an independent 70 % member at +3.1 on new people,
about +1.8 sealed. (ii) REVE fine-tuned on nine people only is at 55.8 % with the lowest error correlation
of any member and a real pretraining effect in the mix. The gap to 65–70 % is what stage A, longer schedules
and seed ensembling have to close; pooled pretraining gave the project its largest single gain when done with
frozen small trunks (+2.9 dev, docs/SEALED_UPLOADS.md:27). (iii) For returning cells the all-public model has
seen the person; this benefit is invisible to leave-one-person-out and has never been measured for a
high-capacity member.

**Honest size.** My estimate: alone 60–67 %; chain +0.5 to +2.0 on new people; sealed +0.4 to +1.5, with
perhaps one chance in five of reaching +1.7. It is still the only candidate with a quantified path there.

**Validation (decisive, people-nested).** PROTEUS leave-one-person-out, ten folds × three seeds, full0 and
full1 primary, within0 secondary; standalone score, correlation with the FBCNet family, then the complete
corrected cap250 replay with the member inside the mix and as a TSEM view, intervals resampling people and
orders. Returning: the `seen_session` split (P04, P10 sessions 2–3) with all-public fits. Scherer cannot be
both training data and external check: run stage A with and without Scherer and keep the without-Scherer arm
for the external comparison.

**Compute.** Stage A: about 40k windows × 10 epochs, under 1 h on one H100. Stage B: 9–30 min per fit on an
A100 (old ledger :8229), 30 fits in one array on Galvani or Ferranti. Package: 138 MB per seed in fp16;
timing on the RTX proxy before any decision.

**Timeline.** Thu 13:30–15:30 smoke (section 5). Thu 15:30–21:00 stage A and the 30 stage-B fits; all-public
fits in parallel. Thu night: chain replays on full0/full1/within0, returning replay. Fri 08:00 read-out and
gate. Fri 08:00–13:00 package build, worker-image run, RTX timing. Fri 13:00–17:00 conformance repeats and
the paired comparison against cap250. Fri 18:00 decision.

**Gate, stated now.** Standalone ≥ 62 % on ten folds and chain gain ≥ +0.5 pp on the full orders with returning
not negative: package it. Because the board keeps our best score, the upload bar is "expected gain positive
and worker-safe", not "interval above zero".

### D2. Agreement-gated refits on top of D1 (dependent amplifier, not a step-change)

**Mechanism.** In-session fine-tuning with true labels is worth +18.5 pp for the nets and +7.5 pp for the
chain (old ledger :6997-7002). On chain labels the same personal model loses 12 points, and the same error
rate drawn at random is even worse (63.6 against 67.8 %, old ledger :7162-7163). So the binding constraint is
label purity at about 136 windows, not the systematic nature of the errors; an independent teacher does not
help by itself. What D1 adds is a second view with low error correlation: windows where the chain and the D1
member agree should be about 90 % pure at about half coverage (0.78 × 0.62 correct-and-agree against about
0.04 wrong-and-agree, assuming independent errors), which reaches the measured requirement of 100 windows
at 88 % purity only after about 190 delivered windows.

**Size and test.** +0.2 to +0.5 pp, concentrated late in long sessions; agreement with the 51 % eye net gave
nothing (old ledger :7464). Test offline on stored posteriors first (purity and coverage of the agreement
set per fifth of the session). Run only if D1 clears 62 % alone.

### D3. Tune for the delivery the sealed set uses (regime fix, enabling rather than step-change)

**Mechanism.** Decide explicitly that sealed delivery is a full within-session shuffle (the organisers'
wording and the 10 Nov score drops both point there, old state :332-341). Then: report and select on full
orders only; remove the run-level count filter from the candidate or prove it never latches; convert every
stage with a recency constant to its exchangeable form (uniform weights over the delivered prefix), starting
with the rebalancer that divides by the "recent" class share (old state :364). Under an exchangeable stream
any forgetting is pure variance.

**Size.** +0.1 to +0.4 pp, CPU only, half a day. Its main value is that D1 and D2 are then judged on the
right instrument. Hedge: keep within0 as a non-regression check in case the shuffle is per run.

### Considered and set aside (with the evidence)

- More label-free adaptation on the present chain (clustering, class-mean alignment, self-training variants):
  measured as purity-bound or negative (old ledger :7488-7492, :1556-1557, :7153-7163).
- Personal high-capacity models for recognised returning people as a separate build: personal nets gained
  +0.17 on the real later sessions of P04 and P10, and cross-session personal nets drop 6 points on P10
  (old ledger :1345-1348, :7007-7009). Returning cells are covered inside D1 by the all-public fit instead.
- Recovering run membership or recording order from EEG drift: purity 0.5, audit risk (section 2).
- A meta-learned in-context session decoder: the right long-term design, untrainable on 14 sessions by Friday.

## 5. The experiment to launch in the next two hours

**Three-arm REVE scaling smoke, four held people (P02, P05, P08, P10), two seeds, plus one returning fit.**
- Arm A: PROTEUS-only fine-tune (reproduces rv1/rv2; control).
- Arm B: stage A on pooled external corpora without Scherer, then PROTEUS.
- Arm C: as B with Scherer in stage A.
- Returning arm: arm B fitted on all public sessions except P04 and P10 sessions 2–3, scored on those.
- Read-outs: standalone balanced accuracy under full0, correlation with the FBCNet posteriors, per-class
  recall, and for the returning arm the difference to the routed chain on the same windows.
- Prediction (written before any result): A 55 ± 4; B − A +5 (80 % range +1 to +9); C − B +1 (−1 to +4);
  returning arm alone 80 (70–88) on P10, 55 (45–65) on P04.
- Rule: B − A ≥ +4 with correlation ≤ 0.5 starts the full D1 wave tonight. B − A < +2 on both seeds stops D1,
  and the last upload goes to the best cap250 variant (hard replacement, +0.2 to +0.4 pp).
- Cost: 24 fine-tunes of 9–30 min plus one stage-A run per arm; fits in two hours on eight GPUs.

It is the most informative run available because it tests, in one wave, the only two ceilings that are both
large and measured (more people, independent member), and its outcome decides what the last upload is for.

## 6. Limits of this review

- No primary source describes any top entry; section 2 is inference.
- The 57/43 cohort split and the 106/46 session count are the repo's assumptions, not a verified census.
- I did not verify which people the packaged base nets were trained on, nor the exact recency form of the
  rebalancer; D3's audit should confirm both.
- The +3.1 estimate for an independent 70 % member (§132) was made on an earlier chain and in recording order.
- REVE licence and corpus declaration were cleared in §280's A0 table (old ledger :8177-8181); recheck that
  REVE's 92 pretraining corpora do not include sealed PROTEUS sessions before any upload.
- The runtime limit on the current worker is unconfirmed (OFFICIAL_CONTRACT.md:71-77); plan for 3,600 s.
