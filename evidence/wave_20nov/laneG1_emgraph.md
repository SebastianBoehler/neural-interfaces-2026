# Lane G1 - class-coupled SessionEM (idea 1 of evidence/council_20nov/research2_codex.md; registered Fri 20 Nov 2026, before any arm was scored)

Question: does solving the three class displacements of the outer SessionEM jointly, with source anchors and one graph edge, beat the independent class means? Is the effect specific to the cognitive edge (SUB-WORD) rather than a generic shared shift or any edge at the same dose?

## Instrument
Code `laneG1/chain/final_g1.py` (copy of `laneS2/chain/final_s2.py`; lane S2 files untouched; release `ni26_laneG1_20261120` on Galvani). Same frozen pieces as lane S2 (GEB gate, FT banks, shadow, unrouted held-new people; stated, not silent), orders full0 and within0, ten held-new people, systems hardcb and ds_all (= hardcb + lite5 FT banks + FBL2 doubling + per-session dispersion; "ds-l5"). Only the two SessionEM updates (the 2-iteration soft-EM `mu` update) are replaced; the likelihood exponent, covariance, TSEM, count filter, CSP stay.

Anchors `A[band, class]`: for every source session, for every prefix size t (rows < t, strictly past) in the band 20-39, 40-79, 80+, the true-label class mean of the prefix-standardized GE features, ((class mean of rows < t) - prefix mean)/max(prefix sd, 1e-3), averaged over t, then sessions within person, then people with equal weight; per class. Source = the labelled dev rows of the OTHER nine people on the same delivered tape (held person excluded; asserted per person); the held person's own labels are never used. The only labels read are those of the nine other people, as source data (`held_labels_opened` stays False).
Update (both EM iterations): `N` = diag soft class masses, `S = Q^T Z`, solve `(N + 20 I + 12 L) D = S - N A`, `mu = A + D` (20 = the shipped m0). Arms (per system, per order):
- `base`: shipped code path (bitwise equal to lane P hardcb / ds_all, asserted).
- `zero`: coupling path with A = 0, L = 0 (reproduces base bitwise; guard, not scored).
- `anchor`: A, L = 0 (`mu = (S + 20 A)/(N + 20)`).
- `shared`: one shared displacement for all classes, `D = sum(S - N A)/(sum N + 60)`.
- `cog`: edge SUB-WORD (classes 1-2), weight 12.
- `wrongWM`: edge word-motor (2-0), weight 12 (the registered wrong edge).
- `wrongSM`: edge SUB-motor (1-0), weight 12 (extra, to check "both wrong edges").
Per-arm checks: every non-zero arm differs from base and from each other on the delivered rows (silent no-op guard); strict-prefix causality is inherited (the EM only sees rows < t; anchors come from other people); reset per session is the existing per-session instance.

## Source-only displacement check (local numpy, no chain)
Per held person p, source anchor `A_p,k` = class mean over the other nine people of per-session z (full-session standardization), displacement `d_{p,i,k}` = person p's class-k mean in interface i (minus anchor). Edge relatedness `r(a,b)` = mean over p and interface pairs i != i' of cos(d_{p,i,a}, d_{p,i',b}), plus the within-interface cos. Reported with a people-resampled 80 % interval for r(cog) - r(wrongWM) and r(cog) - r(wrongSM). Prediction: r(cog) > both wrong edges by +0.05 [-0.05, +0.15].

## Predictions (pp balanced accuracy, both-order mean; people+orders 80 % intervals; seal)
- cog vs base: +0.25 [-0.40, +0.90]. cog vs anchor: +0.15 [-0.30, +0.60]. anchor vs base: +0.10 [-0.35, +0.55].
- shared vs base: -0.15; wrongWM vs base: +0.05; cog minus wrongWM: +0.20.
- ds_all agrees in sign with hardcb for cog vs base.
- Minority class (per session the class with the lowest base-predicted share in ages 20-60; recall of that class at ages 20-60, arm minus base, mean over sessions then people): cog +1.0 pp, anchor +0.5 pp, wrongWM +0.3 pp.
- Cost: em stage < 0.5 ms/window added for the three-class solve (anchor extraction reported separately).
## Rule (before scoring)
A contrast is refuted on its instrument if its people+orders 80 % upper bound <= 0; supported if cog beats anchor AND wrongWM by >= +0.2 both-order mean with the lower bound > 0 and the same sign in full0, within0 and both systems; inside +-0.5 of base undecided (AGENTS.md), neither closed nor confirmed. P04 and P10 reported, neither decides. J and the unverified 57/43 and 20/80 mixes are reported as UNVERIFIED sensitivities. If confirmed: repeat on the live outer RouteB tail with the same reset (not done here).
## Run log
(appended below)
Fri 20 Nov ~21:05 CET (before the waves started). Smoke 3144599 (full0, hardcb base/zero/anchor/cog + ds_all base/cog) passed: zero == base bitwise, base == lane P hardcb and ds_all bitwise (guard 0.0), every moved arm differs from base (max |dp| 1.8-2.0) and from each other, source-labels of the held person excluded (asserted). First attempt 3144595 failed (the release tape has no labels `y|PROTO`; labels now come from the feature file in delivered order, only the nine other people). EM stage time in smoke (2655 rows, one pass): base 1.52 s, cog 1.69 s (hardcb) / 1.91 s (ds_all), anchor 2.08 s (includes the ten anchor extractions at 0.05 s each).
Source-only check (`laneG1/source_check.py`, ran before any chain arm; cos of class displacements, 10 people, 2 interfaces each): cross-interface cog -0.426, wrongWM -0.320, wrongSM -0.381, cog minus wrongWM -0.106 [-0.281, +0.082], cog minus wrongSM -0.045 [-0.193, +0.109]; within-interface -0.271/-0.186/-0.215 (cog minus wrongWM -0.086 [-0.276,+0.122], minus wrongSM -0.056 [-0.224,+0.125]). The registered prediction (cog above both wrong edges by +0.05) is not met; the cosines are all negative because the three displacements sum to about zero under session standardization, so the check is weakly informative. Not read as closing the arms.
Wave: complete full0 3144608, within0 3144609 (all 14 arms each), scorer 3144610 (afterok; writes cells.json and the minority-recall deltas).

## Result (Fri 20 Nov ~21:30 CET; waves full0 3144608 / within0 3144609, scorer 3144610; `laneG1/data/cells.json`, `report.txt`, receipts; analysis `laneG1/analyse_g1.py`)
Guards: base == lane P hardcb and ds_all bitwise (0.0, both orders), zero == base bitwise, every arm moved and distinct. A late label-alignment guard (stage input argmax vs delivered-order labels, 0.727 on the smoke; asserts > 0.6) was added after the waves (smoke 3144683); the wave code differs only by that assert. Local replay cross-check (recording order, B2 posteriors, EM stage only, all ten people, nine-person anchors): base 77.2, anchor 74.7, cog 73.6 accuracy, same sign, so the loss is not an alignment artefact.
Terminal BA vs own base, both-order mean [people+orders 80 %] (full0/within0), P04, P10, unverified-J (all pp):
- hardcb (base 76.42): anchor -5.37 [-7.42,-2.49] (-5.12/-5.62) P04 -6.3 P10 -13.2 J -6.3; shared -9.58 [-13.4,-5.1]; cog -5.86 [-7.96,-2.99] (-5.24/-6.48) P04 -6.2 P10 -13.9 J -6.9; wrongWM -6.22 [-8.81,-2.70]; wrongSM -6.39 [-9.18,-2.95].
- ds_all (base 76.89): anchor -5.54 [-7.45,-2.87]; shared -9.60; cog -6.39 [-8.25,-3.78] (-6.19/-6.59); wrongWM -6.57; wrongSM -6.74.
Contrasts (hardcb | ds_all): cog - anchor -0.49 [-1.05,-0.04] | -0.85 [-1.20,-0.55]; wrongWM - anchor -0.85 [-1.64,-0.01] | -1.03; cog - wrongWM +0.36 [-0.49,+1.04] | +0.18 [-0.91,+1.16]; cog - wrongSM +0.53 [-0.29,+1.26] | +0.35; cog - shared +3.7 | +3.2 (shared worst).
Minority-class recall (ages 20-60, vs base, pp): hardcb anchor -0.9, cog -0.9, wrongWM -0.9, shared -2.5; ds_all anchor -3.7, cog -4.4, wrongWM -4.6 (people 80 % mostly spanning zero for hardcb).
Cost: EM stage 0.60 (base) vs 0.63-0.68 ms/window (coupled; hardcb anchor arm 0.82 incl. ten 0.05 s anchor extractions), i.e. +0.03-0.08 ms/window after anchors (budget 0.5 met).
Reading against the rule: predictions refuted (cog vs base +0.25 predicted, -5.9 measured; cog vs anchor +0.15 predicted, -0.5/-0.9; upper bounds below zero on both systems and both orders): the nine-person source anchors replace the session-mean prior with a wrong centre for the new person (P10 -13), which dominates everything else; the coupling itself does not rescue it (cog and wrongWM both below anchors-only). The registered edge contrast (cog above wrongWM) is +0.36/+0.18 inside noise, same sign in both orders and systems but not by >= +0.2 with a positive lower bound: undecided as an edge effect, and moot as a route because every coupled arm is worse than base. Scope of the refutation: anchors from the nine other dev people at weight 20 (= m0) in prefix-standardized GE space; weaker anchor weight, anchors shrunk toward zero, or coupling without anchors (A = 0, edges only) were not run and are not closed. Frozen parts as in lane S2 (gate, FT banks, shadow, unrouted held-new people). Source-only displacement check: not supportive (see above). Nothing was uploaded.
