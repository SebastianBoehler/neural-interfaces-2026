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
