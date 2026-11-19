# Clean-slate team B · cluster-then-name session model

Code `cleanslate/teamB/` (`submission.py` = Solver and nets, `train.py`, `dev.py` = same streaming protocol as the harness
with every arm kept, `collect.py`). All numbers are DEV on the ten public people; cap250 reference 73.74 held-new (full0).

## Design (from the audit facts, not from cap250's structure)
- The error is person-level: people whose task patterns differ from the population are at chance cross-person but are
  internally consistent (P04: 50 % held-new, 85 % once enrolled). A posterior remap cannot fix that; a session model
  that finds the person's own class groups and only *names* them with the cross-person classifier can.
- So every encoder is trained for two things at once: class logits (naming) and a unit projection under a
  **within-session supervised-contrastive loss** (same-class windows of one session together; nothing asks two people to
  agree). That objective does not need aligned labels across corpora, so other-task corpora can train it later.
- Session model: causal EA over the prefix, whole prefix re-embedded under the current alignment every 4th window,
  session feature normalisation, then a von Mises-Fisher mixture on the projections, initialised and named by the
  ensemble posterior, with a class-balance prior that tightens with prefix length. Arms per window: `enc` (ensemble
  only), `bal` (+ balance prior), `em` (+ mixture). Adaptation gain = `em` − `enc`, same weights, same pass.

## R1 · first whole system (registered Thu 19 Nov 2026 22:40 CET, before any scored run)
System: 9 encoders (ShallowConvNet 1–45 Hz ×3, slow Shallow 0.3–30 Hz ×3, FBCNet 9 bands ×3), 1,200 steps each, public
PROTEUS only, λ_supcon 0.5; session model κ 8, α 1, n_b 24. Held-new P01–P10, orders 71000 and 71001, 28 cells; returning
P04/P10 (session 1 in training, sessions 2–3 streamed). Variants scored in the same pass (label-free settings only):
κ 4/16/32, α 0.5, n_b 8/96; picking among them on the ten people is selection, reported as such.
- Mechanism: if clusterability generalises to unseen people, `em` lifts the low people (P01, P03, P04, P05) most.
- Predictions (held-new cell mean): `enc` 62 % [57, 66]; `bal` − `enc` +0.3 [−0.3, +1.0]; `em` − `enc` **+2.0 [−1.0, +6.0]**;
  `em` vs cap250 −9 pp [−15, −3]. Returning `em`: 80 % [72, 88]. Runtime < 60 ms per window on an A100.
- Refutation: the cluster-then-name mechanism is refuted in this form if `em` − `enc` has a people-80 % upper bound below
  +1.0 pp with the best registered variant; supported if its lower bound is above +1.0 pp at the base setting. Anything
  else is undecided. A whole-session (non-causal) rerun of the mixture and an own-label nearest-centroid score on the
  end-of-session projections are reported as ceilings (diagnostics only).
- Smoke: job 3139282 (30 steps, one net, P03/P04 new + P04 ret) passed the code path; no score read.

### R1 result (Thu 19 Nov 23:35 CET; job 3139336, 12 folds, orders 71000/71001; P10 held-new fold landed last, table below is 9 people / 22 cells and is replaced when R2 reports)
| | enc | bal | em (base) | best registered variant |
|---|---|---|---|---|
| held-new | 64.77 | 64.76 | 64.69 | κ 4: 65.08, +0.31 [−0.37, +1.12] vs enc |
| returning P04/P10 | 73.49 | 74.21 | 74.19 | n_b 96: 74.53, +1.04 [0.00, +2.09] |

Ceilings on end-of-session projections (first order): enc 64.60, whole-session mixture 64.13, **own-label nearest centroid
(leave one run out) 64.61** (P04 51, P03 55, P01 57). Runtime 45 ms per window (2080 Ti / A100 mix).
- Predictions: `enc` 62 → 64.8 (inside). `em` − `enc` +2.0 → −0.07 (inside the interval, at its low end). Returning 80 → 74.2
  (inside, low). Against cap250 73.74: about −9, as predicted.
- Rule: best variant upper bound +1.12 is above the +1.0 bar by 0.1, so formally **undecided**; I read it as failed in
  this form and say why: the projection has no own-label headroom on unseen people. With the session's own labels a
  nearest-centroid rule in it scores what the cross-person classifier scores (64.6 vs 64.6; P04 51 %). The contrastive
  loss was fitted to nine people to its floor (training loss 1.948, the minimum for 7 positives of 23) and learned those
  people, not clusterability. No mixture can extract what the space does not hold.
- What it does not show: that a clusterability embedding cannot exist. A tangent-space logistic regression with own
  labels reaches 72 % on the same people (audit §3), so person-specific structure is there in generic features.

## R2 · clusterability trained on external people (registered Thu 19 Nov 23:40 CET, before scoring)
Change: every step adds six groups from an external bank (`extbank.py`, 93,300 windows, 1,055 sessions, 15 public MI /
mental-task corpora, Scherer excluded so NI-V stays untouched) with the same within-session contrastive loss, class
labels used only inside their own session (λ_ext 1; no label of another corpus is mapped to a PROTEUS class). 2,000 steps.
Side runs, same registration: R2b λ_ext 3; R1s (no external data, 600 steps) to see whether fitting to the floor costs `enc`.
- Mechanism: clusterability is learnable only from many people; 9 → about 500 people should make the projection of an
  unseen person group by that person's own classes.
- Predictions (held-new): own-label nearest centroid 66.5 [63, 71] (R1 64.6); `enc` 65.5 [63, 68]; `em` − `enc` +0.5 [−0.5, +2.0].
- Rule: the learned-projection route stays open only if own-label centroid − `enc_end` is at least +3 pp with a
  people-80 % lower bound above 0. Otherwise it is set aside for this deadline and the session model moves to generic
  features (tangent space), where the own-label ceiling is known to be 72.
