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
