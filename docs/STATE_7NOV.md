# State on 7 Nov 2026 (evening): the starting point

This file supersedes docs/HANDOFF.md (30 Oct) as the entry point. Details are in docs/SEALED_UPLOADS.md (upload log),
docs/AUDIT_SBN.md (audit recipes) and docs/REVIEW_5NOV.md (5–7 Nov evidence and closed directions). The session
scratch is archived in `runs/archive_7nov/` (see its `INDEX.md`; mirrored on Galvani).

## Baseline: sealed submission 1040847, 0.8240266154 (4th)

- Package: `cand_rec_ts2_tsem_route3geb`. ZIP: `runs/upload/sealed_cand_rec_ts2_tsem_route3geb.zip`, SHA-256
  `74433d9b…`, 57,280,711 B. Code as uploaded: commit `c4d600d`, `submission.py` `786bcab0…` (967 lines).
- **Lean code (this cleanup):** `submissions/ensemble/submission.py` `dacee85a…` (781 lines). It keeps only the paths
  of the three uploaded packages: cand_nxif, cand_rec_ts2_tsem and route3geb.
  - On all 2,655 public windows it reproduces each of them byte for byte: labels, member mix, every stage output and
    the final posteriors.
  - The route3geb audit recipe, run with it, rebuilds the shipped `weights.npz`.
  - Evidence: docs/AUDIT_SBN.md, "Reference values: lean code".
  - The other staged packages (routing v1 / v2, route3, route3eb, tsem256 as an upload) are rebuilt with commit
    `c4d600d`.
- Members, mixed with fixed weights (each causal, reset per session):
  - two tangent-space cores, one v2 unit each: core1 (`car_ov17_fs2`, CAR, 17 overlapping bands, full + two
    half-window covariances, causal log-Euclidean recentring, LR C 10) and the Hilbert core (`car_fb7_hfs2`, complex
    analytic-signal covariances, LR C 3);
  - the 35 nets of cand_nxif (ShallowConvNet, FBCNet, FBLight, SincShallow; nets 10–14 at weight 0) and its EEGNeX +
    IFNet TorchScript teacher (weight 0.625);
  - 30 SessionBN TorchScript nets (FBCNet k 2 and k 5, raw FBCNet k 20, SincShallow k 20, FBLight k 5; 6 seeds each).
- Stages after the mix: rebalancing (α 1.15, k 5, last 30 windows), then SessionEM (frontal slow potential + ocular
  events; nmin 20). TSEM follows (exact PC128 of core1's scaled features, λ 0.15). Both carry the gated early boost
  (GEB: ×(1 + 2 h g) and ×(1 + 4 h g); h falls from 1 at t 30 to 0 at t 60; the gate g opens for low-margin
  sessions, 0.20 / 0.35). Routing v3 comes last: from t 60, a gate on the unrouted (shadow) tail matches the session
  against the 10 public people's galleries (τ 0.22, ramp 0.05). It puts the matched person's PS1 head at 3× into the
  core1 slot of a second, routed tail.

## Sealed history and transfer lessons

| Day | Submission | Package | Score | Δ | Runtime |
|---|---|---|---|---|---|
| 1 (5 Nov) | 1034698 | cand_nxif | 0.7794596196 | | 834.7 s |
| 2 (6 Nov) | 1038262 | cand_rec_ts2_tsem | 0.8196888507 | +4.02 | 2,497.2 s |
| 3 (7 Nov) | 1040847 | cand_rec_ts2_tsem_route3geb | **0.8240266154** | +0.43 | 2,687.2 s (44.8 of 60 min) |

- **Big single steps transfer about 1:1.** cand_nxif → TSEM was dev J +3.7 to +3.9 and gave sealed +4.02.
- **Small stacked gains transfer at about 0.6×.** Routing v3 + GEB had an honest dev estimate of +0.75 and gave sealed
  +0.43 (central forecast 0.827).
- Our paired sealed differences are exact: the owner API reports 10 decimals.
- Reading rule: discount selected or stacked gains below about +1 J by 0.6, and treat anything under +0.5 J as a tie.

## Board, uploads and runtime

- **Board** (7 Nov ~19:10 UTC): bryanyahir03 0.838, seohee 0.825, timothewt 0.825, **us 0.824 (4th)**,
  xianglong_bao 0.823.
- **Uploads:** 2 of 5 left, at most 1 per UTC day (the quota resets at 00:00 UTC). The deadline is 21 Nov, 23:59 CET.
  The best score counts, and a failed run uses no quota.
- **Upload policy:** the user decided on 7 Nov to upload again only for a clearly better dev estimate ("push to .83
  first"). Every upload needs explicit authorisation and a live check of quota and phase.
- **Runtime:** 44.8 of 60 min on the real worker. The margin is about 15 min, and the deep members are most of the
  cost. New stages must be cheap, or must pay for themselves with a runtime cut elsewhere (the aim of the lean study).

## The honest gap to rank 1

- **Rank 1:** 0.838 − 0.824 = **1.4 points**. That needs about +1.4 dev J if the step transfers 1:1, or about +2.3 dev J
  at 0.6×.
- **Rank 2–3** (0.825) is within 0.1. That is noise-level: a small real gain could move us up or not.
- **The pipeline:** no candidate has a dev estimate near +1.4. The last passing gates were all +0.2 to +0.5 J: GEB over
  EB +0.18, routing v3 +1.0 (mostly P04), and EB +0.2 to +0.25.
- **The bottleneck is unchanged:** per-session class directions (REVIEW_5NOV §5d). Even with true labels, only about
  half of the +6 J headroom is causally reachable.

## Closed (details in docs/REVIEW_5NOV.md)

- **Data-only knobs.** On cand_nxif: teacher pooling, maps, α / EM grids. On route3geb, the consolidation re-tune:
  α 1.3 and TSEM λ 0.2 both lose.
- **Session-adaptive models other than TSEM:** a learned session-conditioned net, the SPD-BN network (TSMNet-style),
  label shift, per-session member weights, a global stacker, combiner re-tunes, head self-training.
- **Tangent-core extensions:** per-session z-scoring (overlaps TSEM), a third core, Gram-Wiener filter, Sw target,
  person-equal core weights, TSEM add-ons (pseudo-label head, session-PC view, agreement seeds).
- **Deep-member variants:** CAR for the deep members, the 7 Nov SessionBN GPU round (teacher replacement, EEGNeX /
  IFNet SBN, SBN twins, CAR SBN), the M1 teacher corpora (wave 11).
- **Personalisation beyond PS1 routing:** a seen-person specialist, personal deep models in the routed tail, early
  routing, full MAP adaptation, Hilbert personal heads, a gaze-EM prior.
- **Timing tweaks:** position-dependent member weights, an early rebalancing prior.
- **Superseded:** routing v1 / v2 (by v3) and plain EB (by GEB).

## Open or running (7 Nov evening)

- **Lean / single-model study** (scratch `lean/`, `prereg_A.txt`, `prereg_B.txt`; Galvani `ni26-lean*`):
  - **Part A** is exploratory. It covers family ablations, greedy backward elimination with nested
    leave-one-person-out, and seed thinning: fewer SessionBN and teacher seeds within 0.2 J of the full system. The goal
    is the same J at a lower runtime, freeing budget.
  - **Part B** tests single tangent heads (one LR on concatenated cores, small MLPs) against today's two-core slot,
    with honest split-half selection. The gate: standalone new > 69.3 and seen > 66.9, and the chain within 0.3 J.
- **Single deep-model variants** (scratch `single_deep/prereg.txt`; Galvani `ni26-sd*`):
  - **Variants:** FBCNet SBN trained longer and augmented (D1b–d), EEGNeX / IFNet fine-tuned (D2), a wide FBCNet and a
    trainable FIR (D3), and FBCNet with late tangent fusion (D4).
  - **Gate:** standalone single-seed new ≥ 68, or the chain within 0.3 J of the full ensemble at a much lower runtime.
- Neither study had results when this file was written. Neither is archived: other agents were still writing to them.

## Code, evidence and audit

- **Audit recipes:** docs/AUDIT_SBN.md has one for every uploaded package (cand_nxif: docs/AUDIT_NXIF.md). The
  route3geb recipe runs with the lean code. On Galvani it reproduces the original export's harness bitwise, and 32 of
  the 33 package files byte for byte; only `submission.py` differs. The recipes for the other staged packages use
  `c4d600d`.
- **Compatibility checks:** `scripts/check_compat.py --post` (new) compares labels, the member mix and every stage's
  per-window output, including the final posteriors, between a package's own code and the current code.
- **Never commit** datasets, environments, credentials or checkpoints. Local Python runs only through
  `scripts/run_capped.sh`, one job at a time. Use Galvani for anything heavy.
