# Fold-trained cap250

20 November 2026, CET. [Component and artifact contract](../../cleanslate/cap250/SPEC.md) ·
[Sealed assembly receipt](../../friday/CAP250_REBUILD.md) · [Source-fit audit](../audit_20nov/assumptions_model_codex.md#source-fits-and-exposure)

## Compute and deadline

The 39 archived entries contain code, the fitted NPZ and 37 traces. The fit inventory
has 34 ordinary nets (14 active), 24 SessionBN fits (30 traces), six Eye/Rest fits, one
teacher calibration, and four CPU fit families covering both tangent cores, PCA, run
and identity calibration, personal heads and priors. Hardcb adds two F16e hard fits and
three CBraMod-C fits per fold. Published stock excludes all 20 zero-weight models from
inference but retains their fitted arrays in the archive.

| Work | Stock | Hardcb |
| --- | ---: | ---: |
| Outer fits, one fold | 1.28–2.95 GPU-h | 1.60–3.92 GPU-h |
| Outer fits, 10 held-new + 2 returning | 15.4–35.4 GPU-h | 19.2–47.0 GPU-h |
| Nested B2 active fits per calibration roster | 0.90–2.05 GPU-h | Shared |
| 65 unique nested rosters, if Galvani owns all held-new and Leonardo both returning | 58.5–133.3 GPU-h | Shared |
| Nested recording-order B2 streams | 0.8–2.5 GPU-h estimated | Shared |
| Four full-shuffle orders of the 12 outer folds | 0.5–1.2 GPU-h estimated | 0.5–1.2 GPU-h estimated |
| **Production total with shared stock fits, both variants** | **~80–185 GPU-h** | Includes both columns' work and shared calibration |

The all-ten fresh acceptance run budgets 10 additional nested rosters and one stock fit,
about 10–24 GPU-h if run first. Geometry and Hilbert CPU jobs can overlap training.
Historical package-wrapper jobs combined other checks with the fits. Runtime references
include R19e 16 min on A100 (3043396), teacher export 44 min on RTX (3043397), SessionBN
stored fit times 21–67 s per seed (3059164/3059663), Eye/Rest 24–66 s per seed, CBraMod-B
5m20s for all-source stage B (64883820), and real archive replay 147–166 s stock /
180–208 s hardcb for 2,655 public windows. Hard is provisioned at 5–20 min per seed,
including CUDA portability and export checks. The original three-arm F14 triples took
30–41 s on H100. The first smoke should tighten that provisional estimate. [NXIF compute](../../docs/AUDIT_NXIF.md#compute) ·
[Eye/Rest receipts](../v6_build_13nov/train_v6_members.jsonl) · [F14 timings](../../friday/fbc323_train/result-18nov.md) ·
[Hardcb timing](final_stack.md#results-19-nov-jobs-below-all-arms-in-the-full-chain-paired-80--intervals-pp-over-exact-cap250)

At 10:10 CET on 20 November, both Slurm hosts answered. Galvani A100 nodes were mostly
mixed or allocated and the account's MaxSubmit was 2,500. Leonardo reported both
AIFAC allocations active and about 5,737 / 4,672 local accounting hours until their
respective monthly balances. The user requested both sites in parallel. Actual queue
starts are unreserved. At the user's 36-lane planning assumption, 80–185 GPU-h is
2.2–5.2 aggregate GPU-lane hours at full utilization. The 21 h 50 min to Saturday 08:00 CET
is **compute-feasible, delivery unconfirmed**. Source parity, builder integration,
nested geometry completion, input staging and queue limits sit on the critical path.
Implementation is authorized regardless of that verdict.

## Source boundaries

- Production route debias `route_b` was fitted against old B2_pool out-of-person probabilities.
  An old prediction for source person B can have been trained using the current outer held
  person A. We refit the active B2 members and the two cores/PCA inside each outer scope.
  The calibration streams use recording delivery and the pre-route, pre-v6 B2 tail.
  The historical B2 bank used saved DEV-stage posteriors and coordinates. Its numerical
  identity with the new live B2 reconstruction remains an open check. Shared nested jobs
  must have identical person/visit exclusions. [Original route build description](../../docs/AUDIT_SBN.md) ·
  [Prior returning limitation](../../studies/17nov/js_readiness321/returning_galleries.py)
- The m7r teacher trunk checkpoints use PhysioNet MI, Cho2017, Lee2019 MI, Dreyer2023,
  Schirrmeister2017, Shin2017A and Shin2017B. The source-trained 38→3 map and EA reference
  are part of each fold. Sinc m7r initializations have the same external corpus recipe.
  Spectral identity calibration uses six external corpora. CBraMod-C stage A uses external
  corpora including Scherer. Each reuse needs corpus metadata and a file digest. This
  verifies dataset scope; identities across independently collected corpora cannot be
  linked from these metadata. [Trunk provenance](../../docs/AUDIT_NXIF.md#external-data-and-licences-declare-in-the-method-description) ·
  [Spectral sources](../../experiments/bio_id.py) · [CBraMod source stage](../../laneC/train.py)
- P04 returning sources contain 2,278 rows and P10 returning sources 2,270. The target's
  public visit 1 is enrolled, visits 2–3 are excluded. A held-new fold excludes every visit
  of its target. The official metric weights person/visit/interface cells equally.

## Prespecified smoke and comparison

Compare the stock assembly from authenticated archived parts against ZIP
`9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed`. Expect zero
changed public decisions in recording order and a pinned full-session shuffle. Any change
refutes zero-change package parity. A separate all-ten **fresh fit** must meet the same
decision gate in recording order. Record changed decisions, per-visit counts and score
if it misses. Published retraining work documents GPU-dependent nondeterminism, so a
reproduction using archived parts alone cannot establish fresh parity.

For the holdout experiment, run both real Solver packages on all 12 folds under
`71000,71001,71002,71003`, persist canonical row IDs and label-only predictions first,
then apply the official balanced accuracy. An initial hardcb prediction versus fresh
stock is +0.4 pp held-new (80% forecast −0.4 to +1.2) and +0.0 pp returning
(−0.6 to +0.6). A paired people-and-orders interval entirely below zero refutes an
improvement. Report individual orders, P04/P10 and the unverified 57/43 cohort mix
separately. Performance evidence from this experiment concerns reused public
participants with newly fit rosters. [Preregistration format](CLEANSLATE.md)
