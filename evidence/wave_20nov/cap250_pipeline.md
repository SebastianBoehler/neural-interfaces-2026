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
respective monthly balances. The later check at about 10:43 CET found 36 running and 148 pending on Galvani. The
coordinator directed the graph to Leonardo first, with Galvani as overflow. Leonardo
`lrd_all_serial` runs on login08/13. Both kinds of workers therefore use `boost_usr_prod`,
with a GPU reservation even for CPU-only fit steps on Leonardo. The table measures
GPU-use estimates and excludes those CPU-phase reservations. Actual start times remain
unreserved. At the user's 36-lane planning assumption, 80–185 GPU-h is 2.2–5.2 hours
at full utilization. The 21 h 50 min to Saturday 08:00 CET
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
decision gate in recording order and pinned full shuffle 71000. Record changed decisions, per-visit counts and score
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


## Slurm execution record

| Check | Job | Outcome |
| --- | --- | --- |
| Hard reduced seed 0, P01/P02/P03 | Galvani 3141176 | Passed in 18 s, real CUDA fit and bounded CPU trace prefix/reset checks |
| CountFilter, all ten | Galvani 3141177 | Passed in 16 s; max archive deviations W 3.8e−8, scale 3.5e−9, thresholds 1.2e−9 on first implementation |
| Tangent/PC basis, three source people | Galvani 3141178 | Passed in 50 s |
| Historical v3 replay guard | Galvani 3141269 | Completed, printed OK |
| Archived parts assembled with the new builder | Galvani 3141310 | All 1,047 arrays exactly equal and all 38 other payloads byte-identical |
| Hard full seed 0, all ten | Galvani 3141312 | Passed in 40 s |
| Source-only external initializer check | Galvani 3141372 | Verified all 30 m7r files and the spectral model. The m7r metadata names the seven external corpora and no PROTEUS adaptation. The spectral model has 224 people and 673 visits. |
| Galvani array-task dependency | 3141373 → 3141374 | Both array tasks passed and the dependent job completed |
| Leonardo hold/release and individual array-task dependency | 64930908 → 64930920 | The SIF uses CUDA 13. The booster host driver exposed CUDA 12.2, so both tasks failed fast and the dependent job canceled |
| Leonardo native CUDA and dependency retry | 64931727 → 64931728 | Both A100 hard fits passed in 2m46s; task-dependent CountFilter passed in 21s |
| B2 simulator path on real tensors and archive weights | Galvani 3141445 | Passed 31-window stream and 16-window prefix check, with no scoring |
| Native archived Solver decision comparison | 64932077 → 64932082 → 64932085 | Reassembly passed; replay exposed a release missing the official benchmark utilities |
| Official utilities staged, replay retry | 64932707 → 64932711 | Native Python 3.11 environment lacked Benchopt. Existing Python 3.12 environment contains Benchopt 1.10.0, Braindecode 1.8.1 and Torch 2.14.1+cu126. |

The first archive build on Galvani lacked the staged ZIP. The second isolated
one-ULP rounding in 34 mixer weights. The third used the preserved exact constants
from the historical arrays. Its following monolithic stream timed out at 20 minutes
(3141311), so the Leonardo retry splits work by participant. The full official decision
comparison is independent of array/payload parity. The SIF reassembly also passed on
Leonardo (64931863). Its replay successors were canceled after the driver failure was
identified and replaced by the native archive graph. [Rebuild code](../../cleanslate/cap250/archive_check.py)

## Known fresh-fit difference

Leonardo SessionBN all-ten reproduction found that EA FBC and FBLight agree with the
archive within 5e−4 with zero flips over the probe. Sinc changed 1 of 4,530 single-member
decisions, consistent with GPU nondeterminism. Fresh raw FBC fits reproduced the cached
seed-0 state exactly. The archived raw traces differed in 8 of 26 parameters in the
spatial block and changed 17–24 of 755 probe decisions per seed. The archived training
state may be the 13 November twin retrain. Galvani 3141490 is checking that attribution.
The nine-person production-recipe smoke passed for every SessionBN family and seed.
Measured SessionBN time is 0.31–0.35 GPU-hours per fold.
[SessionBN build card](cap250_build_sbn.md)

Leonardo 64933623 evaluated 2,655 windows per order. Recording order had zero changes
(99.618216% balanced accuracy for both packages). Shuffle 71000 changed one decision
in P02 visit 1 (99.362599% archive, 99.322917% fresh). The job produced its full report
and exited at the strict equality check. The ten replay workers completed successfully.

Before swapping families, register the attribution check. Run both orders with the
original ZIP, reconstructed archive, fresh package, each archived family restored
individually in the fresh package, and the fresh raw SessionBN family in the archive.
Expect zero reconstruction changes, reproduction of the original changed row, and
one restored decision from raw SessionBN alone. Expect the archive with only fresh
raw SessionBN to match fresh decisions on both orders. Any failure of these checks,
or rescue of the original decision by another family, refutes raw-only attribution.
A successful attribution lets the fold experiment use fresh stock as its paired
reference. Both arms share stock fits within each fold. The archived-parts gate stays
strict, and the original fresh-versus-archive result remains a parity miss.
