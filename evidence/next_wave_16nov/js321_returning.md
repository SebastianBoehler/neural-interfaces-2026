# §321 returning-person validation: new genuine checkpoint scope

Historical checkpoint identity remains refuted. §281 job3110506 used candidate source `4370f1fc772d1a3377d3758d599b0b364b3231eb`; seed0 frozen-QF reconstruction differed by .005591801/one label for P04 and .005279755/zero labels for P10. Those checkpoints were correctly not persisted. The complete original remote training revision was never established. This failure does not make a newly trained, correctly excluded fold invalid; it prevents calling it the historical fold.

The documented recipe and required inputs exist. `experiments/session_bn.py:make` resolves `sincshallow_ea_ptm7r_rot05_sbn` through `run_cv.resolve`; `deep.AlignedNet._fit` supplies AdamW/OneCycle, 40epochs, lr/wd.001, batch64, rotation.05. Session groups32 are noncausal during training, latent test prior20; causal EA input prior remains5. Original m7r trunks0–5 exist, with their exact current hashes pinned in each training snapshot. No external data is added.

`protocol.folds(...,'seen_session')` holds P04/P10 sessions2+3 out, retains that person's session1 and all other people. P04 train2278/held377; P10 train2270/held385. The driver poisons held labels and weights before fitting. An observer checks every actual training alignment call, its exact row union, and the training-only covariance mean. Saved metadata records seed, row/source/trunk hashes and explicitly denies historical identity.

|Preparation|Actual outcome or job|Immutable workspace|
|---|---|---|
|First new Sinc pilot|3117914 failed75s before persistence: historical class lacks modern `_order` metadata|`ni26_js321_20261117_returning1`|
|Observed-input Sinc pilot|3117928 PASS50s, fit37.186s; own-weight export max2.633e−7, reset/deepcopy pass|`ni26_js321_20261117_returning2`|
|Remaining11 Sinc fits|3117930, concurrency1; canonical P04seed0 reused by proof/hash|`ni26_js321_20261117_returning_sinc_full1`|
|CC30 pilot|3117941 PASS66s, fit33.929s; CPU/CUDA export2.10e−6/4.05e−6, reset/deepcopy pass|`ni26_js321_20261117_returning_cc30_pilot1`|
|Remaining11 CC30 fits|3117998, concurrency1|`ni26_js321_20261117_returning_cc30_full1`|
|Existing source bridge|3118014 CPU strict failure: FB P04 max1.15808e−5/0labels; train-onlyG/classic rows pass|`ni26_js321_20261117_returning_existing1`|
|Device-discriminating source guard|3118024 A100 same-checkpoint/QF ≤1e−5 and self-repeat ≤1e−8; 3118016 cancelled pending to add absolute input locks|`ni26_js321_20261117_returning_existing_gpu2`|

Every new Sinc weight must be used identically in the paired control and JS arms. JS also needs genuine held-session CC30 weights; neither all10 deployment weights nor leave-entire-person-out weights match these returning folds. Existing FB/FBL/raw checkpoints cover both folds and all six seeds. Their three-field schema lacks explicit training metadata, so the source bridge independently checks training-only G/scale, strict class/state load and full frozen probabilities against archived QF outputs. Classic/gate feature `idx_tr` must exactly equal the declared source mask. No label accuracy is read by these checks.

The frozen all10 ZIP cannot directly validate held sessions: its TS, gates, own-stage and other coefficients include those sessions. Use the existing whole returning harness (`studies/13nov/retv6.py`, `studies/14nov/newmember/s281_seen_route.py`, `s281_seen_relabel.py`) with genuine fold-specific classic/gate/own-stage sources. This is a **new paired whole-returning dev comparison**, not numerical identity to the uploaded package or the historical returning metric.

Required consumption sequence before any score:
1. Collect all24 new neural proofs; independently validate all six existing family seeds, all held rows and eight fixed orders. Do not infer full coverage from seed0/within0.
2. Regenerate Sinc frozen streams under a distinct source prefix. Keep explicit source adapters for other families; no archive overwrite or missing-file fallback.
3. Recompute the shared original inner mixture/shadow, gate inference and pseudo-label prefixes with the new Sinc base. Refit **all four** FT families using these prefixes and their genuine source-fold weights. Old FT deltas and old gate/label traces cannot be substituted.
4. Preserve original formulas, source calibration and inner15 weighting in both arms. Apply JS-all only to outer FB removal/CC30 insertion and whole adapted FBL doubling, yielding outer17/core1 share1/17. Recompute each arm's route/own-stage, balancing, filter and causal-prior state independently.
5. Use the existing owning whole scorer, fixed B2_pool six-seed deployment and eight orders. Require same-path-zero/control/order/replay/all-six-CP-weight guards, reset/prefix and actual activity before labels. Register exact contrasts and source maps before scoring. Report P04/P10 separately and returning .2/.8 weighting, with people/order uncertainty. No in-sample replay or inherited new-person gain claim.

Provisional returning JS−new-control prediction is0pp[−.5,+.7], not the selected new-person+.272pp. Full returning accuracy has not been calculated. Source training and numerical guards are preparation only. Frozen files remain unchanged; no upload or promotion is authorized.
