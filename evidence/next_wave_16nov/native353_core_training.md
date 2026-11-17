# §353 native classical source training — prepared, not runtime-qualified

API: `train_classical_core(source, recipe)` in `ni26/core_training.h`.
CLI: `ni26_core_train RAW_SOURCE.pt IMMUTABLE_RECIPES.pt OUTPUT.pt` trains both
actual classical presets, serially, and writes a versioned native tensor artifact.
No Python feature/scaler/head computation or optimizer callback is invoked.

The audit read `experiments/ts_core.py`, `ts_core2.py`, `features.py`, active
`scripts/export_ts.py`/`export_ts2.py`, `export_tsem.py` and the lean TangentCore.
The ACTUAL real core is CAR/ov17/fs2/t0.1/le/k2, C=10. The ACTUAL Hilbert core is
CAR/fb7/hfs2/t0.1/le/k2, C=3. Both reference exporters use sklearn default
LogisticRegression gtol=1e-4, not §343's explicit C=.3/gtol=1e-5. The new fixed
`fit_multinomial_core` path preserves 1e-4; the existing §343 API stays 1e-5.
Private native optimizer presets are typed; no generic public stop knobs.

## Raw source and recipe contract

Source archive keys are exactly native_format=1, x float32[N,41,480], y int64[N],
stream/person nonnegative int64[N], fit_mask bool[N], held_person int64[H].
The input is already in official model-facing channel/unit/window order.
Each selected session must be complete; stream IDs cannot cross people; selected
held people fail. Only selected EEG and labels enter preprocessing or fitting.
Excluded EEG may be NaN and excluded labels may be invalid sentinel values.
Training rows/person/stream IDs and optimizer diagnostics are written with outputs.

Recipe archive contains native_format and ts_/ts2_ prefixes of exactly:
sos, slow_sos, bands, car, nsub, hilbert, shrink, k, slow_cutoff, slow_step.
The trainer rejects learned fields such as W/mu/log_glob and unknown settings.
The two actual band lists, CAR, fs2, k2, shrink.1, 1Hz/step60 and float64 SOS shapes
are checked explicitly. Existing immutable order4 SOS coefficients are permitted
model parameters; native Butterworth filter DESIGN is not implemented here.

A single pass processes selected rows in original order. It accumulates the
window-weighted global mean of matrix logs and separate whole-session log/slow
sums; no person-average weighting or query statistics. Covariances are retained,
per-window logs discarded. Each source session then uses exp(mean(log C)) and its
inverse square root for tangent features, plus slow−whole-session slow mean.
Feature order is band-major full/two-half blocks, then328 slow entries:
44,239 real /35,629 Hilbert. Native scaler and fixed actual-core L-BFGS-B follow.
Covariance buffers are released before scaler/head allocation.
`ClassicalPreprocessor` exposes covariance/log/slow for shared inference reuse;
root's existing TangentCore has not been refactored by this worker.

## Explicit remaining scope

Neural EA G is not used by either CAR core and is not fabricated by this CLI.
Native neural source-G fitting, PCA/TSEM full SVD+svd_flip, personal/gallery heads,
all other learned stage priors, native dataset ingestion and complete training
orchestration remain separate. The cores consume no NumPy RNG. This is not a
claim that external NumPy-RNG-dependent training routines have been ported.

## Proposed minimal real-source registration, BEFORE any new fit

One raw fixture with two COMPLETE actual source sessions (e.g. P01/session1 and
P02/session1, ~380 windows, all3 classes), plus a held person's session whose EEG
is poisoned NaN and labels −1. Actual held EEG for query equivalence is a separate
label-free62-window archive. Fit masks/person/session IDs are explicit; source
row hashes and immutable SOS hashes pin both native and original Python oracles.
Reference only computes Feats/Core on selected source data. Test partial-session
and selected-held-person masks fail BEFORE preprocessing. Run both actual recipes.

First compare a 31-source-window preprocessing canary; use measured cost before
releasing full fits. Engineering prior: complete native+reference two-core fixture
≤180s at4CPU/16GiB; hard10min=.667CPUh, subject to root's remaining353 budget.
Raw feature maxabs≤1e-4/relativeL2≤1e-6, global-log prior maxabs≤1e-5; source scaler
and standardized-feature differences reported explicitly. Require same C/gtol/
zero initialization/ftol/maxls/maxiter, finite independent loss/gradients, and
source plus label-free causal query probability maxerror≤1e-3 with ZEROall label
changes. Record actual iterations/evaluations; different feature rounding may
change trajectories and must not be hidden as a stopping change. Any failed
source/exclusion/feature/posterior guard blocks adoption; no bounds loosened.
No held labels, accuracy, whole-system score or separate heavy allocation.

Current status: C++ syntax-only and diff checks pass; no core-training fit has
run. Root includes `cmake/core_training.cmake` after the base+linear targets when
its integrated build/budget is ready. All owned non-vendor source files <300LOC.
