# §353 training completeness audit — prepared, not source-fit validated

Audit17Nov2026. Native runtime functionality and native training completeness
are separate claims. No new allocation, feature computation, fit, accuracy or
held-label selection was performed for this audit. The new test is syntax-only.

| Stage | Implemented in C++ | Python or imported preparation still required |
|---|---|---|
| Dataset ingestion | Versioned tensor archive reader and strict raw source schema | Official EEG/channel/unit/window extraction and NPZ→tensor migration. The trainer expects canonical float32[N,41,480]. |
| Classical source features | Both actual CAR presets, SOS application, real/Hilbert covariance, source whole-session references/slow centering | Immutable SOS coefficients are migrated from the existing package. Native Butterworth coefficient design is absent. |
| Classical priors/head | Window-weighted source log prior, native scaler and exact native L-BFGS-B; real C10/Hilbert C3, default actual-core gtol1e-4 | No Python callback. End-to-end native **source fitting remains unvalidated**; existing96-window inference/math qualification does not validate source aggregation or fitting. |
| Neural source preparation | Native inference input paths exist | Current training API consumes source_X already scaled/reordered by original Python AlignedNet._fit, plus source-session IDs. Original source G and scale and training-row order are fixture preparation. Root owns its replacement work. |
| Neural initialization | Native RawFBC forward/backward/update consumes named tensors | Current training entry imports `initial_*` from Python build/PyTorch RNG. Root owns the new raw-FBC initializer; its unvalidated work is not counted complete here. |
| Neural sampling | Native training consumes and checks exact1240×64 padded indices, sizes and same-session32-row chunks | Python SessionAlignedNet._batches uses NumPy default_rng permutation, choice for padding, then chunk permutation across40epochs. Exact RNG/sampling remains fixture supplied. |
| Neural optimization | Native AdamW, OneCycle LR/beta schedule, gradients and parameter updates | Initializer/first-update fixture is an oracle, not proof of40-epoch fit equivalence or complete preparation. |
| PCA/TSEM source basis | Native runtime projects with an imported basis | export_tsem.pc_basis computes centered exact gesdd full SVD, svd_flip(u_based_decision=False), then saves float32 basis. Returning basis256 splits128+128; active TSEM consumes128. This PCA uses **no NumPy RNG**. No native PCA trainer exists. |
| Galleries/personal heads | Native routing consumes gallery/prototype/head tensors; online contrast math exists | returning_galleries builds per-session class contrasts, person/session averages, old-OOF q70 bias, basis(pop,bias), and Bio.fold_params. returning_context fits ps1_heads(C.1), session-standardized class priors and ownspace.Personal. No native source-gallery/personal-head orchestration exists. |
| Complete stack orchestration | Classical CLI trains the two actual cores serially | Complete member families, calibration/task maps, learned route/Bio/own priors, PCA/galleries and source preparation are not produced by one native training program. Existing fixed-production-prior sensitivities do not establish all-components source exclusion. |

Owning sources reviewed: experiments/ts_core.py, ts_core2.py, features.py,
deep.py, session_bn.py; scripts/export_ts.py, export_ts2.py, export_tsem.py;
returning_view_fit.py, returning_context.py, returning_galleries.py; native
core_training*, training*, convert_package.py and neural_training_fixture.py.
No neural source/header was changed by this audit.

## Smallest concrete next training parity test

Prepared `native/tests/core_training_fixture.py` and `core_training_source.cpp`,
CMake target `ni26_core_training_source`. This tests the **real core only** on the
existing P04_s23 source fold:2,278 source rows,377 excluded complete-session rows.
P04/session1 remains source; P04/session2+3 are excluded sessions, not an excluded
person. Therefore held_person is empty and fit_mask enforces this actual split.
The test separately rejects partial-session masks and selected-held-person IDs.
Excluded EEG is NaN and labels−1; only original source y_tr is opened.

Existing original fitted reference is
`ni26_js321_20261117_returning_view_fit1/source_view_P04_s23.npz`,
SHA256 `8f52ab55d4dea7eca9b43b610fad2c645c1f3029372570d05ab71b2187123675`;
its exact row SHA is `ca4a869ff981142b692b2a4e2ecb44f075ea2494e398280f103c9a9a36bcea2b`.
Data/ts_core hashes match its original fit.json. No new Python head/scaler/PCA
fit is needed: the test recomputes original source features, requires BITWISE
saved float32 PC256 reproduction, and uses its existing C10 W/b/mean/scale/prior.
The Python oracle exists only for independent comparison. The native active fit
consumes raw source EEG plus immutable SOS recipe, never reference features.

Prospective fixed guards (no measurements yet): all rows/order/exclusion guards;
full44,239-dimensional raw feature maxabs≤1e−4/relativeL2≤1e−6; log-prior maxabs≤1e−5;
source scaler mean/scale maxabs≤1e−4; native fitted source probability maxabs≤1e−3
and ZERO predicted-label changes versus original features/scaler/head. Independent
final objective must match solver-reported objective within1e−10. Record nit/nfev,
final objective/max-gradient, wall/coresec and peak RSS. Any failure blocks adoption;
no stop or tolerance change is planned. No causal-query/full-runtime claim follows
from source-only fit parity; Hilbert and label-free causal query checks follow
only after this first real-core source test passes and budget is rechecked.

Cost proposal, **not released**: one CPU2/16GiB/10min job, maximum1,200 allocated
coresec=.333333CPUh/0GPU. Existing original Python source features took132.215s
at1thread, original C10 head5.990s at6threads. Expected combined original feature
recompute+native source preprocessing/fit≤360s; no speed win is presumed.
The .81GB original feature matrix and ~1.56GB native covariance cache are bounded
under16GiB; existing whole-Python fit/PCA peaked9,011,536KiB but this test has no SVD.
No repeated/fold grid or new Python fit. First failed guard stops the test.

NN-owner audit at15:38:53UTC: all17§353jobs used7,720 allocated coresec=2.144444CPUh;
no running/pending353 jobs. Their root-authorized arithmetic check reserves180
coresec, leaving6,500 before refunds. This proposed1,200 would leave5,300;
root must reconcile actual completed cost and register/release before execution.

Validated now: Python byte-compilation and C++20 LibTorch syntax-only check;
`git diff --check`. Not validated: link of new test, numerical source-training
parity, execution cost, memory, Hilbert fit, full neural fit or full-stack training.

## Released source-only test, BEFORE execution

Root explicitly released the fixed test17Nov2026 after arithmetic3120762 completed.
Aggregate actual7,753coresec; reserve1,200 for neural strict8/31 and1,200 here:
10,153<14,400coresec. This ONE CPU2/16GiB/10min release has no extension/retry.
Immutable snapshot `ni26_core_training_source_v1_20261117`; exact source/header/
reference/data/coefficient/base/transitive-NFS-library hashes checked before submit
and on worker. Worker-local dependencies fingerprinted after placement and checked
at exit. Compile ONE source-test TU with original CMake ABI/flags/link argv and
qualified CAR base70d14978…1cb5; no base rebuild, no Python head/scaler/PCA fit.
The independent source feature oracle and native fit execute serially; time/RSS
recorded separately. Original one-thread PC projection context is preserved.
No new held inputs/labels, accuracy, full-system scoring or source-refit selection.
Script `native/tests/core_training_source.sbatch`; source/build proof is retained.
Original source/full-dimensional feature/PC/scaler/prior/objective/exclusion bounds
above stay fixed; first failure stops, preserves artifacts and blocks adoption.

### Preserved pre-numerical failure and explicit one retry

3120845 FAILED71s×CPU2=142allocatedcoresec. The new executable provenance guard
caught the copied original DT_RPATH resolving the OLD parity_v3 base despite the
absolute link input and LD_LIBRARY_PATH. No original features or native fit ran.
The old snapshot/logs/object remain immutable. Root explicitly authorized ONE
fresh same-cap retry, after link+ldd preflight with the unchanged compiled object.
Only old-build DT_RPATH is replaced by qualified CAR base.parent; no math change.
Actual preflight link+ldd PASS/no EEG execution, unchanged objectSHA
9300162e1558365876008d484ed4d4df82de1a6944161b152e547225cc91e9fb;
correct executableSHA cbee44d037901bd1b91414e5b928e5ceffde4262ebda3fbcdc58dc721c79c5a5.
Retry `ni26_core_training_source_v2_20261117` pins/runs that exact prebuilt binary,
its dependencies, same original source/feature oracle/SOS/reference/strict bounds.
No recompile, no native-base fallback, no second retry. Aggregate7,895coresec used
plus neural1,200/source1,200/root-initializer180 reserved=10,475<14,400coresec.

3120955 FAILED49s×CPU2=98coresec BEFORE source features/fit: the oracle exporter
omitted original producer's required official benchmark_utils import path. Source
repair restores that identical explicit path; import-only preflight PASS. Both
failures remain immutable;240coresec=.066667CPUh total. No further allocation.
Root requested the complete read-only dependency/schema/owner/thread/output audit
before reviewing any release: see native353_training_preflight.md and its JSON.
Native source fitting remains unqualified; numerical tests/bounds are unchanged.

### Explicit final source-test re-release after full dependency review

Root reviewed the83-line complete checklist plus ACTUAL fixture/C++ source and
explicitly authorizes ONE fresh CPU2/16GiB10min execution AFTER the SAME complete
read-only preflight on its final immutable snapshot. This goes past the earlier
no-more-retries rule because BOTH failures were preparation/dependency errors
before any features or fit; DT_RPATH and original official import path now have
individual demonstrated fixes. No numerical result/math/test/threshold changed.
Current actual8,371coresec includes7993+NN198+root-initializer180 (actual SlurmCPU2,
not requestedCPU1); this1,200 maximum leaves4,829 of14,400coresec. Exact corrected
prebuilt executable cbee44d0…9c5a5 reused, same C10/gtol1e−4/source/PC/scaler/
objective/exclusion guards. Snapshot ni26_core_training_source_v3_20261117.
No automatic retry or extension. NN strict FT failure remains separate;
this release is REAL classical source fit ONLY, never NN source updates.
