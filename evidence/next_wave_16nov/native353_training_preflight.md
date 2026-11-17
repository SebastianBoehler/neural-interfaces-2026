# §353 REAL-core source test: complete preflight checklist

**Pre-release checklist snapshot; later source fit3121076 PASS (see training_completeness.md).**
Both earlier failures are preserved:3120845 link provenance
(142coresec) and3120955 missing official Python import path(98coresec). Neither
computed original source features or ran a native fit. Total240coresec=.066667CPUh.
Aggregate§353=7,993coresec before NN3120986 actuals; NN1,200 and root-init180 reserved.
At checklist review, numerical source fitting was still unqualified; the later
explicitly released fit passed. No numerical math/test/bound was changed.

Read-only checker: `native/tests/core_training_preflight.py` executes the ACTUAL
fixture script's sys.path initialization/import AST; it does not call main.
It hashes the input files, reads NPZ array headers and source metadata/reference
arrays/recipe tensors, and verifies all owners, keys, shapes and thread contexts.
It never reads X values or dataset y, computes features, fits, or scores accuracy.
Compact proof: native353_training_preflight.json. Complete transitive module
file/hash manifest is retained remotely beside that proof, not a new code tree.

## Imports and owners — checked against actual executed paths

| Import stage | Complete direct imports and owning roles |
|---|---|
| Fixture standard/runtime | argparse,hashlib,json,pathlib.Path,sys,time,numpy,torch,threadpoolctl.threadpool_limits. Python executable/version and package versions are recorded. |
| Fixture serializer | Insert snapshot native/tools; import convert_package.save_tensors. That module imports argparse,hashlib,importlib.util,pathlib,sys,zipfile,numpy,scipy.signal.butter,torch. Only save_tensors/tensor serialization is called; export_package/native filter design/source_fixture are not called. |
| After actual sys.path assignment | Insert frozen returning2/source/experiments AND official2026-competition/tracks/bci_decoding; then import ts_core. Assert ts_core,features,protocol and benchmark_utils.metrics resolve to the exact pinned owners. |
| ts_core | argparse,json,os,re,sys,time,numpy; joblib.Parallel/delayed; pyriemann.utils.geodesic.geodesic_riemann and utils.mean.mean_riemann; sklearn.discriminant_analysis.LinearDiscriminantAnalysis,linear_model.LogisticRegression,pipeline.make_pipeline,preprocessing.StandardScaler,svm.LinearSVC; protocol; features.bandpass,causal_references,expm_sym,logm,slow_waveform,whiten_tangent. |
| features | pathlib.Path,numpy,scipy.signal.butter/sosfiltfilt. |
| protocol | sys,pathlib.Path,numpy,benchmark_utils.metrics.balanced_accuracy/group_scores. Import is required; no evaluation/metric function is called. |
| official metrics | numpy only. Symlink resolves to official2026-competition/benchmark_utils/metrics.py; official files are read, never edited. |
| Transitive dependencies | All loaded modules after actual imports are listed with resolved file paths/SHA or builtin/namespace marker. Package versions and interpreter binary hash are recorded. Lazy numerical execution itself remains untested. |

Every original input pin is in ACTUAL fixture `PINS` and compact proof:
canonical dataset2be4187f…e1327; fitted source_view8f52ab55…23675;
ts_core0dd250ac…d6e1c/features360f335d…3dabd/protocol390fd6be…14a2c;
official metricsfcc73940…9787d; qualified coefficient archive4bf45aee…ab31c;
fit.json eb3bcd55…6635e; source.sha2561196bab5…00297 and ALLthree listed producer
files returning_view_fit.py c1834050…0566, .sbatch bd7a2312…02da,
returning_assets.py8e706695…b292. Serializer source hash is recorded as an owner;
next immutable source manifest must pin it, actual fixture and checker too.

## Every read field, required schema and purpose

| Input | Exact required fields/types/shapes |
|---|---|
| fit.json | source_rows=2278; query_labels_used=false; accuracy_computed=false. Exact full-file SHA also pinned. |
| canonical NPZ | Xfloat32[2655,41,480],C-order; subjectU21[2655]; sessionU1[2655]; ch_namesU3[41]. Checker reads only X header, never X values. Actual run selects2,278source X rows; dataset y is never accessed. |
| fitted reference indices/labels | idx_tr int64[2278], exact ordered rowSHA ca4a869f…cea2b; y_tr int64[2278], three classes0/1/2. These are SOURCE labels from existing original fit only. |
| fitted reference core | ts_bands f64[17,2]; ts_log_glob f64[51,41,41]; ts_mu/ts_sd f64[44239]; ts_W f64[3,44239]; ts_b f64[3]. |
| fitted reference scalars | ts_k/ts_shrink/ts_slow_cutoff f64[]; ts_slow_step/ts_car/ts_nsub int64[]. |
| fitted reference PCA | tsem_m f64[44239]; tsem_V and route_V_hi f32[44239,128]; pc_tr f32[2278,256]; singular_values f64[2278]. Actual fixture loads every reference key; checker enforces the complete19-key set and schemas. No SVD/PCA refit. |
| qualified native.pt recipe | ts_sos f64[17,4,6]; ts_slow_sos f64[2,6]; ts_bands f64[17,2]; ts_car/ts_nsub/ts_slow_step int64[]; ts_shrink/ts_k/ts_slow_cutoff f64[]. All finite; seven non-SOS metadata fields exactly equal existing source-fit reference. hilbert=int64(0) supplied explicitly. |
| generated source.pt | native_format=int64(1); x f32[2655,41,480] with excluded EEG NaN; y int64[2655] with excluded labels−1; stream/person int64[2655]; fit_mask bool[2655]; held_person emptyint64[0]. |
| generated recipe.pt | native_format plus the nine unprefixed immutable recipe fields above and hilbert0. Test removes native_format before passing strict recipe API. |
| generated reference.pt | native_format; original raw source features f64[2278,44239], probabilities f64[2278,3], rows int64[2278], log_glob/mu/sd as above. Native training never consumes reference features; only the comparison does. |

Split: P04/session2+3 are377complete excluded rows, all other rows source, including
P04/session1. Thus held_person MUST be empty; this is session exclusion, not an
excluded-person claim. Native test separately requires partial-session and
selected-held-person masks to fail before preprocessing. Row order is preserved.

## Execution, output and unchanged numerical checks

- Oracle has absolute existing venv interpreter; OMP/OPENBLAS/MKL/NUMEXPR all1,
  CUDA_VISIBLE_DEVICES empty. Torch1 and both ACTUAL threadpool_limits contexts1
  are checked. Feature extraction and saved PC256 projection are one-thread.
  Original six-thread context was SVD ONLY, outside the saved PC projection.
- Native executable sets Torch2 explicitly, takes exactly source/recipe/reference/
  new-output arguments. No Python runtime call occurs in native fitting.
- Fixed preflight executable cbee44d0…9c5a5 comes from unchanged test object
  9300162e…1e9fb. Actual link+ldd PASS resolves qualified CAR base70d14978…1cb5;
  corrected DT_RPATH directly names that base.parent. Original RPATH failure kept.
  Next allocation must pin executable/base/full NFS dependency manifest and rerun
  exact executable ldd BEFORE oracle. Worker-local OS hashes are placed/checked
  on worker; no login OS identities are substituted. No old-base fallback.
- Slurm cwd must be the fresh immutable snapshot root; native/tests and tools
  retain hierarchy. Its fixture directory and native_core.pt MUST NOT exist.
  Parent must exist and be writable; checker requires≥4GiB free for~1.02GB input/
  reference archives plus output. Serializer uses new native tensor dictionaries.
- Actual oracle must reproduce saved float32PC256 BITWISE before writing fixture.
  Native guards remain rawF≤1e−4/relativeL2≤1e−6, prior≤1e−5, mu/sd≤1e−4,
  sourceposterior≤1e−3 and ZEROlabels; independent objective≤1e−10.
  No new Python head/scaler/PCA fit, held labels, accuracy or tolerance change.
- First failure stops and preserves artifacts. A passing read-only preflight is
  runnable-input evidence, NOT numerical parity, runtime/memory or complete-stack
  proof. No further allocation/retry exists until root reviews and releases it.
