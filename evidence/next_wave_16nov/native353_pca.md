# §353 native PCA/TSEM source basis — implemented, unqualified

No PCA allocation or numerical PCA execution. Native module/header, separate
CMake fragment and common-source comparison are prepared; only syntax/byte
compilation/diff checks pass. Root owns master CMake and source-boundary extraction.
No existing qualified/running binary, frozen package or official submodule changed.

API `fit_tsem_basis(standardized_source)` consumes detached contiguous CPU f64[N,D],
256≤N≤D, and returns f64 mean[D], f32 basis[D,256], ALL f64 singular_values[N] and
queried workspace element count. Artifact split is basis[:,0:128]→tsem_V and
basis[:,128:256]→route_V_hi, contiguous f32; mean→tsem_m. No standardization is
silently fitted here. Caller supplies the same standardized source features as
original export_tsem.pc_basis. Native core/PCA composition remains a later test.

## Actual reference and native dependency inspected

Original source-only PCA owner is frozen returning2/source/scripts/export_tsem.py,
SHA6b45b1de…d181c. Actual function: m=Z.mean(0); scipy.linalg.svd(Z−m,
full_matrices=False,lapack_driver=gesdd); sklearn.svd_flip(u_based_decision=False);
returnm,Vt[:256].T,S. Existing returning source producer casts basis halves to
float32 and saves PC256. No NumPy RNG, Gram method or randomized SVD is involved.
Its feature/projection contexts are1thread; full SVD context explicitly6threads.

Installed SciPy1.18.1 _decomp_svd.py calls _batched_linalg._svd; actual native
_batched_linalg.so imports scipy_dgesdd_. Installed cython_lapack.pxd line438
supplies the exact LP64 pointer signature (int dimensions/workspace). Qualified
native libscipy_openblas-5f890258.so exports scipy_dgesdd_, ilaenv_, LAPACKE
variants and scipy_openblas_get_num_threads/set_num_threads. Reuse the exact
already licensed/provider-pinned native OpenBLAS/LAPACK dependency from linear.cmake:
SHA2a5b4c1a…2981. This is a C/Fortran native call, not a Python/SciPy runtime call.
No new provider, download, terms or service. Runtime notices already accompany
native/vendor/scipy_lbfgsb; no new vendor source was copied.

Native implementation uses direct scipy_dgesdd_ with job='S', same wide full
thin decomposition, optimal workspace query, fixed finite/LP64 guards. One
blocked copy centers into Fortran layout; source is never overwritten. It
applies first-argmax-absolute Vt sign convention to the256 returned vectors and
casts f32, releases U/fullVt/work/centered storage before return. Flipping discarded
U/otherVt rows cannot affect returned mean/basis/S. All singular values, including
small modes, are returned; no threshold/rank shortcut or alternate-driver fallback.
Driver/query failure raises an error. This is not yet numerically qualified.

## Memory and timing boundaries

For existing2278×44239 source: input806,211,536bytes; centered Fortran806,211,536;
thinVt806,211,536; U41,514,272; mean353,912; returnedbasis45,300,736; integer work
72,896 plus EXACT query-determined double workspace. Native input+major buffers
~2.34GiB before runtime/workspace; no whole D×D covariance/eigen allocation.
Reference pc_basis temporarily retains fullVt through its returned V view;
comparison wrapper casts f32 and releases that view to match returned storage.
Both methods include mean/centering/copy/workspace/SVD/sign/cast/free in API time.
Standardizing the common input is separate measured setup. Neither gets a reused
SVD preparation/cache; first and warm calls both count full fitting work.

## Prepared prospective matched test — NOT released

Reuse3121076 original-source feature oracle reference.pt, exactSHAaba119c2…bcff3c,
and existing P04_s23 source_view SHA8f52ab55…23675. pca_fixture.py computes the SAME
original standardized commonZ, reads no raw EEG/y, fits no head/PCA, and saves
source_Z plus original mean/f32 basis/S/sourcePC. Its file/source row hashes pin
all2,278rows/all44,239dimensions. No label, calibration or model selection.

pca_reference.py executes original pc_basis via exact pinned AST function. Order
is reference1→native1/native2 in fresh child→reference2, same source matrix/node,
six-thread BLAS/Torch limits/affinity. Native test asserts actual native-provider
thread count6. Both methods have cold/warm API times; startup/common-input load
excluded from those timers and reported separately by the eventual job wrapper.
Fresh per-method current loaded RSS and high-water before/after fit are recorded
(the high-water can include an earlier parser peak); concurrent
parent/child total belongs to Slurm batch peak, not either method's own memory.
No general all-metric/whole-stack performance claim follows from this component.

Fixed proposed guards for root review: mean maxabs≤1e−12; f32 basis maxabs≤1e−6
against BOTH fresh original function and existing fitted source artifact; ALL
singular values maxabs/max(S)≤1e−10. Float32 source PC projection maxabs≤1e−3 /
relativeL2≤1e−6; basis orthogonality maxabs≤1e−6; full-SVD energy residual≤1e−10;
retained256 projected-versus-singular energy residual≤1e−6. Energy residual is a
norm/variance identity, not a claimed full-matrix reconstruction residual.
Expect tighter≤1e−7 basis/≤1e−4 projection given same native driver/input; excess
registered bound refutes module parity. Reject source mutation, shape, finite,
thread, owner/path/hash/dependency and output-existence failures. No fallback,
relaxed bound, automatic retry or ignored small singular modes.

Cost proposal: ONE CPU6/16GiB≤5min max1,800allocatedcoresec=.5CPUh, no GPU, only
AFTER complete final-snapshot import/schema/ELF/provenance preflight and root
registration/release. Existing original six-thread PCA24.781s predicts four
fits~99s; add2-TU compile/link~40s, immutable input setup/hash/validation~40–60s:
expected180–210s, hard300s. If common-source compile/setup cannot fit, stop before
numerics; no extension. Current root tally9,011coresec before later jobs;
+max1,800=10,811<14,400 (3,589 remaining before later reservations).
These are estimates, not measured PCA runtime/memory or proof of a speed gain.

Files: pca.h/pca.cpp (no source fit or labels); cmake/pca.cmake (root includes
after linear.cmake); tests/pca_source.cpp,pca_fixture.py,pca_reference.py. New
source files remain<300lines. Link/runtime/numerical/complete training integration
are unvalidated. Need a staged exact-base executable and complete input/dependency
preflight before any allocation; earlier source-training failures remain preserved.

## Explicit matched-test release, BEFORE allocation

Root reviewed the code/card and released ONE CPU6/16GiB≤5min allocation after
complete final-snapshot module/schema/field/hash/ELF/thread/output preflight.
Fixed bounds and R-N-N-R/copy/setup/memory boundaries above remain unchanged.
No new head/EEG/labels, no automatic extension/retry. Current actual9,287coresec
includes later FT diagnostic276; reserves FTrepair600+sampler600+PCA1800+
root sourceprep600=12,887<14,400. Immutable ni26_pca_source_v1_20261117.
Narrow build-only preflight compiles2TUs and links exact existing CAR base/provider,
without SVD/feature execution; its actual static build cost is reported separately.
Old3121076 qualified scope stays on its old binary, not the source-boundary refactor.
This test does not qualify raw-source/native-core/PCA composition.
