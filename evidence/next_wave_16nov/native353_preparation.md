# Native raw neural source preparation — source tensor parity qualified

Own C++ preparation calculates source-only covariance G and global scale, groups
source rows in first-seen session order and emits the original raw scaled float32
training tensor. It reads no labels, performs no neural forward or initialization
and does not generate batches. This covers the existing unaligned raw-FBC recipe.

Shared `SourceData`/`select_source_rows` enforces CPU contiguous detached
float32[N,41,480], explicit complete-session masks, nonnegative session/person
IDs, one person per stream and explicit held-person exclusions. Classical fitting
delegates those checks while retaining its exact seven-field archive, label
vector and immutable recipe guards. Its accumulation/scaler/head code is unchanged.
The earlier classical fit remains qualified only for its frozen old binary;
the new extraction is syntax-checked, not yet dynamically integrated there.

## Prospective source-only comparison

ONE CPU2/16GiB≤5min, maximum600 allocated core-seconds. Reuse the qualified CAR
base and NumPy temporal-mean runtime; compile only source.cpp, preparation.cpp
and the source-test TU. Full original module/input/schema/ELF/hash/output preflight
before allocation, actual load-path/end-of-job checks before/after numeric work.

Reuse the original metadata-only oracle with1898 source rows and all40 historical
epoch hashes. The original source7 inventory includes P04–P10; explicit source6
then removes P09. P01 is already outside that inventory. Source6 is therefore
P04/P05/P06/P07/P08/P10, not a generic three-person exclusion. First-seen session
order is P06|1,P07|1,P10|1,P10|2,P10|3,P05|1,P08|1,P04|1,P04|2,P04|3.

Independent original features.covariances computes raw source G/scale. Its scale
must reproduce the original CONTROL trace exactly before any C++ comparison.
Only source X enters math; all excluded EEG is NaN in the native input. No y
array is decoded. Require source order/session IDs exact; G≤1e−8 absolute,
scale≤1e−10, prepared X≤1e−6 absolute/relativeL2≤1e−7. Predict tighter float32
agreement given the already qualified mean primitive; any excess blocks adoption,
with no tolerance change or automatic retry. Partial-session and explicitly-held
selection must fail before preprocessing. No training/accuracy/performance claim.

Budget: actual3539287coresec; reserves600 FTrepair+600conservative sampler+
1800PCA+600here =12887<14400. Source prep/CLI integration and other-family aligned
recipes remain separate; no offline source-update oracle is released by this test.

## Actual source-only result

3121547 COMPLETED0:0 in134s×2CPUs=268allocatedcoresec. All1898 prepared source
windows match the original float32 tensor BYTE_EXACT (relativeL2=0); Gmaxerror
2.9558577807620168e−12. Source scale reproduces CONTROL beforecomparison; native
scale/order/session and originalfixedbounds pass. Both invalid split/person
controls reject. No y, NNinitialization/forward/update or accuracy was accessed.
The first read-only preflight used an unavailable private NumPy header reader;
the public canonical NPY1.0 reader repaired that preflight before any allocation.
Numerical/source definitions were unchanged; both preflight outcomes are retained.
Source/build/input/worker-library checks pass before/end. Classical source guard
extraction is still syntax-only integration in its new build; no new core fit or
whole-runtime/worker/performance claim follows from this source-preparation test.
