# Native raw-FBC initialization — static parity qualified

The native training loop previously imported all initial neural weights from
Python. A source-only C++ initializer now constructs its own weights, biases,
SessionBN state and private CPU random generator. Fixed FIR coefficients remain
immutable model data. FIR coefficient design, sampling, source preparation and
the remaining family recipes are separate unfinished work.

The original constructor consumes default Conv2d weights and bias, then a Xavier
weight redraw, before default Linear initialization. Keeping the otherwise
discarded first weight draw matters for seed equivalence. Native initialization
never seeds or draws from the process-global CPU/CUDA generator.

## Prospective static qualification

One CPU1/16GiB/3min allocation, maximum180 core-seconds. No EEG, labels, forward,
backward, optimizer step, accuracy or full source fit. The separately held neural
first-update oracle is not released by this test.

Independent original `deep.build('fbcnet',41,480,pretrained=False)` followed by
`session_bn.swap_bn(...,False,20)` supplies all26 state fields for seeds0 and1.
Require ALL52 state tensors byte-exact, including fixed FIR numerator/denominator,
weights, biases and BN priors. Also require unchanged global CPU generator state
and explicit rejection of incorrect FIR shape. Predict exact equality; any
mismatch blocks adoption without a tolerance change or automatic retry.

Before submission, original-model capture is smoke-tested with no EEG/forward/
training. C++20 syntax checks cover the initializer, CLI and test. Actual Slurm
snapshot pins source, original producer sources, qualified CAR base and resolved
shared dependencies. Reuse the original CMake ABI/link recipe, explicitly replace
its old base RPATH, and verify actual `ldd` resolution before numeric execution.
Compile only initializer/test TUs. Preserve every failed output.

Budget at release:353 spent7993 core-seconds after source-plumbing failures;
reserve1200 neural strict test +1200 source-fit retry +180 here =10573<14400.
Actual spend and allocation ID must replace these reservations after completion.

Current checks: initializer, CLI and source-test C++20 syntax checks pass; Python
fixture parses and actual original-model capture preflight passes without EEG.

Job3120984 submitted CPU1/16GiB/3min after actual Galvani import/source/dependency
preflight passed. Snapshot `ni26_native353_initialization_v1_20261117` pins13source
files and37original source/library/build inputs. Local original capture also
passed for both seeds with no EEG, forward or optimizer steps. These preflights
do not establish native tensor equivalence; the scheduled test owns that result.

## Actual result

3120984 COMPLETED0:0 in90s. All52 original state tensors match byte-exactly;
the default CPU RNG is unchanged and incorrect FIR shape is rejected. Requested
CPU1 but Slurm actually allocated2:180allocated core-seconds, not90. Maximum
job RSS1,444,532KiB. Static native test0.45s/327,676KiB; original capture9.37s/
697,668KiB includes Python imports and filter design, so these are not a matched
performance benchmark. Source/build/library/end-of-job fingerprints pass.

The source-fit implementation and prepared first-update test now construct initial
weights with this function and an explicit source seed. The independent test
compares original state before any update; production consumes only fixed FIR
coefficients from that recipe. This integration is syntax-checked, not a source
fit or first-update qualification. That oracle remains held until neural parity.
This does not qualify initialization for the other architecture families.
