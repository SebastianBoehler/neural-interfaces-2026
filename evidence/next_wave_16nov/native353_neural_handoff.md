# Native353 neural handoff —17Nov15:18CET

Latest15:55CET:3120517 COMPLETE0:0/110sCPU1; one-window45member diagnostic actual PASS1e−6, max C++prob6.07048e−7/id37, raw members BITWISE; hostPython↔SIFgolden BITWISE ALL45/allinput/Eye/Rest. EAinput C++difference2.38419e−7; Eye0/Rest1.32865e−7. Old3120312 failure was somewhere in known8 prefix, not provenwindow0 (old log had no index). FiveOSchecksum differences identified (ld-linux/libc/libm/libpthread/libutil); initial17sfailedpreflight preserved. Second67sfailedstandalonelink preserved; exactv3productionlink reuse+ldd preflight fixed it. All3diagnosticattempts194coresec=.053889CPUh. No further job queued. Prepared tools/neural_input_arithmetic.py compares existing known8 original/torch-centering/cov/P/h WITHOUTneuralforward/FT/fit; root authorization pending. Native training oracle remains HELD. CMake root-owned.


Root may resume CMake ownership now. CMake changes are limited, uncommitted with root's initial native integration: remove OpenSSL find/link; own hash.cpp/hash test; include existing core_training.cmake; prepared neural source-update test target. No other root files edited. All owned neural code and evidence are committed/pushed through6d73a1e.

Runtime API: native/include/ni26/networks.h (NetBank/EyeNet/RestMember/FineTune); training.h train_raw_fbc. Own source files networks*.cpp/internal.h, finetune*.cpp, session_bn.cpp, training*.cpp/internal.h; hash.h/hash.cpp. Own source files formatted, max252LOC.

Implemented native runtime: original group/member ordering, current-inclusive raw/EA InputStream, plain Shallow/FBC, TorchScript JIT/SJIT/sharedFIR, Eye preprocessing, Rest. Strict original FIR class/dtype/shape/order/digest607a44ee…506c6 remains. TorchScript architectures are migration artifacts; native source architecture training is not complete.

FineTune: own C++ Adam/CE+KL loop, independent native device RNG, strict earlier-window prefixes, same relabel callbacks/schedule/cap250, own module bytes loaded lazily, warm parameters/fresh optimizer, five steps/family and final StreamBN rebuild, same clocks. Refit explicitly AutoGradMode(true) because master runtime has NoGradGuard. Positive active gradient/parameter guards exist.

Source reference3120067 COMPLETE0:0/56s×2CPU=112coresec=.031111CPUh/RSS2302588KiB. Snapshot /mnt/lustre/work/gehler/gfh527/ni26_native353_neural_reference_v1_20261117. Fixture reference/fixture.pt SHA08e368896bca2837d0157744f0d0497a2db8b2cd3d08b5a61f016f2f19e9f3a8. Original P01session1 known8 and forced-unrecognized31, no tasklabels/accuracy. Unknown two real4-familyrefits20/30:40steps/2relabels/dtheta.00999158/zeroerrors. Actual casewall5.274/34.948s. Native test uses this exact fixture.

Original frozen submission37c666a8…b69f/weights7cf13696…88a2 verified. Native assets /mnt/lustre/work/gehler/gfh527/ni26_native353_qualify_v1_20261117/package; native.pt SHA4bf45aee1fa6457e75e504d5acf747a6c6a4e21c99a3266c0c5a514c6b5ab31c.

Full build3120077 FAILED186s/4CPU because conda sysroot could not see host OpenSSL header. Preserved. Own81LOC original FIPS180-4 SHA eliminates this dependency; Linux and local knownempty/abc/multiblock/million-a/real3724FIR/tamper tests PASS. Tests/hash.cpp + tools/hash_fixture.py extract exact existing frozen module coefficients; no generated coefficients/checkpoints/binaries committed.

Recovery3120247 FAILED1:0/265s×4CPU=1060coresec=.294444CPUh/RSS6853580KiB. COMPLETE C++build PASS ALL targets including CUDA-generator/runtime/training/core-training/CLIs. Linux SHA PASS. Neural test FAILED BEFORE first numeric comparison in FineTune reset: getDeviceIndexWithPrimaryContext queried unavailable CUDA driver even for CPU model. Exact failure retained in immutable /mnt/lustre/work/gehler/gfh527/ni26_native353_neural_recovery_v2_20261117/neural.log. Source manifest e4c6feae…69f59, assets2f60f8e3…ba663. Evidence native353_neural_recovery_v2.json.

Fix6d73a1e: release_cache(device) returns before ANY CUDA call for CPU device; CUDA-requested behavior retained. Local syntax PASS. No numerical retry or new allocation after this failure. Parent coordinates next same-source immutable v3/budget; no automatic extension. Could freshbuild at measured265s; do not rerun classical/linear fits already owned by3120215. No jobs remain owned by this branch.

Prepared native/tests/neural_source.cpp requires SAME8/31 fixture, all45 member posterior/input/Eye/Rest/FT≤1e−6, RouteB≤1e−8, exact member order/counters, positive optimizer change, reset/copy/session guards. Linux numerical parity remains UNVERIFIED.

Future local tools/neural_fixture.py also captures emitted runtime_scores, but old3120067 fixture does NOT contain them. Preserve old source/artifact. Future source score fields are known_runtime_scores/unknown_runtime_scores; master runtime test needs selected-case mapping. No whole-runtime/source-worker/A100 qualification yet.

Own raw FBC graph/source loop is implemented but UNRUN. Exact biased within-group32 SessionBN population updates; k20 causal eval; grouped constrained288 spatial, four-segment UNBIASED logvar1152 head; native AdamW and OneCycle LR plus beta1. train_raw_fbc currently consumes imported initial/source_X/full1240-batch fixture. Initialization/source preparation/NumPy PCG64 batch generation and other family source graphs/trainers remain incomplete.

Source-FIRST-UPDATE oracle card appended14:51CET; PREPARED/RELEASEHELD until inference parity. tools/neural_training_fixture.py + tests/neural_training_source.cpp export exact originalA6 initializer/source1898/source6/all1240 batches and LR/beta trace, then ONE original/native source update; no query forward/labels. Body/head actual changes, finite grads, head constraint enforced. Bounds logits/grads1e−5/posterior+state1e−6/schedule1e−15. Direct separate forward/backward/optimizer timings. No source fixture/update/fit run. Parent-approved prospective maxCPU2/5min16G needs separate release.

Important future initialization detail: original Conv2dWithConstraint consumes default Conv initialization, THEN Xavier-uniform redraw of spatial weight; LinearWithConstraint keeps default Linear initialization. A plain native Conv initializer would consume the wrong RNG. Exact original initial-state SHA is pinned by the prepared oracle.

Latest owned commits: c55c7a4 subsystem; e7c0a8c readable independent InputStream assignment;6cb359e source-reference/first-update prep; e15e809 ownSHA; c681ef8 source-body/head activity;6ae243f bounded recovery script;6d73a1e CPU-driver repair. Canonical STATE/STATUS/OVERNIGHT untouched. LEDGER appends remain for root checkpoint. No new accuracy method, upload, provider, GPU fit or frozenfile edit.
