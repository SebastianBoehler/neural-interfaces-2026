# §341 custom C++ solver cost and regression audit

Actual start17Nov2026,08:51:24UTC /09:51:24CET. Galvani-only/local tools; Weco/provider not executed.
Engineering-workflow and Weco benchmark guidance inspected; user’s local-only instruction controls scope.
No new individual-model, full-system-dev, sealed or rank result was measured.
“Original raw neural reference” elsewhere means PyTorchFBCNet/SessionBN; this audit concerns our separate
CUSTOMC++/LibTorch multinomial head and its input validation.

## Existing evidence and why a new profile was needed

The old243–244sC++CPU versus19–20ssklearn comparison changed stopping quality and optimizer/backend.
§311 already measured41.44sautograd versus37.61sanalytical on6logicalCPUs; rawBLAS57.09→44.36s
also changed objective-evaluation count370→186. Rewriting the gradient again would duplicate that evidence.
§309 residentA1001.062s versusCPU238.483s also changes DEVICE and trajectory182versus384evaluations.
It is not a224×language gain or a full-system speedup. Inputs stayed resident DURINGeachfit already.
The existing raw-to-posterior63–64msCPU comparison versus113–117msPython caches FILTERCOEFFICIENTS,
not EEG inputs. Native batched eigensolvers versus Python loops also differ; no intrinsic language attribution.
C++fixed-head/feature parity and research reuse334→333/337 are real uses, not a measured ensemble gain.

## Controlled same-binary profile

Exact source-only P01 nine-person matrix2465×44239float64/Ccontiguous, no P01held labels/accuracy.
MatrixSHAa38ec018aaf2f727916fa402bb082659543dcfbf758d44e6ee3f9819e4ef8b24;
originallibrarybf3298eb03bbfdfe32b84629ae27e54b77b7c5f0088886b42b11400564805bb0.
SameC=.3, zero initialization, meanCE+weight-onlyL2, unpenalizedbias, LBFGShistory10/maxiter2000/
maxeval2500/grad1e−8/change1e−12/strongWolfe. No new solver objective, source split or stopping rule.
3118356PASS235sCPU6; hostgalvani-cn102, affinity[1,2,3,37,38,39], AVX512/MKL2024.2.
Six scheduler logical CPUs are not evidence of six independent physical cores. Node topology/hardware matters.

| Unprofiled full solve, SAMEhost/binary/data | Seconds | Objective evaluations | Maxgradient |
|---|---:|---:|---:|
|1thread|80.345|216|3.73e−7|
|6threads|45.465|370|3.40e−7|

Full-solve ratio1.77×; per-evaluation ratio3.03×. More6thread evaluations offset part of its faster math.
Final objective gap2.85e−10. Full times are upper bounds on first reaching target; stopping trajectories differ.
Historical243s was NOT reproduced. This profile cannot assign that entire historical difference to threading.
One fixed-order full pair is descriptive, not broad performance uncertainty or a language comparison.

Profiled1threadfullsolve79.555s reproduced EXACTsameparameters/evaluation count as its unprofiled control.
Of79.252ssummedselfCPUtime, aten::mm432calls=76.386s/96.38%; copy_=.227s/.29%.
Those432calls are exactly two dense matrix multiplies per216objective evaluations. Ordinary dispatch,
parameter copies and the ONCE-per-fit validation do not explain most full-solver cost on this host.
The data matrix is borrowed viafrom_blob; CPU.to(CPU)aliases it. No repeated0.87GBinput copy in closures.
Backend-internal BLASpacking/traffic was not independently counted; no hardware-bandwidth measurement claim.

The separate CHECKEDobjectiveAPI intentionally validates the full input on EVERYcall. Its median times
were~.995s/1thread and~.319s/6threads. Input-finiteness checks create~874MBabs plus~109MBboolean
intermediates/multiplepasses. This API overhead is NOT repeated by the internal fit closure, which validates once.
Confusing those two paths would falsely attribute full-fit cost to repeated validation/copies.

## One surgical patch, actual old/new regression

Fresh isolated generated head.cpp replaces ONLY inputmatrix torch::isfinite(...).all with a single
std::isfinite scan. Same errors forNaN/+Inf/−Inf, all math/BLAS/optimizer/options/API unchanged.
Original remote snapshots and binaries remain immutable. EXACTtested source now updates future canonical
`studies/16nov/followups/native_core/head.cpp`, SHA d4264facfbb7850a5ca2f727bb0d944ec390e87ea527e8c8153ba0c1e0a605b6.
Freshbinary9675ee3f9beeff87ab4d0a9da77e8fce5d3e6df72e9012737aa4fcdf789f5d8c is CPUvalidated only.
Build3118380PASS46sCPU2; pairedbenchmark3118387PASS107sCPU6 ongalvani-cn123.

| CheckedsourceobjectiveAPI | Old median | New median | Old/New |
|---|---:|---:|---:|
|1thread|1.0647s|.4712s|2.26×|
|6threads|.3080s|.2262s|1.36×|

Mini/full objective and gradient differences EXACT0. NaN/+Inf/−Inf source perturbations rejected byboth.
Old/new fullCPU6fits41.955/41.589s have EXACTsamecoefficients/objective/370evaluations.
The~.9%singlepair time difference is not a demonstrated practical full-solve improvement.
This validation patch reduces checked API overhead; dense solver BLAS cost remains.

Source-B1head-only3118401PASS17s,12counterbalancedAB/BApairs,16sourcewindows, no accuracy.
Actualsourceμ/σ/referenceθ; inverse-normalized features recover original standardized inputs within1e−12.
Oldmedian2.300ms→new.912ms(2.52×), everypairedratio.324–.435; logits/probabilities EXACT0difference.
These are HEADONLYtimes on one host, not remeasured raw-to-posterior64ms/core/chain/worker latency.
Input/scaler/coefficient copies are outside timing; tiny outputs are allocated inside bothtimedcalls.
Slurm roundedrequested1CPU to2allocatedlogicalCPUs; Torch used1thread. Accounting uses actual allocation.

## Reuse, limits and resource account

A validated CPU-only source patch is not GPUvalidation, productionadoption, package or an accuracy gain.
No speculative full CPU solver rewrite is justified by this profile. Existing thread/runtime selection and
residentGPUplacement matter more for full training; each route still needs objective/source/parity checks.
Separate solver341_gpu_plan.md specifies a bounded UNRUN residency/repeated-fit benchmark and exact API.
A full-system score would require complete current-chain evaluation; none was run here.

Actual allocatedCPU=(235×6)+(46×2)+(107×6)+(17×2)=2178coresec=.605CPUh,0A100; alljobs complete0:0.
PeakRSS profile3100560KiB/build1517132/pairedvalidation2973536/head1993524. No data/checkpoints committed.
Reports retain exact matrix/binary/reference hashes, backend/thread/operator counters, fullfit/paired API
metrics and numerical guards. No Weco/API key/provider/newcorpus/terms/upload/accuracy-grid execution.

Actual final phase clock17Nov2026,09:30:27UTC /10:30:27CET; elapsed39m03s since start.
