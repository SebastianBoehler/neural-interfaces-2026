# §321 JS package-readiness audit and preparation

17Nov2026. Status: draft preparation, no candidate promotion/upload. Historical exposure: STATE,
LEDGER, actual§318source/result and v6build records were read. SelectedJS is CC30FBswap+FBL2,
+.2234officialpp[.0738,.3512], people/order+.184[.0323,.3327],7/10up; this is a small selected
reused-dev result, not independent confirmation, a sealed projection or a top-five claim.

## Exact package mapping

Original frozeninput `v66_cp1_cap250.zip`SHA
`9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed`.
BundleSHA`37c666a8a63c043c290bd35c527c26c6cfaf028bbe55c746cb8c685dac9fb69f`.
Read-only ZIP inventory in `js321_v66_inventory.json`;65nets, original raw weight sum1.875.
No frozen `submissions/*/submission.py`, source ZIP or candidate directory was edited.

| Actual active path | Required JS draft behavior | Why a naive edit fails |
|---|---|---|
| `lean/src66/v6chain.py:step62` computes oldmix→shadow→originalroute→gate/out_v5 before mix6 | Keep original self.w, NetBank, shadow, route, gate and v5hist exact | Global member/weight replacement changes recognition and the labels used by FT |
| FBk2 is replay slot6 / `sc_fb`; package net35–40, files`sbn_k2_s0..5.pt` | Outer-only replacement at one originalslotunit by six-seedCC30k2 posterior | FBk5 remains an independent slot; changing allFBCNet traces replaces too much |
| `shufcombo.chain` forms `sc_fb` after frozen re-stream and FTdelta;318run removes entire`sc_fb` | Omit BOTH old frozenFBk2 contribution and its FTdelta from outer mix | Replacing only the frozenposterior leaves an old-family FTdelta unsupported by the newmember |
| FBL is replay slot8 / `sc_fbl`; package net59–64, files`sbn_fblight_ea_rot05_k5_s0..5.pt` | Double the entire adapted FBLgroup after its own clipping/normalization | Doubling frozen FBL alone fails once FTstarts |
| Allfour original FTmembers35/47/53/59 and r15callback read original streams/labelchain | Initially retain original FTcomputations/labels, including unusedouterFBdelta; discard thatdelta only in outer mix | Editing `v6_ft_member`, blobs or r15relabel source changes the experiment |
| Baseline outer total16units=15oldmix+rest1; JSadds oneFBLunit with CC30replacementunit-neutral | Normalize total17: rest1/17, core1share1/17, FBLgroup2/17, CC30group1/17 | Leaving rest/core1 routed share1/16 changes both mixture and personal-head blend |
| `RouteB` receives core1share at construction | Change only outerRouteBshare1/16→1/17; retain original innerRoute share1/15 | A global share change contaminates the original recognition path |
| TailX,RouteB,CountFilter,CausalPers each carry sessionstate | Recompute each JSouter downstreamstate from JSstream; independent reset/deepcopy | Pasting a posterior into frozen control stages does not reproduce JS |

Exact source evidence: `studies/17nov/coral_fbl318/patch.py` removes`sc_fb`, doubles`sc_fbl` and
uses `replay/alib.py:mix` for totalweight/core1share; `studies/13nov/shufcombo.py:258–288` creates
re-streamed/FTadapted slots before that swap. Package `mix6` clips and normalizes each adaptedfamily.
The package uses labelmode0 (inneroriginal output), r15enabled, routedmode1 (FTstops at recognition),
cap250 and CPon. `relabel_prefix` uses the originalroute.tail when labelmode0. Those must stay exact.

For normalized originalweights w, frozenfamily contributions C_FB/C_FBL and adaptedfamily
C*_FBL, construct the same original FTmix while excluding the FBdelta, then subtract C_FB,
add W_FB×CC30 and add one C*_FBL. Final rest mixing with the existing wr=1/16 has mass17/16;
normalize by17/16 and bind outerRouteBcore1share to1/17. DisabledJS must branch directly to the
unmodified method, without re-normalizing, so exactzero identity remains possible.

The original65net bank must remain available for originalinnergates/labels. Thus a faithful draft
adds sixCC30forwards instead of globally deleting sixoldFBforwards. OriginalFBFTmay still cost time
although its outerdelta is discarded. Existing SharedFIRSJit selects exact filename allowlists; new
CC30filenames initially use genericSJit and will recompute FIRunless a separately fingerprinted
private allowlist extension is validated. Study probability swaps do not measure this package cost.

Draw caveat: actual318`stream(fam,bn)` averages each base's ownseedsubset. An allsixseed deployment
corresponds to B2_pool, not simultaneously to the single-/three-seed draws used by the six-base
robustness report. FTalso uses each draw's firstseed. Package/study parity must use the matching
B2_pool or a genuinely reconstructed matchingdraw; never treat the reported six-base average as
one package's deterministic output.

Recognition caveat: preserve originalinneroutputs/gates fromfirstrecognition and inspect windows0–9
separately. A faithful JSouter RouteBmay differ afterrecognition because itsmix andcore1share changed.
Forcing JSoff afterrecognition to guarantee whole-output returningidentity would define another
policy; it cannot silently be called the testedJS. Real returningP04/P10 wholefold remains blocked
by missing valid original Sinc checkpoints; prior reconstruction failed and no substitute is allowed.

## Initial blocker and smallest decisive test

§304`produce.py` saved posteriorNPZs, metadata and parameterhashes, but never saved its trained
networks. Read-only scan of its `runs/` found noPTcheckpoints. Therefore none of the selectedCC30
study weights can simply be exported. All10deployment training alone cannot validate that gap.

Prepare one originalP01held/seed0 CC30retrain using the EXACT immutable304source snapshot
`ni26_s304_coral_prod02_20261116`, manifest
`94a5f8a19c3532017f42fc1e85520e9ca0aa511e2ece1bc40df9f406a65b45aa`.
Original archive metadata: A100-PCIE40GB,torch2.14.1+cu126,1600steps,32.584sfit/34.193stotal,
9sourcepeople P02–P10. Reuse original `make/install/infer` paths, seed0, training/augmentation/RNG,
source row/data/order hashes and same A100family. No P01groundtruth scoring. Persist the newstate,
G/scale/source metadata and referencecomparison before export; never overwrite original outputs.

Historical probability guard: all8orders, sameindices, maxabs≤1e−5 and zero changedpredictions where
reference topmargin>2e−5. Parameterhash equality is reported but not assumed. This is a numerical
reproduction hypothesis, not a dev accuracy prediction. Prior: possible exact/≤1e−5 reproduction,
but nondeterministicGPUtraining may fail; an observed violation refutes exact-reconstruction readiness.
Do not loosen tolerance posthoc. If first historical guard fails, do exactly2additional seed0fits
for the predeclared repeat distribution, record all reference/pairwise differences, and stop. No
checkpoint is selected from those repeats and no all10wave/metric is released.

If historical guard passes, reuse exact owning `device_free_trace` helper andSessionBN`to_stream`.
Predeclared existing-export tolerances: CPUstreamvsCPUeager<1e−4; CUDAstreamvsCPUeager<1e−3.
Require zero savedStreamBNstate, prefix/reset/deepcopyinterleaving checks, nonzero conditionalgradient,
sourceexclusion and immutablehistoricalarchive/assemblerhashes. Exportfailures are recorded and block
readiness. This validates a member, not the outerwholepackage. OneA100allocation4CPU/16GB/8min;
originalfit32.6s suggests firstfit+export~1–2min, three diagnosticfits~2–3min plusimports/queue.

## Subsequent build/checklist, not passed

`all10.py SEED` prepares the exact deployment-only recipe and checks originalFBk2G/scale/k. Commands
once BOTH member and activeouterpath guards pass, from the private workspace on an allocatedA100:

```
for seed in 0 1 2 3 4 5; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python all10.py "$seed"
done
```

Outputs `all10/cc30_all10_s<seed>.pt`, `all10/cc30_all10_k2_s<seed>.pt` and hashedmetadata; all10fit
people are explicitly recorded and these checkpoints cannot be used as heldfold models. The command
currently REFUSES without successful `result.json`, explicit `outer_active_path.json` proofs and
pinned original `base_v66.zip`. No all10training is queued. Sixfull10fits estimate~4GPUminutes before
export from32.6s9personreference, not measured. Missing source/input/trace guards fail loudly.

A COPY-only draft must preserve every original file/hash, add newCC30blobs/keys and source-own modules,
and bundle into a new draftpath. Do not invoke an existing default builder that can write a frozen
submission. No new packageversionname or candidatepromotion is assigned. Requiredbefore-ready:

1. JSoff→originalv66 bitwise posterior/label identity; zero-mix algebra and sourcekey guards.
2. Actualheldfold JSoutermixture versus unchanged §318runner's mix, and stagedstage/CP versus owning
   study with ≤1e−10float64mix/stage tolerance and zero non-tiedlabels; use existing evaluator.
3. CPU thenA100 reference with packagebatchshapes; FTreference tolerance1e−4, no changednon-tied
   final labels attributable to the port. Failed historical source reconstruction blocks this claim.
4. Originalinner gate/shadow/out_v5/r15labels exact; ownouter RouteB17shares; first10before recognition
   andfromfirstrecognition traced separately; originalrecognizedpath and forced-unrecognized tests.
5. Sessiondeepcopy/reset/interleavedprefix invariance for inputs, CC30StreamBN, stages, count andCP;
   no mutable package/module state that changes later-session predictions.
6. Whole returningP04/P10 only with genuine originalcheckpoints; remain unvalidated otherwise.
7. Fullworker-image CPUfunctional replay and A100warm B1+190/378/800window paired timing, worst-case
   gatesoff and actualFT/CPactive, plus firstprocesswarmup. Require zero errors, validparity and
   worst-case official-hour slack≥1.3 using measuredpaired ratio to pinnedv66cap250. Existingv66
   historical worstcase43.5min implies only~6.1% extra before46.15min slacklimit; do not claim that
   estimate is a sealed runtime. Stop before harvesting accuracy/timing benefits after a guard fails.

No in-sample replayaccuracy is a dev gain. No sealed projection or upload follows from this audit.

## Actual guard result and remaining readiness

A100guard3116954 completed0:0, actual03:47:13–03:48:06CET on17Nov (53s allocation;48.24sscript).
P01/seed0/40epochs trained33.007s. Recovered parameterSHA EXACTLY matches historical
`db62cedc6a66b52b3e583288cd71ca3c1a27bf9a2bf94166a6cb7a56eec8c1d9`. All8orders have
maximum probabilitydifference0 and0changednon-tiedpredictions (1520order/window observations).
The first guard passed, so no diagnosticrepeat was run and no tolerance was changed.

CPUexport maximum6.97e−6; CUDAexport7.98e−6; savedreset, prefix and interleaveddeepcopy checks passed.
CheckpointSHA`e5e940e27ac86f6796fac2c674bf06f0de9aaef54681990d544c010346e84b46`;
traceSHA`5fde9531448d1ab088f989e7232f25cbaaef62d0042afdb6efc900a57de73fd8`.
Full report `js321_reference_result.json`. Workspace `ni26_js321_20261117_reference1`;
source manifestSHA`676408c488d2bb150c4ae08463fd349f80a76cf971f04d37f2543a5ae7a7e747`.

The originalZIP was separately copied and hash-checked onGalvani, then extracted into `draft_base`
(39originalfiles), recorded in `js321_base_copy.json`. This is an unchanged inputcopy, not a JSpackage.
Onefold/seed member recovery is nowvalidated. Seeds1–5, the actual outer-only draftadapter, B2_pool
whole-package/study parity, returning folds and worker/timing gates remainunvalidated. No all10wave
was queued; `all10.py` stillrefuses until explicitactivepathproof is present. Nextdecisive test is
the source-only outeradapter/controlidentity plus real matching6seedheldfoldB2_pool mix proof;
recover other required sourcecheckpoints before calling it whole-study/package parity.

Read-only follow-up also verified original304source `ch_names` exactly equal the frozenbundle's
41`CH_NAMES` positions. The source/P01 checkpoint guard is not using a different montage ordering.
Owned continuation note: `owned321_320_317_handoff_17nov.md`.

### 17 Nov continuation: actual six-seed recovery and deployment-relevant contrast

Recovery3117049 passed all six P01 seed identities, exact historical probabilities/parameter hashes, CPU/CUDA export bounds and saved reset/deepcopy guards. Seeds1–5 took33.39/28.98/29.13/29.37/34.44s; total recovery191.17s. Seed0 reused only its previously validated checkpoint/trace. Compact proof: `js321_six_seed_recovery.json`; immutable workspace `ni26_js321_20261117_recovery6`, manifest5a103449b7802e72c2663bc60d2ff793b60a1b1a542669220c61c043d9053ea8.

The preregistered **post-hoc** B2_pool extraction3117050 re-ran the complete owning318 collector before labels. B2_pool control76.279335%, JS76.551732%; official effect+0.272396pp[0.004869,0.527102], people/order+0.277955pp[−0.006993,0.545638],6/10people improve. P04−0.048350pp/P10+0.565349pp are new-person routing-emulation effects, not returning-person package validation. Other five base effects range+0.134053 to+0.301971officialpp. These selected reused-dev results are not independent confirmation. Exact owning metrics/provenance are in `js321_pool_contrast.json`.

Copied-bundle guard3117079 is queued on explicit4CPU/16GiB/20min, immutable workspace `ni26_js321_20261117_active1`. Policy is **JS-all**, matching318: original inner gates/FT labels stay exact, but final output continues outer JS after recognition. The guard separately reports actual final changes before10/afterrecognition; preserving inner gates is not final recognized-output identity. JS-off must exactly reproduce original outputs. The active reference uses real65-member/rest/FT outputs and six distinct recovered CC traces, with all six actual318 draw recipes across8orders captured before stages/accuracy. All10 training remains blocked until these checks pass. No returning, full-worker, later-CP or runtime readiness is claimed.

Prelaunch audit cancelled still-pending3117079 before execution: owning318 runner's post-run LOG diagnostic would fire before the deliberately deferred capture. Immutable active1 remains intact. Separate active2 snapshot3117082 omits only that final diagnostic; the actual mix implementation and nonzero-use assertions remain. No tolerance changed. All10 entry now also requires all8orders×6draw capture proofs tied to the generated bundle and successful six-seed recovery.

### Corrected serialization and conditional unattended preparation

Active3/3117085 failed exact first-window C1 equality. Diagnostic3117087 proved independent original instances bitwise identical, while off/on shared C1≤2.22e−16 andu≤7.99e−14 differences (mix/shadow/out_v5 exact). The private builder's ndarray.copy() silently changed ten original arrays from Fortran to C order. This is a resolved serialization path, not an existing-model or accuracy error; previous guard remains failed. Builder/collector now preserve order='K'. Real no-model filesystem preflight checks all1,047original arrays' dtype/shape/bytes/strides, frozenZIPsha, both generated bundles' compilation and actual official-base import. See js321_filesystem_preflight.json/js321_layout_diagnostic.json. No tolerance changed.

Active4/3117092 separately pins corrected source and adds independent original-repeat exact checks, effective BLAS/Torch settings/RNG/dtype reporting. Separate orchestration3 controller3117093 depends afterok, independently verifies source/build/file hashes, all six historical seeds/exports/state and every8order×6draw actual recipe, then releases deployment-only six-seed CC30 all10 array at concurrency2 and complete collector. Dynamic IDs appear in `ni26_js321_20261117_orchestration3/released.json`; no model is trained without proof. Collector prepares private `prepared_jsall` only after six seed/source/checkpoint/trace/input/state proofs. This is deployment training, never heldfold accuracy.

Separate worker1 observer3117097 depends on3117093 and binds the actual successful owning collector dynamically. CPU4/24GiB/45min and A1004/24GiB/30min probes run in existing isolated workerSIF, only after complete six-seed artifact and source hashes; actual image import/compile preflight passed. Both original and candidate run paired warm B1/190/378/800 real-source length stress, known and forced-unrecognized, reset/repeat and runtime activity/error checks. Historical43.5min and slack1.3 require ratio≤1.061008; first failing ratio stops later harvesting. No sealed forecast, promotion/upload, returning-person or full held-package readiness claim. Missing genuine original returning checkpoints remain unresolved.

Active4/3117092 completed0:0 actual17Nov04:37:18–04:44:05CET (6m47s). Both45window cases passed, including independent original repeat and exact off/on original inner/FTlabel paths, disabled final identity, reset/deepcopy/interleaved prefix and real owningmix≤2.22e−16/share1/17. RNG unchanged; CPUfloat32, BLAS/Torch1, Torchinterop4. Known recognition occurred at10; JS-all changed all10before10 and35postrecognition final-posterior rows. Forced-unrecognized FBLFTdelta0.154301/oldFBFTdelta0.252313 (excluded outer); zero runtimeerrors. CPfused0 is a limitation, not lateCP validation. All8orders×6actual318draw recipe captures passed without accuracy. Source0aeed45c4de5757aae0b7a82219878f8954b8ff0afa0bcf806a321b4f3c2724e; proofs `js321_outer_active_path.json`/`js321_study_recipe_guards.json`.

Controller3117093 completed0:0 in5s and independently verified proofs. At16running/117active it released all10seedarray3117189(%2) and owningcollector3117190. Observer3117097 completed0:0 in2s and queued complete-collector-dependent workerCPU3117196/A1003117197. Exact dynamic IDs/provenance: js321_deployment_jobs.json/js321_worker_jobs.json. This DAG advances unattended if successful and fails closed otherwise; no upload/promotion or package-ready claim.

### 17 Nov input-prior failure and corrected raw fidelity

All6initial all10tasks3117189 failed1:0 after44–68s before checkpoint/export. Explicit normalization guard incorrectly required causalEA input k2, confusing latent SessionBNk2 with AlignedNet.input k5. Owning source-only covariance calculation proves sourceall10G/scale exactly originalnet35 (Gmaxerror0; scale24.282585702079455), actualinputk5. Historical heldweights/prealignedexports, off/inner identity and outermix algebra remain valid; **old k2 raw-bank/study-equivalence proof is invalid**. Deadcollector3117190/workers3117196/3117197 cancelled; no guards weakened or scores read.

Raw-input repair3117225 completed0:0 actual05:00:08–05:01:07CET (59s), separate immutable `ni26_js321_20261117_inputrepair1`. It reuses all6hash-validated historical checkpoint/traces, derives inputk5 from originalnet35 and actual source default, preserves latentBNk2, then streams actual raw heldwindows through owningInputStream and exportedtrace on all8orders for eachseed. All48streams pass maxerror<1e−4/0non-tiedchanges; proof js321_raw_input_repair.json. This corrects raw-input metadata, not weights, model selection or metric.

Newactive5/3117230 runs corrected fullbundle guards. Separate orchestration4 controller3117231 depends afterok, then releases ONEall10seed0 smoke; after actual normalization/input/export/reset/deepcopy/hash proof, remaining controller releases seeds1–5(%2) and completecollector, reusing canonicalpassedseed0. Dynamic IDs are `orchestration4/smoke_jobs.json` then `released.json`. Worker2observer3117232 waits actualnewcompletecollector. Its source now asserts knownrecognition10/FT0/CPstop10 and forcedunrecognized no recognition/stop, exacteligibleFT/CPrefit/fusion counts from actualv66schedule, no overbudget skips/runtimeerrors; reports single-classCPattempts separately. Both baseline/JS satisfy applicable counts. Isolated actualworker import preflight passed before allocation.

The CPUratio bar remains predeclared engineering overhead only: CPUratio×historicalA10043.5min is NOT a validCPU sealed-time estimate; CPU-onlyfailure is not A100timeout evidence. CPU and A100 probes remain independent aftercompletecollector; firstbarfailure stops later rows. HistoricalA100basis itself is conditional, not fullworker-limit proof. No returning/checkpoint repair, heldaccuracy, upload, promotion or readiness claim.

Corrected active5/3117230 completed0:0 actual05:02:07–05:10:03CET (7m56s); raw-corrected fullbundle guard398.57s and all8actual recipe captures passed. Proof `js321_active5_outer_path.json`: originalrepeat/offfinal/inner/FTlabels exact; mixmax3.33e−16, reset/deepcopy pass, actualJS-all recognizedfinal changes retained. Sourceadfe86c94aafd90ad1f6a9310b3a1e3e976d1addd8c6149e76e841011d82d228.

ONEcanonicalall10seed0/3117236_0 completed0:0 in58s: fit34.05s, input_prior_k5/latent_bn_k2, rawInputStreammax1.46e−11, CPUexport7.10e−6/CUDA7.29e−6, reset/deepcopy/hash guards passed. `js321_all10_seed0_smoke.json`. Controller3117237 validated this actualproof before releasing remaining5seedarray3117240(%2) and completecollector3117241. Workerobserver3117239 was rebound afterok3117237 without waiting5minutes, then queued independentCPU3117246/A1003117247 afterok3117241. Corrected IDs in js321_corrected_deployment_jobs.json/js321_corrected_worker_jobs.json; no completedartifact/workerclaim until those guards finish.

### Complete all10 artifact and first CPU worker stop

All six corrected all10seeds completed0:0; totalallocation347s (seed0/remaining1–5 elapsed58/53/53/75/51/57s). Owningcollector3117241 completed0:0 actual05:15:35–05:15:41CET(6s), verifying allseed/source/normalization/checkpoint/trace/export/state proofs before preparing privateartifact `orchestration4/prepared_jsall`. Compactmanifest/proofs `js321_all10_complete.json`; originalsourceZIP/submissions unchanged, no labels/score/promotion.

CPUworker3117246 failed1:0 after4m06s at FIRST known190timingrow: original82.330s, JS90.303s, ratio1.096840>1.061008bar; warmB1medians433.27/474.39ms. Actualisolatedworker loading, disabled17identity, independentreset/repeat, knownrecognition10/FT0/CPstop10 and190CCcalls passed. No runtimeerrors. It stopped before378/800 and forcedunrecognized; their lateCP/FT activation is **UNTESTED**, not failed/passed. `js321_worker_cpu_failure.json` recordsfirststop/imagehash. This is CPUengineeringbarfailure, not CPUsealedestimate or A100timeout evidence.

IndependentA100worker3117247 remains pending; actualv66A100 cap250190/378/800 forcedunknown40.58/78.15/155s grounds25min boundedrequest (~37% over four oldpackage×mode profiletotal; knownmodeavoidsFT). Pending-onlyscheduler request30→25min/hostmem24→16GiB (CPUimageobserved3.12GiB), same4CPU/A100/source/checks, timeoutfailclosed. Forecast06:32:30CET is scheduled estimate, not actualstart/result. No GPUslow/readiness claim. If actualA100barfails, profile measurednewCCbank/CPU-GPUoverlap/sharedfixedFIR before any separately registered optimizedvariant; do not mutate frozenprobe or relaxbar.
