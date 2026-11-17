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
