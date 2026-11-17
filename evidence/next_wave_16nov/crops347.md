# §347 qualified late-cue supervision on the strong raw model

Qualification work began after the17Nov11:42UTC §346transfer failure; first §347EEGjob submitted11:53UTC.
This is additional work, not the scheduled14:51CET research block. No submission/provider/download/newterms.
Canonical STATE/STATUS/OVERNIGHT belong to root; this report owns347evidence.
“Native” here names the original stockPyTorch FBCNet/SessionBN reference, not the customC++ stack.

## What is new, and what was already tested

Not a claim that extra task seconds were globally unused. Actual historical ordinaryEA/ordinaryBN
FBCNet `fbcnet_ea_c23` used2/3s crops: local/crops_s0.log +stored1900/385row banks record56.33%dev_loso,
64.27%dev_session. These lack modern source/order guards and used40epochs with triple rows, not matched steps.
`run_cv.py` also defines raw/c234forms; a configuration is not evidence it ran.
Coreaug and §100 tested classical population/personal heads. §277SSLform2 actually used26channel
EEGNet start0.5/2/3/4crops plus externals; supervised finefit labels stayed on original windows.
Actual312rawexport used separate BaselineCE/OE only. These scopes do not establish a matched
late-labelled-data versus extra-official-repeat comparison on exact strong rawSessionBN/source6.

## Source timing qualification changed the assumption

Six fit people P04/P05/P06/P07/P08/P10:1,898sourcecues, all original rows retained.
Only1,894have an actual trial-end marker(value800) before the next cue. FourBHlastcues lack it.
The BIDS trial_type also calls run-end code32770 `end_trial`; it must not count as an actual800trial end.
True qualified cue→end range7.634–8.080s, feedback start2.626–3.110s.
This contradicts the older7.98–8.01s summary. A fixed3.9–7.9crop crosses one actual end by.266s.
Declared8seconds alone does not qualify the four missing800end cases.

Prospectively selected fixed3.5–7.5s on ALLqualifiedsourcebounds, before new query metrics.
Every crop has480samples/120Hz,1.5s overlap with official1–5s and2.5s outside(62.5%).
No per-trial shift/pad/clip/alternatecrop. Exact1,894eligiblemask shared by extraB/C; the four unanchored
cases remain in the original1,898source rows of every arm, and appear in neither extra view.
Nominal outside-original seconds sum4,735s; these are not independent trials/people and preprocessing
has temporal filter support outside nominal crops. BIDS task classes remain the same three cues.

## First failure and measured repair

3119273FAILED1:0/13sCPU2: metadata/mask and firstfile official-input≤2e−4µV passed, then actualA6
seven-source-window posterior bound≤1e−6 failed. No extra export/training/querytransform/querylabels.
The initial failed record omitted the actual posterior error; that observability gap is preserved.

3119299source-onlydiagnosisPASS/14sCPU2, same7real source windows:

| Numeric path | Raw errorµV | Aligned-input error | A6posterior error | Label changes |
|---|---:|---:|---:|---:|
|cachedtensor|0|0|0|0|
|old312doubleV→µV→float32|1.52588e−5|9.53674e−7|1.635827e−6|0|
|officialfloat32V→µV|3.72529e−9|1.16415e−10|0|0|

Same-cached-input repeat was0, model state unchanged, eval/k20. ActualNeuralSet
`MneTimedArray.from_native` caches float32VOLTS before later `_get_timed_array` scale_factor1e6.
The immutable312helper uses the same filters/resampling but returns float64microvolts first.
The repair recovers volts from that helper, casts float32, then scalesµV. The roundtrip residual is not
bitwise input identity. The original2e−4input/1e−6posterior bounds stayed unchanged.
No physiological signal or class inference follows from this numeric issue.

## Complete raw qualification

3119303PASS0:0/208sCPU2,117.837sscript. ALL60sourceEDFs/1,898official rows matched cached tensors:
max7.62939e−6µV<2e−4. First actualA6sourceprefix posteriors BITWISE.
Same41channelorder,500Hz native notch50/60harmonics/0.1–75Hz full-record filter,120Hzresample,
float32V-before-µV and exact cue/onset/sample/run/class mapping. All1,894late crops finite41×480,
inside declared/task800end/before nextcue, hashlocked onGalvani. No excluded-person raw read or querylabel.
Source original raw RMS31.89346µV versuslate38.53373µV. Keep original native scale25.817935280729536
and G in every training arm; do not conflate source rawRMS with native centered-covariance scale.

| Immutable Galvani snapshot | Source manifestSHA |
|---|---|
|ni26_crops347_prepare_v1_20261117(failed)|e300e05488fc8bf32ed6ec1e5af0ca89cc2dadbbb9be790e31f3da12fd481894|
|ni26_crops347_diagnosis_v1_20261117|ae59b2e344ff49e3c184efda7cfb26277b5c54fdd3edada1ac058e8a20fc746d|
|ni26_crops347_prepare_v2_20261117|969b88ad1e092a0e412d6ef2cc914b1c1d75608e7f3d6bdcb04f6037e45c40b1|

All snapshots are under `/mnt/lustre/work/gehler/gfh527/`.
Qualified `late.npz`SHA c1dcb7979a53525ac5edbff81dcdbe01c9fdc2d40ec166b182ed48da97cd6042;
rowSHA5c884851ab9ed75b0d94d308b81c8b365e406eff93a369e6ac5526ea4729af0e.
Compact reports crops347_{failed,diagnosis,first_file_smoke,qualification}.json contain rawfile/owner/input hashes.
The EEG/cache remain private; no data committed. Actual completed qualification cost470CPUcoreseconds=.13056CPUh.

## Registered next control/cost smoke — not a scientific score

Source-only3119319queued12:06UTC,CPU6/24GiB/5min. Four1epochfits: unmodifiednativeAref31steps;
wrappedA31steps must BITWISEmatch all final weights/G/scale; B/C62steps with matched source/eligible
labels/sessionchunks/RNG and active gradients. No query EEG or querylabels in this smoke.
Full method preserves original1,898rows and interleaves31base/31extra batches in each of40epochs:
Aoriginal1,240steps; Bextraofficialrepeat/Cqualifiedlate each2,480steps, same native constrained head,
AdamW/OneCycle/sourceG/scale/session32chunk policy. Eligible batchcount mismatch stops; no fallback resampling.
Only a passing source smoke and conservative measured≤8CPUh projection can release the two fullfits.
All three exposedqueries P02/P03/P09/all8orders must be produced for both B/C before labels or scoring.
Priors C−B+.5pp[−1.5,+2.5],C−A0pp[−2,+2], selected/exposed; no independent/full-system claim.

Source-controlsmoke3119319COMPLETE0:0/59sCPU6,50.359sscript. All4real fits passed, including
BITWISEnativeAref/wrapperAfinalweights/G/scale and originalA6initializer,62stepB/C, identical
source labels/rows/RNG/sessionchunks, Bcopy/CchangedEEG/changedfinalweights, activebody/head gradients.
Measured conservative two-fit projection1,438.02s; total budget projection11,686.3coreseconds=3.246CPUh<8.
Actual source manifest5e3ca578f502cee17064ebc408bdea38815253f9adf7f3946bfa95686c981824.

Entrypointpreflight3119346PASS0:0/22sCPU2. Actualproducer/collector imports, qualifiedinput and
ALLoriginalA6controlbanks/body/source proofs passed before full fitting or held labels. Exact189queries
perP02/P03/P09; none substituted anA7control. Source ead1aefb6e50b3f453dfaaa369ff071693b1449a9f44da376a00a43a072b8a76,
immutable `/mnt/lustre/work/gehler/gfh527/ni26_crops347_full_v1_20261117`.
The full snapshot's training347.py is EXACTLY the smoked file; other files are producer/collector/dispatch.

Full2fitarray3119363_[0,1]%2released12:15:24UTC/13:15:24CET, CPU6/24GiB/15mineach.
Afterok-BOTH collector3119365CPU2/8GiB/3min. At12:16:32UTC BwasRUNNING44s/CwasPENDING;
the1,438s serial conservative projection is not an observed concurrent or queue completion time.
All40originalA6base-batch hashes and matchedextra/B/Cstate/data/source guards are mandatory.
Publish all3people/all3arms/all8orders only after the complete before-labels proof. No347accuracy yet.
Actual completed qualification/control/preflight cost868coreseconds=.24111CPUh; runningfits excluded.
`crops347_compute.json` owns later resource accounting. Source manifests retained as compact evidence;
the current localsource.sha256 describes current runnable files, not every prior snapshot's manifest.


PendingC memory review at12:20UTC: source-smokeMaxRSS3,153,648KiB(3.008GiB), runningB
2,884,676KiB(2.751GiB). A prospectiveC-only24→12GiB amendment was recorded. The first command's
12G suffix was rejected; before corrected numeric12288could apply, Cnaturally started at13:20:51CET
cn114 with original24GiB. No resource request changed; bothfits keep6CPU/24GiB/15min.
No cancellation/requeue/runningBchange or memory/scheduling speedup claim. ActualjobIDs are
3119364(B,array0) and3119363(C,array1), which matters for correct sstat memory accounting.


Legacy-harness provenance clarification: `AlignedNet.__init__` computes cached PER-WINDOW covariances
for the full loaded official X array, including query rows. `features.covariances` centers within each
window and produces one independent matrix per row; it does not aggregate people or windows.
For these models `align=False`: inference `_aligned` uses only X/scale and never consumes those query
covariances. The fitted G/scale use C[fit] only and match original source6 exactly; optimization/scalers/
readouts use only source6 labels/rows. Thus immutable machine flags `held_transformed:false` mean no
held NEURAL FORWARD, direct-summary or held RAW-EDF processing before the declared gates; they are
not literally proof that no held pointwise cached covariance was computed. The original flags remain
unchanged and this note narrows their wording. No target aggregate, future-window adaptation or hidden
label enters the fitted parameters or predictions through this unused precomputation. This is an
experiment-harness limitation, not a validated package/export path; no package was built by these studies.

## Completed matched result — negative

Both fullfits and collector3119365PASS0:0. All72banks and source/initializer/G/scale/40originalbatch/
matchedextraeligibility/labels/RNG/activepath/checkpoint/prefix/reset guards passed BEFORE347querylabels.
The unchanged original scorer and fixed scheme-equal order summary were used for every arm.

| Exposed person |A originalraw40|B extraofficialrepeat|C late3.5–7.5|C−B|C−A|
|---|---:|---:|---:|---:|---:|
|P02|65.6460%|64.5822%|50.5968%|−13.98540pp|−15.04919pp|
|P03|56.2275%|54.8724%|54.2773%|−.59513pp|−1.95023pp|
|P09|64.0610%|64.8812%|54.2894%|−10.59177pp|−9.77161pp|

Selected3personmeans A61.97817/B61.44526/C53.05449%, descriptive, no peopleCI or independent confirmation.
Extra optimizer work/schedule is controlled byB. Fixed late-label pooling harms this form;
P02/P09 miss the positive C−B prior widely, while P03's smaller effect does not close a family alone.
The all-source-original causalNLL diagnostic wasB.35525/C.54550; recallsB.972/.874/.938 versus
C.943/.705/.800. These learned Cmodels are already less compatible with original source windows.
This does not identify physiological causality, normalization-prior contamination, representation conflict,
or the optimum training loss. There is no full-system/rank/returning/worker/promotion claim.
No crop/weight grid or current-system consumer is justified by these negatives.

FullBscript389.307s/training356.204s; C488.413s/training449.128s. Differentnodes/resources prevent a
paired speed interpretation. Settled allocation B461s×6CPU, C523s×6CPU, collector28s×2CPU.
All347allocations total6,828coreseconds=1.89667CPUh,0A100, noownedjobsremain.
BfullMaxRSS7,068,768KiB(~6.74GiB),C6,869,512KiB(~6.55GiB): the earlier3GiBsource-smoke/earlyB
sample did NOT bound fullfit plus repeated-deepcopy verification memory. No12GiBupdate took effect;
both actually used24GiB. Do not infer a3×full-memory margin from the early sample.

Exact compact crops347_result.json/before_labels.json/compute.json retain allpeople/orders/control hashes.
Originaldatasets/checkpoints/prediction arrays remain private. Originalfailedqualification is preserved.
Actual collection12:34:06UTC/13:34:06CET. §346+347work since11:23:35UTC spans70m31 throughcollection,
with6,828+734=7,562allocatedCPUcoreseconds=2.10056CPUh, not70minutes of scheduledprimary research.
Root requests a separatelyregistered§349source-only phase/ONEBNmoment mechanism diagnostic next;
there is no new jointhead/training/normalization-grid authorization from this result.
