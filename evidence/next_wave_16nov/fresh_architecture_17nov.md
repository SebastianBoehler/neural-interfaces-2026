# Fresh architecture review, 17 November 2026

Actual research start: **04:13:54 UTC / 05:13:54 CET**.
Actual research end: **04:30:14 UTC / 05:30:14 CET**; substantive phase-boundary review lasted **16min20s**.
Later operational continuation is recorded separately below.
Fresh execution worker: user-selected GPT6.1sol, new context with task packet and canonical files.
The worker read STATE → LEDGER → OFFICIAL_CONTRACT and the scientific-discovery skill.
This is **artifact-informed fresh-context research**, not blind replication or cross-provider agreement.
The brief explicitly suggested revisiting foundation models; that suggestion was retained.
No new source score is an outer, current-system or sealed result.

## Reconstruct the problem before choosing an architecture

The hard observable is a current **41×480 float32 EEG window**, four seconds at120Hz,
starting one second after a command cue. The three labels distinguish motor imagery,
mental subtraction and silent word generation. These are different cognitive strategies,
not left/right/tongue motor-imagery classes. Their generators can differ in spatial power,
slow potential, eye/muscle behavior and nonstationary within-window organization.
No assumption that artifacts are irrelevant is imposed by the official tensor contract.

The official preprocessing bandpasses0.1–75Hz before120Hz resampling, in microvolts.
The available waveform cannot acquire new physical bandwidth by upsampling to200Hz.
Within-window filtering/bidirectional modeling may use all480 samples of that current window.
Bidirectionality across delivery windows would use future information and is forbidden.
Delivered-window order is shuffled/fixed/unknown; chronology and interface metadata are unavailable.
Adaptation is current-session-only, including module globals, and must survive deepcopy/reset.
The official metric balances classes inside person/session/interface cells, then weights cells equally.
Ten people/14sessions give few independent population units despite2655 windows.

Our export [`scripts/export_windows.py`](../../scripts/export_windows.py) calls the official
NeuralBench task and saves all splits with metadata. It does not itself spline-interpolate
PROTEUS windows. The §282 note describing our entire export as spline-interpolated is not
established by this exporter; external dataset conversion explicitly does interpolate missing
channels. I did not infer physical PROTEUS sensor rank from that note or modify the tensor.

The [MEG study on exactly these cognitive strategies](https://pmc.ncbi.nlm.nih.gov/articles/PMC10171552/)
(29March2023) finds distinct beta-power engagement across motor, language and arithmetic regions.
Its18-person MEG/beamforming and within-study classifiers are physiological motivation only.
A scalp-EEG decoder has different observability, artifacts and transfer conditions.

## What actual execution covers

| Executed root | Actual test | What remains unanswered |
| --- | --- | --- |
| Strong geometric/spectral decoder | Core1 band/half-window covariance tangent features plus LR; coherent adaptation chain | Not evidence that all useful EEG is stationary second-order structure |
| Imaginary spectral features (§313) | Whole decoder and full chain; imaginary beats matched real but not strong BASE reliably | Nonlinear temporal organization and learned temporal filters were not isolated |
| Temporal summaries (§314/315) | Fixed random asymmetric filters, max/positive fractions and1s bandpower; weak whole gain, no reliable chain gain | Learned continuous selective dynamics and discrete topology sequences were not tested |
| Raw nets/pretraining (§312) | Two small architectures, matched raw-pretraining controls; whole54.7%best FBC supervised, guarded consumer no gain | Not a general raw-learning ceiling; §323 causal-normalization mechanism materially changes inference |
| Normalization (§323) | Four fitted models, same-weight fixed/causal inference,80fits; best whole59.0% | Large normalization effect repairs only part of the raw-vs-strong gap; does not identify missing representation |
| Existing EEG foundation models | OriginalLaBraM/CBraMod/REVE frozen (§71); CBraMod5forms and REVE4forms FT (§280), kit integration (§282) | New2Nov LaBraM-at-Scale weights/architecture scale not tested by those runs |
| Vision prior (§319/324/325) | FixedPCA/full/intensity/frequency forms negative; learned display separately owned | This worker did not repeat or claim novelty for learned rendering |
| Conditional heads (§326/327) | Strong fixed core, source penalty doses, two exposed query people | Learning objective changes on fixed feature map, not another representation root |

The last several projects therefore have wider implemented coverage than simply more heads.
However, large run counts within a fixed representation do not establish broad end-to-end search.
The major recurring bottleneck is a weak whole comparator: improvements over40–60%decoders
have not reliably improved the76.22%current-system new-person reference.

## Ranked cards

### 1. New released EEG prior with retained patch structure: LaBraM-at-Scale

**Changed assumption.** Replace the old frozen EEG encoder with a newly released larger
pretraining prior; retain channel/time patch structure and then adapt the representation.
This is not another head on the current strong tangent feature map.

[Paper](https://arxiv.org/abs/2611.03165), **2November2026**;
[official code](https://github.com/AlbertMentat/LaBraM-at-Scale), revision
`59e4e7ea2d855cda5573c312b380b109f1856764`;
[official weights](https://huggingface.co/Pan-Junyu/LaBraM-at-Scale), revision
`72189644cef9a91b7b436c5f37f083c1366e2f21`, liveAPI200, ungated/public, modified8November.
Code/weight metadata and actual license text explicitly grant MIT terms.
The release distinguishes last pretraining checkpoints from pretraining checkpoints selected
using the23-task downstream benchmark. Both contain pretrained encoders/pretraining heads,
not23 task-finetuned encoders. The latter still add benchmark checkpoint selection.
We choose default Atom-L,46.3Mencoder parameters,212001576B including pretraining heads,
SHA`a28e97f7bd73bd7ff55f3b5d680f48807c335cf3134d1281228a061056aaf52b`.
Its published aggregate is not our prediction; hyperparameter selection uses seed42test,
and the release compares favorable LPpooling/readout settings.

**Input/coverage/rights.** Atom uses200Hz,200samplepatches,µV/100; Molecule instead z-scores.
Amplitude-preserving Atom is deliberate because slow means and absolute power carry task signal here.
The published corpus inventory contains Dreyer2023 and EEGMAT. No literal PROTEUS/Dreyer2026/
nm000290 match was found, which does not prove the same people never occur under another corpus.
Never call our people unseen during upstream pretraining without participant-level evidence.
Existing §71 frozen-originalLaBraM is not this checkpoint; §280FT covered CBraMod/REVE.

**Small decisive first test.** Registered§329: no-label realP04-window source guard, actual public
checkpoint download/hash, exact channel map with unknown-name failure, strict encoder load,
retained164×400patch outputs, random same architecture, amplitude/z-score contrasts,
lastsample gradients/reset/deepcopy/prefix and warmA100B1 timing. No new classification score.
The checkpoint's postnorm must load into `norm`, not disappear while a fresh `fc_norm` is added.
Exact lm/projection/mask pretraining keys are listed separately; missing encoder keys fail.
An unused image/text `logit_scale` scalar absent from pretraining is explicitly removed.

**Next prospective source experiment, not yet authorized by this card as a scored run.**
Same seven fitpeople/two previously exposed queries, allfit/scaler/prior excludingqueries/P01.
Pretrained/random × pooled/retained-four-timepatch readout, with/without identicalraw287;
fixed head settings and equal fitsteps; include existing same-source causal rawSessionBN
and strong source geometric decoder contexts. Then one fullFT form only after an active,
credible source comparator. Broad selected prior: pretrained−random+3pp[−5,+10];
FTwhole65%[50,78], not an outer accuracy prediction. Refutation: random matches/exceeds
pretrained on both query people under same rawblock and readout, or frozen control is too
weak to support escalation; ambiguous effects remain undecided. No automatic outer matrix.
Whole alternative must get coherent session adaptation before comparing to the chain;
posterior-only addition is a separate component experiment.

**Resources/readiness.** One≤5minA100preflight,6CPU16GiB; predictionB1≤10ms[1,30],
GPUpeak<2GiB[.2,4]. Actual timing pending. Remote environment lacks `timm`; pinned0.9.16
wheel SHA`bf5704014476ab011589d3c14172ee4c901fd18f9110a928019cac5be2945914`
is installed into private assets/runtime only. Torch/torchvision are not changed.
No provider, login, acceptance, upload, shared-environment mutation or new package promotion.

### 2. Continuous selective dynamics on raw EEG: a small state-space whole decoder

**Changed assumption.** Learn input-dependent propagation/forgetting over continuous EEG
rather than fixing band covariance, random filters or a discrete state alphabet first.
[NeuroMambaNet](https://www.mdpi.com/2504-2289/10/11/385), **11November2026**,
provides a recent physiological dual-stream example. Its nestedLOSO motor-imagery result
is not our three-strategy task. [EEGMamba](https://arxiv.org/abs/2407.20254), July/October2024,
uses subject-grouped fivefolds; these dates are not recent-days discoveries.
The original [Mamba paper](https://arxiv.org/abs/2312.00752) motivates selective dynamics;
linear scaling is not decisive for only480samples. Content-dependent temporal organization is.

**Actual code/rights/readiness.** New paper's [official repository](https://github.com/div7OWO/NeuroMambaNet)
revision`11aed8f7f220dcf0d3104d4563b4c84c968a728b` exists with checkpoints and strictLOSO
script. APIlicense=None and noLICENSE file: do not copy its model/weights.
Its released MAMBO uses hard-coded22channelMamba2 blocks and fixedchunk sizes; it is not a
41channel drop-in. The split script admits validation-subject identity is unavailable in
prepared pickle blocks, limiting independent split audit. Stock
[Mamba code](https://github.com/state-spaces/mamba) is Apache2; exact version/extension and
worker compatibility still require a separately pinned preflight. Current env lacks
mamba_ssm and pytorch_wavelets. No cost estimate is presented as measured.

**Concrete bounded study proposal.** Learn41→64sensor projection; local temporalstem;
two bidirectional selectiveSSMblocks; mean+variance temporalreadout; identical source-fixed
amplitude scale and explicit rawmean/power path. Hiddenstate resets every currentwindow.
All480current samples available in both directions, no across-window recurrentmemory.
Compare matched-depth/parameter TCN and fixedSSM/nonselective blocks, same2seed schedule,
same rawbranch and normalization; include with/without causalSessionBN as an explicitfactor.
No hard-coded motor-only beta gate; MI/SUB/WORD need different spatial/spectral cues.
First smallsource checkP02/P03 after realfit smoke: whole65%[50,78], selective−matchedTCN
+1pp[−4,+6], no claim source effect is outergain. Refute selective mechanism if fixedSSM/
TCN matches it under unchanged rawbranch; check source optimization before family conclusions.
Planningbudget oneA100≤15min sourcepilot after extensionpreflight, not yet submitted;
actualB1mustbe measured. This is lower readiness than Atom-L and genuinely different learning.

### 3. Spatial topography tokens and their dynamics: executed small §328 probe

**Changed assumption.** Represent each sample as one of sixteen source-learned scalp
patterns, with polarity-invariant topography, then classify occupancy/dwell/orderedtransitions.
Covariances do not uniquely specify a non-Gaussian state sequence.
[Atoms of Thought](https://arxiv.org/abs/2605.20182), **19May2026**, is motivation:
its sixchannel/1000cluster sleeppretrained tokenizer and CNNreadout differ from our pilot.
Its public article does not supply an inspected reproducible tokenizer artifact or establish
our person-held protocol. No paper code, weights or restricted dataset was used.
A [mental-arithmetic microstate study](https://www.nature.com/articles/s41598-020-79423-7)
(11January2021) concerns performancegroup discrimination, not MI/SUB/WORD.

**Actual protocol.** P04–P10fit only;2048GFPpeak samples/person,16sphericalcentroids,
seed328/thirtyupdates, currentwindowCAR+fixed2–30HzFFT, no query labels duringfit.
QueryP02/P03 are previously exposed sourcepeople; P01 excluded throughout.
Raw287mean/sixbandpower is identical across concatenated controls. Fixed balancedLR C=.01,
sourceStandardScaler and official protocol.score use interfacecells. Within-window tokens
are stateless across windows, so any deliveryshuffle has exactly the same posteriors.

| Whole source decoder | P02cellBA | P03cellBA | Two-person mean |
| --- | ---: | ---: | ---: |
| raw287 |44.775|44.218|44.497|
| occupancy only |39.673|30.308|34.990|
| raw+occupancy |46.321|50.850|48.585|
| raw+ordered occupancy/transitions/dwell |43.785|49.756|46.770|
| samefullfeatures, shuffledtokens |47.799|44.079|45.939|
| samefullfeatures, randomcodebook |43.196|44.982|44.089|

Ordered−rawoccupancy **−1.815pp** misses+1[−4,+6]prior in direction; ordered−shuffled
+.831pp has opposite signs across people. Learned−random+2.681pp is positive in this
weakdecoder; occupancy-only is weak. Whole46.77misses65[50,75]prior. The previous same-source
rawSessionBN context in§324 is62.30%pooledpersonBA, a different aggregation, still muchstronger.
No evidence that orderedmicrostate dynamics solve the current system's residualerror;
noouterexpansion. This exact weaksummary form is negative; learnedtoken sequence models,
polarity-sensitive representation and adaptive-spatialmapping remain different untestedforms,
which need evidence beyond keeping a family nominally open.

**Execution/evidence.** Initial3117279 failedimport beforelabels because snapshotprotocol
expected benchmark_utils on its path. Immutablev2 corrected explicitpath;3117280 completed0:0
in36s. No-score real-data invariance/prefix/amplitude/shuffle/finalsample/unitcentroid
checks passed before query labels. Sourcehash`4dade504b4011b6697c3a2a78d073b342b05cce70581ec24da173e950b1c3527`.
[Result](microstate328_source_results.json);34.25smeasuredwork,1.50GiBpeakRSS;
CPUwarmB1median1.863ms includes allsixfeaturebanks plus orderedhead, a conservative pilotpath.
Not a GPUworker/package runtime or quality claim. Sourcefiles111/87lines.

## Independent validation instruments and selection limits

- **Within PROTEUS:** sourceP02/P03 have been exposed repeatedly. They can diagnose
  mechanisms but cannot independently confirm the selected architecture. A newall10LOSO
  matrix is still development evaluation on familiar people. Preserve actualpeople/order
  bootstrap and officialcell weighting, reportP04/P10 separately; neither alone closes a root.
- **Shin2018 cognitive data:** usable as a source-held corpus/person test of semantic
  MI/SUB/WORD mechanisms, beyond adding trainingrows. §317 already uses external cognitive
  supervision, so prevent reusing those fitpeople as independent validation; reserve explicit
  people beforefit and audit actualchannelinterpolation. Exact heldinventory/terms required.
- **Scherer2015:** exactthree-strategy externaldomain check possible with organiser clearance,
  but sourcehighpass≈4Hz and19sharedrealchannels make it a deliberately different instrument.
  Failure there does not refute lowfrequency PROTEUS information. Its CC-BY-NC-ND exception
  remains declaration-required; this worker downloaded nothing and made no new legal claim.
- **EEGMAT:** subtraction-versus-rest validates sensitivity to cognitive engagement, not
  threeway commands. Atom's corpusinventory includes it, so it cannot establish zero-shot
  unseen-corpus transfer for Atom. Holdpeopleout of our adaptation; disclose upstream exposure.
- **Other MIcorpora:** useful to diagnose sensor/normalization/pretraining defects, but their
  hand/foot labels cannot substitute for cognitive-command validation.

## Decision and retrospective

Allocate next representation effort to the new released Atom prior after its actual preflight,
then a controlled small source experiment and credible whole comparator. Keep raw selectiveSSM
as a separately engineered root; do not copy an unlicensed hard-coded22channel release.
The microstate pilot supplied an actual changed-information test and a negative result;
its cheap latency does not justify further accuracy matrices without a stronger form.

Recurring failure to avoid: weakwholemodel+positivecomponent is repeatedly mistaken for
headroom against the strongsystem. A family requires its own coherent adaptation and a
credible matchedcontrol; source-only success merely earns the nextregisteredquestion.
All current submissions, officialsubmodule and root-owned STATE/STATUS remained unchanged.
This extra phase-boundary review does not replace the nextscheduled06:51CET researchblock.

## Operational handoff after the timed review

- §329 preflight **3117332 pending(Priority)** at04:30UTC, immutable
  `/mnt/lustre/work/gehler/gfh527/ni26_labram329_v3_20261117`, source
  `936aecfa1040d3c75c767feef448da3f17662b5f908587b59d8a2405ec21118f`.
  Earlier setup-only pending3117282/3117323 were canceled before allocation to preserve
  strict pretrainednorm loading and isolate missingtimm; they produced no score.
- §329 separately registered frozen sourcepilot **3117338 afterok:3117332**, immutable
  `/mnt/lustre/work/gehler/gfh527/ni26_labram329_source_v1_20261117`, source
  `0e37013afcb5a071113dc2da284e52c36a6895e5a3232e339db9d60ebea68199`.
  It refuses missing/failedpreflight and verifies immutableassets/code/dependencies.
  Complete9heads plus originalsame-source40epochrawreference precede querylabels;
  pooled400/retainedfourtimepatch1600 × pretrained/random × explicitrawbranch.
  These are queued states, not completed checks or accuracy results.
- Current private-snapshot imports and own source files compile locally. This does not
  establishA100 compatibility; that is the pendingpreflight's explicit purpose.
- No automaticouter/fullFT/newpackage follows either sourcepilot. Inspect actual guards,
  referencequality, both exposedquery people and source/revision hashes before decisions.
