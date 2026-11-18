# State: the starting point

This is the entry point. It is written so that a new session needs nothing from any chat history. Where an older
document contradicts it, this file wins.

**Read in this order:** this file; [LEDGER.md](LEDGER.md) (index of what is adopted, built, open and closed, and where new
runs are pre-registered; the full records are PREDICTIONS_11NOV.md and PREDICTIONS_12NOV.md);
[SEALED_UPLOADS.md](SEALED_UPLOADS.md) (upload log and procedure); [../OFFICIAL_CONTRACT.md](../OFFICIAL_CONTRACT.md).

## 1. Where we stand

**18 November scoring audit:** historical shuffled full-system comparisons retain recording-order
transformer and shallow EA inputs. Their `STRICT` guard exempts these members. Treat those comparisons
as **historical full-system DEV with partial delivery replay**, including the three-seed cap250 stock.
Exact identity checks do not establish full delivery causality. Repeated public/external cohorts and
new orders also do not provide fresh-person confirmation. The training-cache repair and next validation
priorities are in the [scoring audit](../evidence/next_wave_16nov/scoring_audit_18nov.md).
The packaged solver's causal behavior is a separate question; no staged ZIP changed in this audit.

[Friday research queue, comparisons and first results](../evidence/next_wave_16nov/FRIDAY_20NOV.md).

### Wednesday 18 November lunch decision

[Overnight checkpoint, live jobs and noon candidate decision](../evidence/next_wave_16nov/OVERNIGHT_18NOV.md). The current plan calls for a penultimate candidate by Wednesday noon and reserves Friday evening for the final decision. The sealed best remains v5.1b.

**Wednesday lunch recommendation: v6.6cap250.** With three eye and three rest seeds, its historical full-system gain against v5.1b was +1.380 pp on ten held new people and +0.757 pp on the assumed returning-person mix. The combined public sensitivity is +1.110 pp. Sealed v5.1b remains 81.9767%. A bounded learned-history addition to cap250 yielded +0.021/+0.002 pp, with both intervals crossing zero. The protected JS policy lost 0.556 pp against cap250 on nine external Scherer participants. See [paired public evidence](../evidence/next_wave_16nov/aux431.md), [full-system memory](../evidence/next_wave_16nov/history443.md), [protected JS](../evidence/next_wave_16nov/protected436.md), and [verified package and Friday decision](../evidence/next_wave_16nov/lunch_recommendation_18nov.md).

The dedicated [whole waveform decoder](../evidence/next_wave_16nov/wave379.md) scored 41.69% with learned memory, versus 44.12% with a matched static decoder and 76.28% with the full frozen stack. A separately matched [multiscale decoder](../evidence/next_wave_16nov/scale380.md) scored 44.74% versus 44.84% for learned waveform filters. These are ten-person public development results. [Recent methods and protocol comparison](../evidence/next_wave_16nov/recent_methods_18nov.md).

The [A6-initialized joint decoder](../evidence/next_wave_16nov/joint382.md) answered a different question on four held people. With the original filterbank fixed, training the spatial convolution, session normalization and classifier together gave 59.705% on Galvani and 58.125% with a second seed on Ferranti. True session history helped by 5.892 and 5.293 points versus blank history. Original explicit A6 scored 60.785%, and the full frozen system scored 81.216% on the same people. The native C++/CUDA implementation and these experimental PyTorch CUDA fits have separate validation records.

The [lunch recommendation](../evidence/next_wave_16nov/lunch_recommendation_18nov.md) records exact package hashes and submission conditions. On nine external Scherer people the exact candidate gained 0.493 pp over v5.1b on the first two orders and 0.040 pp on two fresh orders. Both paired intervals include zero. The [whole-worker pruning experiment](../evidence/next_wave_16nov/qualification408.md) saved about 2.4% on two paired A100 runs. The staged ZIP retains its original implementation.

### Current campaign: 17 November 2026, late CET

The user authorizes ongoing implementation, broader experiments and justified follow-ups on Galvani.
Top five remains the objective.
Account limits:32jobs running/272submitted with28reserved; preserve unrelated jobs. Never upload automatically.
Report every gain as FULL-SYSTEM DEV, SEALED LEADERBOARD or INDIVIDUAL-MODEL, with baseline/cohort. For competition updates, give the corresponding 0–1 leaderboard units, the comparator and the dated entry-rank implications where available. Label transfer assumptions and keep standalone gains separate from full-system score scenarios.
“Custom stack” means C++/CUDA. “Original raw neural reference” means the stock PyTorch336/338 model.
Use the [current handoff](../evidence/next_wave_16nov/HANDOFF_17NOV.md) for a fresh session.
The [solution-space review](../evidence/next_wave_16nov/solution_space_review_17nov.md) records
current scoring priorities and the evidence that would change them.
The [earlier detailed status](../evidence/next_wave_16nov/STATUS_17NOV.md) captures past sources and jobs.

**Submission and rank.** Uploaded v5.1b scored .8197668759 in 1,287 s. The official leaderboard mirror updated
17 November at 16:44 CET and displayed .841 at fifth place, about 2.12 percentage points above our score.
Counting the two ETRI_EnSeption entries once gives a provisional fifth-team benchmark of .837, a 1.72-point gap.
Prepared v6.6cap250 is unchanged: `runs/sealed/final_zips/v66_cp1_cap250.zip`, SHA256
`9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed`.
The v6.6 .824–.829 sealed-score scenario is unvalidated.
[Official leaderboard](https://neural-interfaces26.github.io/leaderboard.html#bci)

**Best measured full-system change.** Selected§318 CC30FB-slot replacement plusFBLfactor2:
76.2177→76.4410% new-person FULL-SYSTEM DEV, +.2234percentagepoints[80%+.0738,+.3512];
people/order+.184[+.0323,+.3327],7/10up. Actualsix-seedB2_pool deployment comparison is
76.2793→76.5517%,+.2724officialpoints[+.0049,+.5271]; people/order+.278[−.007,+.546],6/10up.
Selected/reuseddev, not independent confirmation, returning-person validation, sealed gain or proven synergy.
Completed305/308/310/312/313/315/316/317 full-system studies establish no useful control gain.
Original301training-head and307independent-refit parity failures remain failures; later diagnostics do not erase them.
Independent312CPU/GPU matrices remain separate; no mixed folds or duplicate CPU consumer.

**What the recent mechanisms establish.** §323complete80fits: adapting session statistics improves individual
raw-decoder accuracy by3.898/4.588points, but every added-member arm−control full-system interval spanszero.
Two-seed ensembling preserves that individual gain; it is already absent beforeCP. About84–85% of newmember
corrections overlap ensemble successes. Calibration improves; overlap is descriptive, not a sufficient cause.
§332same-weight mean/variance/donor intervention reproduces originaloutputsBITWISE: variance-only carries
+3.921/+4.607INDIVIDUAL-MODELpoints, mean-only≈0. Selfstats beat source-donor. ActualONEBNlayer only.
§333label-free probes: pretrainedAtom retains diagonal information better thanrandom under a fixedlinearprobe,
but offdiagonal prediction usually fails to beat source-mean. Full44239targets retained; no information-absence proof.
§334generative adaptation49.891% loses to its static58.875% INDIVIDUAL-MODELcontrol. §335source-held
calibration improveslogloss versusuniform, but geometry/permutation mixture loses tosemantictranslation.
Residualscaling can activate probabilities while selecting nullgeometry. No generative-family closure.
§336originalrawneuralreference61.776%, relaxedhead62.827%, classweighted62.343%, changedbatches62.932%
on exposedP02/P03. §338combinations62.516/62.013/61.973% are belowhead-only62.827%; allhurtP03.
P09source-held reference64.061/head62.747/head+weight65.179%, so the earlier headlead does not transfer uniformly.
§337P09 INDIVIDUAL-MODEL reference64.061%, covariance-pretext63.736%, moment62.512%, shuffled61.669%,
extra-supervised63.441%. Covariance beats weaker pretexts but not the originalreference; no FULL-SYSTEM gain.
Post-finefit reconstruction does not improve. Oneperson/seed/64steps; missingprefineweights limit attribution.
Earlier visual/EEGpriors, spatialoccupancy and rawstate-space pilots did not beat strongerreferences in testedforms;
these are limited forms, not architecture-family ceilings. §340finiteFIRbank is fullrank480/condition2916:
frequencyattenuation is not mathematicalinformationabsence before compression. CARsensitivity is not taskcausality.

**Completed full-system normalization test — §339.** All20nestedsource8seedbanks/30source-onlytemperatures,
48members/64cells/384CPoutputs and originalsource/zero/order/route/endpointguards pass; scorer3118434 COMPLETE.
Control76.217673%; rawfixed/both/variance FULL-SYSTEM DEV deltas+.03846/−.03295/−.03086pp;
calibratedfixed/both/variance+.04531/−.05331/−.04812pp. ALLsix people/order intervals spanzero.
Calibrated−raw fixed+.00685/both−.02036/variance−.01726pp are uncertain. No established system gain.
Two-seed INDIVIDUAL-MODEL fixed54.727/both58.782/variance58.817%: the ~4pp gain did not stack here.
Probability/decision changes rule out an inactive addition; they do not explain the attenuation causally.
Actual24.06CPUcoreh/.2075A100h. New-person routingemulation only; no packagedreturning/sealed result.
[Owner handoff](../evidence/next_wave_16nov/calibration339_full_handoff.md) owns all contrasts/source limits.

**Candidate qualification — §321.** CorrectedrawEAprior5 versuslatentBNprior2 proofs/all6all10models pass.
JS-all preserves innergates/FTlabels but changes outermix even afterrecognition. No finalrecognizedidentity claim.
SharedFIR/inputreuse preserve original45active nets and20inactiveweightsets; controlledCPU/GPU semantics exact.
CompleteA100known/unknown190/378/800functionality passes; warm12AB/BApairs pass6.10%engineeringbar,
medianoverhead2.43/1.19/2.62%, max4.91%. Oldfailedbitwise/timing/CPU45mintests remainfailed.
LongCPUfunctional3117756/7 COMPLETE2h27/2h32; collector3117884 passes. No CPU timing or sealed runtime proof.
All60NEWgenuineheld-sessionPRIMARY-NNweights/72traces and576B1SIFbanks pass source/export/reset/order guards.
Bothfold-specificsourcecontexts pass. Historical checkpoint identity remains unproven, not silently substituted.
ActualRouteB/TSEM/OwnStage/CF/CP engineeringbridge3118487 uses originalfrozenclasses and explicit contexts.
BothneworiginalsourceCore/PCAfits/nativeSIF raw31canaries/full8order raw256banks3118623 pass.
All28newnormalBN sourcefits+originalShallow/FBCexport guards, newcontexts/sourcegalleries and source-onlyteacher
taskmaps pass. Actualprivate3arm/source-inner/recognition/FT/rawview/softwareguards pass; completeconditional comparison runs.
Root explicitly chose FIXED-PRODUCTION-PRIOR CONDITIONAL ENGINEERING sensitivity: frozenMAD/teacherEA-G/
OOFcalibration AND originalrest/eyeauxiliaryNN remainfixedproductionpriors. Primary/innerNN sourceexclusion does
not covertheseauxiliaries. A positive cannot establish held FULL-SYSTEM DEV/rank/promotion. No score yet.
Fullsourcebridgeguard3118899 PASS6m20, off/base/inner/gates/FTlabels/reset/prefix exact; JSrecipe≤2.22e−16.
Initialcellcohort failed own6-versus4cell metadata assertion BEFOREinference; preserved, notofficialmetricbug.
RepairedALL16exactinputpreflight passed. ExplicitCPU fullcell3119222_0 PASS12m56/RSS4.04GiB; bothsessions
originalinner/off/recipe/raw256/prefix/immutable guards pass; rec60/4FT andrec10/0FT actuallyactive.
ALL16CPUcells/collector3119253 COMPLETE; ALLsource/frame/zero/rawview/row/8orderguards beforegold, replayOK.
CONDITIONALreturning primaryUNIFORM8orders: P04base89.6958→JS89.0308%(−.6650pp), P1095.9603→95.7666%(−.1937pp).
Returning.2/.8weighted−.2879pp,95%DESCRIPTIVEpeople/order[−.8766,−.1131];7/8weightedordersnegative.
Bothdecline, but conditionaltwopersonresult isnotindependentgeneralization/sealedharm/rankproof. Keepnewperson
selected+.2724separate. Equal-scheme1/3 supplement is read-only/separate; no primaryswap or promotion.
§350 COMPLETE conditional returning policy: JS only while runtime-unrecognized reduces the JS-all regression,
but remains below baseline: weighted −.11934pp[95%descriptive−.28038,−.00109], P04−.19741/P10−.09983.
Versus JS-all +.16857pp; same retained-production exposure, not independent DEV/sealed/rank evidence.
All16 whole cells, original guards and replay pass. No further policy variants or promotion.
[Owner evidence](../evidence/next_wave_16nov/js321_returning.md) records these measured limits.

**Custom stack and new readout exploration.** PairedCPUclassical-core63–64ms versus cachedPython113–117ms
is a narrow~1.8× inference result. ResidentA100linearhead1.062s versus CPU238.483s changesdevice/trajectory.
§341same-binaryCPUrepeat80.35s(1thread)/45.47s(6), historical243snotreproduced;96.4%time ismatrixmultiply.
One-linefinite-validation patch preserves exactmath, improvescheckedAPI2.26×/1.36× and head-onlyB12.30→.91ms;
fullCPUsolve41.96→41.59s(singlepair), no practicalfullsolve/core/accuracy claim. Future sourceonly, activebinariesunchanged.
GPUresident/batchedreuse remains a plan. Neuralnetworktraining stillusesPyTorch, not a completeC++NNrewrite.
§343 CPU repair COMPLETE, explicit future `FullReference` API: fusedC++value/gradient with SAME existingSciPy
L-BFGS-B/C=.3/stops, notfullyC++NNtraining. WarmP01sklearn19.7467→16.4362s(−16.76%), CPU117.3624→93.3867
coresec(−20.43%); independentP10cold15.4332→10.7565s(−30.30%). ALL1534completeheld-stream predictions
(P01/P10, within0/full0) matchlabels, maxprob6.65e−8; source/fullgradient/finite/scaler/oldPythoncanaries pass.
FreshsampledfitRAMincrement77,796→71,748KiB(−7.77%); singleprobes/differentloadedRSS preventgeneralabsoluteclaim.
Old243snotreproduced; oldtighter40–47s and v1heldparityfailurespreserved. QualifiedV4SHA7a2715ad…939bd,
reportsolver343.md/handoff. Allownjobsdone/2.451CPUh/0GPU; runningbinaries/submissions unchanged, noaccuracygain.
§342 COMPLETE57s/32fits: pretrained pooled44.630/temporal40.086/spatial51.347/full44.199% onexposedP09.
Spatial−pooled+6.717 INDIVIDUAL-MODELpp, but below strongerhistoricalraw64.061%; no systemgain/promotion.
Randommatchedspatial41.002%; exactsource/CPU-GPU probabilityguards pass beforelabels. Oldbatchgeometry
featureanchor failure preserved/explained; no historicalrandomidentity claim. Reportspatial342.md owns scope.
§344 COMPLETE: P09 INDIVIDUAL-MODEL originalA64.061/sequentialB63.736/concurrentC61.175/derangedD61.891%.
C improves finalcovariance-probe error versusallcontrols, yet loses2.886ppvsA. Betterretention didnotimprove decoder.
All16snapshotprobes/strongA,Bidentity/source guards pass beforelabels; mixedtrajectory, nogeneralforgettingclaim.
Actual3.948CPUcoreh/0A100; noautomaticweightgrid. §345 COMPLETEhonestnestedteacher taskdistillation:
P09INDIVIDUAL-MODEL raw64.061/KD61.665/deranged63.051/teacher63.382%; actualteacher<raw, nothistorical69%.
Allsource/gradient/1240step/RNG/eightorderguards pass, allocated2.186CPUh/0GPU. NoFULL-SYSTEM DEV/rankgain.
§346 COMPLETEdirect-summarypilot: A6raw64.061/matchedrefitB60.686/C+369signedslow+highpower65.447/
run-derangedD60.640% P09INDIVIDUAL-MODEL. C−A+1.386pp,7/8orders; pooledNLLworse, notfullsystem/calibrationgain.
OriginalAall8BITWISE, allguardsbeforelabels; .151CPUh/0GPU. FrozenNO-REFIT extension3119224 COMPLETE:
P02A65.646/C64.164%(−1.482pp); P03A56.227/C51.391%(−4.837pp). Samefrozenmethod/source6; allguardsbeforelabels.
P09positive DIDNOTtransfer; thisreadoutnotready/fullsystemexpansionunjustified. §347late-supervision follow-up below.
Sourceinventoryfinds7,592nominalseconds outsideofficialcrop, NOTgloballyuntested: oldEA2/3scroptraining and
§277SSLstart4alreadyusedsome. §347newquestion is exactrawSessionBN/step-matchedlate versusrepeatedofficialsupervision.
Sourcequalification3119303 PASS: ALL60EDFs/1898officialrows matchtensorbounds, A6sourceprefixBITWISE;1894actualended
late3.5–7.5crops qualified,4unanchoredextrasomittedBOTHextraarms. Earlierrounding-orderfailurepreserved/diagnosed,
no loosenedbounds. NativeofficialG/scale fixedallarms, noqueryfits/labels. Sourcecontrol3119319 PASS59s:
Aref/AwrapperBITWISE, B/C31base+31eligibleextra batches/RNG/labels/chunks/initializer match, activebody/head.
Actualentrypreflight3119346 PASS; fullB-repeat/C-late40epoch2480step fits3119363[0,1]%2 released→3119365collector.
§347FULLCOMPLETE: originalA/repeatedofficialB/lateC INDIVIDUAL-MODEL P02 65.646/64.582/50.597%,
P03 56.228/54.872/54.277%, P09 64.061/64.881/54.289%. C−B−13.985/−.595/−10.592pp, all3peopleworse.
Allmatched2480step/source/initializer/40base-batch/72bankguards beforelabels; source-originalNLLC.5455>B.3552.
Latepoolingharms thisform, extraofficialupdatesdonotexplainlargeharm; no rawdataabsence/familyclosure/systemclaim.
Actual1.897CPUh/0GPU/alljobsdone; fullfit+deepcopyRSS~6.7GiB exceededearliersmoke3GiB; bothstayed24GiB.
§349COMPLETE: own-official versusmatchedmixedBNprior changesP02−.808/P03−.138/P09+1.087INDIVIDUAL-MODELpp;
stillbelowstrongAby15.857/2.088/8.685pp. Genericrecalibrationchanges0accuracy; originalB/CALL16banksBITWISE.
Priorreplacementdoesnotrecoverlate-trainingdamage, norclosealllate-data mechanisms.
Sourceprecisionlimit: legacyAlignedNet precomputesPOINTWISEcovsforallpublicX(includingheld), notcrossrowstats;
rawunalignedinferenceignoresheldcovs, G/scale/weights useONLYsource. Oldheld_transformed:false meansnoheld
NEURALforward/directfeatures/rawEDFprocessingbeforegates, notliteralnopointwisecovcache; immutableproofsunmodified.
§348CMxmetadataqualification COMPLETE:8308rows/201recordings/24people mapped;16guaranteednativechannels.
EveryMIcrop.5sannotatedsound/MA2–3s: condition/promptconfounded, nouniversalrule/WORDclaim oraccuracy.
Actualold12m8cpretrainedteacherusedcorpus; old+1.67wasincrement, notteacheraccuracy. §351semanticmatchedqualification COMPLETE: ShinWG≠SUB; SchererWORD/SUBsame-run metadataqualified.
§352COMPLETE frozenPROTEUS-only→Schererbinary: covariance59.030% versuspower53.260%,
EXTERNALWORD/SUB+5.770pp[80%people/day/order+3.958,+7.541],8/9peopleup. SourceLOPOcov66.899%,
10/10abovechance; sourcegatepassed beforeexternalgold. Allbanks/source/reset/orderguards pass.
Clinical/language/acquisition/QC/campaignselection/19-channelcapacitylimits remain; noFULL-SYSTEM/rankgain.


**Current work: lean native baseline plus independent accuracy waves.** The image-specific C++
worker matches the frozen inference and adaptation replays in §366. Its public A100 speedup depends
on the workload. Complete CUDA training parity and sealed worker timing remain open. Scientific
experiments continue on frozen public DEV.

**Full C++ port — §353.** Native artifact/DSP/cores/tails/routing/Own/CF/CP/neuralbank/FT/masterruntime exist.
All earlier failed builds/proofs remain preserved. CPU neural known8/unknown31 now PASS all originalbounds:
3121389 fixes FT Adam operation order, all308 paired source trace fields BYTE_EXACT;2refits40steps/2relabels,
reset/copy/source/activity guards pass. This is component CPU proof, not whole-runtime/long-session/A100 proof.
CAR96source arrays, REALcore2278source fit, initializer52state tensors, sampler1240batches/40epochhashes/
41RNGstates and PCAmean/basis/ALLsingulars/sourcePC qualify at their recorded scopes. PCAoneRNNRpair is
only3.4–3.6%lower API time; memory inconclusive. Raw source prep3121547 reproduces all1898float32sourceX
BYTE_EXACT, Gerror2.956e−12, originalscale/order/session/exclusion guards pass. Dataset/source6 metadata
uses explicitP04/P05/P06/P07/P08/P10, unsortedfirstseen sessions; P01 is outside initialsource7 inventory.
RawFBC source training composes native preparation, initialization and sampling. Job 3121976 fixed the FIR layout
and matched all gradients and state tensors at the first CPU update. The six-update check 3122002 failed at step 6.
At step 2, identical gradients and second moments isolate the first difference to the AdamW first moment. Using the
reference lerp on the recorded inputs restored exact step-2 moments and weights. Shared Adam repair 3122019
matched every checked field at updates 1, 2 and 6. Complete CPU fit 3122086 matched all 1,240 losses and 17
final tensors exactly. Both checkpoints gave identical posteriors and labels for 31 unseen P01 windows through
the reference causal decoder. CUDA training and a matched raw-to-fit speed comparison remain open.
[Training evidence](../evidence/next_wave_16nov/native353_training_prefix.md)

The complete C++ runtime passed the 31-window CPU source prefix in 3121996 after a fresh-session copy repair.
All labels matched, final scores differed by at most 1.14e−8, recognition matched at index 10, and interleaved
copies and reset replay were exact. This case recognized early, with no FT/CP refits. Lean package build 3122004
separates offline training from the two deployed shared libraries. An install-only repair removed the host RPATH.
Clean worker replica A100 adapter check 3122018 passed exact 31-window label parity on both repeats.
Warm native inference took 11.65 s versus 4.50 s frozen in this pair. Profile 3122028 locates 6.65 s in
the neural bank on one window. Worker counterfactual 3122043 changed only TorchScript clone to deepcopy
in the stateful neural bank. Warm replay fell from 11.77 s to 5.09 s on the same A100, all 31 labels exact.
Imported TorchScript is a runtime migration artifact. All-family training, gallery fitting and full orchestration
are open. [Earlier worker evidence](../evidence/next_wave_16nov/native353_worker_prefix.md)

**Current-source A100 worker benchmark — §360.** Rebuilt package 3122062 passed installation.
The worker replica matched every label and activity counter on two fresh copies of 31 ordinary and
64 forced-unrecognized windows. The latter performed five fine-tuning refits and three personalization refits
per copy. Job 3122084 matched all 2,655 outputs across 14 shuffled sessions. All recognized at index 10.
C++ replay took 449.87 s versus 415.08 s frozen on the same A100. Throughput fell 7.73%.
Long forced adaptation job 3122085 matched refit counts but failed exact parity at 1/378 and 4/800 labels.
Each arm varied at one index on the shared prefix across fresh copies. Diagnostic 3122389 reproduced the
same cold/warm differences with deterministic CUDA on a second A100. With TorchScript graph optimization
disabled in both arms, job 3122401 matched all 330 active-adaptation labels across two fresh repeats. The
port has yet to free evaluation time against frozen under default settings.
Neural posterior transfer batching matched both AB/BA pairs and saved 0.45–0.95% of replay time. Separately,
parallel tangent cores matched both pairs and saved 1.39–5.66%. The two optimizations missed their paired
3% thresholds. Corrected vectorized filtering in 3122414 matched exact filter bytes, labels and counters in
both run orders and saved 5.76–6.40% versus the prior C++ build. Direct frozen comparison 3122422 matched
all 578 labels and counters in each order, saving 1.77–2.17% of replay time. The extended diagnostic 3122423
with both graphs unoptimized matched 378 labels and differed at 2/800 despite equal activity. Full default
timing 3122513 matched all 2,655 labels/counters. Solver replay was 411.78 s frozen versus 413.01 s vectorized
C++ (+0.30%); wall from input loading was 416.84 versus 417.42 s. The broader pair yielded zero headroom.
Trace 3122514 showed numerical drift entering the outer mixture after FT and amplified in personalization.
The first five hard refit targets matched in 3122538. Python scalar CUDA Adam reduced the 150-window
outer-mixture error by 27%. In a 70-window diagnostic with scalar Adam and exact shared fit inputs/weights,
C++ matched the Python outer mixture to 2.17e-8. Default long SIMD worker 3122604 reproduced the prior mismatch
at 1/378 and 4/800 labels; counters and status logs matched. It replayed in 297.42 s versus 308.30 s for frozen.
Exact parity failed. This does not qualify added evaluation budget.
[Current evidence](../evidence/next_wave_16nov/native_worker360.md)

**Adaptive repair — §366.** CUDA foreach Adam and rest SessionBN deepcopy give exact 378 labels in
3122631/33 and four residual 800 mismatches despite equal counters. The actual shared-input 70-window
optimizer test matches frozen outer scores to 2.17e-8. Ordinary pair 3122643 matches 2,655 labels and
work counters and saves 2.85% replay time (427.32→415.14 s). This is about 103 seconds per ordinary
solver hour at the measured mix. The active long workload is still unqualified. The largest source of
observed confidence drift is the Hilbert core, with band-filtered input differences preceding FFT.
Float32 reflection is corrected in d0acaa8. The private pad+DUCC pair 3122768 matches all 2,655 ordinary
labels and counters but takes 0.78% longer (406.07→409.24 s). The exact complex reduction brings the
long replay to 378/378 and two disagreements at 800. C++ CP on identical inputs matches all 1,178 labels
below 1e-12. The worker overlay with corrected Euclidean alignment restores all 1,178 labels and every
work counter against a fresh frozen arm in 3122783. Same-A100 solver replay takes 314.67→304.52 s
(3.23% saved, 116 s per matched solver hour); wall from public data load takes 319.64→309.61 s.
Same-build ordinary pair 3122794 matches all 2,655 labels and counters on 14 sessions.
Replay takes 402.22→397.99 s (1.05% saved, 38 s per matched solver hour); wall takes
407.58→402.35 s (1.28%, 46 s per matched wall hour). The additional real-centering control
3122787 also matches every long-stream label and counter, with no ordinary timing.
[Exact worker evidence](../evidence/next_wave_16nov/native366_worker.md)

**Native experiment throughput — §368.** Two independent short active replays share one A100
with exact labels and work in 3122839. C++ takes 142.50 s serial and 108.62 s overlapped.
Frozen takes 147.58 s serial and 116.32 s overlapped. Peak device samples during overlap
are 3,039 MiB C++ and 3,927 MiB frozen. Both fit. This measures development throughput
for 96-window jobs, with full sealed execution still separate.
[Native experiment evidence](../evidence/next_wave_16nov/native_iteration368_369.md)

**Native policy changes — §369.** The experiment varies one existing artifact scalar without a
library rebuild. Native skip matches stock bitwise over eight held-person DEV orders and reduces
measured personalization-stage time by 4.1%. A first fit at window 90 changes DEV by +0.0128 pp
[80% −0.1079,+0.1350]. Both variants match the 190-window A100 anchor. Unequal template warmup
leaves whole-worker speed open. One short ordinary CUDA profile (§370) measures 3,075 launches
per window, with launch APIs accounting for 14.4% of profiled wall. Graph or fusion speed is open.

**Parallel accuracy — §354–356.** §354 complete after two fixed-route recoveries. Gaussian factorization lost to
the strong raw decoder by 7.44 pp [80% −10.28, −4.62] across three exposed new-person folds. Gaussian versus
the matched learned factorizer is undecided. [§354 report](../evidence/next_wave_16nov/factorization354.md)

§355 complete after the single-window replay guard. Lag 42.742% versus instantaneous 39.338%, aligned
envelopes 37.076% versus offset control 40.272%. All four whole decoders scored below 55%. The tests
leave open the use of lagged relations inside a strong decoder. [§355 report](../evidence/next_wave_16nov/relations355_results.md)

§356 complete. On the six-base FULL-SYSTEM DEV measure the unchanged chain scored 76.2177%, the PROTEUS-only
WORD/SUB correction 75.7174%, and the external correction 75.3670%. External versus PROTEUS-only −0.3504 pp
[80% −0.5645, −0.1621]. MI labels stayed exact. Set aside this fixed terminal fusion. [§356 report](../evidence/next_wave_16nov/external_binary356.md)
The [coverage audit](../evidence/next_wave_16nov/exploration_primary_sources_17nov.md) also identifies an
unordered-prefix decoder, matched-cost control and lagged relations in a strong decoder.
§357 source-only readiness 3122027 passed exact prefix and reset guards plus all eight label-blind orders.
It used one source person for the new readout. Held accuracy awaits a cross-fitted comparison.
[§357 card](../evidence/next_wave_16nov/causal_prefix357.md)
§358 repaired four-person label-free bank job 3122038 passed all 757 rows and 32 causal banks.
All 24 archived comparisons retained identical labels. Four-person held readout comparison §359 completed
as job 3122050. Untouched A6 scored 60.785%, current-only 59.187%, and prefix 59.676%. Prefix versus
matched control +0.489 pp [80% −0.052, +1.037] and versus untouched A6 −1.109 pp [−2.296,+0.088].
This exposed INDIVIDUAL-MODEL pilot gives no evidence of a full-system gain.
[§358 evidence](../evidence/next_wave_16nov/causal_prefix357.md#358-four-person-input-qualification) · [§359 card](../evidence/next_wave_16nov/causal_prefix359.md)
§361 fractional CSP completed all 48 full frozen-chain controls bitwise. The intervention changed 934 decisions
but moved the six-base new-person score by −0.0015 pp [80% −0.0469, +0.0490]. On B2_pool it moved −0.0571 pp
[−0.1761, +0.0747]. Returning equality is structural on the earlier conditional replay. Keep the standard
estimator. [§361 evidence](../evidence/next_wave_16nov/fractional_csp361.md)
§362 measured a B2_pool label-assisted choice upper bound of +1.60 pp versus base and +1.33 pp versus JS.
The branches disagree on 682 of 21,240 predictions. The returning proxy bound was +0.10 pp versus base.
§363 tested one causal margin selector on those complete branch outputs. It moved six-base DEV −0.009 pp
versus always-JS [80% −0.035, +0.022]. The outer held person was excluded from the selector fit, while
archived source branch networks could include that person in their upstream training. The effect is exposed
DEV. No selected rule or benchmarked shared-branch worker emerged.
[§362 evidence](../evidence/next_wave_16nov/branch_ceiling362.md) · [§363 evidence](../evidence/next_wave_16nov/branch_selector363.md)
§364 completed the matched 50 ms relation test in frozen B2_pool, eight orders and 10 held-person folds.
Untouched 76.27934%, unadjusted S0C 76.32610%, instantaneous 76.39725%, lag 76.38662%. Lag versus
instantaneous −0.01063 pp [80% −0.10262, +0.10458], people/order +0.04295 [−0.05706, +0.14901].
All frozen-zero controls bitwise and all 24 new routes error-free. Lag benefit undecided. Public new-person
scope; genuine returning and worker latency unmeasured. [§364 evidence](../evidence/next_wave_16nov/lagged364.md)
§365 moved causal personalization refits toward active use. Skip-only preserved frozen scores bitwise
across eight orders and saved 18.44% of cached personalization wall. A later refit within the original
prefix budget gained +0.0201 pp [80% −0.0173, +0.0579]; equal-call recycling +0.0102
[−0.0447, +0.0734]. Both gains are undecided; cached stage savings do not measure worker wall.
[§365 evidence](../evidence/next_wave_16nov/cp_schedule365.md)
§367 fixed arithmetic and geometric pooling of the two complete branches yielded
76.3733% and 76.3765% on six-base DEV. The existing JS branch scored 76.4410%. The
geometric-minus-JS delta was −0.0645 pp [80% −0.1243,+0.0107]. Both pools leave
the comparison undecided, with no demonstrated gain over JS. [§367 evidence](../evidence/next_wave_16nov/pooling367.md)
The user moved ownership to the active Codex chat and archived the previous chat's automation.

## 2. Hard bounds (from the official code and rules)

- **Input:** one window per call, float32 `(1, 41, 480)`: 4 s at 120 Hz, starting 1 s after the cue. Output: one label
  (0 motor imagery, 1 mental subtraction, 2 word generation). No labels, no person, session or interface metadata.
- **Stream:** one participant-session, delivered in a fixed evaluation order that is not recording order (organisers, 13 Nov); Graz and BrainHero share the session; state resets between sessions.
- **State:** the evaluator deep-copies the loaded model at every session boundary. Nothing may cross sessions.
  Inside a session everything causal is allowed (adaptation, fine-tuning, identity inference from EEG).
- **Metric:** balanced accuracy per participant × session × interface cell, then the plain mean over cells.
- **Runtime:** 60 min by rule (the worker's real timeout is 80 min; plan for 60). The package is a ZIP with
  `submission.py` plus data files; no extra Python files.
- **Worker image:** Python 3.12, torch 2.14 + CUDA; the inspected image does not install compilers, numba or cupy.
  Bundled native code is permitted: Bruno Aristimunha confirmed a Python-imported C++ pipeline and wheels on
  Discord, 20 Oct (screenshot supplied by the user, 16 Nov). A compiler absent from the worker does not prohibit
  prebuilt binaries. Native extensions still need exact worker compatibility and session-reset validation; neither
  a native prototype nor a CPU replay proves Linux/A100 compatibility. See the
  [clarification and proposed native-stack scope](../evidence/next_wave_16nov/native_stack.md).
- **Uploads:** each needs the user's explicit go-ahead and a live check of phase and quota. Never cancel a running
  sealed submission: it costs a slot. Codabench has a bot check; the user opens the page in Chrome.
- **Data:** public datasets are allowed if their terms permit. Scherer 2015 (CC BY-NC-ND) was cleared by the
  organisers on Discord on 11 Nov; it must be declared in the method write-up.
- **Order (organisers, 13 Nov, Discord, verbatim):** "1) Window order. Please don't assume recording chronology.
  'Previously delivered' means earlier in the evaluation stream, not necessarily earlier in recording time. The same
  evaluation order is used for all submissions, with state reset between sessions. Without going into details, but to
  avoid tricks we discovered of solving the track without using EEG, we removed chronology. Hence why we reran all
  submissions earlier this week. 2) Labelled public windows during adaptation. Public labelled data may be used for
  training. Evaluation-time adaptation is of course restricted to the current session's delivered windows, without
  labels, as mentioned in the FAQ. 3) Any models trained on permitted public data are allowed, whatever its training
  strategy on those public data. So Yes."
  **Consequences:** the sealed stream has not been in recording order since the rescoring of 10 Nov. §62's conclusion
  ("the sealed order is still close to acquisition order") is withdrawn: the fingerprint failing to match meant we did
  not find the order, not that there is none. Before13Nov the development protocol used recording order. The current primary dev
  protocol is non-chronological within a session, scored under several schemes with fixed permutations
  (ledger §164); recording order is a secondary column. No scheme is primary: the 10 Nov rescoring does not identify one,
  and its per-package drops fit full better than within unless half a point was a uniform backend effect (correction
  under §169). Evaluation-time adaptation may use only the session's delivered
  windows, without labels. v5.1b and the combination fit nothing on labelled public data during a session (checked
  13 Nov: the own-space stage and the personal heads load precomputed parameters).
- **Cue-order regularity** (Graz runs follow a weakly shuffled WORD → SUB → MI cycle): not raised with the organisers;
  it goes into the final method write-up (user decision, 11 Nov).

## 3. The system (v4.1; the uploaded v5.1b adds the own-space stage of §6)

**Members**, all causal, reset per session, mixed with fixed weights (arithmetic mean of posteriors):
- core1: tangent-space features (common-average reference, 17 overlapping bands, full and two half-window
  covariances, causal log-Euclidean recentring), logistic regression. The strongest member (69 % alone on new people).
- Hilbert core: complex analytic-signal covariances in 7 bands, logistic regression.
- About 35 plain nets (ShallowConvNet, FBCNet, FBLight, SincShallow) and 30 SessionBN nets (batch norm with the
  session's own causal statistics).
- External teacher (weight 5): EEGNeX and IFNet trunks pretrained on 7 motor-imagery corpora; their head outputs go
  through a small logistic regression to the 3 classes.

**Stages after the mix:**
1. Rebalancing (divides by the recent class share; worth 5.75 J).
2. SessionEM: per-session class templates on frontal slow-potential and ocular features.
3. TSEM: per-session Gaussian class templates in the top 128 PCs of core1's features, seeded by confident outputs.
4. Gated early boost in windows 30–60 for sessions with low early margins.
5. Routing: a returning person is recognised and their personal head (logistic regression on core1 PCs, trained on
   their public sessions) enters a second tail at 3× in core1's slot. v3 recognises by class directions from window
   60; **v4.1 adds a label-free identity gate** (spectral shape, two-covariance model fitted on 224 external people,
   threshold 0.26) from window 10. Externally: 86.7 % correct accepts at 1.9 % false accepts.

**Packages:**

| Package | Where | Status |
|---|---|---|
| `cand_v51b_own` | `runs/upload/sealed_cand_v51b_own.zip`; `submissions/ensemble_v5b/submission.py` | Uploaded12Nov,.8197668759; frozen |
| v6.6cap250 | `runs/sealed/final_zips/v66_cp1_cap250.zip`; exactSHA above | Prepared; recordedMac/A100 checks; real sealed-worker transfer unknown |
| §321JS-all | Private Galvani prepared artifact and separate optimization snapshots; owned handoff above | Experimental; returning validation/runtime qualification incomplete; no frozen ZIP/version promotion |

Earlier builds and timing tables are in [V6_BUILD.md](V6_BUILD.md), [V52_BUILD.md](V52_BUILD.md)
and [SEALED_UPLOADS.md](SEALED_UPLOADS.md). Their dated checks are historical, not current job status.

Never edit `submissions/*/submission.py` in place: their SHA-256 values are pinned to uploaded or staged packages.
Package builders: `scripts/add_bio_gate.py`, `scripts/check_bio_gate.py`, `scripts/build_sealed.py`; audit recipes in
[AUDIT_SBN.md](AUDIT_SBN.md), [AUDIT_NXIF.md](AUDIT_NXIF.md).

## 4. How results are measured

**Dev metric J** = 0.5 × new + 0.5 × seen, person-equal mean of cell balanced accuracy.
- new: leave-one-person-out over the 10 public people (`all_loso`).
- seen: P04 and P10, trained with session 1, tested on sessions 2–3 (`seen_session`). They are the only people with
  more than one public session.
- Six dev bases (different seed sets of the nets): `B2_pool`, `B2_s345` for exploration; `B2_s012`, `B2_s0`, `B2_s1`,
  `B2_s2` for confirmation only after a gate passes. People halves for nested selection: G1 = P01 P02 P05 P07 P09,
  G2 = P03 P04 P06 P08 P10.

**Replay** (CPU, about 5 s per base):

```bash
.venv/bin/python replay/check.py
```

It must print the six reference values (B2_pool 80.0131, B2_s345 79.8474, B2_s012 80.1446, B2_s0 79.1770,
B2_s1 79.7214, B2_s2 80.2093) and `OK`. That is the v3 chain. The uploaded v4.1 is replayed by
`studies/11nov/rt4/route4.py runs/replay_data/bio_model_psd_d64.npz BASES` (mean J 80.44, +0.58 over v3).
- Code: `replay/` (tracked). Inputs: `runs/replay_data/` (60 MB, gitignored; manifest `replay/data.sha256`; copy on
  Galvani in `runs/diag/replay_data/`).
- To test a member: `A.EXTRA[name] = {proto: P}` then `A.system(base, fams, extra=[(name, w)])`; see
  `studies/11nov/ttt/score.py` for replacing a member slot.

**The six bases are not independent** (ledger §63): B2_pool is the mean of B2_s012 and B2_s345, B2_s012 the mean of
the three single seeds, and only the five SessionBN slots differ between them. "Confirmed on fresh bases" means
robust to the net seeds, nothing more. The uncertainty that matters is across people: report "new" with a person
bootstrap interval (standard error 0.5–1.1 for a typical change) and P04 and P10 separately.

**What dev numbers are worth on the sealed set:**
- Noise: historical8permuted-view nulls gave−.22 to+.18J. Small effects remain uncertain, not closed;
  use people/order intervals and approximately±.5person noise rather than a hard+.3gate.
- Small stacked gains transferred at 0.6–0.7×; v4.1 at 0.46× (+0.58 dev, +0.267 sealed).
- Dev "seen" historically weightsP04/P10 equally. The current reporting proxy is57%new people,
  with returning people approximately.2P04+.8P10, perAGENTS. This cohort proxy is not a verified sealed census.
- Report P04/P10 separately; neither person alone closes a direction. Eight pinned delivery orders and all
  original controls are required for current shuffled-delivery comparisons.

## 5. What the measurements say, and how firmly (revised 13 Nov; ledger sections in brackets)

Read every line with its test in mind: "alone" means not scored in the chain; "posterior-only" means the new part's
outputs were added while the session stages and the routing kept using the old core's features. The map
([SOLUTION_SPACE_MAP.md](SOLUTION_SPACE_MAP.md)) tags all closures by test quality.

**Where large ceilings have been measured** (the places worth several attempts in several forms):
- **More training people:** core1 rises +8 points per doubling with no flattening [§39, §48]. Substitutes tried so
  far (generated variation, people-efficient estimators, Scherer rows) were ≤ 0, but all at core level, alone
  [§46b, §47, §49, §53].
- **Members whose errors are independent of ours:** an independent member at 70 % is estimated at +3.1 for new people
  [§63]. The external teacher is the only such member; its design dates from 1–2 Nov and was last judged on a measure
  dominated by one person.
- **Labels inside the session:** +1.6 to +2.9 through fine-tuned nets or session-own members, +3 through a head
  [§28, §35, §38, §81]. With the chain's own labels the best form so far reaches +0.46 for new people [§108].
- **The session's class geometry:** aligning a session with its true class means lifts a simple classifier from
  64.6 % to 89.1 % (whole session, not causal); one label-free form as an add-on member failed [§103c, §104].
- **Session-own representations:** 81–82 % with a session's own labels against 74.9 % in the population PCs; P04
  reaches 74–80 % [§36]. The own-space stage uses this for returning people.
- **Choosing per session** among stage variants or member weightings: +1 to +2 with an oracle; no label-free rule
  found yet [§76, §117, §118].

**What held up repeatedly:**
- Extra members added at a fixed weight lower the chain even when strong alone; member weights and stage constants
  sit at an optimum **for this chain** [§31, §66, §69]. This says nothing about a redesigned chain.
- Dev resolves about ±0.3 to ±1. Gains seen only on the two exploration bases have vanished on the other four three
  times [§54b, §58b, §61b]. The six bases share people and differ only in net seeds [§63].
- Dev "seen" is two people. Upload 5 suggests the other returning people gain about +0.6 per session from the
  own-space stage, so they are not like P10.

Historical closure audits and counterexamples are in [SOLUTION_SPACE_MAP.md](SOLUTION_SPACE_MAP.md).
They constrain claims about tested forms, not the scope of new whole alternatives.

**Where the error is:** P04 as a new person 54 %, the other nine 75–90 %; the first fifth of a session 73 % against
77–80 % later; word generation is the weakest class; returning people lose 8 points in Graz feedback runs against
calibration runs [§67, §105].

## 6. Current candidate lineage

v5.1b's own-space stage on recognized sessions is frozen in the uploaded package. v6.6cap250 adds the
validated local development combination documented inV6_BUILD; its sealed transfer is unmeasured.
§321JS adds only the selected outerCC30/FBL recipe and still needs qualification. Older v5.2 variants,
research-window restrictions and unqueued experiments from13Nov are historical, not present decisions.
Use current source paths and explicit guards; do not transplant old favorable numbers across package versions.

## 7. Working rules and infrastructure

- **Pre-register** every run in [LEDGER.md](LEDGER.md) before scoring it: mechanism, numeric prediction, refutation
  rule. Report misses as misses.
- **Local runs:** only through `scripts/run_capped.sh LOG CMD…`, one job at a time (the Mac froze on 22 Oct). CPU
  analyses and chain replays are fine locally; net training on MPS is about 6× slower than a 2080 Ti.
- **Galvani** (`ssh mlcloud-galvani`, work dir `/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026`):
  - The user types the 2FA code; agents never re-authenticate. Check with `ssh -O check mlcloud-galvani`. ControlPersist43200 is12-hour local idle retention, not
    verified VPN/server expiry; network changes can close access. Ask renewal only on an actual failure.
  - Submit the current owned source-locked batch script with its declared device/resources; do not substitute
    a generic GPU partition list for a device-specific registration. Measure costs for the actual full task.
  - Downloads from Galvani run at about 0.2 MB/s: fetch small result files only.
- **Commits:** commit and push freely; no AI author line. Never commit datasets, environments, credentials or
  checkpoints. Keep the `2026-competition` submodule pinned.
- **Historical Weco:** both11Nov searches overfit their visible half. No Weco provider/service is authorized or
  running in the current campaign. Existing local study artifacts are historical evidence, not provider use.

## 8. Where everything is

| What | Where |
|---|---|
| Entry point, ledger, upload log | `docs/STATE.md`, `docs/LEDGER.md`, `docs/SEALED_UPLOADS.md` |
| Detailed record of 11 Nov (33 sections) | `docs/PREDICTIONS_11NOV.md` |
| Detailed record of the night of 11–12 Nov (§34–§61) | `docs/PREDICTIONS_12NOV.md` |
| Critical review of 12 Nov: sources of gains, what stopped on 7 Nov, located losses, self-imposed limits | `docs/REVIEW_12NOV.md` |
| Study scripts of 12 Nov (index in its README) | `studies/12nov/` |
| Closed directions 5–10 Nov | `docs/REVIEW_5NOV.md` |
| History (superseded; may be wrong) | `docs/archive/` (its README lists the wrong assumptions) |
| Chain replay | `replay/` |
| Study scripts of 11 Nov | `studies/11nov/` (the ledger cites them as `runs/archive_scratch_11nov/<same path>`) |
| Training, features, exports | `experiments/`, `scripts/` (only code that packages, audit recipes or the ledger still reference; the rest was removed on 11 Nov, see `docs/archive/README.md`) |
| Identity gate | `experiments/bio_gate.py`, `experiments/bio_id.py` |
| Compact evidence logs | `evidence/` (11 Nov: `evidence/wsp_bio1_11nov/`, `evidence/seq_bio_11nov/`) |
| Local only (gitignored) | `data/`, `runs/` (`runs/upload/` ZIPs, `runs/replay_data/`, `runs/ttt/` outputs, `runs/archive_scratch_11nov/` full scratch mirror, 3.5 GB) |
| Galvani outputs | `runs/diag/{ttt,rest4,aec,sbnx,w12,schr,steeg}` under the work dir |

## 9. Current user decisions and autonomy

The user authorizes continued code changes, broader whole architectures/objectives, parallel Galvani experiments,
and justified follow-ups overnight. The incumbent is not a search boundary. Preserve and report the best measured
artifact while exploring different mechanisms. Use the global scientific-discovery skill and actual coverage evidence.
Every4hours perform at least15actual minutes of substantive primary-source research; extra reviews do not erase
scheduled cadence. Record actual time and distinguish retained history from independent or blind context.

The provisional decision checkpoint was 17 November at 18:00 CET. No upload is authorized by the
research campaign. Obtain explicit upload authorization and check live phase/quota/storage before submission.
Earlier upload counts/deadlines/board estimates require live verification; do not infer available quota anonymously.
Do not ask optional implementation questions while the user sleeps. Continue work from the active Codex chat
and the current ledger and Slurm records.
