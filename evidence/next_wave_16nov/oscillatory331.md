# §331 oscillatory raw decoder — complete source-only result

**Completed:** CPU3117541passed0:0,17Nov06:28:34–07:01:48CET(33m14allocation,1990.22sscript),6boundCPUcores/24GiBrequested ongalvani-cn117,MaxRSS2633212KiB(2.51GiB). NoGPUfallback/externalassets. AllL/F/S40epochs/checkpoints/guards/predictionbanks andunchangedR40epochs/eightorders completed BEFOREP02/P03querylabels. These are selected/exposed source-development scores, not a completeouter/current-system/returning/sealed result. Nooutermatrix ornewtrainingqueued.

| Whole source decoder | P02 officialcellBA% | P03 officialcellBA% | Personmean% | Fitseconds |
|---|---:|---:|---:|---:|
| L — local,no recurrence |46.3542|45.4491|45.9017|101.48|
| F — staticoscillatory recurrence |50.5388|47.7050|49.1219|815.72|
| S — input-selectivedecay |50.0179|46.6980|48.3580|802.02|
| R — unchangedrawSessionBN,k20 |66.6533|56.8990|61.7762|260.13including8-orderinference|

Raverageswithin/full/runs equally overthe8pinnedorders; L/F/Sarestateless,B1queryinference. Actualowningofficialparticipant×session×interfaceequal-cellBA, notearlierpooledBA. S−F−.7639pp(P02−.5208,P03−1.0069); F−L+3.2202pp(P02+4.1846,P03+2.2558). Recurrence/frequencypath ispositive againstthislocalcontrol; inputselectivityisnot. Onlytwoalreadyexposedpeople; nointerval, independentconfirmation or familyclosure is warranted. Squalityprior64%[50,76]wasnotmet.

SourceP04–P10only(2087windows),queriesP02/P03(189each),seed331/finalepoch40/noqueryHPO. SamebatchscheduleSHA-listverified acrossL/F/S. WeightedsourceCEfirst→last:L1.0862→.3789,F1.0763→.3172,S1.0762→.3024. Finalonline sourceclassrecallsL[.921,.889,.869],F[.926,.913,.889],S[.938,.916,.895]; these are onlineepochdiagnostics, not heldperformance or proofconverged. BetterSsourcefitdidnottransferbetter. Generalization/normalization/regularization remaincompetingexplanations; no causaldiagnosis from thispilot.

## Numerical and active-path evidence

Independenttorch.complexdirect480-steporacle, fullclassifieroutputs ANDallparameter/inputgradients: initialfloat64max5.55e−17/8.88e−16; float32max5.96e−8/2.38e−7. InitialS/Foutputsexact. Total25760parameters; activepathsL23584/F23648/S25760. LfrequencygradNone, L/Fdtgradexactzeroasdeclared; Ltauactivevia drivegain, soL isnotall-decay-off orparameter-useequivalent. F/Sfrequencyactive; trainedSdtweightschangedmax.3153/.3331, Fdtremainedzero. Allotherdeclaredactiveparametergroups changed; sourcegradientcurvesretainedremotely/compactsummary.

FinalB1/B4posteriorerror≤1.193e−7,zero nontiedargmaxchanges; amplitude/finalsample/repeat/interleaving/deepcopy/immutable-stateguards passed. Allactualprobe transitionmoduli finite<1; Smax.997059. Rprefixerror5.834e−7,per-sessionreset/deepcopy/stateguards passed. BidirectionalityusesONLYcurrentcomplete480samples, notlaterwindows. F/Sstatezeroeverywindow; no persistentlatent/sessionstate.

Post-fitnumericalreview3117574passed0:0/4s, noqueryscore/newfit: actualtrainedScheckpoint onONEfitP04window hasrho temporalvariation.08560/.08386 acrossits2blocks. SAMEindependentoracle atgenuinelynonzero learnedselectivity:float64output2.22e−16/gradient6.66e−16,float32output1.19e−7/gradient1.79e−7. Thusinitialidentitydidnot hide a nonzero-selectivity scanfailure. Proof`oscillatory331_learned_oracle.json`; this post-hocnumericalcheck isnot independentperformanceevidence.

## Source, budget and limits

Scientificv2workspace`/mnt/lustre/work/gehler/gfh527/ni26_oscillatory331_v2_20261117`, sourceSHA`b0fec93e6c3c6e3f66cc1d967e2c3b7ba1c87e01ca8308d3d56a6f5e0fe8fde9`; model.pyunchangedfromprepared70df557. FullremoteJSONSHA`863b18d18248042ac020013d8234a9dc3aad8b19741f55d10c737a1e032fe972`; local`oscillatory331_results.json`isacompactsummary retainingthisfullreporthash, all40sourceepochs andgradientranges. PredictionarchiveSHA`ff090769cc90aee9bdc549de0ab12b2b45127bce5e442ffb7f16b7795f038713`. L/F/ScheckpointSHA21fde106…c5a1/136bff84…7773/c7caf6a3…66b9; exacthashesinJSON. Rcheckpointwasnotexported; itsrecipe/source/input/posteriorandbefore/afterin-memory-stateguardsare recorded, aweakerpersistenceinstrumentthanL/F/S.

Firstguard3117534failedbeforepilot/querylabels: timingprobe passedunsortedindices to theexistingstrictlabelreader. Preservev1log`oscillatory331_guard_v1_failed.txt`. V2sortsONLYthe32engineeringprobeindices; noarchitecture/trainingbatch/tolerancechange. Correctedguard3117539passed32s, proofSHA`04d01e3469b9bc6a19f1bccbd93bd5d912743ecbba7b997dbb519e8698c27125`.

Costgatecorrectlyblockedautomatic45minpilot:1.5×warmL/F/Smediansteps×66batches×40epochs+1.5×Roneepoch×40+120s=3188.62s(53.14min)>36minlimit. Separatelyregistered75minallocation/60minprojectionauthority, wrapperSHA`3b2f5d9bf8ef12accb8c15fa95018ee7fca69dfcbaa52e2d8f7d94a508981094`, changedONLYbudgetpredicateinmemory; originalscience/guardimmutable. Actual33m14isnotaretroactivechange tothatgate. Root'slaterconditionalparallelreplacementdidnotfire because3117541alreadyRUNNING; preservedserialjob, no duplicates. Unexecutedparallelplan isnot a run.

SourceRMS/current-windowCAR onlyinbody; same287mean/logpowerbranch preservesamplitude andall480samples.331bands1–4…30–55Hz/sourceRMSarenotbitwise324's.1–4/quantilescaled287. P01excluded fromfittedstatistics/scalers/training/queryaccuracy. ExistingRlibrarymaterializesper-windowcovariancecacheforallloadedX, butG/scale/trainingselectONLYsourceidx(deep.py285–301); donotclaimP01array/per-windowcachewasnevermaterialized. NoP01labels orcross-personaggregateleakfound. Risaqualityinstrument, not a matchedarchitecturecontrol.

Owning`experiments/protocol.py`SHA`390fd6bea0ddc9c53e34fdabb094b1a269ff45b07b4a5542a8135b44d5314a2c`; official`benchmark_utils/metrics.py`SHA`fcc739400806a74259f248618501376a4fe53f2b5a40ccdc20db6a9e2369787d`. Actualmodulepathsasserted; no evaluator/submodulechange. Existingselected-labelreader35edab56…c4dcusedonlyallowedrows. State/reset/scaler/source/checkpointguardsarenot worker/packagevalidation.

**Decision:** do not expandthisweakwholeformtoouter/chainautomatically. Selectedstaticrecurrencepositive staysopen; failedS−FdoesnotcloseSSMs. Beforeanotherqueriedvariant, use aninnerfit-personholdout toseparate optimization/epochchoice from normalization/domainshift, withoutP02/P03tuning; anydifferentnormalization/inferenceproposal needs itsownregistration. Noextraepochs/width/headselectionornewscientificjobwereadded.

Contextactuallyread: sharedprojecthistory, STATE→LEDGER→OFFICIAL_CONTRACT, scientific-discovery/engineering-workflowskills, actual331prepared/source/reference/officialmetriccode, realguard/pilotoutputs. No blindfreshcontextclaim. Mathematicalgroundingonly:[S5](https://arxiv.org/html/2208.04933v3),[S7](https://arxiv.org/html/2410.03464v1); directlog-depthscanisnot theiroptimizedkernel/theorem/reproduction. [NeuroMambaNet](https://www.mdpi.com/2504-2289/10/11/385)/[LuMamba](https://arxiv.org/abs/2603.19100)arecontextonly; no modelcode/weights importedorlicensingpermissionassumed.
