# Campaign status: 17 November 2026, 18:50 CET

Reporting: FULL-SYSTEM DEV / SEALED LEADERBOARD / INDIVIDUAL-MODEL must be explicit, with comparator.
Custom stack=C++/CUDA; original raw neural reference=stockPyTorch336/338.

Submission unchanged: uploaded v5.1b **.8197668759**. Prepared v6.6 cap250 remains unchanged;
ZIP SHA `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed`.
Last verified board01:12 CET:16th entry/15th distinct team; leader.859, fifth entry.839/team.837.
A later public refresh returned a bot challenge, so no newer verified rank is available.
[Observed board](live_board_17nov.json). The v6.6 .824–.829 scenario is unvalidated sealed transfer,
not a measured score. No upload or candidate promotion; top five remains an objective.

## Current-system evidence

All differences below are percentage points on repeatedly used new-person development people.
Intervals are80% and resample people and delivery orders. These are not independent confirmations.
Original PC128/shadow/FT/routing remain fixed in posterior-only interventions; returning tails are separate.

- **§318 selected joint JS:** conditional-CORAL30 FB-slot swap plus FBL factor2 gives76.217673→76.441042%.
  Official+.2234[+.0738,+.3512]; people/order+.184[+.0323,+.3327],7/10up.
  Selected after earlier studies, small relative to person variation; interaction spanszero, no synergy claim.
  [Report](coral_fbl_s318_results.txt), SHA `9abebd5069186fc659730f2f48d92c7036db49ac17265e5574d508435b130874`.
  Actual six-seed deployment is B2_pool:76.279335→76.551732%,+.2724official[+.0049,+.5271];
  people/order+.2780[−.0070,+.5456],6/10up. Within/full/runs effects+.0203/+.6640/+.1496;
  sealed delivery-scope transfer remains uncertain. [Deployment extraction](js321_pool_contrast.json).
  P04/P10 are new-person folds, not returning-session results.
- **§318 attribution:** V6Fr+.2264 versus finalCP+.2234; effect change−.0030[−.0923,+.0871].
  Most net corrections are subtraction/word generation. True-class probability margins fall despite more
  correct argmax labels; Brier change uncertain. This does not identify an earlier causal mechanism.
  [Selected diagnostic](coral_fbl_s318_diagnostic.txt).
- **§305 native head:** fixed port preserves76.218%; converged head75.966%, people/order−.113[−.259,+.028].
  No gain; original failed training-head parity remains failed. [Report](native_core_s305_results.json).
- **§308 nonlinear heads:** best tanh76.1048 versus control76.2177%; all arm−control intervals spanzero.
  Quadratic versus projected linear also changes width/ridge geometry, so does not isolate nonlinearity.
  [Report](strong_heads_s308_results.json).
- **§310 entropy:** flatten76.013 versus76.218%; flatten−control people/order−.0718[−.2236,+.0805].
  Flatten beats sharpen slightly, not the current system. No gain established.
- **§314/315 temporal:** whole temporal bank49.158 versus matched46.975%,+2.182official points;
  chain TFD−control−.110official, people/order+.008[−.158,+.182]. No established system gain.
  [Temporal chain report](s315_314_results.txt).
- **§313 spectral:** whole IMAG62.956/REAL60.983/BASE62.716%; IMAG−REAL+1.974official,
  people/order+1.992[+.450,+3.527],8/10up. VersusBASE uncertain. BASEmatches305converged,
  not306historical; the gap does not support a feature/cache bug. Complete chain IMAG76.1050/BOTH76.1210%
  versus76.2177%; no established control gain. [Chain report](spectral_s313_chain_results.json),
  SHA `7c1c7eb509d42b6219a9a0f64bd7ad50f09be6b7d574cb772ade3f9fff636266`.
- **§316 common-anchor adaptation:** scalar−fixed people/order+.0806[+.0134,+.1479], but
  scalar−current−.0929[−.2416,+.0512]. No established control gain. Original307parity remains failed.
- **§312GPU raw-pretraining integration:** complete54cells/scorer3116969 passed all guards.
  FBC pretrained-finetuned versus supervised FT−FS+.0007official[−.0639,+.0676],
  people/order+.004[−.061,+.073]; Shallow ST−SS+.0409[−.0350,+.1076],+.014[−.066,+.092].
  Every arm−control interval spanszero. [Chain report](raw312_chain_results.txt), SHA `8628c4ff8ab0256975f7e1a5d45d85b405115097029094ee46991a78fba547e6`.
- **§323 normalization integration COMPLETE:** scorer3117171 passed05:30:42 CET/30s;
  registered interaction reporter3117174 passed05:31:26/24s. All complete source/archive/zero/order/
  routing/six-CP guards passed before metrics. Owning scorer loads labels before collect; interactions collect first.
  No score-selection leakage or numerical miswire is established by this ordering. Control76.217673%; all eight arm−control intervals spanzero.
  Official differences P0F+.0385/P0C−.0329/P1F+.0793/P1C−.0032/S0F+.0899/S0C+.0345/
  S1F+.1089/S1C+.0602. Causal−fixed P0C−P0F−.0714[−.1344,−.0056] official, but people/order
  −.053[−.119,+.013],2/10up. S0C−S0F−.0555[−.1608,+.0473], people/order−.036[−.135,+.062].
  All chain factorial-interaction intervals spanzero. Whole-model causal gains did not transfer into useful
  additions in this form. Selected diagnostic3117400passed1m24: two-seed ensembling preserves+4.055/+4.742pp
  causal gains; effect is already absent atV6Fr(P−.0787/S−.0651pp), finalCP effectchange+.0073/+.0096,
  both intervals span0. Thus CP is not the primary attenuation location. About85.16%/83.67% of member
  corrections are already control-correct. Member NLL/Brier/ECE improve, so worsening calibration is not
  observed. Error overlap is descriptive, not a sufficient causal explanation or family closure.
  [Selected diagnostic](norm323_attenuation_results.json); no new policy was fit.
  [Report](norm323_chain_results.txt), SHA `9e82b8082948702b600687b455e9efcd027b1698ae21955deb9e5faa0196ae34`;
  [interactions](norm323_chain_interactions.json), SHA `9104644305c5b1a3f2ffa6a39d382e2e096dea5ae0fb4207b88996617be388d0`.

## Whole alternatives and source mechanisms

A weak individual decoder versus the stronger ensemble does not alone close its architecture family.
Matched causal contrasts, source quality, whole systems and full-system integrations are distinct instruments.

**§312 GPU whole:** correctedcollector3116911 passed all160fold/source/checkpoint/row guards before labels.

| Encoder | Supervised S | Frozen pretrained F | Pretrained/fine-tuned T | Extra-supervised C |
| --- | ---: | ---: | ---: | ---: |
| FBCNet |54.694%|45.934%|53.899%|52.384%|
| Shallow |50.062%|45.606%|50.321%|51.511%|

T−C people/order FBC+1.657[−.459,+4.001], Shallow−.591[−2.126,+.851]; T−S also uncertain.
[Whole report](raw312_gpu_whole_results.json). One strong source P02 result did not establish whole quality.
Seed changes both initialization and source-validation person/epochs. Source-only body/skip probes do not
support skip dominance. Failed importcollector3116577/purged-IDbridge3116075 remain recorded; their
module-path/orchestration repairs preserved original producer source and metric.

**§317 external cognitive supervision:** complete60foldcollector3117176 passed6s, privateconsumer3117177 passed8s.
Whole per-seed averaged official A(extraPROTEUS)52.8361/B(external stop-gradient)53.6952/
C(continued external gradient)53.7346%. C−B+.03943official, people/order+.5513[−.7633,+1.9531],6/10up.
Large seed differences; not two-seed ensemble accuracy or isolated auxiliary gain. C−A+.89846official.
[Owned evidence](aux317_handoff_17nov.md). Its complete guarded current-system comparison is recorded below; no further317job remains.

**§322/323 normalization:** selected322source pilot found matched SessionBN gains5.4–8.3points on exposedP02/P03,
but training and inference changed together. Complete32380fits/20folds/128per-seed streams separate those phases:

| Training | Skip | Fixed inference | Causal inference |
| --- | --- | ---: | ---: |
| PopulationBN | off |54.545%|58.443%|
| PopulationBN | on |55.130%|58.014%|
| SessionBN | off |53.820%|58.408%|
| SessionBN | on |55.450%|59.045%|

Same-weight causal−fixed no-skip official+3.898[+2.216,+5.875]/+4.588[+2.692,+7.092];
people/order+4.437[+2.093,+6.814]/+5.654[+3.081,+8.247],8/10and9/10up.
Training-mode contrasts alone uncertain. Skip attenuates causal-minus-fixed contrast, but causal skip effects
spanzero; this does not prove skip harm. [Whole](norm323_whole_results.json), [interactions](norm323_interactions.json).
The selected[prefix diagnostic](norm323_prefix_results.json) uses fixed full-cell/class weights; bins sumexactly.
Most score contribution occurs after40windows, where most windows occur. No age threshold/easier-late-window claim.
Whole scores average independent seed metrics; chain members average seed probabilities before prediction.

**§319/324/325 visual prior:** CPU/A100 strict-load/offline/input/amplitude/padding/gradient/reset guards passed.
Full41/16row warmB1 encoder394/134msCPU and12.96/4.69msA100; not whole-package timings.
Fixed324full-intensity pretrained42.462% versus random36.452/direct53.936/rawSessionBN62.299% across exposedP02/P03.
Full channels/frequency views did not consistently restore quality. [Factorial](visual_s324.md).
Learned325295parameter display passedCPU3117222/38s, A1003117226/8m39s:
pretrained42.462→49.639% (+7.177), almostallP02(+14.336/P03+.017); learnedrandom41.320,
learneddirect54.890/fixeddirect55.142%, rawreference62.299%. Identical raw sidechannel does not rescue it.
Learning is active; robust both-person prior gain is not established. Three epochs/unverified crossfitteacher
convergence limit conclusions. Minibatch-loss changes alone do not prove optimizer failure. No outer expansion.
[Learned-display evidence](visual_s325.md); source queries were reused and initializer previously selected.

**§326/327 native source-logit invariance:** seven source peopleP04–P10, P02/P03query-only/P01excluded.
CPUprep3117223 passed3m07, A1003263117224 passed37s; full44,239features/sourceobjective/analytic-autograd/
finite-difference/gauge/zero/prefix guards. Absolute strengths nearly suppressed the offset penalty.
Separately registered loss-relative3273117238 passed25s. P02zero/1%/10%:78.913/78.229/77.609%;
P03:59.042/58.921/59.577%. No consistent gain. The1%fit reached2500evaluation cap despite small finalgradient;
retain solver-limited scope. No outer wave. [326](conditional_s326.md), [327](conditional_s327.md).

**§328 new spatial-state representation:**16source-learned polarity-invariant directions; whole-window occupancy,
dwell/transitions, no assumed delivery chronology. CPU3117280 passed36s, staticcontrol3117354 passed12s.
Raw28744.497%; raw+learnedoccupancy48.585, ordered46.770/shuffled45.939. Static gain+4.089pp is real in this
weak source form, but raw+randomoccupancy48.317 and raw+angularcov48.714 recover it. Cov+learned47.925,
cov+random47.950; no learned conditional-covariance gain. No higher-order/dynamic mechanism established,
no outer expansion or universal family closure. [Static controls](microstate328_static_results.json).

**§320 reliability:** H60.115/U61.156/static61.669/learned62.173% on repeatedly usedP02.
Learned beats uniform, but H failed registered68.60quality premise; no outer expansion. [Report](reliability320.md).

## Candidate engineering and active jobs

**§321 corrected JS-all artifact:** [Owned handoff](owned321_320_317_handoff_17nov.md) owns exact source hashes/IDs.
Original inputEA prior5 is distinct from latent SessionBN prior2. Initial all10array3117189 failed before export;
old raw-bank equivalence invalid, recovered weights/prealigned export and dev scores retain their scope.
Corrected raw repair3117225 passed59s; all6historical heldweights/all8rawstreams, max8.607e−7/zero nontied changes.
Active5/3117230 passed7m56: exactoff/inner/FTlabels/gates/reset/deepcopy, activeFTdeltas, all8orders×6recipes.
One all10seed3117236_0 passed58s, remaining5array3117240 and owningcollector3117241 passed; privateartifactcomplete.
Original1047arrays preserved byte/dtype/shape/stride; prior private C-order copy explained tinyinner differences.
Outer-only adaptedFBk2swap/FBLdouble/17unit normalization preserves original inner65nets/weights/FT/gates.
JS-all continues after recognition; exact inner identity is not final recognized-output identity.
InvalidhistoricalSinccheckpointsremainrecorded; genuineNEWprimaryheldfoldfits belowreplace nohistoricalidentityclaim.

CPUworker3117246 **failed first runtime bar**: known190base82.330→JS90.303s, ratio1.09684>1.061008.
Off17/reset/knownrec10/FT0/CPstop10/190CCcalls passed; STOP means378/800/forcedunknown/lateCP untested.
CPUratio×historicalA10043.5min is not CPU sealedtime or A100timeout evidence.
Independent A100worker3117247 completedFAILED1:0,05:48:20–05:52:32 CET/4m12.
Known190ratio1.03104 passed; known378base49.409→JS61.152s,ratio1.237657>1.061008 failed.
WarmB1median131.41→164.12ms at378; knownrec10/FT0/CPstop10/378CCcalls passed.
STOP means800/forcedunknown/lateCP untested. This relativebar failure is not an actual timeout/whole-limit proof.
Controller3117342 verified actualGPUbar failure/source/image and released one6minA100componentprofile3117399.
Profile3117399 passed: all6FIRfingerprints match; oneFIR+sixindependenttails versus sixfullSJit calls has
EXACT0.0 probabilityerror across two alternated-reset17window runs. Auxiliary-networktime.2509/.2441s→
.07674/.07639s(68.7–69.4%less). This is component timing, not whole-worker or sealed speedup.
Separate FIR-only and readonly-current-input variants preserve all original tensors/paths. ActualNetBank
instantiates45nonzero-weight networks;20declaredinactive arrays remain covered by unchangedNPZhashes.
OldstandardGPUselfcontrol exhibits6.25e−8reset drift beforeFT and3.02e−4final drift withFT;0labels.
Deterministicold215control3117572passes EXACT0 alloutputs/FTparameters/generators, and controlled
old→FIR3117581/reuse3117582 proofs pass EXACT0 through20FT/10CP/125fusions, originalinputs/gates/labels.
Oldbitwise failures are preserved; no bound was fitted to noise and production flags remain unchanged.
Separate standardA100 first-stop optimizedworkers pass known190/378/800(maxratioFIR1.0553/reuse1.0493),
thenfail unknown378(FIR1.3115/reuse1.2477). Those rows are relativebar failures, not observed timeouts.
CompleteFUNCTIONAL-onlyA1003117602FIR/3117601reuse now PASS all6known/unknown190/378/800rows.
Unknown800has24FTrefits/480steps/noover,39CPrefits/710fusions. Reuse800absolute134.875→137.939s is
DESCRIPTIVE, not a timing-pass claim. Firstunknown190payscommoncoldFTsetup, so its~.5ratio is not speedup.
CounterbalancedAB/BA timing3117650 COMPLETE43m59, all12pairspass6.10%bar:
medianoverhead190/378/8002.43%/1.19%/2.62%, max4.91%. Actual800base156.69–158.72s/JS161.47–164.38s.
Coldsetupseparate/orderconfounded,nospeedup. Original45minlimitunchanged; TIMEOUT-onlyretrycorrectlyNO_RELEASE.
IndependentCPU2153117573PASS81m17, exactold→FIR/reuse with20FT/10CP/125fusions, buffer/generator/resetproofs.
OriginalCPU3117585/7TIMEOUT45minpreserved; separatefullFUNCTIONAL-only3117756/7 COMPLETE2h27m06/2h32m09,
collector3117884passesall6length/mode/source/activityguards. Functional completion is not CPU timingqualification.
All60NEWgenuineheldPRIMARY-NNweights/72traces+576B1SIFbanksPASSEDownsource/export/reset/order/weightguards,
bothsourcefoldcontextspass. Historicalidentityunverified; oldCPU/GPUstrictQFfailurespreserved. Actualfrozen-stage
RouteB/TSEM/OwnStage/CF/CP engineering3118487PASS5m59, exactcomposition/offzero/inner/gate/labels, JSactive
pre+postrec;rec10/FT0/CPstop10. ENGINEERING_ONLY(cachedrecordingfoldview/originalall10inner), notRETURNINGDEV.
BothnewsourceCore/PCAfits3118591/3118601 pass~3min/8.4GiB, sourceheldrowsabsent; nativeSIF raw31canaries3118613 pass.
Full8order raw256banks3118623/newsourcecontexts3118631/ALL28normalBNfits+originalShallow,FBCexports pass.
Source-onlyteacher maps/galleries/private3arm/source-inner/recognition/FT guards pass. No scoreyet.
Bridge is CONDITIONAL ENGINEERING: fixedproductionMAD/teacherEA-G/oldOOF ANDoriginalrest/eyeauxNN retainexposure.
Raw256/sharednewinnerFT/wholepairedprotocol stillrequired; a positive cannot establish held FULL-SYSTEM DEV/rank.

**§317 private chain COMPLETE:** all30cells/control/zero/source/route/six-CP guards passed, scorer3117358passed19s.
C−control+.048820official[−.114472,+.218121], people/order+.036[−.150914,+.231166],5/10up.
Actual auxiliary contrast C−B+.002274official[−.049902,+.056308], people/order+.012[−.046749,+.070972],7/10up.
No established useful system gain; weak-backbone/external-family limits remain. [Complete report](aux317_chain_results.txt).

**§312 independent all-CPU COMPLETE:** all160folds/source/splits/checkpoints/raw/8order guards pass;
collector3117428completed06:11:07–06:12:07CET/1min. FBC S52.249/F47.604/T54.816/C55.421%;
Shallow S50.407/F44.913/T49.812/C50.830%. T−C people/order FBC−2.338[−4.775,+.125],Shallow−.973[−2.418,+.529].
FBC T−S+2.140[+.740,+3.561], but compute-matched C−S+4.477[+2.376,+6.660]; no isolated pretraining gain.
[Independent CPU report](raw312_cpu_whole_results.json), SHA`bfb6abdf0c489032a77f97d7800b2151ef3ac7f9f5e94e03e417f71a0f259f70`.
No device-fold mixing/CPUchain/automatic outer wave. [Scheduling/source record](raw312_schedule_handoff.md).

**§329 new released Atom-L prior COMPLETE source instrument:** CPU3117540/A1003117543 strict425-key/input/
gradient/reset proof passed. A100B1~14.493ms/593MB. ActualCPU/A100fittedheadprobabilityerror1.4667e−5,
0argmaxchanges; featuremaxabs.00282955 passes combinedatol.002+rtol.001, notabsolute.002 alone.
Initialdynamic-module and latercachedmetric-module failures are preserved. Canonical fixedC/rawreference repeat
3117559passed109.26s using exact savedencoderbanks; no identity to lost process-localoutputs is claimed.
All heads/scalers/rawcheckpoint/querybanks/source metadata now saved beforemetriclabels.
Official source-personmean: raw44.236%; prepool39.836/random36.313; preretained43.587/random41.938;
prepool+raw44.738/random43.645; preretained+raw45.955/random46.023. BestfixedpretrainedP0247.347/P0344.563
versus random43.733/48.313, mixed. StrongrawSessionBN66.873/57.604→62.238mean is better.
Retained+rawsourcefit88.884%/NLL.432,52–85LRiterations; no solvercap explains weakquery quality.
[Canonical source result](labram329_source_result.json). Defaultweights atom/atom-large.safetensors SHAa28e97…;
ff09aeda… is the benchmark-selected best-full-ft file, which was NOT used. No device-selected source-score duplication.

**§330 fixed fullFT/frozen controls COMPLETE:** A1003117553passed0:0,625.08s/3.542GBpeak.
All4fits/12epochs/actualbodyupdates/CPU-A100fittedheadparities preceded query scoring.
Pretrainedfrozen43.141%, pretrainedFT45.299%; randomfrozen43.957%, randomFT39.705%.
PreFT−randomFT+5.594pp is positive on bothP02/P03(+5.337/+5.851), but randomfit remains optimizer-limited:
sourcepreFT100%/loss.001286/gradient.00387, randomFT98.946%/.08149/.3949.
PreFT−prefrozen+2.158pp is mixed(+5.585/−1.269). Whole45.299% remains belowraw62.238%; no outer expansion
or familyclosure. These are exposed source people, not independent/returning/sealed results.
[CompleteFTresult](labram330_source_result.json), [checkpoint replay](labram330_checkpoint_replay.json)
reproduce all original official person/interfacecell scores exactly. [Owned handoff](labram330_handoff.md).
No new candidate or score projection follows.

**§331 independent raw oscillatory decoder COMPLETE:**3117541passed33m14, all40epoch L/F/S/Rfits,
checkpoints/banks/state/officialguards before querylabels. Local45.902/static49.122/selective48.358%,
unchangedrawSessionBN61.776% across exposedP02/P03. Static−local+3.220pp bothup; selective−static−.764bothdown.
SourceCE improvesL.379/F.317/S.302, but transferweak; not optimizer-convergence or familyclosureproof.
Postfit3117574independentdirect480stepcomplex oracle onactualtrainedselectivity passes output/gradients;
float32≤1.79e−7, actualrho varies. FinalB1/B4≤1.193e−7, parameteractivity/state/modulusguards pass.
Originalcostgate53.14>36min preserved; separate75minserialallocationunchangedscience, parallelreplacement
neverfired afterlaunch. No duplicatefits or outerexpansion. [Complete scope](oscillatory331.md).

**§332 mean/variance/donor intervention COMPLETE:**collector3117655passed07:59:41CET/25s,
all20person×seedcells/80originalcheckpoints/10views×8orders. OriginalF/Bprobabilities BITWISE0error,
originalwholemetrics reproduced. Mean-only−fixed P0.000pp[0,0]/S+.014[−.031,+.062].
Variance-only−fixed P+3.921[2.241,5.881]/S+4.607[2.720,7.085]officialpp;
people/order+4.437/+5.661. Selfboth−sourcedonor P+4.267[2.682,6.327]/S+5.849[3.950,8.481].
Mean×varianceinteraction CIspans0. ActualONEBNlayer; featurevarianceadaptation supports thissavedweightform,
not a physiologicalcause/newchain gain. Initialfloat32moment guardfailure/transparentf64oracleamendment
preserved; normalizationarithmeticunchanged. Completecompactsourceevidenceownedbyfresh_solution_audit.

**§333 label-free information-retention probe COMPLETE:**3117678guard4s/3117681pilot25s.
All9savedAtom/rawbanks fit fixedridge.1 on P04–P10 to fullsource-fixed44,239cov/slowteacher, no labels/newNNfit.
Allmaps/guards beforequeryerror. PrepooldiagrelativeMSE.864/.670 versusrandom.898/.887(P02/P03),
retained.944/.668 versusrandom1.004/1.045. Offdiagonalprepool1.157/1.053, retained1.304/1.248;
retained+raw1.467/1.190. Ratio1 is source-mean predictor. Diagonal recovery improves; offdiagonal lineartransfer
is weak. Thisfixedridge/exposedcrosspersonprobe doesnotproveinformationabsent or explain classificationcausally.
[Complete diagnostic](retention333.md), [allerrors/guards/hashes](retention333_results.json).

**§334 coherent whole generative model COMPLETE:**3117620passed95s, all7sourceheld6-personrefits,
full7refit/48banks/guards BEFORE querylabels. Static58.875%, translation/mixture49.891%, hardMAP49.737%.
Sourceheld selectedη.25NLL7.315 vs uniform1.099, severelymiscalibrated. ActualfittedCPU/A100featureposterior
3.14e−14/0nontiedchanges. Full44,239invertiblegeometry retained. Source-onlyaudit3117674 finds
residualclasscontrast.00769 vssemantic63.3895 underD−2scaling; g/πposteriorsnearlyuniform, translationactive.
Mixture−translationprobmax1.02e−7: aggregatehypothesisactivity failed to establish eachlatentuseful.
Negativefixedmodel, notgenerativefamilyclosure. [Complete evidence](generative334.md).
§335source-held-only20-controlstudy3117719→3117726 COMPLETE: bestsemantictranslationη.01
NLL.938013/Brier.559839/official56.468%, NLLbeatsuniformonall7people. Bestmixtureη.01ρ1NLL.938245
isworsethansemanticcontrolonall7;ρ100.956346/ρ10000.964811. Largeρactivatesresidualevidence butP04audit
findsg0nullgeometry/b0posterior1andπuniforminall3sessions. Actual652CPUcoreseconds=.1811CPUh,
NOA100/P02/P03inference/scoring. Usefulcalibrationdoesnotestablishusefulgeometrymarginalization.
§336 COMPLETE five40epochSOURCE7arms3117844→3117845: nativeA61.776%, noheadmaxnormH62.827%(+1.050pp),
uncenteredscaleU61.157%(−.619), classbalancedW62.343%(+.567), changedbatchscheduleB62.932%(+1.156).
H/W improve bothpreviouslyexposedP02/P03; B worsensP03. All1360steps/init/RNG/source/optimizer/reset/prefix guards
beforelabels; A matches331R. Actual284–312s/arm, notarchitecture/ensemblegain. [Evidence](reference336_handoff.md).
§338 combinations COMPLETE: INDIVIDUAL-MODEL source7mean HW62.516/HB62.013/HWB61.973vsH62.827%.
EverycombinationlosesP03. Source6heldP09A64.061/H62.747/HW65.179/HB61.630/HWB61.327%; no consistentgain.
All10new+4reusedcells beforemetrics; no selectedcombination/full-systempromotion. [Report](combinations338_results.json).
§337 fourmatchedpretextfits+collectorCOMPLETE: INDIVIDUAL-MODEL/P09 original64.061/M62.512/C63.736/
shuffled61.669/extraCE63.441%. C−M+1.224pp,C−shuffle+2.067,C−original−.325. No FULL-SYSTEM DEV gain.
Nativewordinraw337reportmeansORIGINALPYTORCHREFERENCE. CovpostfinefeatureMSEworseall3groups.
Actual2.205CPUcoreh/0A100;64pretextsteps,oneperson/seed,prefinecheckpointsabsent, no familyclosure.
[Evidence](pretraining337.md). §340 numericalfeature/learningaudit usesactualFIR/source-onlyinputs,noaccuracy:
finitebank4320×480rank480,cond2916,signalattenuation≠absence; CARchangesrawNNfeatures1.188sourceSD,
whileteacher~1e−6invariant. No taskcausality/NNsufficiency claim. MissingprefineNN/auxweights limitsforgettingattribution.
§339 COMPLETE: ALL20nestedsource8banks/30source-onlyT/48members/64cells/384CPoutputs and original
source/zero/order/route/endpointguards pass before finalscorer3118434. Control76.217673%; official FULL-SYSTEM DEV:
rawF+.03846[−.09626,.17026], rawB−.03295[−.18929,.12581], rawV−.03086[−.18945,.13084]pp;
calF+.04531[−.07573,.16648], calB−.05331[−.20539,.10204], calV−.04812[−.20169,.10722]pp.
Every controlcontrast interval spanszero. Cal−rawF+.00685/B−.02036/V−.01726pp are uncertain.
INDIVIDUAL-MODEL two-seedmean F54.727/B58.782/V58.817% is unchangedbytemperature; ~4pp did not stack.
Activeprobability/decisionchanges ruleoutno-op, notcausalexplanation. Sameweight/fullstages/sixdependentCPbases;
newpersonroutingemulation, notpackagedreturning/sealed. Actual24.06CPUcoreh/.2075A100h. No ownedjobs/promotion.
[Completecontrasts/guards/source](calibration339_full_handoff.md), [report](calibration339_full_results.json).
§341complete: samebinary/sourceCPUrepeat80.35s1thread/45.47s6,historic243notreproduced;96.4%matrixmultiply.
Validation-onlypatchpreservescoefficients/loss/grad/predictions EXACT; checkedAPI2.26x/1.36x, headB12.30→.91ms.
FullCPUsolve41.96→41.59s(singlepair), no practicalfullsolver/core/worker/accuracyclaim. Futurecanonicalsourceonly;
activeremotebinariesunchanged. [Evidence](solver341_handoff.md). GPUresident/batchreuseplanunrun.
§342 COMPLETE3118608/57s: P09 INDIVIDUAL-MODEL prepool44.630/time40.086/spatial51.347/full44.199%;
matchedrandom36.369/36.782/41.002/42.511%. Spatial−pooled+6.717pp, but below originalraw64.061% contextualreference.
All32source-selectedGPUheadfits/guards beforelabels, sumfit11.46s, peak1.114GB; no worker/languagecomparison.
Failedfeatureanchor3118549preserved; sameweightsB3→B8 explainsquerynumerics. Unchangedprobabilityguards pass.
No outerexpansion/systemgain/promotion. [Finalreport](spatial342.md) owns all8arms/controlscope.
§343 CPUrepair COMPLETE V4: customfusedC++value/gradient + SAMEexistingSciPyL-BFGS-B/layout/C=.3/stops.
WarmP01sklearn19.7467→16.4362s(−16.76%), core117.3624→93.3867(−20.43%); P10cold15.4332→10.7565s(−30.30%).
ActualcountsP0188eval/82iter andP1071/65match. Full1534held-stream predictions/P01,P10/within0,full0 ALLlabels
match, maxprob6.65e−8; fouroldPythonfeaturecanariesmax3.77e−6/relative5.59e−9 passFIRST. Noheldlabels/accuracy.
FreshfitRAMincrements77,796→71,748KiB(−7.77%), loadedbaselineDIFFERS121,024KiB; no attributingabsolutepeakdrop.
Singlefreshprocess/method/twowarmfits/method/onesharednode, noenergy/hardwaregeneralization.
V1spanheldFAIL/old243snonreproduction/V2slow/V3qualification preserved. V4lib7a2715ad…939bd, futureexplicitAPI
only; noactiveproducer/package/submodulechanges. Total2.4511allocatedCPUh/0GPU/allownjobsterminal.
[Finalreport](solver343.md), [fullstreamproof](solver343_full_held.json), [handoff](solver343_handoff.md).
§344 COMPLETE: P09 INDIVIDUAL-MODEL A64.061/B63.736/C61.175/D61.891%; C−A−2.886pp and worseNLL.
Finalcovariance-probeC improvesall3groupsversusA/B/D, but nottaskaccuracy. ALL16maps/source/strongA,Bidentity
beforelabels. Mixedprefine/finaltrajectorydoesnotshowgeneralforgetting. Actual3.948CPUh/0A100/no344jobs.
[Fulltrajectory/limits](retention344.md); no automaticauxweightgrid. TeacherC++backend, NNtrainingPyTorch.
§345 honesttaskdistillation COMPLETE: P09INDIVIDUAL-MODEL raw64.061/KD61.665/deranged63.051/teacher63.382%.
KD−raw−2.396pp/KD−deranged−1.386pp; teacheritselfdoesnotbeatraw, not historicaltenfold69%.
Allnestedsourceprior/scaler/head/P09exclusion/native40/zero/gradient/eightorderguards beforelabels.
NLLwindow-mean(.8272/.8526/.8457/1.0954), not344balancedNLL. Actual2.186allocatedCPUh/0A100/nojobs.
Pythoncostprojectionfailure/wrapperGPUhardcodefailurepreserved; explicitCPU0nativefeaturesqualified/run.
[Report](distillation345.md); noautomaticgrid orfamilyclosure.
§346 directsummary COMPLETE .15056CPUh/0A100: P09INDIVIDUAL-MODEL A6raw64.061/Brefit60.686/C65.447/D60.640%.
Cadds369signedslowtrajectory/high-bandpowerfeatures; Dsamewidthsource-run-labelblindderangement.
C−A+1.3859pp(7/8orders), C−B+4.7608/C−D+4.8072pp; strongerAcontrolincluded, all8AprobabilitiesBITWISE.
PooledNLLC.953>A.827: notcalibration/fullsystem/independentgain; relatedinfoalreadyincore/SessionEM.
FrozenNO-REFIT extension3119224 COMPLETE: P02A65.646/B57.746/C64.164/D56.638%; P03A56.227/B51.770/C51.391/D51.883%.
C−strongA−1.482/−4.837pp, sameA6/source6/heads/scalers/C=.01/0newfits; all4×2×8guards beforeeitherlabels.
P09positive DOESNOTtransfer; no fullsystemexpansion. Allthreepeoplepreviouslyexposed/selectionlimitsexplicit.
Next boundedrawtask-outsidecrop sourceaudit checksactualoldshiftedcropcoverage and tensor/cue provenance first.
BIDSsourceinventory500Hz/41EEG/0EOG/EMG finds7,592nominalseconds outsideofficialcrop, NOTgloballyuntested.
ActualoldordinaryBN-EAcrops2/3s and§277SSLstart4usedsome; §347exactrawSessionBN/step-matchedlatecrop gap registered.
§347 qualification3119303 PASS208s:60sourceEDFs/1898officialrows/1894latecrops; fixed3.5–7.5fitsactual800ends,
4unsupportedextraendsexcludedBOTHextraarmsbutoriginalrowsretained. Float32VOLTS→µVorder restoresA6prefixBITWISE;
oldhelperroundingposterior1.636e−6>1e−6 failurepreserved. No loosebounds/querylabels/newdataindependenceclaim.
Sourcecontrol3119319 PASS59s: actualnativeAref/Adisabled BITWISE; B/Csameinitializer,31base+31extra,source/labels/RNG/
normalization/chunks, activegradient/updates. Conservativefull2fit1438s,total3.246CPUh<8; no automaticoutermatrix.
Actualsame-sourceproducer/controlpreflight3119346 PASS; fullCPU6/24GiB15minB/C array3119363%2→CPU2/3mincollector3119365
released13:15CET. OriginalA6control retained; allB/C×3people×8orders plus sourcematchedproof beforelabels. No347scoreyet.
Approx7–12minconcurrent+queuebasedsourcecost, actualcompletionnotassumed. BCOMPLETE356sfit/389sscript withall
2480step/40base-batch/source/8orderguards, noquerylabels; Cstarted13:20:51cn11424GiB. Pending-only12GiBrightsizing
attemptfailedbecauseitstarted, NOresourcechange/cancel/requeue/speedupclaim. [Ownedevidence](crops347.md).
§347FULLCOMPLETE/allmatched/source/72bankguards beforelabels: P02A65.646/B64.582/C50.597%,
P03A56.228/B54.872/C54.277%, P09A64.061/B64.881/C54.289%. LateC−B−13.985/−.595/−10.592pp.
Original-sourceNLLC.5455>B.3552; C alreadylesscompatiblewithsourceofficialviews. No positive/familyclosure/fullsystemclaim.
Actual1.89667CPUh/0GPU, nojobs. Fullfit/deepcopyMaxRSSB6.74/C6.55GiB >3GiBsmoke;24GiBunchanged.
Next349 ONEBNphase/sourceaudit+onefrozen-source-normcounterfactualpreparing, ≤1CPUh, no newtraining/grid.
LegacyAlignedNet computesPOINTWISEcovcacheonallpublicX inclheld, unusedbyrawunalignedinference; no crossrowfit/
querylabels/parameterstats fromheld. Literalheld_transformed:falseoverstatescacheabsence; interpretednarrowlyinownreports.
§348 COMPLETEmetadata qualification:8308rows/201recordings/24people exactexportmapping,16guaranteednativechannels;
everyMIcrop.5sannotatedsound/MA2–3s. CMxcancondition-stresstest, notuniversalWORD/SUB/3classvalidation.
Existing12m8ccheckpointsusedCMx; old+1.67comparisonincrement NOTteacheraccuracy. No EEGaccuracy/fit.
[Qualifiedscope](external_instrument348.md). New351 auditssemanticmatchedShin/Schererdata/trialindependence beforeanyfit.
Sharedv4teacher remainsimmutable, reused by333withoutduplicatebuild.

## Current implementation and new completed results

§349 COMPLETE: frozen official-phase BNprior minus mixed-phase control changes P02−.808/P03−.138/P09+1.087
INDIVIDUAL-MODELpp, still below strongA by15.857/2.088/8.685. OriginalB/C ALL16banksBITWISE;
genericrecal controls0accuracychange. No recovery of latepooling damage, noall-late-data closure.
§351metadata qualified SchererWORD/SUB. §352 COMPLETE PROTEUS-only frozen externalbinarycov59.030%
versuspower53.260%,+5.770pp[80%people/day/order+3.958,+7.541],8/9up. SourceLOPOcov66.899%,10/10abovechance
beforeexternalgold. Allbanks/source/reset/orderguards beforelabels. External clinical/language/hardware/QClimits;
notFULL-SYSTEMDEV/three-class/sealed/rankgain. [Report](external_pair352.md).

User requests full own-C++ active training+inference. Newnative/ ports every active runtime class, including
fine-tuning, plus native linearoptimizer and ownrawFBC/SessionBN/AdamW-OneCycle source-loop. Implementation
is not qualification. Sourceprep/sampler/other-familytraining/PCA/gallerypipeline remain incomplete.
TorchScript migration artifacts execute in C++; that does not claim their architecture definitions were rewritten.
All C++ targets build and SHA/FIR guards pass; earlier failures stay recorded. Native linear3120215 passes
82iterations/88evaluations/maxprob6.38e−9/zeroheldlabels; one4thread19.823s is not matched whole-stack speed evidence.
Routing/OwnStage/CF/CP3120312 passes including10CPrefits125fusions/reset/deepcopy. CAR repair3120535 matches all96
NumPy CAR arrays exactly and passes original core/features/tail/reapply bounds with zero label changes.
[Core precision proof](native353_core_precision.md) is host-source component qualification only.
Temporal-mean repair3120986 matches all1599means byte-exact and passes known8, then unknownFTfbraw
exceeds unchanged1e−6 at1.1296823e−6. Exact failing FTwindow was not logged; bounded paired optimizer/state
diagnosis is authorized. PythonAdam lerp versus C++mul/add is a verified implementation difference, not a proven cause.
Own raw-FBC initializer3120984 ALL52tensors byte-exact for seeds0/1, globalCPU RNG unchanged;180actualcoresec.
REAL classical source fit3121076 ALL2278rows/features/prior/scaler/objective/exclusion guards pass;
sourceprob8.53e−9/ZEROchangedlabels, noquery/accuracy. Native95.5s is unpaired, not a speed benchmark.
Actual shared3539011coresec=2.5031CPUh; FTdiagnostic reserved≤600. SourceNNfirst-update stays held.
Other-family initialization/sampling/sourceprep/PCA/gallery/whole-worker compatibility remain incomplete.
§350 COMPLETE all16cells/guards/replay: conditional policy−baseline −.11934pp[95%descriptive−.28038,−.00109],
P04−.19741/P10−.09983; policy−JS-all +.16857pp. Both people remain below baseline; no promotion/rank claim.
[Final policy evidence](policy350_results.md); actual6.2575CPUh, all jobs complete, no further variants queued.

## Coordination and cadence

Root owns canonicaldocs/nativeintegration; external_supervision_17nov now prepares exact native sampling/RNG;
fresh_information_346 owns nativeNN/FT/training; performance_solver343 owns optimizer/classicaltraining/PCA.
External351/352 ownercompleted. Preserve allimmutable snapshots/failedbars and sharedledgerappenditions.
BothautomationsACTIVEthrough17Nov18:30UTC; Galvanionly≤32running/272submitted/28reserve.
SSHworks; jobscontinueafterclientloss. Provisional18:00CETcandidatecheckpoint, userretainsuploadauthority.
Scheduledresearch14:58:15–15:15:16CET COMPLETE17m01, [report](research_4h_17nov_1458.md). Next≈18:58CET.
The16:51CET delayed research trigger did not create another paper block; native implementation continued.

Latest18:50CET: native CPUneural+FT3121389 PASS including308byte-exacttrace fields; sampler/PCA/rawsourceprep qualified (sourceX1898BYTE_EXACT). Composedrawsourceupdate stillfails convbiasstate7.98e−5; r2operatorcounterfactual refuted/noeffect. Stageforward/backward diagnostic authorized, no fullfit/relaxedbound. Known353Slurm11301coresec plusseparatePCAstaticbuild73.81actualCPU. §354 ninefull40epochfits3121766/%8:8running/1pending after measuredsafe pending-onlyRAMresize; §355all40fitscomplete, actualCPU B1replay3121783→3121784 queued, qualityunknown. B64guardfailedpreserved; unchangedwholemetric/scorer. No newpackage/rank/promotion. Nativeimplementationtodaytarget, parityfirst, parallelscientificwavesindependent. See factorization354.md,relations355_handoff.md andnative353_neural_handoff.md.
