# Campaign status: 17 November 2026, 11:32 CET

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
Missing/invalid genuine original Sinc heldfold checkpoints block full returningP04/P10 validation.

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
All60NEWgenuineheldNNweights/72traces+576B1SIFbanksPASSEDownsource/export/reset/order/weightguards,
bothsourcefoldcontextspass. Historicalidentityunverified; oldCPU/GPUstrictQFfailurespreserved. Actualfrozen-stage
RouteB/TSEM/OwnStage/CF/CP engineering3118487PASS5m59, exactcomposition/offzero/inner/gate/labels, JSactive
pre+postrec;rec10/FT0/CPstop10. ENGINEERING_ONLY(cachedrecordingfoldview/originalall10inner), notRETURNINGDEV.
BothnewsourceCore/PCAfits3118591/3118601 pass~3min/8.4GiB, sourceheldrowsabsent; nativeSIF raw31canaries3118613 pass.
Full8order raw256banks3118623%1 run; newsourcecontexts3118631 pass. Four normalBN pilots pass; remaining24
genuineShallow/learnedFBCfits3118637%2→28assetcollector3118663 queued, sameweights botharms. No scoreyet.
Bridge scope is explicitly CONDITIONAL ENGINEERING with fixedproductionMAD/teacherEA-G/OOFcalibration.
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
§343 user-requestedCPUsolverrepair ACTIVE: SAME-node warmbaseline3118664 sklearn19.016s/112.84coresec,
native40.848s/243.96coresec. Source-span fullsmoke3118696 fits9.24s includingGram/eigen3.91s; rank2464,
onlycenteringnullremoved; fulloriginalobjective.0015985027/gradient2.48e−7, probdiff1.49e−4/zerosourcelabelchanges.
Provisionalcomponentresult; repeatedcontemporaneouspairs/isolatedmemory/heldinputproofpending. Noaccuracygain.
§344 finalrealzeroaux/RNG/costsmoke3118695 PASS58s; C/Didenticallambda/groupedderangement/actualnative8updatetwin.
FourCPU6fits3118755A/B+3118756C/D RUNNING→3118759collector; allsnapshotprobes/strongA/Bidentity beforelabels.
Initialcoststops preserved; hardtotal6.725CPUh/0A100. TeacherC++backend reused, NNtrainingPyTorch; noqualitygainyet.
Sharedv4teacher remainsimmutable, reused by333withoutduplicatebuild.

## Coordination and cadence

Galvani only,32running/272submitted with28reserve; verifylivecapacitybeforeeachdispatch.
SSHworks;12hourlocalidleretention isnotverifiedserver/VPNexpiry. Submittedjobs surviveclientloss.
Rootowns canonicaldocs/342collection; external_supervisionowns321; performance_solver343owns343;retention344owns344.
Previous339/340/341/342owners completed; no duplicatefits or unperformedcross-providerclaim.
Usefollowup_task towakecompletedowners. Preserveallsharedledgerblocks/immutableworkspaces.
Global scientific-discoveryskill applied; personalmasterdbaab0e. No cross-provider orWecoproviderexecution.
NativepairedCPU63–64msvscachedPython113–117msisnarrow;A100solve1.062svsCPU238.483schangesdevice/trajectory.
Noenergy/end-to-endclaim. Scheduledresearch10:49:35–11:04:39CET COMPLETE15m04;
[report](research_4h_17nov_1051.md). Next≈14:51CET; no duplicate delayed10:51trigger. Canonicalhistoryretained,notblind.
BothautomationsACTIVE; latest researchmotivates344, qualifiesCMx7-MMsource-taskidentity before newdatawork.
Next17Nov18:00CET decisioncheckpointprovisional;nonewqualifiedsubmission/topfiveproof.
