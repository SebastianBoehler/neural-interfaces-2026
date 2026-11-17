# Campaign status: 17 November 2026, 06:20 CET

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
[Owned evidence](aux317_handoff_17nov.md). The guarded full-current-system consumer is running independently.

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
Separate immutable same-weight optimization is being prepared; fresh raw/active/reset/paired-worker guards
remain necessary. Separate readonly-current-input variant also registered. Initial guards3117415/3117429
failed before inference on a newly assumed persisted stream column; owning protocol derives stream fromsubject/session.
Failures preserved/deadcontrollers canceled; realNPZ/loginCPU schema smoke precedes replacementGPUguards.
No tolerance relaxation, candidate promotion or new accuracy score; failed source remains intact.

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

**§329 new released EEG prior:** MIT LaBraM-at-Scale Atom-Ldefault pretraining encoder46.3M, release2Nov2026.
Defaultpretrain weights are not the benchmark-selected best-full-ft epoch. Training overlap unknown.
A100preflight3117332 pending, forecast06:58 CET. Independent explicit2080Ti compatibility→ninefrozenhead
sourcepilot3117344 pending, forecast06:23:56 CET; one20minallocation, CPU-fitted-readout guard beforequerylabels.
Identical pretrained/random×two pooling×raw factor, rawonly plus unchanged rawSessionBNreference;
P04–P10fit/P02P03exposedqueries/P01excluded. No329numerical/accuracy result yet, no mixed-device bank.
Separate full-finetuning form§330 is being prepared prospectively, before329scores; release requires actual guards,
not favorable accuracy. [Fresh architecture review](fresh_architecture_17nov.md). No new outer matrix automatically.

**§331 independent raw state-space decoder PREPARED:** two32-wide oscillatory blocks over all480samples,
input-selective versus matched static and local controls, identical source-fitted287branch. No pretrained/code assets;
not Mamba/S5/S7 reproduction. Source-only40epochs/P04–P10fit/P02P03exposedqueries/P01excluded.
Independent complex direct-recurrence output/gradient, active/inactive-path and actual-cost guards precede a bounded
CPU pilot; no numerical/source score yet. [Prospective scope](oscillatory331.md). This is a different architecture,
not another incumbent head. The user authorizes preparation/execution; no automatic outer/chain expansion.

## Coordination and cadence

Galvani only,32running/272submitted with28reserve. Latest05:47CET account snapshot:37active/7running,
no dead dependency observed; verify capacity before each dispatch. SSH renewed01:02CET still works.
ControlPersist43200 means12-hour local idle retention, not VPN/server expiry. Submitted jobs survive client loss.
Root owns canonical STATE/STATUS/handoff; external_supervision owns321/317; fresh_architecture owns328/329/330;
fresh_solution_audit completed323diagnosis/CPU312 and now owns331numericguard/sourcepilot. Use followup_task to wake completed owners.
Do not overwrite others' ledger blocks or immutable snapshots; reconcile shared files before commit.

Global scientific-discovery skill applied; personal masterdbaab0e. No cross-provider execution.
Native pairedCPU inference63–64ms versus cachedPython113–117ms is narrow; A100solve1.062s versus CPU238.483s
also changes device/trajectory. No energy/end-to-end efficiency claim. No Weco service/provider was run.
Last scheduled research02:48:22–03:09:39 CET lasted21m17s; [report](research_4h_17nov_0250.md).
Extra artifact-informed fresh-chat review05:13:54–05:30:14 CET lasted16m20; canonical history retained, not blind.
It does not replace next scheduled≥15min block around06:51 CET. Both50mincampaign/4hresearch automations active.
Next17Nov18:00CET decision checkpoint remains provisional; local selected gains do not establish topfive readiness.
