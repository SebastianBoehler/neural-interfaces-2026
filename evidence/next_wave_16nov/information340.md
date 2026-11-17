# §340 source-only information and learning audit

Actual start17Nov2026,08:31:01UTC /09:31:01CET. Initial budget≤1CPUcoreh; no accuracy grid.
“Original raw neural reference” means the original PyTorch FBCNet/SessionBN recipe, not our C++stack.
This audit measured filtering and feature sensitivity on12real SOURCEwindows, without class labels,
P09EEG, P02/P03inference, optimizer updates or new individual/full-system accuracy results.

## What is actually saved

The337artifact tree contains fourFINAL model.pt files, final populationBN embeddings and prediction
banks, plus ridge maps. model.pt holds native modelstate/G/scale, no pre-finetune or auxiliary head.
The producer saves only after64pretext+40supervisedepochs; its auxiliary head remains a local variable.
Consequently pre-finetune/auxiliary/source-held reconstruction observables are MISSING, not measured.
The existing fine.initial_state_sha256 DOES identify the original post64pretext native state:
`3981dc104410449adfee1b34aeb84fc82918959d2d55f9470c2b21748acccc57`.
Finalcovariance model checkpointcdfb792f…45401, finalstate4ab0cef0…b5f9b, originalnodegalvani-cn123.
The earlier337 after-finefit fixed linear-probe errors do not locate failed learning versus forgetting.
Nor do333/337's linear errors establish information absence or a raw/foundation model family ceiling.

## Actual filter and teacher compatibility

Audit3118265PASS40sCPU6(9.095sscript); operator3118278PASS20sCPU6(1.044sscript).
Actual frozen coefficients restored from the original raw neural reference architecture:
9FIRbands4–8/8–12/.../36–40Hz,199/199/133/99/81/67/57/51/45taps, denominatoridentity.
Installed FilterBankLayer uses FFT convolution with SAMEzero-padding, not an ideal band selector.
ImplementationfileSHA d86be001cb22a4be5caa8a8b9eae2c9b8149691b6607528e3e3e070a185e3068.
Teacher has17overlapping bands1–3/2–4/3–6/.../36–45/40–58Hz and a separate1Hz-lowpass slow waveform.
Eachband contributes full+twohalf-window trace-.1-shrunk covariance blocks, not only pooled power.
Sourcefixed COMMON3reference/scaler, all51blocks+328slow, are unchanged.

| Frequency Hz | Largest of9actual FIR amplitude gains |
|---:|---:|
|0|4.07e−8|
|.5|.002603|
|1|.002874|
|2|.004437|
|3|.499815|
|4|.996031|
|6|1.001229|
|40|.997969|
|45|.501242|
|50|.003237|
|58|.002137|

Nominal4–40Hz is not a sharp support boundary. Transition bands preserve some3Hz and45Hz signal.
Low/upper-edge teacher targets have very different gains from the nominal passband, but are not proven absent.
More strongly, the actual finite480-sample linear filterbank operator(4320×480 PERCHANNEL) has full
numerical rank480 at relative singular-value threshold1e−12: σmin.000552113,σmax1.610022,condition2916.11.
Actual-source FFT versus independently assembled direct-convolution operator differs by≤2.10e−7.
Thus this finite linear PREspatial bank is injective in the measured numerical model, despite attenuation.
Inverse conditioning/quantization/optimization and later spatial/nonlinear/logvariance compression are
separate questions. This is not proof that the final1152features preserve task information or permit stable
recovery of all teacher statistics. It rules out the simple claim “the first filters mathematically erase all low EEG”.

## Real source interventions

Twelve fixed windows: first2globalrows from eachP04/P05/P06/P07/P08/P10, no source-class selection.
Identity restoredFINALfeatures match saved337embeddings within1.19e−7; recomputed teacher/cache4.17e−7.
IdealFFTprojection is applied within each current4secondwindow; its periodic boundary assumption matters.
The actual original filter still uses zero-padding. Feature changes use SOURCE6finalCfeaturestandard deviations;
teacher changes use unchanged COMMON3teacherstandard deviations. Different representations' RMS numbers
are sensitivity descriptions, not comparable classification utility or mutual information.

| Intervention | Removed raw energy fraction | Finalfeature RMS | Teacher diag/offdiag/slow RMS |
|---|---:|---:|---|
|remove0–1Hz|.7907|.1026|.0446/.1191/1.0492|
|remove1–4Hz|.1175|.1622|1.3630/.5278/.1366|
|remove4–40Hz|.0822|9.8730|8.8049/1.5792/.0289|
|remove40–58Hz|.0317|.2043|1.2294/.5688/.0168|
|keep4–40Hz|.9178|.2784|2.0218/.7692/1.0563|
|CAR|.6815|1.1884|5.00e−7/1.86e−6/3.91e−8|

For removing0–1Hz, squared filtered-difference/input-difference gain is.003197overall but2.536e−6
in center samples100:380. The longest FIR radius is99, so finite edges dominate this low-frequency leakage.
Removing1–4 gives gain.1001overall/.09467center;40–58gives.1667/.1601. These nonzero responses directly
counter a hard-band-absence explanation. The4–40removal changes the learned summaries far more strongly.
CAR removes common-mode signal that the teacher excludes by construction. The original raw neural
reference remains sensitive to that direction. This is target-irrelevant variation, not proof of task
irrelevance, artifact physiology, a sufficient generalization cause, or the proper normalization remedy.
Only12orderedsourcewindows/one final model were tested; no population/label/generalization inference.

## Smallest prospective learning-versus-forgetting instrument — NOT RUN

Repeat only ORIGINALC64pretext, with no extra epochs, no final40refit, no accuracy query or objective grid.
Use immutable337source/teacher/64batchtrace/seed0/CPU6 and preferably originalnodegalvani-cn123.
Instrument a pure snapshot callback before final40, saving nativeweights, auxiliaryhead and populationBN
source/held-Xfeatures. Require the existing post64 native-state hash AND original64loss/bodygradient
trace agreement BEFORE using the reconstructed stage. A mismatch means failed historical reconstruction;
report it, without claiming the original missing checkpoint was measured. No historical auxiliary checksum
exists, so auxiliary identity cannot be certified by an old auxhash; save its new hash and document that limit.

Then compare original raw neural reference40, reconstructedC64 and existingC64→40 using the SAME frozen
teacher, source6 splits, probe class/hyperparameters and target groups. Fit each stage's probe on SOURCE6
only to measure accessible information, and separately transfer a SINGLEsource-fitted frozen probe across
stages to measure basis/readout drift. A frozen probe failing while a stage-refitted probe succeeds is not
information loss. Include actual held-X auxiliary reconstruction before finefit and complete-source reconstruction
adequacy;64training-step losses from different batches are not that measurement.

Expected repeat cost≤60sCPU6 plus sourcefeature/probe≤60sCPU6, conservative≤.20CPUh. A failed hash gate
stops stage interpretation. This is a prepared proposal, not automatically launched by the initial1CPUh audit.
The source-only result would decide between low/slow task-target compatibility, insufficient learning and
finefit forgetting; no architecture change, moreepochs, P09label/precision/HPO or broader matrix is authorized.

Total allocated auditcompute(40+20)s×6=360CPUcoreseconds=.100CPUh, no A100; peakRSS2049332/1316440KiB. Read-only import inspection
uses additionalsmall login CPUtime, not training; total remains well below1CPUh. No live340jobs remain.
Full frequency/singular spectra, per-band/group interventions, exactsource rows/filter/checkpoint hashes in
information340_audit.json/operator.json; original remote snapshots immutable. No datasets/checkpoints committed.

Actual final audit clock17Nov2026,08:49:57UTC /09:49:57CET; elapsed18m56s since start.
