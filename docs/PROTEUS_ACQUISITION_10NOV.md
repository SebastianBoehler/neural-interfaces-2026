# PROTEUS acquisition, signal and decoder review · 10 November 2026

## What the hardware records

PROTEUS used a 64-channel Brain Products actiCAP slim and actiCHamp setup. The
release contains 41 scalp EEG channels at 500 Hz, with M1 (left mastoid) as
reference and FpZ as ground. The released traces are DC-coupled EDF+ with
per-file digital scaling. The earlier metadata audit on the seven source
people found active Ag/AgCl electrodes, extended 10–10 channel names, the same
channel order for both interfaces, and no recorded EOG or EMG. The coordinates
available to the loader come from a nominal montage. There are no measured
electrode positions or run-by-run contact impedances in the audited sidecars.
[Dataset deposit](https://github.com/nemarDatasets/nm000290#readme) ·
[Pinned loader](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralfetch-repo/neuralfetch/studies/dreyer2026proteus.py) ·
[Source sidecar audit](../runs/rank1_8nov/runtime_review/electrode_drift_wave/RECORDING.md)

The gel makes an ionic connection from skin to the Ag/AgCl electrode.
Electrode polarization and skin potential can add slow offsets, and their
impedance varies with frequency and contact. The active circuit sits at the
electrode and buffers the high-impedance contact before the cable. This
reduces sensitivity to pickup on that lead, but cannot undo a change in skin
potential or an artifact already present at the contact. Two simultaneous
impedance experiments show why a universal impedance threshold is weak
evidence. Ferree et al. found no significant band amplitude change up to
40 kΩ in their system, while Kappenman and Luck measured more low-frequency
noise at high-impedance sites, especially in warm, humid conditions.
[actiCAP flyer](https://www.brainproducts.com/download/flyer-acticap-slim-snap/) ·
[Active/passive electrode comparison](https://doi.org/10.1016/j.jneumeth.2014.05.012) ·
[Ferree et al.](https://pubmed.ncbi.nlm.nih.gov/11222977/) ·
[Kappenman and Luck](https://pubmed.ncbi.nlm.nih.gov/20374541/)

A useful measurement model for channel `i` is
`x_i = (L_i − L_M1)s + (a_i − a_M1) + ε_i`. Here `s` represents neural
sources, `L` represents how head tissues spread their fields, `a` covers
biological and contact voltages, and `ε` covers residual pickup, amplifier
noise and export error. Ground sets the recording system's common-mode
operating point. M1 is the reference used for the recorded differences.
Common-mode rejection suppresses voltage shared at the inputs. Unequal or
changing contact impedances can convert common pickup into differential
signal before that rejection helps. A noisy reference can affect all
channels. Subtracting the channel mean cancels an exactly common additive
term, and also removes broad spatial components. It can distribute a single
large channel artifact through the rest of the montage.
[Extracellular field physics](https://doi.org/10.1038/nrn3241) ·
[Experimental motion and common-mode model](https://pmc.ncbi.nlm.nih.gov/articles/PMC11479364/) ·
[Manufacturer discussion of reference choices](https://pressrelease.brainproducts.com/multicenter-studies/)

The actiCHamp series data sheet lists 24-bit conversion per channel, nominal
EEG input impedance above 1 GΩ at DC, and 100 dB common-mode rejection under
its stated test standard. Those are series specifications. The dataset does
not identify the amplifier revision, internal acquisition rate or exact
hardware anti-alias and decimation settings. The 500 Hz figure describes the
released stream. EDF+ stores integer samples with physical and digital
minimum and maximum fields, so exported quantization can be coarser than the
converter. The deposit flags one run with a resolution coarser than 1 µV
per count, final record padding marked `BAD_ACQ_SKIP`, and 13 cued runs with
fewer trials than usual. Those documented issues warrant source quality
checks. They do not establish that converter noise or missing samples drive
classification errors.
[actiCHamp specifications](https://www.brainproducts.com/solutions/actichamp/#technical-specifications) ·
[EDF format](https://www.edfplus.info/specs/edf.html) ·
[Release issues](https://github.com/nemarDatasets/nm000290#known-issues)

## What survives the official pipeline

The pinned competition adapter uses the NeuralBench streamed task and
NeuralSet's run-level extractor. The loader reads the released traces
without re-referencing. The extractor applies a zero-phase 50/60 Hz notch
expanded to in-range harmonics at 500 Hz, then a 0.1–75 Hz band-pass.
It calls MNE's resampler on the run before slicing the cue windows.
The pinned local MNE version defaults to FFT resampling with an
anti-alias cutoff at the new Nyquist frequency. At 120 Hz that is 60 Hz,
so the upstream 75 Hz limit does not imply useful 60–75 Hz information
in the challenge input. The band-pass filters contiguous sections
separated by the specified acquisition-skip annotations.
[Pinned competition adapter](https://github.com/neural-interfaces26/2026-competition/blob/a66bc23d1a0e686b895c093a642b45c74970758b/tracks/bci_decoding/datasets/bci_studies.py) ·
[Pinned extractor](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralset-repo/neuralset/extractors/neuro.py) ·
[MNE resampling behavior](https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html)

Each example is 41 × 480 samples in microvolts, with no extra scaling or
clamping. It spans cue+1 to cue+5 seconds. Feedback starts near cue+3
seconds, so both pre- and post-feedback activity are present. The source
annotations give event-marker times, rather than a measured display or
audio latency. Run-level filtering before cropping can spread transient
activity across a crop boundary. A pre/post comparison is therefore an
approximate timing contrast. The solver can act causally over arriving
windows even though preprocessing the released signal has this temporal
support. The official metric averages balanced accuracy over
participant/session/interface cells. No interface, cue, or run label
is passed to the model as an inference feature.
[Pinned task configuration](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/eeg/_motor_imagery_stream/config.yaml) ·
[Track 2 scientific FAQ](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md) ·
[Source timing audit](../runs/rank1_8nov/runtime_review/electrode_drift_wave/RECORDING.md)

Four seconds at 120 Hz give Fourier bins 0.25 Hz apart. Actual resolving
power is coarser because of window length and tapering. A 4-second window
contains roughly 32–52 cycles at 8–13 Hz. Filtering makes successive
samples correlated, so 480 samples do not provide 480 independent
observations of a covariance matrix. A 41-channel covariance has 861
unique entries. A two-second crop reduces the evidence for each entry
further. These calculations motivate shrinkage or structured pooling
and favor compact tests before increasing model capacity.
[Thomson on spectral estimation](https://doi.org/10.1109/PROC.1982.12433)

## What may carry class information

Synaptic transmembrane currents contribute strongly to extracellular
potentials. Scalp electrodes record spatial mixtures of many sources
through conductive head tissues. In a small unilateral motor imagery
study, central mu and beta rhythms desynchronized over the contralateral
sensorimotor region. PROTEUS asks for the non-dominant hand, so a fixed
left-versus-right rule would assume unavailable information. A separate
two-day MEG experiment found stronger beta decreases for subtraction in
left superior parietal regions and for word generation in left temporal
regions. This supports a band and region hypothesis. The MEG result does
not map one-to-one onto this scalp montage, task implementation, or person.
[Field origin](https://doi.org/10.1038/nrn3241) ·
[Motor imagery EEG experiment](https://doi.org/10.1016/S0304-3940%2897%2900889-6) ·
[Subtraction and word generation MEG experiment](https://doi.org/10.1002/hbm.26284)

Several distinct processes can mimic useful class features:

| Process | Likely footprint and discriminating check |
|---|---|
| Eye motion and blinks | Frontal slow fields, lateral eye-movement patterns, or short saccadic muscle transients. Compare matched frontal projection or masking with a spatial control, but treat all inferred events as proxies without EOG or gaze. [Simultaneous EEG and eye-tracking experiment](https://doi.org/10.3389/fnhum.2012.00278) |
| Jaw, forehead, scalp or neck muscle | Activity can spread across electrodes and overlap beta, so a high-frequency mask is an incomplete check. Compare power-only and covariance gains across frontal/temporal and central/posterior families under the same fit. [Measured facial EMG spectra and topography](https://pubmed.ncbi.nlm.nih.gov/12948787/) |
| Contact movement, gel bridging, or cap position | Pops, drift and frequency-dependent gain can change a channel; a bridge can make neighbors nearly identical. Inspect pair-difference variance and persistence, and test a fixed physically plausible channel-pair blend against equal-count rewired controls. This estimates sensitivity to the stated simulation. [Motion measurements](https://pmc.ncbi.nlm.nih.gov/articles/PMC11479364/) · [Bridging measurements](https://pmc.ncbi.nlm.nih.gov/articles/PMC3943722/) |
| Shared source or reference | A single field can make many electrodes covary at zero lag. Real coherency can be a useful feature but is not evidence of cortical connectivity. Compare common-reference-invariant and matched control views while holding the readout fixed. [Volume-conduction experiment](https://pubmed.ncbi.nlm.nih.gov/15351371/) · [Decoder-weight interpretation](https://pubmed.ncbi.nlm.nih.gov/24239590/) |
| Interface and feedback | Inria describes BrainHero as a gamified Graz derivative with music during training. Cue appearance, gaze, motor tension, practice and feedback may all covary with the interface or run stage. Sham versus online comparisons within a source participant can describe this shift. They do not isolate any one cause. [Inria project report](https://radar.inria.fr/report/2024/potioc/index.html#brainhero-for-the-anr-proteus-project) · [Release description](https://github.com/nemarDatasets/nm000290#readme) |

The resulting model hypothesis is a regularized, band-specific sensor
representation with separate power and spatial-covariance views. Preserve
relative spatial structure while testing sensitivity to amplitude,
reference and localized corruption with one controlled change at a time.
Leave subjects out of the entire fitting pipeline. Compare an exact
baseline, a candidate and controls on identical source rows. The seven
source people each contribute one session, so this can test generalization
to source people and interface strata, but not establish cross-day
robustness. No protected participant data were opened for this review.

## Measurements on the seven source people

The source audit read 56 EDF headers and matched the 1,323 official
source windows to their raw 500 Hz samples using run and window onset.
The EDF resolution was 0.468 µV/count at minimum, 0.573 at the median,
0.933 at the 95th percentile and 1.271 at maximum. Only P06's online
Graz run 6 had a channel above 1 µV/count. The median step would
correspond to 0.165 µV RMS if quantization error were uniform and
uncorrelated. That calculation concerns export precision.

Across the four interface/feedback combinations, the median ratio of
power at 49.5–50.5 Hz to adjacent 46–48 and 52–54 Hz bands was
37.9–41.1 in the EDF and 0.469–0.590 after official preprocessing.
Residual mains power is a low priority relative to spatial mixing.
The processed 25–45 Hz to 8–25 Hz power ratio showed no large
consistent BrainHero increase confined to the frontal channel group.
Band power alone cannot tell muscle from neural activity.
[Source audit code](../runs/rank1_8nov/runtime_review/proteus_signal_audit/acquisition_audit.py)

A synthetic 20 ms or 200 ms voltage pulse passed through the local
notch, band-pass and resample operations left a peak roughly 5.5–5.7%
as large as its main peak in the 250 ms preceding onset. One
sample's impulse had a larger relative pre-response. The values
illustrate the filter response. They do not measure cue leakage in
PROTEUS. A crop with a guard interval is preferable for a causal
interpretation of the raw recording.

In the official source windows the spatial mean of 8–25 Hz EEG had
about 0.79–0.82 times the typical channel RMS. Replacing the spatial
mean by a channel median changed the reference by about 0.11–0.12
times typical channel RMS at the median, with similar values in
BrainHero and Graz. The shared component is large. These figures
do not tell how much comes from the mastoid, shared neural fields,
or other sources. A matched reference test can measure predictive
sensitivity without assigning the component to any one source.

In the source-only half-window probe, the typical absolute change
in Fisher-z rank correlation was 0.244 for BrainHero and 0.241 for
Graz at 8–13 Hz, and 0.152 and 0.148 at 13–25 Hz. These are
participant-equal means of within-person medians. The corresponding
typical absolute whole-window values were 0.83–0.88. The result
motivates testing per-edge reliability. It cannot separate estimator
noise from a genuine change during the task.

## How the experimental results change the model choice

At the second-half readback, Galvani had 16 of the 48 overnight
results. Two auxiliary bootstrap seeds on the same seven people
measure training variability, not independent replication across
new people or later sessions. The additional waves had not yet
become eligible. The figures below are balanced-accuracy percentage
point changes from the fixed 10% comparator.

| Signal or readout | Source evidence | Design implication |
|---|---|---|
| Dense mid-band rank covariance | 3083051 gained +0.706 overall and +1.041 BrainHero over six draws. The lead over same-band ordinary covariance was only +0.130. In 3083135 rank alone remained +0.780 BrainHero, ahead of fusion on all six draws. | Keep the frequency range and broad spatial information. Measure the independent contribution of reference choice and rank transforms. |
| Log-covariance readout | First two overnight draws gave +0.716 overall and +1.135 BrainHero for the rank logistic control. The rank class-template model gave +0.348 overall. | A regularized discriminative head is the strongest starting point. The generative head's assumptions did not buy better accuracy so far. |
| Lag energy | Bounded energy repeated at +0.587 overall in 3081432, with +0.800 on Graz. In the first two overnight draws, arithmetic half-window energy gave +0.556 overall and geometric pooling gave +0.370. | Preserve lag energy as an independent comparison. More aggressive suppression of changes within the trial can discard useful information. |
| Hard-coded sensor topology | The physical graph lost −0.487 overall and lost to rewiring. Mirrored Hann pairs gained +0.364, only +0.055 ahead of one rewired control. | Keep dense covariance in the immediate comparison. Scalp spread does not justify a specific local graph here. |
| Learned temporal and interaction models | Patch ordering lost −0.223 overall and −0.520 BrainHero in the first two draws. Markov transitions and trainable products did not beat the best matched controls. | More capacity and finer timing are lower priorities on these source data. Their measured performance does not establish that all temporal features lack signal. |
| Equal fusion and gating | 3083135 traded BrainHero gains for Graz gains. 3083136 gave +0.624 overall vs +0.627 for equal fusion and +0.622 for a shuffled-target selector. | Improve the feature estimator before adding another learned routing rule. |

The next architecture should keep a fixed spectral-power anchor and
a dense, regularized mid-band covariance branch. Keep branch weight
and inference budget fixed. Covariance represents pooled second-order
patterns expected from a spatial mixture. Full-window estimates can
still vary because band-limited samples are correlated and a single
contact transient can affect an entire window. A regularized classifier
can learn which directions separate the commands from a limited number
of people. It should not be interpreted as a source map.
[Source campaign evidence](PROTEUS_SOURCE_10NOV.md) ·
[Motor-imagery covariance decoder](https://pubmed.ncbi.nlm.nih.gov/22010143/) ·
[Robust EEG covariance estimation](https://pubmed.ncbi.nlm.nih.gov/28410052/) ·
[Filter-bank variance readout](https://pubmed.ncbi.nlm.nih.gov/33018625/)

## Follow-up protocol

1. **Reference sensitivity.** Train the same mid-band rank decoder
   under the recorded mastoid reference, common-average reference,
   a 20% trimmed reference and a same-size fixed channel-subset
   average. All arms share CAR-based power. Evaluate clean windows
   and two label-independent perturbations. Add the same window-local
   pulse either to every channel or to one chosen channel. Compare
   the change in auxiliary probabilities and in balanced accuracy.
   The archived overlap solver remains fixed in the stress checks.
   Shared-pulse cancellation is an algebraic check. Local-pulse
   retention and clean score can distinguish robustness from loss
   of useful shared structure. This is a sensitivity test with one
   stipulated noise amplitude. [Robust reference methods](https://pubmed.ncbi.nlm.nih.gov/24975291/)
2. **Covariance reliability.** Compare the whole-window Fisher-z
   rank correlation, the mean of two half-window estimates and a
   shrinkage estimator that moves inconsistent edges toward their
   fitted source median. Fit the typical noise and signal variances
   using only the current fit fold. Compare against an edge-permuted
   reliability score that preserves the same shrinkage recipe, and
   against power alone. The half difference is an uncertainty proxy
   and may also reflect a real within-trial change.

Use six fresh auxiliary seeds 4800–5300, three archived overlap
seeds, the same person-disjoint source folds, and exact fixed 10%
comparator parity. Both tests replace the same 10% vote. The clean
promotion bar is +0.8 overall, +0.5 on BrainHero, nonnegative Graz,
+0.3 over every named matched control, and a gain in five of six
auxiliary averages. P04/P09/P10 remain outside fitting, selection
and scoring. No Codabench upload is scheduled.

The acquisition and evidence review ran from 18:00:33 to 18:11:11
Helsinki time. The first half audited the recording chain; the
second half compared source experiment results and designed the
tests. Slurm accepted array 3083698, with reference sensitivity
in task 0 and covariance reliability in task 1.

## First follow-up result

Task 3083698_0 completed on all six auxiliary seeds. Clean source
gains were +0.639 points overall, +0.873 BrainHero and +0.405 Graz
for the recorded mastoid view. Ordinary CAR gave +0.471 overall,
+0.843 BrainHero and +0.099 Graz. Trimmed CAR gave +0.020 overall,
+0.156 BrainHero and −0.116 Graz. The fixed-subset control gave
+0.207 overall. The trimmed arm missed the frozen source gate.

The shared pulse changed the mastoid auxiliary posterior by mean
total variation 0.0758. It changed each rereferenced posterior
by approximately 0.000002–0.000003, as expected. The same pulse
placed at one electrode caused more change, even after reference
subtraction. The clean results favor preserving the recorded
spatial view for this head. They do not assign its predictive
information to mastoid voltage, neural fields, or artifacts.

Task 3083698_1 also completed. Whole-window Fisher-z rank covariance
was the strongest arm at +0.752 overall, +0.710 BrainHero and
+0.794 Graz. Arithmetic half-window pooling gave +0.679 overall.
Adaptive shrinkage gave +0.551 and the edge-shuffled shrinkage
control gave +0.545. The +0.006 matched difference provides no
support for the edge-specific reliability weighting in this test.
Against the untransformed mastoid rank arm on the same six draws,
Fisher-z gained +0.113 overall (four of six), lost −0.163 BrainHero,
and gained +0.389 Graz. This trade-off warrants keeping both as
comparison arms. Neither experiment cleared its promotion gate.

The separate overnight batch still had 16 of 48 results at the
last check.

## Second covariance batch

Slurm accepted array 3083704. Each of its four tasks evaluates six
fresh auxiliary draws, 5400–5900, on the same source participant
folds and three archived overlap seeds. Each arm replaces the same
10% vote and shares the same power features.

| Task | Approach | Fixed comparison |
|---|---|---|
| 0 | Refine the full-window rank representation by applying Fisher-z only at 13–25 Hz | Raw in both bands, Fisher-z in both, and Fisher-z at 8–13 Hz only |
| 1 | Refine the logistic prior by encouraging the same electrode-pair weights across the two bands | Independent weights and a matched constraint on rewired pairs |
| 2 | Estimate partial correlations from a ridge precision matrix | Same ridge followed by ordinary correlations, and whole-window Fisher-z rank correlations |
| 3 | Classify each trial by its Gaussian covariance likelihood with class models shrunk toward a shared estimate | Class models restricted to a common eigenbasis, logistic on the same covariance with the same power vote, joint Fisher logistic, and power alone |

The precision model asks whether conditional relationships predict the
command after accounting for the other electrodes. They can still
reflect spatial mixing, reference, and artifacts. Ridge shrinkage
makes the inversion well-conditioned. The likelihood model treats
the measured covariance as the signal to explain and fits each
class's covariance from source training rows. Its discriminative
control uses the same class logit scale and power vote. These tests
use the recorded mastoid view and do not require inference metadata.
[Covariance conditioning](https://doi.org/10.1016/S0047-259X%2803%2900096-4)

All controls, arms and the source promotion threshold are fixed
before scoring. P04/P09/P10 stay out of fitting, tuning and scoring.
The run does not schedule a Codabench upload.

Task 3083704_0 finished. The prespecified candidate, Fisher-z at
13–25 Hz alone, gained +0.448 points overall. Applying Fisher-z at
8–13 Hz alone was the strongest comparison at +0.829 overall,
+1.054 BrainHero and +0.604 Graz. Its lead over untransformed
rank covariance was +0.154 overall, below the +0.3 matched-control
bar. The result supports an independent comparison of the two
frequency treatments before any promotion.

All four tasks subsequently completed. The table shows percentage
point changes versus fixed 10%, averaged over the same six draws.

| Arm | Overall | BrainHero | Graz |
|---|---:|---:|---:|
| Fisher-z at 8–13 Hz only | +0.829 | +1.054 | +0.604 |
| Same-pair cross-band constraint | +0.657 | +0.876 | +0.438 |
| Rewired constraint | +0.669 | +0.644 | +0.694 |
| Ridge partial correlation | +0.326 | +0.198 | +0.455 |
| Class covariance likelihood | +0.114 | +0.262 | −0.033 |

The rewired constraint led the Fisher arm on mean Graz, but
beat it in only two of six Graz draws and none on BrainHero.
No alternative had a higher mean on both interfaces. Ridge
marginal correlation beat it on both interfaces in one draw,
while losing in the mean on each. None of the prespecified
candidates passed its gate.

The overnight batch still had 16 of 48 results. Its rank logistic
control gained +0.716 overall, +1.135 BrainHero and +0.298 Graz
across auxiliary draws 4200 and 4300. These are different draws
from the six in 3083704. To remove that ambiguity, the next
source-only run replays the overnight control on those two
draws and tests it against the 8–13 Hz Fisher arm on eight fresh
draws, 6000–6700. It also compares a fixed ±0.95 cap in the
lower band, both with and without Fisher-z, to test whether
the result depends on the most correlated pairs. The cap
is a diagnostic; the unmodified 8–13 Hz Fisher arm is the
preregistered confirmation candidate.

Slurm accepted the confirmation as 3083806. Task 0 runs the
eight fresh draws. Task 1 verifies the two archived overnight
control scores and compares the methods on those same draws.
