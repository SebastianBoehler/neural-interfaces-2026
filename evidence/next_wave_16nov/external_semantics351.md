# §351 Semantic and provenance qualification

Status: **complete metadata audit; no accuracy or training**. Started 17 November 2026 at
12:40:29 UTC. Initial finding sent within ten minutes. Applied scientific-discovery and engineering-workflow.
This audit does not replace the scheduled 14:51 CET research block.

**Scherer WORD-versus-SUB is the best existing semantic match.** Both tasks occur within randomized
runs with identical beep/cue timing. Shin2018 provides genuine word generation versus baseline, with
matched trial timing, but contains no arithmetic/subtraction condition. Combining it with another
corpus would confound task with corpus. No such combined classification is proposed.

## Shin2018: a useful task, but no matching binary pair

The [primary paper, 13 February 2018](https://www.nature.com/articles/sdata20183?error=cookies_not_supported&code=aa)
describes initial-letter word generation and low-load fixation baseline, randomly interleaved within
three runs. Each trial has a two-second instruction, ten-second task, and 13–15-second rest. Beeps and
fixation during the task are shared. The other paradigms are n-back and discrimination/selection,
not subtraction. Participants are 26 new healthy adults, not the earlier Shin2017 cohort.

The [owner description](https://api-depositonce.tu-berlin.de/server/api/core/bitstreams/ff7415bb-fb47-461b-aba4-44488d0fe9e1/content)
explicitly defines raw task-onset markers 16 as verbal fluency and 32 as baseline.
[Owner code](https://github.com/JaeyoungShin/simultaneous_EEG-NIRS/tree/daac0f165792ac3eb8faaf004aac86db2ef049df),
revision `daac0f165792ac3eb8faaf004aac86db2ef049df`, 26 June 2019, connects word1/2/3 to WG.
Its internal B/C variable names differ from the published dataset letters; use actual file/task semantics.

The current [DepositOnce record](https://depositonce.tu-berlin.de/items/eb49d988-33d4-4eac-9ed3-36dcd711f9d2)
is DOI `10.14279/depositonce-5830.2`. Existing `shin_terms_17nov.json` retains owner-page hashes:
historical item says CC0, current item CC BY 4.0. Retain attribution; no new terms were accepted.

**New raw-metadata check:** all 3,120 exported rows map exactly to 26 people × three runs × 20 trials
× two crops. All 78 headers declare 1,000 Hz, 30 channels and µV. The 16 native target channels are
`F1 F2 FC1 FC2 FC5 FC6 C3 C4 Cz CP1 CP2 CP5 CP6 P3 P4 Pz`.
No per-recording channel rejection occurs here. Exported `onset` denotes marker or marker+4;
actual pixels start one second later. Row identity includes person/run/trial/crop; two crops are
not independent trials. Participant 017/run2 has the known filename-reference typo handled by the
existing exporter. This audit changes no raw headers.

The older 13-person `shin2018wg/windows_1-13.npz` uses classes `[S16,S32]`, with WG label zero.
The newer 26-person `shin2018/windows.npz` uses `[baseline,WG]`, with WG label one.
Numeric labels cannot be transferred between these exports without their class names.

Actual prior consumers: §278 fitted within-corpus LOSO (66.7% historical BA) and pooled WG rows into
PROTEUS WORD, which hurt in its tested form. §317 used first crops only, people 6–26 for auxiliary
fitting and 1–5 for validation, with a task-private WG/baseline head. These results do not constitute
PROTEUS-only frozen external validation. However, baseline is not either competing PROTEUS task,
so a direct two-class PROTEUS score on Shin would have no valid negative label.

## Scherer2015: paired semantic instrument with declared limits

[Primary acquisition paper, 18 May 2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0123727):
WORD means generating Spanish words from a presented initial letter. SUB means repeatedly subtracting
a one-digit number from an initial number between 15 and 30. Nine people with stroke/spinal-cord injury
participated on two different days. All five tasks were randomized within runs. This supports semantic
matching, while language, initial arithmetic range and clinical population remain transfer differences.
The primary analysis excluded artifact-contaminated trials/channels; the present export includes the
available raw rows rather than inheriting those analysis exclusions. Raw files lack a rejection mask.

The [owning description](https://lampx.tugraz.at/~bci/database/004-2015/description.pdf) defines raw labels
1 WORD, 2 SUB, 3 NAV, 4 HAND, 5 FEET. Each run has 25 trials, five per class. Task cue starts three
seconds after trial start, disappears after 1.25 seconds, and imagery ends at trial+10 seconds.
Both conditions share beeps; there is no ongoing spoken-prompt difference. Acquisition is 256 Hz,
0.5–100 Hz bandpass and 50 Hz notch, with mastoid reference. The pinned exporter stores cue onset and
cuts cue+1 to +5 seconds. That crop includes the final 0.25 seconds of the visual cue.
The three-second preparation period cannot supply a four-second silent pre-cue control.

[Owner catalog](https://bnci-horizon-2020.eu/database/data-sets) identifies CC BY-NC-ND 4.0.
Current STATE records the organizer's 11 November clearance and attribution requirement. This is the
existing competition-policy record, not a new licensor waiver or new terms acceptance. The proposed
use evaluates PROTEUS-trained models; it does not fit on or redistribute Scherer data.

**New raw-metadata check:** all 2,130 exported rows map exactly to raw person/day/trial and recover
142 runs across nine people and 18 days. Every recovered run has 25 trials with five of each class.
Within-run trial gaps are 12.50–13.50 seconds; run boundaries exceed 20 seconds. E/day2 and H/day2
have seven runs, all others eight. This explains 2,130 rows versus nominal 2,160, without assuming
randomly missing trials. Export `run=0` is therefore replaceable with verified grouping metadata,
without changing EEG pixels. WORD/SUB contributes 1,420 rows, ten per recovered run.
The 19 native target channels are retained; the other 22 interpolated channels must be excluded.
The small MAT-v5 reader decompresses the container but skips numeric X payloads entirely.

The historical exporter uses `trial/fs+3` rather than explicitly converting MATLAB one-based sample
positions. This possible 1/256-second convention difference is preserved and reported, not silently
corrected. New tests must use exactly the frozen exported pixels and their documented cue convention.

## Artifact and next action

[Compact evidence](semantics351_metadata.json) pins exact code/source hashes and counts.
Galvani `runs/diag/external_semantics351/audit.json` retains all recording headers, markers and raw hashes;
`shin_rows.tsv.gz` and `scherer_rows.tsv.gz` retain complete row/group/crop manifests.
One real smoke preceded the full audit. Full audit: 29.30 seconds wall, 28.67 CPU seconds, zero GPU.
No EEG numeric array was decoded; no model or statistical signal score was computed.

The minimum next test is §352: fixed PROTEUS-only WORD/SUB learning on true native channels,
matched power/full-covariance readouts, person-held PROTEUS sanity, then frozen Scherer evaluation.
No Scherer fitting, calibration, hyperparameter selection or pretrained teacher is allowed.
This measures condition discrimination across a distinct clinical cohort; it cannot establish
physiological causality, current-system benefit or independent confirmation of campaign-selected gains.
