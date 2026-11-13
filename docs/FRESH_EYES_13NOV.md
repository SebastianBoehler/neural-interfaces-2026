# Fresh-eyes review, 13 Nov 2026

An independent check of whether the search has been narrower than the rules require. It was written in two phases.
Phase 1 used only the official material (submodule `2026-competition/` at `origin/main`, `bddaff3`, 11 Nov) and the
public dataset; no project note, ledger, evidence file or git history had been read when it was committed. Phase 2
(below, added afterwards) compares it with what the project did.

Measurements quoted in phase 1 are my own crude ones: linear models on hand-made features, per-session feature
standardisation over the whole session (not causal). They show the relative size of effects, not attainable scores.

---

# Phase 1: from the rules and the data only

## 1a. The allowed solution space

Each row quotes the governing text or code. "FAQ" is `codabench/pages/scientific_faq_bci.md`, "Terms" is
`codabench/pages/terms.md`, "Guide" is `codabench/pages/participate.md`, "Track" is `codabench/pages/competition_bci.md`.

### What is fixed

| # | Constraint | Governing text or code |
|---|---|---|
| 1 | The model sees only the EEG tensor, one window per call, shape `(1, 41, 480)`. | FAQ: "`predict(X)` receives only the EEG tensor. Participant, session, run, interface, and label metadata are not passed to the model." `benchmark_utils/streaming.py`: `pred = np.asarray(to_numpy(current.predict(X[i:i + 1])))` |
| 2 | The input preprocessing is fixed. | FAQ: "Source signals are notch-filtered at 50 and 60 Hz, band-pass filtered at 0.1–75 Hz at their recorded rate, resampled to 120 Hz, and expressed in microvolts. No learned scaling or clipping is applied." and "a submission cannot request raw 500 Hz sealed data." |
| 3 | One answer per call, final when returned. | `streaming.py`: `y_pred.append(pred[0])` directly after each call; nothing lets a model revise an earlier answer. FAQ: "One integer class index per window with shape `(B,)`". |
| 4 | A stream is one participant × session; nothing crosses it. | FAQ: "Each new session starts from a fresh copy of the submitted model and calls its optional `reset_state()`." and "State, windows, statistics, and parameter updates must not cross the session boundary." |
| 5 | Causal use of the stream only. | FAQ: "a model may use the current and previously delivered windows, but no future window or label." |
| 6 | Anything fitted during evaluation uses the session's own windows. | FAQ: "Any evaluation-time transformation or fitted statistic must use only the current and previously delivered windows of the current session, without labels." |
| 7 | No sealed data, no closed clinical data, no tampering. | Terms rule 4: "The sealed test split and non-permitted closed clinical datasets are not allowed." Rule 6: "accessing the sealed test split, or modifying the scoring harness, will result in immediate disqualification". |
| 8 | One hour, one GPU, for the whole test pass. | Terms rule 5: "The evaluation container must complete its test pass within 60 minutes on one A100 GPU." `competition_bci.yaml`: `execution_time_limit: 3600`. |
| 9 | One `submission.py`, worker-image packages only. | Guide: "Additional Python modules are not supported." and "Nothing is installed during evaluation". Also: "do not download competition data or write into the submission directory." |
| 10 | Seven sealed uploads, one a day, best one counts. | Terms rule 3: "7 submissions total in Track 02 ... at most 1 submission per team per day per track". `competition_bci.yaml`: `submission_rule: Force_Best`. |
| 11 | External data must be public, permitted and declared. | Terms rule 4: "Public datasets may be used when their original terms permit competition use. ... Declare every external corpus, and a compute estimate". |
| 12 | The top three must be reproducible. | Terms rule 7: "we rerun your training pipeline from the committed configuration and rescore the resulting weights against the sealed split. Scores within ±2σ of your submitted number stay on the prize roster". |
| 13 | Score: balanced accuracy per participant × session × interface cell, cells averaged equally. | `objective.py`: `cells = np.stack([info["stream_id"], context], axis=1)`, `balanced_accuracy_cell_mean=float(cell_scores.mean())`. |

### What is explicitly allowed

| # | Freedom | Governing text |
|---|---|---|
| A1 | Label-free adaptation inside the session, including training. | FAQ: "Causal, label-free adaptation using the current and previously delivered windows of the current session is allowed, including statistics-based alignment or gradient updates inside `predict`." |
| A2 | State across runs and interfaces of a session. | FAQ: "State may persist across Graz and BrainHero runs in the same session." |
| A3 | Any preprocessing on top of the delivered tensor, and any preprocessing of the public data. | FAQ: "You may preprocess the public training data differently and perform additional transformations inside `predict`." |
| A4 | Any public corpus whose licence permits it; resting runs too. | Terms rule 4 (above). FAQ: "Eyes-open and eyes-closed runs may be used for training or analysis". |
| A5 | Unlimited training compute; up to 15 GB of artefacts in any format. | Terms rule 5: "Train on whatever you have." Guide: "`weights.pt` # trained parameters, with any name or format" and "optional non-Python artifact". |
| A6 | Pretrained models, shipped or fetched. | Guide: "Other weights may download during each submission and count against the one-hour evaluation limit, so prefer shipping them when feasible." (So the worker has network access; only competition data may not be downloaded.) |
| A7 | Training on all ten public participants. | FAQ: "represented participants contribute only their already-public sessions". The 60/20/20 split "remains available for local validation"; nothing requires it for training. |

### Not rules

These are habits a team could easily adopt. I found no text behind any of them.

- **"Train only on PROTEUS."** Terms: "Training data is suggested to stay within the track." A suggestion.
- **"Use the public train/val/test split."** It is offered for local validation only (FAQ, above).
- **"Decode brain activity only."** No text restricts which property of the 41-channel signal is used. Eye movements,
  muscle activity and responses to the feedback display are all inside the tensor.
- **"One network."** `load_model` returns one object; it may hold any number of models, stored data or experts.
- **"The window must be used whole, at 120 Hz, as given."** A3 says otherwise: crop, re-reference, filter, split.
- **"Be fast per window."** Only the one-hour total exists. Its per-window value depends on the hidden test size,
  which the leaderboard's `Duration (s)` column lets a team estimate from its own uploads.
- **"Do not work out who the participant is."** The FAQ says identity is "not passed to the model"; it does not
  say a model may not infer it from the EEG. The Guide's "Do not rely on undocumented dataset names, paths, subject
  identifiers" sits under "For portability" and is about `meta` and paths. Recognising a represented participant from
  their signal and using their public sessions is consistent with "represented participants contribute only their
  already-public sessions". I mark it **allowed by silence**; it is a visible design choice and should be declared.
- **"The same decision rule for every window."** Nothing prevents a rule that changes with position in the session.
- **"Deterministic inference."** Not required, though rule 12 makes seeded inference prudent.

### Grey areas (ask the organisers before relying on them)

- **Fitting on shipped, labelled public data during evaluation.** Shipping the public windows in the ZIP is not
  forbidden, but row 6 read literally bars an evaluation-time fit that uses anything except the session's own
  windows. Precomputed experts chosen or weighted by session statistics stay on the safe side of that sentence; a
  supervised refit on stored source data inside `predict` may not.
- **Heavy fitting inside `load_model`.** The Guide asks for "a fully trained model". Row 12 audits a training pipeline.
- **Module-level state.** A global variable would survive `copy.deepcopy`; row 4 forbids using it. Not grey: prohibited.

## 1b. What the data-collection pipeline implies

Sources: the dataset `README.md`, BIDS sidecars and `*_events.tsv` under
`data/neural_compet/Dreyer2026Proteus/download/nm000290/`, `neuralfetch/studies/dreyer2026proteus.py`, the task
config `_motor_imagery_stream/config.yaml`, and `data/proteus_windows.npz` (2,655 windows, 14 sessions).

**Recording.** Brain Products actiCHamp with a 64-electrode actiCAP slim (active Ag/AgCl); 41 channels are released.
They cover AF3/AF4 and the F, FC, C, CP, P rows plus PO7/PO8; there is no Fp, F7/F8, T7/T8 or occipital electrode.
Reference is the left mastoid (M1), ground FpZ, DC-coupled, 500 Hz, stored in µV. The loader "does not filter,
re-reference, resample or clean artifacts"; electrode positions are the standard montage, not measured.

**Session.** Eyes-open and eyes-closed rest (3 min each, never streamed), then six cued runs in acquisition order:
runs 1–2 are the two calibration runs (`AcquisitionGraz`, `AcquisitionBH`, either first, "sham feedback"), runs 3–6
are four online runs with real feedback (two per interface, in any of several orders). A classifier-training
trigger (`OVTK_StimulationId_Train`) is in the event vocabulary, so the online feedback comes from a decoder fitted
on that day's calibration runs.

**Trial.** Graz: fixation at −1.5 s, beep at −0.5 s, cue at 0, feedback from 3 s, end at 8 s, exactly 12.5 s between
cues. BrainHero: no beep, 10–12.5 s between cues, slightly jittered events. The model's window is cue + 1 s to
cue + 5 s: two seconds of task alone, then two seconds with feedback on screen.

**Counts (measured from the event files).** Graz calibration: always 36 trials, 12 per class. BrainHero calibration:
nominally 36 but cut by a time limit near 380 s, so 30–36 trials, almost balanced. Online runs: 30 trials, exactly
10 per class (two 36-trial exceptions). A session is about 189 windows; a cell is about 96 (one calibration run and
two online runs of one interface). Calibration windows are therefore roughly 37% of every cell and arrive first.
Class order within a run is a random permutation of the balanced set; no two of the 84 runs share a sequence.

**What a decoder can exploit.**

1. *The session is its own best training set, but unlabelled.* Band power per channel, linear model: 0.52 across
   participants, 0.77 within a session (5-fold). Covariance tangent space, 8–30 Hz: 0.46 against 0.71. The signal
   is there; what fails to transfer is the mapping from signal to class. Anything that recovers a session-specific
   mapping without labels addresses the largest gap in the problem.
2. *A represented participant's public sessions.* For the two participants with three sessions, a later session
   scored with other people's data only: 0.40, 0.43, 0.65, 0.53. With their own earlier sessions added and weighted
   eight-fold: 0.58, 0.65, 0.78, 0.75. The same crude features, +12 to +22 points. Identity is not supplied, but a
   mean band-power fingerprint already picks the right participant out of ten in 3 of 4 later sessions after five
   windows and 4 of 4 with the whole session.
3. *Slow signals.* Activity below 4 Hz in 0.5 s bins scores 0.47 across participants and 0.66 within a session; the
   first two seconds carry more than the last two. Four frontal and temporal difference channels alone reach 0.43.
   Part of this is likely ocular or tied to the cue and feedback display. It is a separate information source from
   the oscillatory one and no rule excludes it.
4. *Balanced runs.* Exact balance makes the class prior known. As a sequential prior (exact filtering over the
   remaining counts, run boundaries given) it is worth only +0.004 at any base accuracy from 0.55 to 0.92: measured,
   not worth building. Its real use is as a constraint for unsupervised fitting (equal-size clusters, bias removal).
5. *Order.* Calibration first, online later; the interface alternates by run. A model can track where it is.
6. *The 30–48 Hz band.* Present after the pipeline (Nyquist is 60 Hz) but weak here: 0.35 across participants.

**What it must be robust to.** A lateralised reference (re-reference inside the model); DC coupling with a 0.1 Hz
high-pass, so slow drifts and electrode pops survive; standard-montage positions; one run with coarse amplitude
resolution; shortened runs; the change from sham to real feedback inside every session (frustration, error
responses, strategy changes); two interfaces with different displays and timing; a cold start in every session, with
no rest recording to align on.

## 1c. The system I would build from scratch

**The central design fact.** Each session delivers about 189 unlabelled windows in order, every answer is final, and
a within-session model beats a cross-participant model by about 25 points on the same features. So the system is a
session-level estimator, not a window classifier. A window classifier is only its cold-start component.

**Components.**

1. *Front end, fixed.* Average reference; filter bank (1–4, 4–8, 8–13, 13–20, 20–30, 30–48 Hz) to shrinkage
   covariances; a separate low-frequency time-domain stream; sub-windows before and after feedback onset.
2. *Session state.* Running reference covariance per band for recentring; running feature moments; a buffer of every
   earlier window with its embedding and soft label; a running posterior over "which represented participant, or
   new".
3. *Cold-start experts, trained offline on all ten participants plus external corpora.* Different model classes on
   purpose: a tangent-space linear model on recentred covariances; an SPD network with domain-specific batch
   normalisation (the test session is a new domain with running statistics); a compact convolutional network on
   aligned signals; a fine-tuned pretrained EEG encoder. Trained with windows cut at many offsets from the raw
   500 Hz trials, not only at cue + 1 s.
4. *Participant experts.* For each of the ten public participants, heads fitted with their own sessions weighted
   up. Mixed by the identity posterior, with an explicit "new participant" component so a wrong match costs little.
5. *Session learner.* An unsupervised, class-balanced model of the session's own embeddings (online EM over three
   components, or prototype updates, or gradient steps on a pseudo-label loss), started from the experts' soft
   labels. Its weight in the final decision grows with the number of windows seen.
6. *Sequence model.* A causal transformer over the session's embedding history, meta-trained on real and synthetic
   sessions to do step 5 implicitly. Highest ceiling, most data-hungry; built after 5 works.
7. *Decision.* Calibrated mixture of 3–6 with running removal of class bias. Use the full hour: the per-window
   budget is 3,600 s divided by the test size, which the leaderboard duration of earlier uploads reveals.

**Information sources and why.** The window (cold start); the session history (the 25-point gap); the participant's
own public sessions (+12 to +22 points where they apply); external corpora with the same mental tasks (ten
participants and 2,655 labelled windows are the binding shortage for anything learned across people); the raw
500 Hz public recordings (more training windows per trial, rest and inter-trial periods for self-supervision).

**External corpora I would look at first.** Same lab, same Graz protocol: the large Dreyer 2023 left/right imagery
set. Same task family: BNCI 2015-004 (Scherer 2015: word association, subtraction, imagery). Mental arithmetic and
word generation against rest with imagery in a sibling set: Shin 2017 A/B and Shin 2018. Licences need checking
one by one (rule 11).

### First ten experiments

Each is scored on a development measure that has two separately reported parts, because the sealed set has two
populations: (N) held-out participants; (R) later sessions of P04 and P10 with their earlier sessions in training.

| # | Experiment | What would change my mind |
|---|---|---|
| 1 | Build the two-part measure and its noise level (seed-to-seed and participant-to-participant spread). | If R and N move together for every change, one number is enough. |
| 2 | Accuracy by position in the session (windows 1–20, 21–72, 73–189) for the best current model. | If the curve is flat, adaptation is not working and is the first thing to fix; if only the first 20 are weak, stop working on cold start. |
| 3 | Adaptation ladder on one fixed encoder: none; recentring; plus bias removal; plus balanced prototypes or EM; plus gradient steps; against the within-session supervised ceiling. | If the supervised ceiling is within 2 points of "recentring only", the session learner has no room and I drop components 5–6. |
| 4 | Participant experts on R with oracle identity, then with inferred identity. | If oracle identity gives under 2 points on a strong encoder, drop component 4. If inferred identity loses more than half the oracle gain, improve the fingerprint first. |
| 5 | Open-set identification: ten known participants plus "new", from the first 5, 10, 20 windows, tested across sessions (P04, P10) and with held-out participants as "new". | If false matches on new participants exceed about 10%, gate softly or only after 20 windows. |
| 6 | Information sources inside the strong model: remove the slow stream; remove 30–48 Hz; first half against second half of the window. | If removing the slow stream costs under 0.5 points, it is redundant with the rest. |
| 7 | Training windows at many offsets from the raw trials (0.5–4 s after cue) and from rest periods as a fourth "none" class. | If more offsets do not help N, the shortage is participants, not windows. |
| 8 | External corpora: each one as pretraining and as joint training with its own head. | If none adds a point on N, stop; ten participants are then enough for these model classes. |
| 9 | Separate treatment of calibration and online windows, and of the two interfaces (detected from the stream). | If one model is within noise of four, keep one. |
| 10 | Meta-trained sequence model over the session against the hand-built session learner of experiment 3. | If it does not beat the hand-built learner on both parts, do not ship it. |

Already measured and set aside: the balanced-run sequential prior (+0.004).

**Where I expect the gain to be, in order.** Represented-participant handling; in-session unsupervised learning;
more participants through external corpora; diversity of the cold-start experts; everything else.

---

# Phase 2: against what was done

Read for this phase: `docs/STATE.md`, `docs/LEDGER.md` (index and headings), `docs/SOLUTION_SPACE_MAP.md`,
`docs/REVIEW_12NOV.md`, the five reports in `evidence/map_13nov/`, and the sibling sessions' notes of 13 Nov.
Ledger sections are cited as §n. Sizes are in points of the sealed score (score × 100) unless marked "dev".

## 2.0 Verdict

**The search was not too narrow in ideas.** Every component of my phase-1 design exists in the project, usually in
a stronger form than I sketched, and several things I proposed had been tested and found empty before I proposed
them. Two of my phase-1 statements were wrong where the project is right (2.1). The project's own map of 13 Nov
(`SOLUTION_SPACE_MAP.md`) already names most of the self-imposed limits; its reading of the rules agrees with mine
almost line by line.

**The narrowing is real, but it is in the funnel, not in the ideas.** Everything is judged by one instrument (ten
people, two of them returning, one of those an outlier), enters one way (as an addition to a chain whose constants
were tuned around its current members), and is released through uploads that were treated as something to protect.
About 150 runs since 7 Nov were aimed at effects the instrument cannot resolve. I add four specific points the
record does not contain (marked **New** in 2.3); none of them is a hidden shortcut of several points.

## 2.1 Where phase 1 was wrong or late

- **Cue order.** I wrote that class order within a run is a random permutation. For Graz runs it is not: positions
  follow a WORD → SUB → MI cycle with agreement 0.58 in calibration runs and 0.55 in online runs (chance 0.33);
  BrainHero runs show none (0.34–0.38). Measured for this review from the event files. The project found this in
  October and bounded it (§9, §55).
- **In-session learning.** I ranked it second. The project measured that the chain already equals a supervised
  model trained inside the session on its own features (75.9 % against 76.6 %, §30), that clustering a session
  reaches 44–65 % (§30), and that true labels are worth +1.6 to +2.9 dev while the chain's own labels are worth at
  most +0.4 (§81, §83, §101). My crude self-training gave +2 on a 55 % model, which says the same.
- **Balanced-run prior.** My +0.004 agrees with the project's oracle bound (+0.38 dev with true run boundaries).
- **SPD network, sequence model, many-offset training windows, external same-task corpora.** All tried (2.4).

## 2.2 `docs/STATE.md` section 2 against the primary sources

| STATE §2 item | Primary source | Verdict |
|---|---|---|
| Input `(1, 41, 480)`, from 1 s after the cue; one label out; no metadata | FAQ: "`predict(X)` receives only the EEG tensor." | Rule, correct. |
| Stream is one participant-session; "runs in order, windows in recording time" | `objective.py`: "its runs in recording order"; `bci_studies.py`: `_ORDERS = {"dreyer2026proteus": _proteus_run}` | True of the public code. It contradicts the "Order" item lower in the same section; see that row. |
| Deep copy at each session; nothing may cross | `streaming.py`: `clone = copy.deepcopy(model)`; FAQ: "must not cross the session boundary" | Rule, correct. |
| "Inside a session everything causal is allowed (adaptation, fine-tuning, identity inference from EEG)" | FAQ: "Causal, label-free adaptation using the current and previously delivered windows of the current session is allowed, including statistics-based alignment or gradient updates inside `predict`." Also: "Any evaluation-time transformation or fitted statistic must use only the current and previously delivered windows of the current session, without labels." | **Wider than the text.** Fine-tuning is covered. Identity inference has no primary text (the permission is a Discord answer). "Everything" hides the second sentence, which limits fitting with shipped labelled data. |
| Metric: cell balanced accuracy, mean over cells | `objective.py` | Rule, correct. |
| "60 min by rule (the worker's real timeout is 80 min; plan for 60)" | Terms rule 5; `execution_time_limit: 3600` | 60 is the rule. 80 is an observation with no primary source; correctly marked as not to be relied on. |
| ZIP with `submission.py` plus data files, no extra Python files | Guide: "Additional Python modules are not supported." | Rule, correct. |
| Worker image: Python 3.12, torch 2.14, no compilers, numba or cupy; "only pure-torch speed-ups are safe" | `tools/Dockerfile`: `FROM python:3.12-slim`, `ARG TORCH_CHANNEL=cu130`, torch unpinned | **Not a rule**: an observation of one image build, and torch is unpinned. The "pure-torch" line is a sensible self-imposed limit. |
| Uploads need the user's go-ahead; never cancel a running one | Terms: "Finished or cancelled by you: counts as one submission." | Cancelling: rule. Go-ahead: project policy. The same paragraph also says "A submission with final status `Failed` does not count." |
| Public data if terms permit; Scherer 2015 cleared on Discord | Terms rule 4: "when their original terms permit competition use" and "Original dataset and model licences remain in force." | The rule is right. **The clearance is not a primary source**, and organisers cannot waive a licensor's terms (non-commercial, no derivatives). No uploaded package contains Scherer as far as the upload log shows; check again before anything trained on it ships. |
| "Order (changed 10 Nov): the organisers removed every guarantee that windows arrive chronologically within a session" | Deleted from the FAQ in `8734b2f`: "There is no shuffling within a stream: runs are ordered by run number and windows are strictly chronological within each run." Still at `origin/main`: `streaming.py` raises `"... windows must come in time order."` when a recording's onsets go backwards, and the FAQ calls `objective.py` "and active phase configuration ... the executable references". | **Overstated.** The prose promise is gone. The public evaluator still enforces time order inside each recording. What cannot be checked is the private sealed adapter, which decides the order of runs. The project's own test (§62) found no reordering that fits its sealed scores. Risk: real for run order, small for order within a run. |
| Cue-order regularity | the data | A fact (confirmed above), not a rule. No text allows or forbids using it. |

Two limits are missing from section 2 and belong there: the audit ("we rerun your training pipeline from the
committed configuration", Terms rule 7), and the evaluation-time fitting sentence quoted in row 4.

## 2.3 Where the tested space is narrower than the allowed space

Already in the project's map, confirmed here: 36 % of the run time and 0.4 % of the package size are used;
uploads were never used to measure a component; every new part entered as posteriors into a frozen chain; several
directions were closed inside the noise of the measure.

**New 1. The returning-person stage was sized on two people when real two-session people exist outside PROTEUS.**
Every sealed gain from 7 Nov on came from work on returning people (+0.43 together with the early boost, then
+0.27 and +0.36). The stage strength (β 0.10) was
kept by a clause on P10, who is at 91 % and has little to gain; P04 wants 0.15–0.40 (§79, §111). The two real
people put the optimum at 0.10–0.15 (§95), the within-session proxies near 0.25 (§111), and those proxies survive
at about 0.19× on the sealed set. External people were used for the identity gate (224 of them) but never for the question the stage depends on: how far to trust a person's
earlier session in a later one. Scherer 2015 has nine people with two sessions and the same task family and is on
disk; Lee 2019 has 54 people with two sessions. I found no run of the personal stage across real external session
pairs in the ledger.

**New 2. Adaptation with a labelled source anchor was never tested, because of a rule nobody asked about.** The
in-session fine-tuning uses the chain's own labels only. The standard form of unsupervised adaptation fits on
labelled source data and unlabelled target data together, which keeps the pseudo-label errors from feeding
themselves; that is the failure §83 describes. It needs shipped public windows inside `predict`, which the FAQ
sentence in row 4 may forbid. The map lists the question (§6, question 1); I find no record that it was put to
the organisers.

**New 3. The cue cycle was tested in its weakest form.** As a per-window prior it adds little because the chain
is already confident on most windows (§9: at most +0.4 dev, one base, and weight 1 lost 2.9, which points to
uncalibrated fusion more than to an empty signal). Its distinctive property is different: it is the only signal
about the labels whose errors are independent of the EEG, and it accumulates over a session. Agreement with the
cycle over one session's Graz windows ranks a labelling that is 85 % right above one that is 55 % right in 93 % of
simulated comparisons, 75 % above 33 % in 98 %, but 85 % above 75 % in only 68 %: it finds gross errors, not
small ones. That is a label-free criterion for the per-session choices whose oracles are worth +1 to +2 dev and for which "no label-free rule" was found (§76, §117, §118), and a check
on sessions where the population model names the classes wrongly (P04's slow features read at 30 %, P05's swapped
classes). It is a protocol artefact, not decoding. Whether to use it is the owner's decision (2.6).

**New 4. Upload choice follows expected value, though the scoring rule rewards variance.** The board keeps the
best score (`submission_rule: Force_Best`). A candidate that is worse on average but has a wider spread is worth
more than a safe +0.2. Nothing in the record selects candidates that way.

## 2.4 My phase-1 design against the record

| Phase-1 component | In the project | Verdict |
|---|---|---|
| Average reference, filter-bank covariances, slow stream, half-windows | core1, Hilbert core, slow block | Present, stronger. |
| Recentring and running session statistics | causal log-Euclidean recentring, SessionBN | Present. |
| Participant experts mixed by inferred identity | routing v3, identity gate v4.1, own-space stage v5.1 | Present, and the only source of sealed gains since 7 Nov. Thinly validated (New 1). |
| Session learner (EM, prototypes, gradient steps) | SessionEM, TSEM, fine-tuning | Present; label-limited, as measured. |
| Convolutional experts, pretrained encoder | about 65 nets; five foundation models, 35–54 % alone | Present. |
| SPD network with session batch normalisation | one own implementation, J 54 against 68.5 (6 Nov) | **Weak form**: a 14-point deficit in one implementation looks like a defect, not a verdict. Low prior for a chain gain all the same, since it would share the chain's training people. |
| Sequence model over the session | meta-trained net (6 Nov, parity at best); §70 confounded; §98 below chance on one seed, not debugged; §141 fails by 8–12 in a consistent basis | Tried four times; one honest failure (§141), three weak tests. Ten people are too few to train it; I would not spend more here. |
| Training windows at many offsets; rest as a class | negative; +0.1 to +0.3 dev | Tested. |
| External corpora with the same tasks | Scherer, Shin, Dreyer, seven imagery corpora, in several roles | Tested widely. A search for this review found no public same-lab corpus with these three tasks. |
| Use the whole hour | 21.5 of 60 min | Absent so far; fine-tuning will use it. |
| Trial-reliability subspace (my own late idea: directions stable within a trial and varying between trials, as a label-free session space) | not in the record | **Tested for this review: negative.** In 8 dimensions the supervised within-session accuracy is 0.60–0.62 against 0.65–0.66 for plain PCA, and 3-means with oracle naming 0.36–0.38 against 0.48: the most reliable directions are session drift, not class. With drift removed (adjacent-trial differences) it only equals PCA (0.64 against 0.65–0.66; clustering 0.40 against 0.48). |

## 2.5 Readings that look like habit

- **"Extra members lower the chain even when strong alone."** This is a property of a fixed-weight average
  followed by stages tuned to its calibration, as STATE §5 now says. It was used for a month as if it were a
  property of the members.
- **Dev J as the unit of account.** Half of J is two people; P04 is about 30 % of J and perhaps 3 % of the sealed
  set. The "0.4–0.6× transfer" quoted for every lever is mostly this weighting, not a property of levers. Decisions
  would be clearer on two separate numbers: new people (person-equal, with its interval) and returning people
  without P04.
- **Bars inside the noise, in volume.** The measure resolves about ±0.3 to ±1. A pre-registered bar of +0.3 does
  not make a +0.29 a refutation or a +0.36 a finding. The map lists nine such cases.
- **"Keep compute busy."** Full queues produced many screens of effects below resolution. Compute use is not the
  constraint; the instrument and the two uploads are.
- **"The leaders' margin is returning people."** A conjecture (`REVIEW_12NOV.md` §7) built on a cohort text the
  organisers withdrew. Nothing measures it.
- **"In-session learning is exhausted."** True in the population's feature space. In a session's own spaces a
  supervised model reaches 81–82 % (§36); the project knows this and has no label-free route to it. That is the
  honest statement, and it is different from "exhausted".

## 2.6 What might not be allowed

1. **Reconstructing the cue generator (§27; "other families not tried" in the map).** If it worked it would
   deliver the label of the current and of future windows. FAQ: "a model may use the current and previously
   delivered windows, but no future window or label." I would not continue it and would not ship it.
2. **Any use of the cue cycle, including New 3.** No text covers it. It is knowledge of the protocol, not of the
   sealed labels, and STATE records the owner's decision to describe it in the write-up and not to raise it. Using
   it without asking risks the audit; ask first.
3. **A deliberate failure as a probe.** "A self-check that fails loudly is free by rule" (map §4, item 5) is fine
   for integrity checks (wrong montage, non-finite output). A failure that depends on a hypothesis about the sealed
   data would read one bit of it per attempt at no quota cost. Terms rule 3 states the cap exists to prevent
   "blind trial-and-error optimization on the test set"; rule 6 is open-ended ("including but not limited to").
4. **Identity inference.** Allowed by a Discord answer only. Declare it prominently in the method description.
5. **Reproducibility.** Rule 7 reruns the training pipeline. Rebuild recipes exist for v4.1 only; teacher
   pretraining scripts were untracked on a cluster that is now unreachable and were rebuilt on another on 13 Nov
   (§148; the retrained teacher reproduces the stored one). A top-three score that cannot be rebuilt is removed
   from the prize roster. This is the largest non-score risk in the project.
6. **Licences.** The gate's 224 external people and the teacher's corpora must all appear in the declaration
   with their licences (one note says Stieger's licence "has not been checked"; Shin 2017 is GPL).

Nothing in the uploaded packages, as described, breaks a rule I can quote.

## 2.7 Ranked list

Gains are my estimates of the sealed effect, with the reason. None is a promise; the first is the project's own
plan, which I rank first because it is also the best measurement available.

| # | What | Plausible gain | Why |
|---|---|---|---|
| 1 | **Upload 6: the fine-tuning + eye-net view + rest member candidate, with recognised sessions bit-identical to v5.1b**, once the research window closes and it has a worker-image replay. More dev draws will not sharpen it. | +0.2 to +0.6 | Dev: new people +0.84 to +0.95 on four draws (§152); new people are person-equal in dev and sealed, so they transfer more directly than J. Because recognised sessions are untouched, the sealed difference equals (share of unrecognised sessions) × (gain there): the first measurement of the split the whole plan assumes. |
| 2 | **Size the returning-person stage on real two-session people outside PROTEUS** (Scherer 2015 three-class subset first, Lee 2019 second): stage strength, which members, how trust should fall with the session gap. One day. | 0 to +0.4 | Returning people produced every sealed gain since 7 Nov and are about 40 % of cells on the assumed cohort; the two real people say 0.10–0.15, the proxies 0.25, and nothing else can arbitrate. Rule stated in advance: if at least 7 of 9 Scherer people want more than the β 0.10 equivalent, raise it; otherwise the question is closed on eleven people, not two. |
| 3 | **Upload 7 as the upside candidate**, not the safe one: item 1 plus the stronger returning-person dose from item 2 and fine-tuning using the remaining run time. | option value | Best-of scoring: a worse upload costs nothing but the slot. Choose by the chance of beating the current best by a clear margin. |
| 4 | **Three questions to the organisers today**: (a) is the order of runs inside a sealed session acquisition order; (b) may a package fit on shipped labelled public data together with the session's windows; (c) can the Discord answer on identity inference be confirmed. Protocol regularities only if the owner reopens the decision of 11 Nov. | decides items 5–7 | Each answer opens or closes a direction; the project has got answers this way before. |
| 5 | If (b) is yes: **fine-tuning with a labelled source anchor** (public windows in every in-session step, target windows unlabelled). | unknown; 0 to +1.0 on new people | True session labels are worth +1.6 to +2.9 dev and the chain's own labels at most +0.4; the gap is the pseudo-label feedback that a source anchor is designed to damp. Untested here, and published gains of this kind have shrunk to a fraction inside this chain before. |
| 6 | Only if the owner reopens it and the organisers agree: **the cue cycle as a session-level criterion**, not as a window prior: choose among stage variants or member weightings per session by agreement with the cycle on Graz windows, and flag sessions where the population names classes wrongly. Run the oracle-free dev test first. | 0 to +1.0 on new people | The selection oracles are +1 to +2 dev with no label-free rule; this is the only label signal independent of the EEG. Needs run boundaries and interface without metadata, and acquisition order on the sealed set. |
| 7 | If (a) is no or unanswered: **keep an order-robust variant ready** and decide on upload 7 with it in view. | insurance, 3–5 points at stake | The chain loses that much under a reordering (§62). |
| 8 | **Finish the rebuild pipeline before 21 Nov**: every trained part of the best package reproducible from tracked code on a reachable machine, external corpora and licences listed. | prize eligibility | Terms rule 7. |
| 9 | **Stop screening effects under about +0.5 dev.** Report new people and returning-without-P04 separately; drop J and the six-base count as decision criteria. | time | 2.5. |

Not worth more effort, on the evidence: a learned sequence model over the session; a proper SPD network; more
foundation models; the balanced-run prior; the trial-reliability subspace; more seeds.

**Plainly:** I looked for a missed approach of several points and did not find one. The team's suspicion is right
about how results are judged and released, and wrong about the breadth of what was tried.
