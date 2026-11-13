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
