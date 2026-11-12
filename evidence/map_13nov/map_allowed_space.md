# Allowed solution space for Track 2 (BCI decoding): what the official rules and code permit, forbid, or leave open

Nothing was edited, created, trained or fetched. All findings come from files on disk.

## 0. Sources, revisions and gaps

**The pinned submodule pages are stale; the current rules are already in the submodule's local git object store.**
- The working tree is `a66bc23` (14 Oct 2026). Its pages still describe the October rules.
- `origin/main` is `bddaff3` (11 Nov 2026, 01:35 CET), fetched but not checked out. I read it with `git -C 2026-competition show origin/main:<path>`.
- Between the two, only docs and YAML changed for Track 2. `benchmark_utils/`, `tracks/bci_decoding/objective.py`, `datasets/`, `requirements.txt`, `tools/` and the ingestion and scoring programs are byte-identical (`git diff --stat a66bc23 origin/main`).

Citation keys: **[pin]** is the working tree; **[latest]** is `origin/main` at `bddaff3`, with line numbers from `git show … | cat -n`. Code citations are valid for both.

Prefix `C` = `/Users/sebastianboehler/Documents/GitHub/neural-interfaces-2026/2026-competition`; `R` = `/Users/sebastianboehler/Documents/GitHub/neural-interfaces-2026`.

| Key | File |
|---|---|
| terms | `C/codabench/pages/terms.md` |
| faq | `C/codabench/pages/scientific_faq_bci.md` |
| desc | `C/codabench/pages/competition_bci.md` |
| guide | `C/codabench/pages/participate.md` |
| timeline | `C/codabench/pages/timeline.md` |
| reg | `C/codabench/pages/registration.md` |
| yaml | `C/codabench/competition_bci.yaml` |
| phasecfg | `C/codabench/phases/warmup/bci_decoding/config.yaml` |
| ingestion | `C/codabench/ingestion_program/ingestion.py` |
| scoring | `C/codabench/scoring_program/scoring.py` |
| objective | `C/tracks/bci_decoding/objective.py` |
| adapter | `C/tracks/bci_decoding/datasets/bci_studies.py` |
| streaming | `C/benchmark_utils/streaming.py` |
| base_solver | `C/benchmark_utils/base_solver.py` |
| nb_task | `C/benchmark_utils/nb_task.py` |
| data | `C/benchmark_utils/data.py` |
| metrics | `C/benchmark_utils/metrics.py` |
| Dockerfile | `C/tools/Dockerfile` |
| nbconfig | `R/.venv/lib/python3.12/site-packages/neuralbench/tasks/eeg/_motor_imagery_stream/config.yaml` (NeuralBench 0.4.0) |
| nfstudy | `R/.venv/lib/python3.12/site-packages/neuralfetch/studies/dreyer2026proteus.py` |
| Discord (secondary) | `R/runs/archive_scratch_11nov/discord_11nov/discord-research-brief.md`, `wider-discord-context.md`, `claude-handoff.md` |

**Gaps:**
- **Website not read.** `rules.html`, `prizes.html`, `tracks.html` and `participant-guide.html` on neural-interfaces26.github.io are not on disk. `C/README.md` l.14 names the website as ground truth for "tracks, datasets, schedule, rules". Prize conditions and the website's cohort description are therefore unverified.
- **Sealed phase bundle is private.** Only `codabench/phases/warmup/` exists in the repo. The sealed phase ships its own benchmark copy, config and optional `datasets/*.py` (ingestion l.3-6, l.68-74; `C/tools/create_bundle.py` l.92-94). The sealed adapter, ordering and context definition cannot be verified from public code.
- **Organiser statements on Discord** exist only as our second-hand summaries, not as an export.
- **Nothing after 11 Nov 01:35 CET** is covered.

## 1. Training data

| Item | Verdict | Governing text |
|---|---|---|
| Public PROTEUS release, all 14 sessions of all 10 people, including the warm-up validation and test participants and their labels | ALLOWED by implication; no explicit Track 2 sentence | Only "the sealed test split and non-permitted closed clinical datasets are not allowed" (terms [latest] l.38). "The public PROTEUS release remains the reference training and local-validation data" (faq [latest] l.26-27). "Represented participants contribute only their already-public sessions" (l.37-38). The split "remains available for local validation and does not reveal the sealed allocation" (l.77-80). nbconfig l.8-9 calls it a "local subject-disjoint benchmark on the training release, not the official split". |
| Rest runs (eyes open, eyes closed) | ALLOWED, explicit | "Eyes-open and eyes-closed runs may be used for training or analysis under the dataset licence" (faq [latest] l.68-73). |
| Continuous recordings beyond the official windows (other offsets, 0–1 s cue period, 5–8 s trial end, inter-trial, raw 500 Hz) | ALLOWED by the preprocessing sentence; not named explicitly | "You may preprocess the public training data differently" (faq [latest] l.100-101). Licence is CC BY 4.0 (l.51-52). Trial structure: "cue at 0 s, feedback from 3 s, trial end at 8 s" (nbconfig l.37); recordings are DC-coupled (nfstudy l.37). |
| BIDS entities (participant, session, task, interface, run) | ALLOWED for training and analysis | "Available in the public BIDS files for local splitting and analysis. They are retained by the evaluator but not given to the model" (faq [latest] l.56-58). |
| External public datasets | ALLOWED if the original terms permit competition use; must be declared | terms [latest] l.38, Rule 4. The pin said "any publicly available, redistributable dataset"; "redistributable" was dropped on 26 Oct (commit `2b53aec`). |
| Pretrained and foundation models | ALLOWED; licences stay in force | "Original dataset and model licences remain in force" (terms [latest] l.38). guide [latest] l.452-461 lists REVE, BENDR, BIOT, CBraMod, SignalJEPA. `C/tools/staged_hf_models.txt` l.4-5 stages `facebook/dinov2-giant` and `brain-bzh/reve-base` on the workers. |
| Closed clinical data; the sealed split | FORBIDDEN | terms [latest] l.38 and l.48 (Rule 6: immediate disqualification). |
| Private, self-collected, non-clinical data | UNSPECIFIED | Rule 4 speaks only of "public datasets". guide [latest] l.348-349 says "another study — or your own data", which in context is a Benchopt dataset file. |
| Non-commercial or no-derivatives licensed data (e.g. Scherer 2015) | UNSPECIFIED in primary sources | Our clearance rests on Discord answers recorded second-hand (`R/docs/STATE.md` l.40-41; Discord brief l.84-88). |
| Synthetic data, augmentation | UNSPECIFIED (not mentioned anywhere) | — |
| Manual labelling or annotation of public data | UNSPECIFIED (not mentioned) | — |
| Training compute | ALLOWED, uncapped | terms [latest] l.40-42. |
| Declaration duty | REQUIRED | "Declare every external corpus, and a compute estimate, in the method description" (terms [latest] l.38). |

## 2. What the model receives at inference

**`predict(X)` receives only X.**
- The call is `current.predict(X[i:i + 1])` with no other argument (streaming l.123).
- X is float32 (nb_task l.268), on `meta["device"]` (guide [latest] l.173), with shape `(1, 41, 480)` (faq [latest] l.90-91).
- Content: 4 s starting 1 s after cue onset; notch at 50 and 60 Hz; band-pass 0.1–75 Hz at the recorded rate; resampled to 120 Hz; microvolts; "no learned scaling or clipping" (faq [latest] l.87-92; nbconfig l.12-18, l.37-38).
- Since feedback starts at 3 s, the window covers 2 s before and 2 s during feedback in runs that give feedback (nbconfig l.37; nfstudy l.24-26: each session has two acquisition runs and four online runs with feedback).
- No EMG or EOG: "41 EEG channels only" (faq [latest] l.62-64).

**`meta` has exactly eight keys, passed once to `load_model`:** `sfreq`, `ch_names`, `chs_info`, `n_chans`, `n_times`, `n_classes`, `device`, `submission_dir`.
- Construction: nb_task l.492-503; adapter l.158-165; objective l.59-78; base_solver l.57-75; guide [latest] l.149-155.
- `chs_info` is only `[{"ch_name": …}]`, with no positions (data l.90-100).
- "Same contract, runtime values may differ" between phases (guide [latest] l.222).
- "Use only `meta` and the provided batches. Do not rely on undocumented dataset names, paths, subject identifiers, or fixed dimensions" (guide [latest] l.229-230).

**Not passed:** person, session, run, interface, labels, time stamps, trial index.
- "Participant, session, run, interface, and label metadata are not passed to the model" (faq [latest] l.122-123).
- The evaluator keeps `subject_id`, `record_id`, `onset`, `stream_id`, `context_id` in `info` (nb_task l.279-288; data l.12-21).
- `train_loader` is `None` in inference-only runs (objective l.73-76).

**Run boundaries and interfaces:** nothing marks them. The only boundary signal is the session: a new model copy and `reset_state()` (streaming l.95-105). Inferring run, interface or position from the signal or a window count is not addressed for Track 2. The Track 3 FAQ explicitly calls an elapsed-window count "valid causal context" (`C/codabench/pages/scientific_faq_sleep.md` [latest] l.94-95).

**Technically reachable but forbidden:** the submission runs in the same process as the objective, which holds the labelled test loader, and staged data is mounted at `/app/data` (Dockerfile l.57-61). Reading either is "accessing the sealed test split" or "modifying the scoring harness" (terms [latest] l.48).

## 3. Statefulness

| Scope | Verdict | Governing text |
|---|---|---|
| Within a session: any causal, label-free state (buffers of earlier windows, statistics, alignment, gradient updates) | ALLOWED | "Causal, label-free adaptation using the current and previously delivered windows of the current session is allowed, including statistics-based alignment or gradient updates inside `predict`" (faq [latest] l.123-126). |
| Across Graz and BrainHero runs of one session | ALLOWED | "State may persist across Graz and BrainHero runs in the same session" (faq [latest] l.114-115). |
| Across sessions | FORBIDDEN | "State, windows, statistics, and parameter updates must not cross the session boundary" (faq [latest] l.126-127). |
| Future windows, sealed labels | FORBIDDEN | faq [latest] l.94-96, l.103-105. |

**How sessions are separated.**
- At each new `stream_id` the evaluator drops the previous copy and makes `copy.deepcopy(model)` of the object returned by `load_model`, then calls `reset_state()` if it exists (streaming l.27-43, l.95-105).
- The original model is never called (streaming l.67-70; `C/benchmark_utils/test_streaming.py` l.50-51).
- A model that cannot be deep-copied fails the run (streaming l.32-39).
- All streams run sequentially in one process (one `benchopt run`, ingestion l.117-122).

**What the evaluator does not reset and the documents never mention:** module-level globals, class attributes, the `Solver` instance, files outside the submission directory, torch global RNG, cuDNN and CUDA caches, a custom `__deepcopy__`.
- Carrying evaluation-derived information across sessions through any of these is forbidden by the FAQ sentence above, though not technically prevented.
- Evaluation-independent caches (JIT warm-up, shared read-only weights) are not addressed.

**Stream order.**
- Between sessions: "the order of independent session streams is not exposed as a model signal" (faq [latest] l.113-114); "evaluation order remain[s] private" (l.40).
- Within a session, public code: runs by BIDS run number, windows by onset (adapter l.71-82; nb_task l.296-328); contiguity and time order within each recording are checked (streaming l.95-121); test loaders are never shuffled (nb_task l.376-406).
- Within a session, sealed phase since 10 Nov: NOT GUARANTEED. The FAQ now says only "the same evaluation order is used for all submissions" (faq [latest] l.111-113). "Causal" was redefined as "the current and previously delivered windows" (l.94-96).

## 4. Compute and packaging

| Item | Verdict | Governing text |
|---|---|---|
| Time limit | 60 minutes | "The evaluation container must complete its test pass within 60 minutes on one A100 GPU. Evaluations that exceed this execution limit time out" (terms [latest] l.42). `execution_time_limit: 3600` (yaml [latest] l.79). |
| How time is measured | Whole run, not per window | The limit is Codabench's container limit. The leaderboard `duration` is wall time around the whole `benchopt run` subprocess plus copying the result (ingestion l.126-134). It includes data loading, `load_model`, every deep copy, every `predict` and the metric. It excludes Hugging Face cache seeding (l.124) and queueing. Benchopt's own timeout is off (`no_timeout: true`, phasecfg l.13). |
| Hardware | One A100 by rule; CPU count, RAM and GPU memory UNSPECIFIED | terms [latest] l.42. "Worker time and memory limits still apply" (faq [latest] l.127) gives no number. Dockerfile l.27-31 targets "G7 Blackwell GPUs", so the hardware class is not pinned by code. |
| Image | Python 3.12-slim, torch from the cu130 channel (unpinned), `requirements.txt` stack | Dockerfile l.25, l.33-41; `C/requirements.txt` l.8-29. yaml [latest] l.14 still says `neural-compet:v3`. |
| Installing packages | NOT AVAILABLE | "Nothing is installed during evaluation, so `submission.py` may only import what the worker image already carries" (guide [latest] l.36-39). The `requirements = ["pip::…"]` hint in base_solver l.48-50 is a silent no-op on the server. |
| Extra Python files | NOT SUPPORTED | "`submission.py` must contain … all Python inference code. Additional Python modules are not supported" (guide [latest] l.30-32). |
| File layout and types | Flat ZIP root; weights "with any name or format"; "optional non-Python artifact" | guide [latest] l.24-34. Official baselines ship joblib pickles (`C/tracks/bci_decoding/solvers/reve_probe.py` l.102-105). |
| Compiled code (shipped `.so`, C++ extension) | UNSPECIFIED | No primary text. python:3.12-slim has no compiler. |
| Package size | Up to the free space of a 15 GB profile, shared across tracks | terms [latest] l.44; guide [latest] l.45-50. |
| Submission directory | Read-only; writing there FORBIDDEN | guide [latest] l.13, l.40-41. Writing elsewhere is not mentioned. |
| Internet | Hugging Face downloads are contemplated; downloading competition data FORBIDDEN; everything else UNSPECIFIED | "Other weights may download during each submission and count against the one-hour evaluation limit" (guide [latest] l.459-461). The cache seeding exists "while letting participants experiment with new ones" (ingestion l.50-65). "Do not download competition data" (guide [latest] l.40). |
| GPU use | ALLOWED; device is `cuda` when available | data l.75-87. Models may also run on CPU, as the official sample does (`C/solution/bci_decoding/submission.py` l.25-28). |
| Multi-threading, multi-processing | UNSPECIFIED | The evaluator's loader uses `num_workers=0` (adapter l.103-106). |
| Fine-tuning at inference | ALLOWED within the FAQ limits | "Label-free test-time adaptation is allowed only as specified in your track's Scientific FAQ" (guide [latest] l.42-43). The pin's "do not train" during `load_model` and `predict` (guide [pin] l.30) was removed on 10 Nov. |
| Training on shipped labelled public data at evaluation time (in `load_model` or in `predict`) | UNSPECIFIED | See ambiguity 7. |

## 5. Output and metric

- **Return value:** "one integer class index per window with shape `(B,)`: 0 … motor imagery, 1 … mental subtraction, 2 … word generation" (faq [latest] l.133-136).
- **What the code checks:** only that the result has a leading dimension of 1 (streaming l.123-128) and that the flattened predictions number one per window (objective l.82-84). There is no dtype or range check.
- **Probabilities:** not possible. A `(1, 3)` output triples the flattened length and raises.
- **Revising earlier outputs:** not possible. Each prediction is appended at once (streaming l.129).
- **Abstention:** not possible and penalised. Any value outside the true classes becomes an extra class with recall 0, because classes are the union of true and predicted labels in the cell (metrics l.45-61).
- **Consequence of the union rule:** if a cell lacks one true class, predicting that class lowers the cell's score.
- **Cells:** a cell is `(stream_id, context_id)`, i.e. participant × session × interface (objective l.86-89; adapter l.48-55). The score is the plain mean over cells, "so every subject-session-context combination contributes equally regardless of its number of evaluation windows" (desc [latest] l.47; metrics l.64-89).
- **Reported alongside, not ranked:** pooled balanced accuracy, accuracy, session mean (objective l.93-100; faq [latest] l.144-145).
- **Leaderboard columns:** cell mean (desc), pooled, accuracy, duration (asc) (yaml [latest] l.96-111). Tie-breaking between equal cell means is UNSPECIFIED.
- **Feedback in the sealed phase:** only status and aggregate scores; logs, predictions and scoring output are hidden (terms [latest] l.64; yaml l.84-86).

## 6. Submission process

- **Quota:** "7 submissions total in Track 02", "at most 1 submission per team per day per track", shared by all members of the Codabench organisation (terms [latest] l.13-15; yaml l.80-81). The time zone of "day" is not stated.
- **What counts** (terms [latest] l.23-28):
  - Each accepted ZIP counts, "even if you upload the same model again".
  - Queued or running reserves a slot.
  - Final status `Failed` does not count, for both the daily and the total limit.
  - Finished or cancelled by you counts.
  - Deleting does not restore a slot.
  - Organiser reruns do not count.
- **Leaderboard:** best-of, automatic. `submission_rule: Force_Best` (yaml [latest] l.93). "Only the best eligible sealed score from you or your team is retained for the final ranking, subject to the reproducibility audit" (terms [latest] l.34). "Team members may appear as separate entries" (l.32).
- **Teams:** each participant uses their own account; a team is a Codabench organisation (terms [latest] l.9). "Each participant may belong to only one team and cannot join or submit for multiple teams" (reg [pin] l.20; the page now only says registration is closed). Several teams from one lab, or one person with several accounts, are not addressed.
- **Deadline:** 21 Nov 2026, 23:59 CET, which is 22:59 UTC (terms [latest] l.19; yaml l.78).
- **Audit:** top three teams per track, 22–30 Nov. "We rerun your training pipeline from the committed configuration and rescore the resulting weights against the sealed split. Scores within ±2σ of your submitted number stay on the prize roster" (terms [latest] l.52). σ is not defined.
- **Audit materials:** "method description, training and inference code, and pre-training disclosures" (terms [latest] l.56). "Training and inference code, configuration, weights and data disclosures ready when the sealed phase closes" (timeline [latest] l.42). Audit permission is limited to verification (terms l.60). The team leader handles the audit (reg [pin] l.22).
- **Prize eligibility:** organisers are ineligible (terms [latest] l.5). Other prize conditions are on the unread website page.

## 7. Protocol knowledge, identity inference, leaderboard probing

- **Cue order, per-run class balance, feedback, interface:** UNSPECIFIED. No primary sentence mentions exploiting protocol regularities. The general permission is the adaptation sentence (faq [latest] l.123-126).
- **Identity inference from EEG:** UNSPECIFIED in primary sources. The FAQ says only that participant metadata are not passed (l.122-123). The permission we rely on is second-hand Discord: Bruno's answer of 27 Oct to a covariance-profile routing question, and an organiser answer of 10 Nov (Discord brief l.38-39, l.73; handoff l.30-32).
- **Leaderboard probing:** not prohibited as such; intent is stated. "The five-submission sealed cap limits repeated tuning against hidden test data. Sealed evaluation measures generalization, not blind trial-and-error optimization on the test set" (terms [latest] l.17). "Exact cohort size, identities, session allocation, and evaluation order remain private" (faq [latest] l.40).
- **Side channels:** deliberately encoding sealed-set properties into the score, the public duration, or a conditional failure (which costs no quota) is not addressed. It runs against that stated intent and may fall under Rule 6.

## 8. Rule changes after the pin (from the submodule's own git history)

| Date | Commit | Change |
|---|---|---|
| 26 Oct | `2b53aec` | Warm-up extended to 1 Nov; sealed phase 5 Nov 00:01 to 21 Nov 23:59 CET (was 28 Oct to 21 Nov AoE). Sealed total cap 5 (the pin's YAML said 10). `Force_Best`; logs hidden. "Failed does not count" replaced the pin's "a failed upload still costs you one of the day's submissions" (guide [pin] l.66). Rule 4 reworded. "Best of each team's last five sealed submissions" (timeline [pin] l.15) replaced by "best eligible sealed score". |
| 2 Nov | `423b5af` | The track description dropped "the same 20-participant, six-session acquisition programme" (desc [pin] l.61). |
| 4 Nov | `6f1a05f` | The FAQ dropped "the ten participants represented … with ten participants absent" and added "exact cohort size … remain private". |
| 4 Nov | `4a8c943` | Each participant's best score is displayed; the team's best counts. |
| 6 Nov | `7160f93` | Sealed quota changed from per participant to shared per team. |
| 10 Nov | `8734b2f` | Track 2 quota 5 → 7. Rescoring after "a technical fix on the Track 02 sealed evaluation backend". Every ordering guarantee removed: "there is no shuffling within a stream: runs are ordered by run number and windows are strictly chronological" deleted from the FAQ; "in recording order" and "chronological" deleted from guide, desc and track README; the FAQ no longer says the evaluator implements "ordering checks". "Do not train" removed from the guide. |
| 11 Nov | `bddaff3` | Rescoring marked complete. |

**Where our notes go beyond or behind the primary sources:**
- **80-minute real timeout** (`R/docs/STATE.md` l.34; `R/docs/COMPETITION.md` l.5): no primary source. It rests on Discord and on our 3,732 s run finishing.
- **"Inside a session everything causal is allowed (adaptation, fine-tuning, identity inference from EEG)"** (`R/docs/STATE.md` l.32): the FAQ supports label-free adaptation and gradient updates on delivered windows. Identity inference is Discord-only, and "everything" is wider than the text.
- **"1 per UTC day"** (`R/docs/STATE.md` l.17): the rule says "per day"; UTC comes from our reading of Codabench source.
- **Scherer clearance:** Discord-only.
- **Worker image `v4-sealed`, "PROTEUS has 22 participants", organiser sealed baselines** (`R/docs/REVIEW_5NOV.md` l.14-23): not in the repo's primary files.
- **`R/OFFICIAL_CONTRACT.md` l.36-37** (runs by BIDS number, windows by recording time) describes the pin and is no longer guaranteed for the sealed phase. `R/docs/STATE.md` l.42-45 has this right.
- **`R/OFFICIAL_CONTRACT.md` l.95-96** ("keep train/test subjects disjoint") is our own hygiene rule for the baseline replication, not an official restriction.
- **`R/docs/COMPETITION.md` l.28-33** (quota 10, "quotas apply to the account") and l.94 ("redistributable") are superseded; its header note says so for the quota.
- **Supported as written:** 7 uploads per team; failed runs are free; cancelling costs a slot; best-of; no extra Python files; deadline 22:59 UTC; removal of the order guarantee in `8734b2f`.

## (a) Degrees of freedom the rules clearly leave open

**Training**
1. Train the final model on every public session of all ten public people, including the warm-up validation and test participants and their labels (terms [latest] l.38; faq l.26-27, l.37-38).
2. Use the public labels for model selection, stacking weights, thresholds and calibration (same citations).
3. Use the eyes-open and eyes-closed rest runs, for example for self-supervised pretraining, per-person spectral galleries or noise models (faq [latest] l.68-73).
4. Use any preprocessing of the public data: raw 500 Hz, other filters, other window offsets and lengths, the whole 8 s trial, inter-trial data (faq [latest] l.100-101; nfstudy l.32-38).
5. Use BIDS entities and event files as auxiliary targets or grouping: person, session, interface, run number, acquisition versus online run (faq [latest] l.56-58).
6. Use any public external dataset whose terms permit it, and any pretrained model under its licence, if declared (terms [latest] l.38).
7. Use any training codebase and unlimited training compute (`C/tracks/README.md` l.145-150; terms l.40-42).
8. Use any model family, including scikit-learn models shipped as joblib (the official baseline does).

**Inference**
9. Keep every earlier window of the session and recompute anything from the full history at every call: clustering, EM, refits, alignment (faq [latest] l.103-105, l.123-126).
10. Do gradient updates inside `predict`, including on the model's own pseudo-labels, as long as no sealed label is used (faq [latest] l.123-126).
11. Carry state across the Graz and BrainHero runs of a session (faq [latest] l.114-115).
12. Apply arbitrary transformations to the current window, including test-time augmentation over crops or shifts (faq [latest] l.100-101).
13. Use the feedback-period half of the window; it is part of the delivered tensor (nbconfig l.37-38).
14. Make predictions depend on the session's prediction history, for example class-share rebalancing; outputs are hard labels with no calibration requirement (objective l.80-100).
15. Spend the time budget unevenly: the limit is on the whole run, not per call, so heavy work at `load_model`, at chosen windows or in `reset_state()` is open (ingestion l.126-134; streaming l.40-42).
16. Use both CPU and GPU; nothing restricts threads (data l.75-87).
17. Read `meta["ch_names"]` to be robust to a different channel order (guide [latest] l.222).

**Packaging and process**
18. Ship up to the profile's free storage, in any file format: large galleries, memory banks, many models (guide [latest] l.24-34, l.45-50).
19. Load non-staged pretrained weights from the Hugging Face Hub at evaluation time, at the cost of run time (guide [latest] l.459-461).
20. Fail deliberately on a self-check (wrong channels, no GPU, non-finite output) instead of finishing with a degraded score: a `Failed` run costs no quota, daily or total (terms [latest] l.25).
21. Upload riskier candidates without leaderboard downside: the best score is kept automatically (yaml [latest] l.93; terms l.34). The audit applies to the best one.
22. Stochastic training is compatible with the audit, which tolerates ±2σ (terms [latest] l.52).

**Metric-driven**
23. Every cell counts equally, so short cells and each interface of each session carry the same weight as long ones (desc [latest] l.47).
24. Never emit a label outside 0–2, and note that predicting a class absent from a cell is penalised (metrics l.56-61).

## (b) Ambiguities worth asking the organisers

1. **Within-session order.** Is the sealed order still acquisition order, or are windows or runs permuted? The FAQ guarantees only "the same evaluation order … for all submissions" (faq [latest] l.111-113), while the unchanged public `streaming.py` l.106-121 still enforces time order within a recording.
2. **What the 10 Nov "technical fix" changed** (faq [latest] l.13-20). It is not described.
3. **Runtime contract.** 60 minutes in the rules and YAML (terms [latest] l.42; yaml l.79) against the 80 minutes we observed. Which clock applies (container or the `duration` of ingestion l.126-134), and does a timeout end as `Failed`, costing no quota (terms l.25)?
4. **Audit tolerance.** "±2σ of your submitted number" (terms [latest] l.52): σ of what, over how many reruns?
5. **Audit scope for pretrained parts.** "We rerun your training pipeline from the committed configuration" (terms [latest] l.52). Must external pretraining be rerun, or are pretrained checkpoints inputs? How are licence-restricted corpora handed over?
6. **If the best score fails the audit,** does the team's next-best sealed score count? "Best eligible sealed score" (terms [latest] l.34) against "drop off the prize roster" (l.52).
7. **Labelled public data at evaluation time.** May a package ship public PROTEUS data and refit on it together with the session's unlabelled windows, in `predict` or `load_model`? "Any evaluation-time transformation or fitted statistic must use only the current and previously delivered windows of the current session, without labels" (faq [latest] l.103-105); the removed "do not train" (guide [pin] l.30) against the current wording (guide [latest] l.40-43).
8. **Identity inference and personal models for represented participants.** Primary text is silent (faq [latest] l.122-127); the guide says not to "rely on … subject identifiers" (l.229-230); the permission is Discord-only.
9. **Protocol regularities.** Is using cue-order or per-run class-balance structure acceptable? Nothing addresses it, and the order guarantee was removed on 10 Nov.
10. **Process-level state.** Are evaluation-independent caches that survive the deep copy acceptable (JIT warm-up, shared read-only weights through `__deepcopy__`, module globals)? The rule (faq [latest] l.126-127) names "state, windows, statistics, and parameter updates"; the mechanism copies only the model object (streaming l.27-43).
11. **Network.** Is general internet access available and allowed beyond Hugging Face weights (guide [latest] l.36-40 against l.459-461)?
12. **Resources.** Memory, CPU cores and GPU memory: "worker time and memory limits still apply" (faq [latest] l.127), with no figure anywhere.
13. **Compiled artefacts** shipped as "non-Python artifact" (guide [latest] l.26, l.30-32): not addressed.
14. **Sealed cells.** Is the context still exactly Graz or BrainHero, and does every sealed cell contain all three classes? The adapter is private, and the union rule (metrics l.56-61) makes this matter.
15. **Cohort.** The description still points to the website's dataset directory for the cohort (desc [latest] l.55), while the FAQ says cohort size is private (faq l.40).
16. **Day boundary** of the one-per-day limit (terms [latest] l.15): time zone not stated.
17. **Private non-clinical data and non-commercial licences** (terms [latest] l.38): Rule 4 covers only "public datasets" whose "original terms permit competition use".
18. **Teams.** Several teams from one lab; one team showing two models under two accounts (terms [latest] l.9, l.32).
19. **Tie-breaking** on equal cell means; is duration a tie-breaker (yaml [latest] l.96-111)?
20. **Using sealed feedback to infer cohort structure** (ten-decimal owner scores, duration, conditional failure): not addressed; intent in terms [latest] l.17 and faq l.40.
21. **Sealed benchmark revision and image tag.** The phase bundle carries its own benchmark copy (ingestion l.3-6), and the YAML still names image `v3` (yaml [latest] l.14).
