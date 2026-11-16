# State: the starting point

This is the entry point. It is written so that a new session needs nothing from any chat history. Where an older
document contradicts it, this file wins.

**Read in this order:** this file; [LEDGER.md](LEDGER.md) (index of what is adopted, built, open and closed, and where new
runs are pre-registered; the full records are PREDICTIONS_11NOV.md and PREDICTIONS_12NOV.md);
[SEALED_UPLOADS.md](SEALED_UPLOADS.md) (upload log and procedure); [../OFFICIAL_CONTRACT.md](../OFFICIAL_CONTRACT.md).

## 1. Where we stand

### Update: 16 Nov 2026

This update supersedes conflicting compute, candidate and research-window statements below. Older scores, rank,
deadline and upload counts are historical snapshots; check the live competition before relying on them.

- **Compute:** The user directs new project work to Galvani. Leonardo was temporary. Galvani access currently uses
  eduVPN and authenticated SSH; access can expire. Keep heavy training on Galvani and respect its user limits.
- **Submission:** v5.1b remains the last recorded uploaded package. The prepared next candidate is v6.6 cap 250,
  `runs/sealed/final_zips/v66_cp1_cap250.zip`, SHA-256
  `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed`. Its Mac replay and recorded A100 checks
  passed; the real Codabench worker and sealed session lengths remain unverified. The user uploads it; do not upload
  without explicit authorization and a live phase/quota check.
- **§281 stack:** Galvani smoke 3110322 and replay guard 3110386 passed. Control array 3110398, treatment array
  3110399 and personalisation array 3110400 completed. Scorer 3110971 passed its zero, order and provenance guards:
  new-person changes were coral -0.069, B1 -0.095 and both -0.220 points; all 80% intervals cross zero. No measured
  improvement; undecided within the noise. The registered returning-person test is not
  yet a complete v6.6 routed-tail evaluation; see `evidence/s281_16nov/returning_gate_trace_handoff.md`.
- **Other runs:** Repaired CORAL, B1 and combined arrays are 3109744–3109746. The 96 returning-base tasks are
  submitted in 3110234, 3110275, 3110358 and 3110408. Check Slurm and the ledger for current state before acting;
  queued work is not a result. §290 core-only screens and completed base-only model screens are in §290 and the
  16 Nov evidence directory; they are not full-chain closures.
- **Returning preparation:** Trace and prefix array 3110497 and SincShallow checkpoint parity diagnostic 3110506
  have finished: seven trace tasks passed; runs0 needs a zero-refit smoke/retry. Both Sinc reconstruction folds
  failed frozen-output parity, so no substitute checkpoints are usable. Fold-stage diagnostic 3110614 passed for
  its narrow P04/within0 cell. Their workspaces, guards and validation limits are in the ledger.
- **Live board, 16 Nov about 15:12 CET:** uploaded v5.1b is 15th displayed entry (14th after merging ETRI's two
  entries). Fifth displayed entry is 0.839; fifth distinct team is 0.837. Source: live Codabench Results, sealed
  phase. A rough v6.6 projection of 0.824–0.829 is an unvalidated sealed-transfer estimate, approximately 10th–13th
  on that board, not top five. New-person §281 results do not improve this projection.
- **Research review, 16 Nov:** [solution-space and recent-method review](../evidence/research_block_16nov/REPORT.md)
  maps actual rules to executed experiments and current primary research. Whole alternative systems need not keep
  the incumbent features, stages or routing. Proposed directions include task/style factorization, learned geometry
  and training through causal adaptation; none has a measured new full-system gain. The review records a §98
  transform confound without claiming it caused the original result. These are proposals for user review; no new
  implementation or experiment was started during the block. New compute remains Galvani-only.
- **Parallel wave §292, 16 Nov:** The user subsequently authorized four wider lanes. [Current jobs and evidence](../evidence/next_wave_16nov/README.md)
  cover full-system EEGNet/distillation/CORAL comparisons, a population-anchored adaptive MLP, CBraMod task/person
  factorization, and learned SPD correction. New production jobs are on Galvani. The MLP form completed and lost
  1.48 points against its static head (80% people/order interval −2.07 to −0.88); its static decoder was also weak.
  SPD correction also completed: official cell BA42.14→43.13%, full-bias versus frozen person/order effect+1.25
  points [.41,2.10]. Its whole decoder remains below the validated new-person incumbent76.22%. Neither result
  establishes a package gain or closes the wider family. Factorization and full-system comparisons remain pending.

- **Best sealed score:** 0.8197668759, submission 1050185, package `cand_v51b_own` (v4.1 + own-space personal stage),
  12 Nov, 1,287.0 s (21.5 of 60 min). Previous best 0.8161546323 (1048575, `cand_v41_bio`).
- **Rank:** 8th by account, 7th by team, on the board of 12 Nov 21:35 UTC: 0.847, 0.846 (one team, ETRI, two
  accounts), 0.830, 0.826, 0.825, 0.825, 0.821, **0.820 (us)**, 0.819, 0.819. Third by account needs +1.0; third by
  team needs about +0.6. The field moves daily.
- **Target (user, 13 Nov):** rank 1 is proof that such scores are reachable, not an upper bound; the user would aim
  for 0.9. Size explore studies by their ceiling (§5), and keep reporting the measured stack honestly.
- **Deadline:** 21 Nov 2026, 22:59 UTC. **Uploads:** 7 per team, 1 per day; 5 used, **2 left**. Codabench storage on
  the account is tight: only 1048575 and 1050185 remain there; check free space before any upload.
- **Research window (user): until about 14 Nov 02:00 UTC** no new package candidate is built and no upload decision
  is taken; research, iteration, profiling. Two tracks, two thirds exploit and one third explore (§9).
- **Compute:** Leonardo (`cineca-leonardo`); the user lifted any cap ("use as much compute as needed"); keep GPUs busy
  with parallel waves, also overnight. Galvani is unreachable until the user logs in with 2FA; the worker image lives
  there, so worker-image replays are missing for everything built since 12 Nov.
- **What is measured against v5.1b as uploaded (13 Nov, the only reference from now on):**
  - The eye-net view and the rest member on v5.1b's own members: J +0.05 (new +0.34, seen −0.24, P04 −1.4) [§151].
  - v5.2's personal nets in package form: about 0 to −0.4 seen, and P04 loses every time [§105b, §140].
  - **Fine-tuning (cum10) + eye-net view + rest member, on sessions the gate does not recognise: meets its bar on all
    four augmentation draws** [§152]. New people +0.84 to +0.95 (without P10 +0.58 to +0.70, 80 % person intervals
    above 0); with recognised sessions left exactly as v5.1b, dev J +0.42 to +0.48. Fine-tuning alone is about 0 beyond
    P10, and the three parts are strongly super-additive. The cheap gC5 schedule keeps about two thirds of the gain
    (J +0.28 to +0.36, recognised sessions as v5.1b).
  - The ceiling of the eye-net kind of view is about +3.7 more (oracle, two draws) [§155]; better views are the
    explore lever [§158, §159].
  - The larger stack numbers of 12–13 Nov (J +0.5 to +0.95) were against study bases about 0.7 below v5.1b on
    returning people [§151]; the line above replaces them.
- **Map of tested against allowed space:** [SOLUTION_SPACE_MAP.md](SOLUTION_SPACE_MAP.md) (13 Nov) lists the gaps, the
  closures that rest on weak tests, and the backlog of ideas never run.

## 2. Hard bounds (from the official code and rules)

- **Input:** one window per call, float32 `(1, 41, 480)`: 4 s at 120 Hz, starting 1 s after the cue. Output: one label
  (0 motor imagery, 1 mental subtraction, 2 word generation). No labels, no person, session or interface metadata.
- **Stream:** one participant-session, delivered in a fixed evaluation order that is not recording order (organisers, 13 Nov); Graz and BrainHero share the session; state resets between sessions.
- **State:** the evaluator deep-copies the loaded model at every session boundary. Nothing may cross sessions.
  Inside a session everything causal is allowed (adaptation, fine-tuning, identity inference from EEG).
- **Metric:** balanced accuracy per participant × session × interface cell, then the plain mean over cells.
- **Runtime:** 60 min by rule (the worker's real timeout is 80 min; plan for 60). The package is a ZIP with
  `submission.py` plus data files; no extra Python files.
- **Worker image:** Python 3.12, torch 2.14 + CUDA; the inspected image does not install compilers, numba or cupy.
  Bundled native code is permitted: Bruno Aristimunha confirmed a Python-imported C++ pipeline and wheels on
  Discord, 20 Oct (screenshot supplied by the user, 16 Nov). A compiler absent from the worker does not prohibit
  prebuilt binaries. Native extensions still need exact worker compatibility and session-reset validation; neither
  a native prototype nor a CPU replay proves Linux/A100 compatibility. See the
  [clarification and proposed native-stack scope](../evidence/next_wave_16nov/native_stack.md).
- **Uploads:** each needs the user's explicit go-ahead and a live check of phase and quota. Never cancel a running
  sealed submission: it costs a slot. Codabench has a bot check; the user opens the page in Chrome.
- **Data:** public datasets are allowed if their terms permit. Scherer 2015 (CC BY-NC-ND) was cleared by the
  organisers on Discord on 11 Nov; it must be declared in the method write-up.
- **Order (organisers, 13 Nov, Discord, verbatim):** "1) Window order. Please don't assume recording chronology.
  'Previously delivered' means earlier in the evaluation stream, not necessarily earlier in recording time. The same
  evaluation order is used for all submissions, with state reset between sessions. Without going into details, but to
  avoid tricks we discovered of solving the track without using EEG, we removed chronology. Hence why we reran all
  submissions earlier this week. 2) Labelled public windows during adaptation. Public labelled data may be used for
  training. Evaluation-time adaptation is of course restricted to the current session's delivered windows, without
  labels, as mentioned in the FAQ. 3) Any models trained on permitted public data are allowed, whatever its training
  strategy on those public data. So Yes."
  **Consequences:** the sealed stream has not been in recording order since the rescoring of 10 Nov. §62's conclusion
  ("the sealed order is still close to acquisition order") is withdrawn: the fingerprint failing to match meant we did
  not find the order, not that there is none. Every dev number so far was measured in recording order. The primary dev
  protocol is now non-chronological within a session, scored under several schemes with fixed permutations
  (ledger §164); recording order is a secondary column. No scheme is primary: the 10 Nov rescoring does not identify one,
  and its per-package drops fit full better than within unless half a point was a uniform backend effect (correction
  under §169). Evaluation-time adaptation may use only the session's delivered
  windows, without labels. v5.1b and the combination fit nothing on labelled public data during a session (checked
  13 Nov: the own-space stage and the personal heads load precomputed parameters).
- **Cue-order regularity** (Graz runs follow a weakly shuffled WORD → SUB → MI cycle): not raised with the organisers;
  it goes into the final method write-up (user decision, 11 Nov).

## 3. The system (v4.1; the uploaded v5.1b adds the own-space stage of §6)

**Members**, all causal, reset per session, mixed with fixed weights (arithmetic mean of posteriors):
- core1: tangent-space features (common-average reference, 17 overlapping bands, full and two half-window
  covariances, causal log-Euclidean recentring), logistic regression. The strongest member (69 % alone on new people).
- Hilbert core: complex analytic-signal covariances in 7 bands, logistic regression.
- About 35 plain nets (ShallowConvNet, FBCNet, FBLight, SincShallow) and 30 SessionBN nets (batch norm with the
  session's own causal statistics).
- External teacher (weight 5): EEGNeX and IFNet trunks pretrained on 7 motor-imagery corpora; their head outputs go
  through a small logistic regression to the 3 classes.

**Stages after the mix:**
1. Rebalancing (divides by the recent class share; worth 5.75 J).
2. SessionEM: per-session class templates on frontal slow-potential and ocular features.
3. TSEM: per-session Gaussian class templates in the top 128 PCs of core1's features, seeded by confident outputs.
4. Gated early boost in windows 30–60 for sessions with low early margins.
5. Routing: a returning person is recognised and their personal head (logistic regression on core1 PCs, trained on
   their public sessions) enters a second tail at 3× in core1's slot. v3 recognises by class directions from window
   60; **v4.1 adds a label-free identity gate** (spectral shape, two-covariance model fitted on 224 external people,
   threshold 0.26) from window 10. Externally: 86.7 % correct accepts at 1.9 % false accepts.

**Packages:**

| Package | Where | Status |
|---|---|---|
| `cand_v51b_own` (v4.1 + own-space personal stage) | `runs/upload/sealed_cand_v51b_own.zip` (`4a0bc427…`); code `submissions/ensemble_v5b/submission.py` (`0259d1ea…`) | **Uploaded 12 Nov, best: 0.8198** |
| `cand_v52_pnets` / v5.2b (v5.1b + personal nets on recognised sessions) | `runs/sealed/cand_v52_pnets`; code `submissions/ensemble_v52/`; notes `V52_BUILD.md` | Built; GPU replay equals the Mac replay; warm paired runtime 1.40× of v5.1b (v5.2b 1.27×); GPU check passes against a reference with the package's own batching; **worker-image replay missing**; not uploaded |
| `cand_v41_bio` (routing v4.1) | `runs/upload/sealed_cand_v41_bio.zip` (`53fc1679…`); code `submissions/ensemble_v4/submission.py` (`45f9f2fd…`) | Uploaded 11 Nov, 0.8162 |
| `cand_v51_own` (the first build of v5.1) | Galvani `runs/sealed/cand_v51_own/`; code `submissions/ensemble_v5/submission.py` (`34eba3b0…`) | Superseded by `cand_v51b_own` (same outputs) |
| Fast package (v3 outputs, half the runtime) | `submissions/ensemble_fast/submission.py` (`ccbce506…`) | Base of v4.1; not uploaded by itself |
| route3geb (v3) | code `submissions/ensemble/submission.py` (`dacee85a…`) | Uploaded 7 Nov, 0.8135 after rescoring |

Never edit `submissions/*/submission.py` in place: their SHA-256 values are pinned to uploaded or staged packages.
Package builders: `scripts/add_bio_gate.py`, `scripts/check_bio_gate.py`, `scripts/build_sealed.py`; audit recipes in
[AUDIT_SBN.md](AUDIT_SBN.md), [AUDIT_NXIF.md](AUDIT_NXIF.md).

## 4. How results are measured

**Dev metric J** = 0.5 × new + 0.5 × seen, person-equal mean of cell balanced accuracy.
- new: leave-one-person-out over the 10 public people (`all_loso`).
- seen: P04 and P10, trained with session 1, tested on sessions 2–3 (`seen_session`). They are the only people with
  more than one public session.
- Six dev bases (different seed sets of the nets): `B2_pool`, `B2_s345` for exploration; `B2_s012`, `B2_s0`, `B2_s1`,
  `B2_s2` for confirmation only after a gate passes. People halves for nested selection: G1 = P01 P02 P05 P07 P09,
  G2 = P03 P04 P06 P08 P10.

**Replay** (CPU, about 5 s per base):

```bash
.venv/bin/python replay/check.py
```

It must print the six reference values (B2_pool 80.0131, B2_s345 79.8474, B2_s012 80.1446, B2_s0 79.1770,
B2_s1 79.7214, B2_s2 80.2093) and `OK`. That is the v3 chain. The uploaded v4.1 is replayed by
`studies/11nov/rt4/route4.py runs/replay_data/bio_model_psd_d64.npz BASES` (mean J 80.44, +0.58 over v3).
- Code: `replay/` (tracked). Inputs: `runs/replay_data/` (60 MB, gitignored; manifest `replay/data.sha256`; copy on
  Galvani in `runs/diag/replay_data/`).
- To test a member: `A.EXTRA[name] = {proto: P}` then `A.system(base, fams, extra=[(name, w)])`; see
  `studies/11nov/ttt/score.py` for replacing a member slot.

**The six bases are not independent** (ledger §63): B2_pool is the mean of B2_s012 and B2_s345, B2_s012 the mean of
the three single seeds, and only the five SessionBN slots differ between them. "Confirmed on fresh bases" means
robust to the net seeds, nothing more. The uncertainty that matters is across people: report "new" with a person
bootstrap interval (standard error 0.5–1.1 for a typical change) and P04 and P10 separately.

**What dev numbers are worth on the sealed set:**
- Noise: 8 permuted-view null arms gave −0.22 to +0.18 J. Treat anything below +0.3 as noise.
- Small stacked gains transferred at 0.6–0.7×; v4.1 at 0.46× (+0.58 dev, +0.267 sealed).
- **Dev "seen" weights P04 and P10 equally. On the sealed set P04 probably has 3 of about 46 returning sessions**
  (if the organisers' withdrawn cohort text holds: 20 people × 6 sessions, 10 of them new). Every returning-person
  gain so far is P04's. Read returning-person levers by P10's response.
- New people are person-equal in both, so "new" transfers more directly. P04 as a new person is 54 %; everyone else
  75–90 %.

## 5. What the measurements say, and how firmly (revised 13 Nov; ledger sections in brackets)

Read every line with its test in mind: "alone" means not scored in the chain; "posterior-only" means the new part's
outputs were added while the session stages and the routing kept using the old core's features. The map
([SOLUTION_SPACE_MAP.md](SOLUTION_SPACE_MAP.md)) tags all closures by test quality.

**Where large ceilings have been measured** (the places worth several attempts in several forms):
- **More training people:** core1 rises +8 points per doubling with no flattening [§39, §48]. Substitutes tried so
  far (generated variation, people-efficient estimators, Scherer rows) were ≤ 0, but all at core level, alone
  [§46b, §47, §49, §53].
- **Members whose errors are independent of ours:** an independent member at 70 % is estimated at +3.1 for new people
  [§63]. The external teacher is the only such member; its design dates from 1–2 Nov and was last judged on a measure
  dominated by one person.
- **Labels inside the session:** +1.6 to +2.9 through fine-tuned nets or session-own members, +3 through a head
  [§28, §35, §38, §81]. With the chain's own labels the best form so far reaches +0.46 for new people [§108].
- **The session's class geometry:** aligning a session with its true class means lifts a simple classifier from
  64.6 % to 89.1 % (whole session, not causal); one label-free form as an add-on member failed [§103c, §104].
- **Session-own representations:** 81–82 % with a session's own labels against 74.9 % in the population PCs; P04
  reaches 74–80 % [§36]. The own-space stage uses this for returning people.
- **Choosing per session** among stage variants or member weightings: +1 to +2 with an oracle; no label-free rule
  found yet [§76, §117, §118].

**What held up repeatedly:**
- Extra members added at a fixed weight lower the chain even when strong alone; member weights and stage constants
  sit at an optimum **for this chain** [§31, §66, §69]. This says nothing about a redesigned chain.
- Dev resolves about ±0.3 to ±1. Gains seen only on the two exploration bases have vanished on the other four three
  times [§54b, §58b, §61b]. The six bases share people and differ only in net seeds [§63].
- Dev "seen" is two people. Upload 5 suggests the other returning people gain about +0.6 per session from the
  own-space stage, so they are not like P10.

**Not established, although earlier versions of this file said so:**
- "Small levers do not stack": tested once, on two returning-person levers [§57b]. The stacking test of all
  replicated small gains is §130.
- "In-session learning on the chain's own labels is closed": the augmented-input form gains for new people on every
  base [§101, §106, §108].
- "The chain gains only from a person's own labels": true of the forms tried; the teacher is a counter-example.
- "The leaders' margin is unexplained" is a statement about us, not a bound.

**Where the error is:** P04 as a new person 54 %, the other nine 75–90 %; the first fifth of a session 73 % against
77–80 % later; word generation is the weakest class; returning people lose 8 points in Graz feedback runs against
calibration runs [§67, §105].

## 6. Levers (state on 13 Nov)

**Uploaded: the own-space personal stage (`cand_v51b_own`, sealed +0.361 over v4.1).** For windows the gate routes to
public person j with ramp r, the routed output is multiplied by exp(0.10 · r · mean clipped log posterior of three
models fitted on j's public sessions in j's own space: FBCSP + LDA, own tangent space + logistic regression, slow
time courses + LDA). Dev seen +1.79 (P04 +3.5, P10 +0.1) [§45]; dev-to-sealed transfer 0.41×.
Code: `submissions/ensemble_v5b/submission.py`, `scripts/add_own_stage.py`, `scripts/check_own_stage.py`,
`scripts/check_perm_stage.py`; study code `studies/12nov/ownspace.py`, `v5_replay.py`.

**Measured, not uploaded:**
- **Personal nets for returning people (v5.2 / v5.2b):** the dev numbers (+0.17 [§78], +0.55 with stage 0.15 [§119])
  gave the nets to whole sessions from window 0 in both tails. In package form (only after the gate routes, routed
  tail only) the reviewer measures −0.05 (P04 −0.10, P10 0.00) [§140]; PKG2 re-scores it on six bases [§140]. The
  §105 expectation (+0.20) therefore has no dev evidence behind it. Built and checked on GPU; no worker-image replay.
- **In-session fine-tuning for new people** (augmented inputs, consistency term, the chain's labels): +0.37 dev;
  warm-start form (each refit continues the previous one, 10 steps) +0.46 at half the cost; one or two fine-tuned
  nets per family keep +0.26 / +0.34 [§101, §106, §108]. "Six bases" adds nothing here (the scorers swap the slots
  that differ between bases; deltas correlate 0.83–0.99): the person interval for cum10 is −0.56 to +1.52 (95 %) and
  P10 as a new person carries it (+3.5; the other nine +0.12) [§139]. The cheapest form so far, gC5 (warm start,
  5 steps every 10 windows, GPU-drawn augmentation), is +0.59 with an 80 % person interval of +0.04 to +1.14 and
  +0.34 without P10 [§113]. Fresh-seed replications run on 13 Nov [§113, §135]. A package runs it only on sessions
  the gate never routes (the gate routes no new-person window in the dev replay) [§138]. Cost on an A100: about 142 s per session for all 30 nets, 20 to
  39 s for the reduced forms [§107].
- **Small positives, combined (13 Nov):** in package form against a study base, FT + eye-net view + rest member gave
  J +0.51 to +0.95 on three seed sets [§140, §145]. That base (study twins, personal nets, stage 0.15) sits about 0.7
  below v5.1b on returning people. **On v5.1b's own members the eye-net view and the rest member give J +0.05**
  (new +0.34, seen −0.24, P04 −1.4) [§151]. Fine-tuning on the shipped members is not measured yet. The design notes
  are in [V6_DESIGN.md](V6_DESIGN.md).

**The next candidate** is decided after the research window, and only from arms scored against v5.1b as uploaded
(shipped members, shipped gate, stage 0.10; `studies/13nov/pkg.py` reports every arm against it first). v5.2 / v5.2b
are not candidates as built (§105b). The open question is fine-tuning on the shipped members (§152).

## 7. Working rules and infrastructure

- **Pre-register** every run in [LEDGER.md](LEDGER.md) before scoring it: mechanism, numeric prediction, refutation
  rule. Report misses as misses.
- **Local runs:** only through `scripts/run_capped.sh LOG CMD…`, one job at a time (the Mac froze on 22 Oct). CPU
  analyses and chain replays are fine locally; net training on MPS is about 6× slower than a 2080 Ti.
- **Galvani** (`ssh mlcloud-galvani`, work dir `/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026`):
  - The user types the 2FA code; agents never re-authenticate. Check with `ssh -O check mlcloud-galvani`. The master
    connection drops after about 12 h or on network changes.
  - Submit pattern: `sbatch --account=gehler -p 2080-galvani,2080-preemptable-galvani,a100-preemptable-galvani,v100-galvani
    --exclude=galvani-cn109,galvani-cn111,galvani-cn121,galvani-cn122 --gres=gpu:1 --cpus-per-task=3 --mem=32G`,
    time limits ≤ 1 h backfill best. `scripts/cloud/pack.sh` packs tasks into one job.
  - One member task (12 folds of a small net) takes 2.5–6.5 min; the queue, not training, is the bottleneck.
  - Downloads from Galvani run at about 0.2 MB/s: fetch small result files only.
- **Commits:** commit and push freely; no AI author line. Never commit datasets, environments, credentials or
  checkpoints. Keep the `2026-competition` submodule pinned.
- **Weco** (CLI 0.3.27) is set up in `.weco/`; both searches of 11 Nov overfitted their visible half. The one-file
  session stage in `.weco/session-stage/optimize.py` matches the chain replay to 1e-15 and is reusable.

## 8. Where everything is

| What | Where |
|---|---|
| Entry point, ledger, upload log | `docs/STATE.md`, `docs/LEDGER.md`, `docs/SEALED_UPLOADS.md` |
| Detailed record of 11 Nov (33 sections) | `docs/PREDICTIONS_11NOV.md` |
| Detailed record of the night of 11–12 Nov (§34–§61) | `docs/PREDICTIONS_12NOV.md` |
| Critical review of 12 Nov: sources of gains, what stopped on 7 Nov, located losses, self-imposed limits | `docs/REVIEW_12NOV.md` |
| Study scripts of 12 Nov (index in its README) | `studies/12nov/` |
| Closed directions 5–10 Nov | `docs/REVIEW_5NOV.md` |
| History (superseded; may be wrong) | `docs/archive/` (its README lists the wrong assumptions) |
| Chain replay | `replay/` |
| Study scripts of 11 Nov | `studies/11nov/` (the ledger cites them as `runs/archive_scratch_11nov/<same path>`) |
| Training, features, exports | `experiments/`, `scripts/` (only code that packages, audit recipes or the ledger still reference; the rest was removed on 11 Nov, see `docs/archive/README.md`) |
| Identity gate | `experiments/bio_gate.py`, `experiments/bio_id.py` |
| Compact evidence logs | `evidence/` (11 Nov: `evidence/wsp_bio1_11nov/`, `evidence/seq_bio_11nov/`) |
| Local only (gitignored) | `data/`, `runs/` (`runs/upload/` ZIPs, `runs/replay_data/`, `runs/ttt/` outputs, `runs/archive_scratch_11nov/` full scratch mirror, 3.5 GB) |
| Galvani outputs | `runs/diag/{ttt,rest4,aec,sbnx,w12,schr,steeg}` under the work dir |

## 9. Decisions waiting for the user, and rules they set

1. **Next upload (v5.2 or a later candidate).** Not before about 14 Nov 02:00 UTC (research window, below). v5.2 is built and being validated; its sealed expectation is pre-registered in ledger §105. Every upload needs the user's explicit go-ahead and a live phase, quota and Codabench-storage check (v5.2 needs about 85 MB free).
2. ~~Two leading accounts~~: ranks 1 and 2 are one team (ETRI_EnSeption, two models; REVIEW_12NOV §7).

**Two tracks inside the window (user, 12 Nov night):** "nice! exactly these findings we want so keep pushing now on that path but also always with some ratio keep exploring completely new properly reasoned from the ground up fitted solution to our problem understanding". **Ratio chosen: two thirds exploit, one third explore**, in compute and in attention. Exploit = the cheap in-session fine-tuning for new people (v52-build session: reduced forms, A100 batching, then raising the gain with the free runtime). Explore = designs new in kind, reasoned from the measurements, scored as whole systems or as oracles that bound them (this session). Every status report says what ran on each track.

**Research window (user, 12 Nov about 22:00 UTC, until about 14 Nov 02:00 UTC):** no new package candidate is built and no upload decision is taken before then; research, iteration, profiling and pushing performance only.

**Rules set by the user on 12 Nov** (also in `AGENTS.md`): no gate or closure is final (check what actually ran before
citing one); cleanups and reviews must hunt silent no-ops (options parsed but not applied); the codebase exists to
optimise the final submission. Time stamps in the ledger come from `date -u`.
