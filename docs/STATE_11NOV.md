# State on 11 Nov 2026 (updated 16:10 UTC): the starting point

This is the entry point. It is written so that a new thread needs nothing from the chat history. Older documents are
in docs/archive/ (see its README for the assumptions in them that are now known to be wrong).

**Read in this order:**
1. This file.
2. docs/PREDICTIONS_11NOV.md: the pre-registered ledger of every 11 Nov run, with results filled in.
3. docs/SEALED_UPLOADS.md: the upload log, the rescoring and the board.
4. docs/REVIEW_5NOV.md: every closed direction since 5 Nov; §5d and later is the most recent.
5. docs/AUDIT_SBN.md: audit recipes and the fast-package parity.

## 1. Where we stand

**Deadline:** 21 Nov 2026, 22:59 UTC (23:59 CET).

**Quota:** 7 sealed uploads per team, at most 1 per UTC day. We have used 3, so **4 are left**. No upload has been made
since 7 Nov.

**Best sealed score:** submission 1040847, `cand_rec_ts2_tsem_route3geb`, **0.8134828319 after the 10 Nov rescoring**
(0.8240266154 before). Its code is commit `c4d600d` (`submission.py` `786bcab0…`). Rescored duration 3,732 s.

**Rescoring (organiser commit 8734b2f6, completed in bddaff3d):**
- All Track 2 sealed runs were rerun after a backend fix.
- Our three uploads moved −0.72, −1.17 and −1.05 points.
- 23 of the 25 submissions present on both board snapshots dropped (median −0.6), so the change hit the whole field.
- Our order-fingerprint replay rejected every reordering hypothesis (H1–H8; ledger §2).
- What the fix changed is unknown. Resets and segmentation (per-interface resets, merged sessions) were never tested.

**Board, 11 Nov 10:47 UTC** (snapshot of all 182 rows in `runs/archive_scratch_11nov/discord_11nov/`):

| Rank | Team | Score | Duration |
|---|---|---|---|
| 1 | seohee (ETRI) | 0.846 | 2,930 s |
| 2 | sungjunwang | 0.830 | 1,222 s |
| 3 | xmqxmq | 0.826 | 1,458 s |
| 4 | bryanyahir03 | 0.824 | 968 s |
| 9 | **us** | **0.813** | 3,732 s |

- seohee's SDE26-v2 gained +2.5 in a single upload, from 0.821, and its runtime rose about 4×.
- Top 3 needs about +1.3 sealed points. Small dev gains transfer at about 0.6–0.7×, large ones at about 1×.

## 2. Packages

| Package | Location | Status |
|---|---|---|
| route3geb (shipped) | `runs/upload/sealed_cand_rec_ts2_tsem_route3geb.zip`; code at commit `c4d600d` | The best upload |
| **Fast package** | `runs/sealed/cand_rec_ts2_tsem_route3geb_fast/`; ZIP `runs/upload/sealed_cand_rec_ts2_tsem_route3geb_fast.zip` | Built, not uploaded |
| **Routing v4 checkpoint** | `runs/sealed/cand_v4_bio/`; ZIP `runs/upload/sealed_cand_v4_bio.zip` `94185c2f…`; code `submissions/ensemble_v4/` | Staged 11 Nov, replays pass, **not uploaded**; dev +0.59 J over route3geb (ledger §14) |
| Lean reference code | `submissions/ensemble/submission.py` `dacee85a…` (committed 9491882) | Reproduces all 3 uploads |

**Fast package details:**
- ZIP SHA-256 `cfd1a4b4…f43f417d`, 57,280,590 B; `submission.py` `ccbce506…`, 904 lines. It contains the same 32 data
  files as route3geb.
- Its runtime code is Codex's shared FIR bank plus CPU/GPU overlap.
- **Parity:** every stage is bitwise equal on all 2,655 public windows.
- **Speed:** A100 solver time 0.41× of route3geb; whole run 0.53× (ABBA job 3086004). Estimated sealed run 19–35 min.
- **Code:** committed as `submissions/ensemble_fast/submission.py` (7a2f9e8), with the per-file SHA-256 manifest.
- It is the base for every future upload. Upload only with your explicit go-ahead, after checking the live quota and
  phase.

## 3. Pipeline (unchanged since 7 Nov)

**Members:**
- two tangent-space cores (core1 and a Hilbert core);
- about 35 plain nets (ShallowConvNet, FBCNet, FBLight, SincShallow);
- 30 SessionBN nets;
- the external-MI teacher (EEGNeX + IFNet pretrained on 7 MI corpora, weight 5).

Every member is causal and resets per session; members are mixed with fixed weights.
- **core1** (`car_ov17_fs2`): common-average reference, 17 overlapping bands, full plus two half-window covariances,
  causal log-Euclidean recentring, logistic regression (C 10).
- **Hilbert core** (`car_fb7_hfs2`): complex analytic-signal covariances in 7 bands, logistic regression (C 3).
- **SessionBN nets:** batch norm uses the session's own causal running statistics (FBCNet k 2 and k 5, raw FBCNet
  k 20, SincShallow k 20, FBLight k 5; 6 seeds each).

**Stages after the mix:**
- **Rebalancing** (α 1.15, k 5, last 30 windows): divides each posterior by the recent class share. It exploits the
  class balance inside a run and is worth 5.75 J.
- **GEB SessionEM:** per-session class templates on frontal slow-potential and ocular features (nmin 20).
- **GEB TSEM:** per-session Gaussian class templates in the top 128 PCs of core1's features, seeded by the 70 % most
  confident past outputs (λ 0.15).
- **GEB (gated early boost):** raises both exponents in windows 30–60 for sessions with low early margins.
- **Routing v3:** from window 60, a gate matches the session against the 10 public people (τ 0.22, ramp 0.05) and
  puts the matched person's own head (PS1) at 3× into the core1 slot of a second tail.

**Dev metric:** J = 0.5 × new (all_loso, 10 people) + 0.5 × seen (P04/P10 sessions 2–3), person-equal cell-mean
balanced accuracy.
- **Bases:** B2_pool and B2_s345 for exploration; B2_s012, B2_s0, B2_s1 and B2_s2 for confirmation, opened only after
  a gate passes.
- **Selection:** nested over people halves G1 = P01 P02 P05 P07 P09 and G2 = P03 P04 P06 P08 P10.
- **Chain replay:** lean library `runs/archive_scratch_11nov/lean/alib.py` via `secop/env.py`. Reference J: B2_pool
  80.0131, B2_s345 79.8474.

## 4. What 11 Nov established (closed; details in the ledger and REVIEW_5NOV)

- **The session stage is at a local optimum.** The overnight programme ran 29 pre-registered arms and none passed; the
  null calibration covers all of them. Summary: `runs/archive_scratch_11nov/night/SUMMARY.md`; ledger §5.
- **Diagnosed headroom:**
  - Errors are pair confusions: MI↔WORD 37 %, SUB↔WORD 36 %, MI↔SUB 27 %. The true class is the runner-up in 73 %.
  - WORD recall is the lowest (73.7 %).
  - P04 holds 263 of 622 new-person errors.
  - Some family is right on 76–92 % of errors, but label-free selection has failed every time.
- **More seeds do not help:** single-seed bases average J 79.7, three-seed bases 80.0, and the pooled base 80.0.
- **The second-opinion head fails:** shuffled heads lose as much as the real ones.
- **ST-EEGFormer fails** (ledger §4): standalone new 53.85 against a bar of 66; chain +0.16 nested. With CBraMod,
  LaBraM and REVE, that makes four pretrained foundation models closed.
- **Architectures already tried** (REVIEW_5NOV §5b; OPTIMIZATION_R3):
  - FBCNet, IFNet, TIDNet, EEGNeX, CTNet, EEGConformer, ATCNet, Deep4, EEGTCNet, EEGITNet, MSVTNet, Inception-ERP;
  - an SPD net with SPD batch norm, a session-conditioned net, a single big model;
  - Codex's source screens.
  - **Never tried:** recurrent nets, Mamba, LMDA, EEGPT/BIOT, and SessionBN versions of TIDNet, CTNet or EEGConformer.
- **Graz cue order is not random** (ledger §9, new on 11 Nov):
  - In all 42 public Graz runs the cues follow a weakly shuffled WORD → SUB → MI cycle. Position alone predicts the
    label in 56 % of windows (chance 33 %). BrainHero runs are properly random.
  - For our chain it is worth at most +0.4 J even with the true run position: the chain's confidence does not mark its
    own errors, so the prior has no reliable place to override it. Not pursued.
  - A weaker base gains about +3 points on Graz cells, so other teams may be using it.
- **Spectral person identification** (ledger §10): closed-set identity is in the spectrum (6/6 by window 60), but a
  raw correlation score cannot reject new people early. The discriminant version is running (§5, BIO1).
- **Synthetic data:**
  - Round 3's "synthetic people" was a feature-space trick and came out negative.
  - Augmentations: spatial rotation was adopted (rot 0.05); frequency scaling and mirroring were tested, mirroring
    refuted; crops rejected.
  - A generative EEG model was never tried. Prior: low, since it cannot add class information it never saw.

## 5. Running (state at 14:55 UTC on 11 Nov)

**Galvani access** dropped at about 14:30 UTC (network changes on the Mac) and was renewed by the user at about
15:15 UTC. If it drops again: `ssh mlcloud-galvani`, the user types the 2FA code; agents must never re-authenticate.

| Job | IDs | State when access dropped | What to do with it | Ledger |
|---|---|---|---|---|
| **TSCH** Scherer as an eighth teacher corpus | pretraining arrays 3087564 (m8s EEGNeX), 3087565 (m8s IFNet), 3087566 / 3087567 (m7c controls), 6 seeds each; head export 3087589 chained after them | 2 of 24 running | Fetch `runs/diag/tsch/heads.npz`; build the teacher member (balanced logistic map C 0.01 per fold from seed-mean head log-probs, EEGNeX + IFNet concatenated, as `deep2/lib.teacher_fn`); swap it for alib member 10 at weight 5; compare m8s with m7c | §18 |
| **SCH** leftovers | array 3086622 | one FB task (SUB+WORD only, seed 1) missing | Nothing: scored without it, gate failed | §7 |

**Local:**
- **Weco** run `ea473cae-53bd-41fd-963b-fa11446fd3f1` finished (60 steps, 56 edits scored, 4.1 credits): one edit was
  +0.11 on the half it optimised and −0.11 on the hidden half; nothing beats the fitted head (ledger §13). The task in
  `.weco/wsp-core1/` is reusable for another target.
- **CSCH** (core1 with Scherer rows, §21): finished, refuted (chain −0.9 to −1.1).
- **Slot weight of the personal head under v4** (§22, exploratory): 2 is 0.5 worse on every base; 4.5 is +0.12 on
  average with mixed signs. The shipped 3 stays.
- **Independent review of the v4 package:** started 16:05 UTC; report goes to `runs/bio1/review_v4.md`.
- **Access:** renewed about 15:15 UTC by the user; the head-export watcher resubmits the export if the last
  pretraining task fails. TSCH scorer: `runs/archive_scratch_11nov/rt4/score_tsch.py HEADS.npz`.

**Scorers written on 11 Nov** (all need the old scratch path, see below): `runs/archive_scratch_11nov/rt4/route4.py`
(routing v4 replay; env TAU, PSC, G3, KAPPA), `stress.py`, `simdiag.py`, `score_core.py`; `runs/archive_scratch_11nov/sbnx/score.py`;
`evidence/wsp_bio1_11nov/wsp_score.py`.

**Old scratch path** (hard-coded in the chain replay library):
`S=/private/tmp/claude-501/-Users-sebastianboehler-Documents-GitHub-neural-interfaces-2026/95eb4b0e-c5db-4ae9-81b3-afeeeac7221a/scratchpad`.
If macOS has purged it, restore it with `mkdir -p $S && rsync -a runs/archive_scratch_11nov/ $S/`. Run local scoring
only through `scripts/run_capped.sh`.

## 6. Decisions waiting for you

**Scherer 2015 shipping:**
- **Licence:** BNCI 004-2015 is CC BY-NC-ND 4.0 (licensor TU Graz).
- **Rule 4:** "public datasets may be used when their original terms permit competition use".
- **Organiser guidance on Discord (27 Oct, off-topic):** Bruno said non-commercial datasets can train prize-winning
  entries, and answered "Can win" for third-party NC weights. This is not a Scherer-specific ruling.
- **Your Scherer-specific question** was posted on Discord on 11 Nov; there was no answer yet.
- **No derivatives:** this clause only applies when material is publicly shared (Creative Commons' guidance), so it
  matters only if we publish the weights.
- **Unverified reading:** an ineligible upload falling back to an earlier eligible one is our reading, not confirmed.

**Decided on 11 Nov:**
- **Scherer 2015:** the organisers answered the Scherer-specific question on Discord ("super okay"). Scherer-trained
  members may ship if they pass their gate; the corpus and its CC BY-NC-ND licence go into the method description.
- **Commits:** commit and push freely, with no AI author line (as in the 10 Nov commits). The fast package's code is
  committed in `submissions/ensemble_fast/` (7a2f9e8).
- **Cue-order artefact:** no question to the organisers. We follow the published rules and document the finding in
  the final method write-up.
- **Fast package:** no upload for speed alone. It goes up as the base of the next real improvement.
- **Repository:** the agent owns the codebase and cleans up stale documents and assumptions as it goes.

**Uploads:** each one needs your explicit go-ahead.

## 7. Facts worth keeping

**Organiser repo:**
- Ten commits since our pinned `a66bc23` (up to bddaff3d). For Track 2 they are documentation and config only: quota
  7, Force_Best, deadline, FAQ wording.
- `streaming.py`, the BCI `objective.py`, the data adapter and the requirements are unchanged. The pin can stay.
- The FAQ dropped the promise that windows are strictly chronological; it now says "the same evaluation order is used
  for all submissions".

**Runtime and submission handling:**
- The rules and config say 60 min. The organisers describe an actual 80-min worker timeout (a Codabench bug), which is
  why our 3,732 s run finished. Plan for 60.
- **Never cancel a sealed submission manually:** it costs a slot.

**Worker image** (replica `runs/cloud/ni26_worker.sif`, v3; the sealed v4 image is unpublished):
- Python 3.12.15, torch 2.14.1+cu130, triton 3.8.0, scipy-openblas.
- No gcc/g++/nvcc, numba, cython or cupy.
- A C++ extension cannot compile there, and torch.compile probably fails too. Only pure-torch speed-ups are safe.

**Discord:**
- It is the organisers' official contact (https://discord.gg/yZv8KqKMpH). GitHub issues #41 and #52 on licences are
  unanswered.
- Disclosed approaches overlap ours: session recentring, covariance-based person routing (identity inference from EEG
  is not prohibited), pseudo-labels, and C++ functional connectivity.
- The leaders' methods are undisclosed.
- **Codex's Discord brief** (copied into `runs/archive_scratch_11nov/discord_11nov/`):
  - `discord-research-brief.md`;
  - `wider-discord-context.md`;
  - `claude-handoff.md`;
  - `track2-leaderboard-2026-11-11.csv`;
  - `our_question_11nov.md` (the Scherer question posted on Discord).

**Where experiment time goes (profiled 11 Nov):**
- **Training is tiny; the queue is the bottleneck.** One member task (12 folds of a small net, 40 epochs) takes 2.5–6.5
  min on a 2080 Ti: about 30 s per fold at roughly 50 optimiser steps per second, which is Python overhead, not GPU
  load. The 21-task W12 array needed 65 min of wall time for its first 11 tasks (56 min of compute, run 1–2 at a
  time) and then stalled on fair-share priority.
- **Fix 1, packing:** `scripts/cloud/pack.sh NAME P TIME "CMD" TASKS…` runs many tasks in one job, P at a time on one
  GPU, and logs GPU utilisation every 20 s. One queue wait instead of one per task.
- **Fix 2, partitions and limits:** short time limits (≤ 1 h) backfill; `2080-preemptable-galvani` often has idle
  nodes, and our tasks skip finished outputs, so preemption only costs the task in progress.
- **Fix 3, the Mac:** the training code runs on MPS (M4 Max, 36 GB) and CPU-only analyses run locally. One capped job
  at a time through `scripts/run_capped.sh`.
- **Chain scoring** is not a bottleneck: one full chain replay takes about 4 s per base locally.
- **Inference** (sealed worker, per window): about 77 ms with the fast package; the CPU tangent path (two cores'
  eigendecompositions, 40 ms) is the critical path, and GPU eigh brings the whole run to 0.79×. Runtime is not
  binding today; it becomes binding only if a new stage adds a second tail or more members.

**Sealed cohort:** the organisers removed the cohort numbers. An earlier text said "20 participants × 6 sessions, ten
new participants", which implies roughly 57 % new people and 43 % later sessions of represented people. That is
unconfirmed. If true, returning-person sessions are a larger share than our 2-person seen proxy suggests.

**Local runs:** only through `scripts/run_capped.sh`, one at a time. The Mac shares RAM with the GPU and froze on
22 Oct.

**Planning lesson:** headroom estimates mislead. On L2D, the forecast was about 3rd and the result 13/110.

## 8. What 11 Nov (afternoon) added, and what is open

**Routing v4 checkpoint (staged, not uploaded; ledger §11, §14).**
- A label-free identity gate (spectral shape, two-covariance model trained on 224 external people) in front of the
  personal heads. External, session level: 1.9 % false accepts, 86.9 % correct accepts at τ 0.25 (the v3 gate: 10–42 %).
- Dev: +0.59 J over route3geb on six bases, all from starting the personal head about 50 windows earlier.
- Replays pass on A100 and in the worker image; runtime 1.02× the fast package.
- You decided not to lock it in yet.

**What limits returning people now (ledger §15, §17, §19).**
- Within a session a personal head gains +5.8 on average over ten people, and five of ten gain at least half of P04's.
  Across sessions P04 keeps 29 % and P10 nothing: the personal model's robustness to a session change is the limit,
  not identification.
- Personal information pays in proportion to the person's weakness (seen three times). A per-person trust weight is
  the natural refinement and cannot be validated with two dev people.
- A personal template prior in TSEM gives +0.19 J (P04 on every base, P10 slightly negative): not adopted, first to
  revisit if a sealed upload shows that returning-person levers transfer.

**Closed on 11 Nov (afternoon), each for the method class tested:**
- weight-space edits of core1 (ten hand edits, learned block gains, and a 60-step Weco search): the head is at its optimum;
- Scherer rows in core1's head (hurts by about 1 J);
- pair specialists (W12), untried SessionBN families (SBNX), Scherer as member training data (members +1–3 points,
  chain 0);
- similarity routing for new people, a forgetting recentring reference, a personal recentring prior, a more
  regularised personal head (wins at core level, loses in the chain).
- **Pattern:** stronger members or heads do not move the chain (four independent cases). Core-level results do not
  predict chain results for personal models.

**Open:**
1. **TSCH** (Scherer in the teacher): the only running bet on new people; prior about 15 %.
2. **Returning-person robustness across sessions:** nothing tested works yet; needs a chain-level idea.
3. **The upload decision on v4:** an independent package review is still to do before any upload.
4. **Segmentation hypothesis for the rescoring** and **GPU-eigh speed:** unchanged, low priority.

**Honest standing:** rank 3 needs about +1.3 sealed points. v4's dev gain is +0.6 J on the half of the metric that is
returning people; its sealed value depends on how the eight single-session public people respond.

## 9. Where everything is

**Docs:**
- current, in `docs/`: this file; PREDICTIONS_11NOV; SEALED_UPLOADS; REVIEW_5NOV; AUDIT_SBN / NXIF / M7R4; COMPETITION;
- history, in `docs/archive/`: STATE_7NOV, the optimisation rounds, strategy reviews, handoffs and Codex's PROTEUS
  notes. Its README lists the assumptions in them that are now wrong.

**Archives (gitignored, on this Mac only):**
- `runs/archive_scratch_11nov/`, 3.5 GB: a full mirror of the session scratch. It covers steeg, w12, schr, gpueig,
  night, order, secop, lean, fastpkg, discord_11nov and the earlier rounds.
- `runs/archive_7nov/`.

**Codex campaign:**
- `runs/rank1_8nov/` (STATUS.md, runtime_review/, next/).
- Codex thread rollout `~/.codex/sessions/2026/11/08/rollout-2026-11-08T09-19-26-01a1ba06-bca0-7d53-be32-4ea3bf78bef4.jsonl`.
- Discord review originals in `~/Documents/Codex/2026-11-11/applications-mentioned-by-the-user-appshot/outputs/`.

**Galvani** (`/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/runs/diag/`): w12, schr, gpueig, steeg, order11nov,
deep2.

**This chat** (95eb4b0e), for reference only:
`~/.claude/projects/-Users-sebastianboehler-Documents-GitHub-neural-interfaces-2026/95eb4b0e-c5db-4ae9-81b3-afeeeac7221a.jsonl`.
