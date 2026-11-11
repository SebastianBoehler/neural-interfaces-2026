# State on 11 Nov 2026 (updated 11:50 UTC): the starting point

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

## 5. Running on Galvani (state at 11:50 UTC on 11 Nov)

**Collecting results:**
- Galvani access is `ssh mlcloud-galvani`. You type the 2FA code yourself; agents must never re-authenticate.
- The ControlMaster was renewed at 10:15 UTC on 11 Nov and drops after about 12 h (around 22:15 UTC).
- The queue is slow: jobs wait on fair-share priority. Everything below shares one user queue.

**Before running an older scorer:** the W12 and Scherer scorers hard-code the scratch path
`S=/private/tmp/claude-501/-Users-sebastianboehler-Documents-GitHub-neural-interfaces-2026/95eb4b0e-c5db-4ae9-81b3-afeeeac7221a/scratchpad`.
If macOS has purged it, restore it first with `mkdir -p $S && rsync -a runs/archive_scratch_11nov/ $S/`. Run local
scoring only through `scripts/run_capped.sh`.

| Job | IDs | State | What it tests | Ledger |
|---|---|---|---|---|
| **W12** pair specialists | array 3086618 (21 tasks) | 9 done at 11:40 UTC | SessionBN nets trained on one class-pair boundary re-split the chain's pair mass | §6 |
| **SCH** Scherer 2015 as training data | array 3086622 (24 tasks) | all pending; Slurm estimates up to 12 Nov ~10:00 UTC | Members retrained with Scherer at loss weight 0.1 / 0.25, against PROTEUS-only twins | §7 |
| **BIO1** person identification | 3087013 (CPU) | pending | Two-covariance discriminant trained on 6 external multi-session corpora; open-set accepts at 5–60 windows | §11 |
| **SBNX** untried SessionBN families | smoke 3087028, array 3087029 (16 tasks) | pending, behind SCH | TIDNet, CTNet, EEGConformer, ATCNet, EEGSimpleConv, FBMSNet, SCCNet, Deep4 with session normalisation | §12 |

**Scoring commands:**
```
bash $S/w12/fetch.sh
NI26_MAX_GB=8 scripts/run_capped.sh $S/w12/score.log .venv/bin/python $S/w12/score_w12.py     # fresh=1 only after a gate pass
bash $S/schr/fetch.sh
NI26_MAX_GB=8 scripts/run_capped.sh $S/schr/score.log .venv/bin/python $S/schr/score_schr.py NJ=2
```
- BIO1 writes `runs/diag/bio1/bio1.json` and `bio1-3087013.out` on Galvani.
- SBNX writes `runs/diag/sbnx/probs/sbnx_<arch>_k<k>-<proto>-s<seed>.npz`; it has no scorer yet (adapt the `mt` arm of
  `score_w12.py`: extra member at weight 0.5 / 1 / 2).

**GPU-eigh speed probe** (ledger §8; finished, not adopted):
- ABBA timing (3086723): whole run 87 s against 110 s for the fast package (0.79×), same in-sample score.
- The parity job (3086722) ended with a failed bitwise assertion; labels were identical. Whether the difference
  matters was not established. Only worth resolving if new members need runtime.

To cancel: `ssh mlcloud-galvani scancel 3086618 3086622 3087013 3087028 3087029`.

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

## 8. Open directions

1. **Returning-person identification (BIO1 → routing v4).** The largest lever identified on 11 Nov.
   - Routing + GEB gained +0.55 sealed in total; a rough split puts routing at about +0.2, against about +0.9 if every
     returning session were routed and gained like the P04/P10 mean.
   - The shortfall is either weak identification on sealed (fixable) or other returning people gaining nothing from a
     personal head, as P10 does not (not fixable). Public data cannot tell which.
   - If BIO1 passes its gate: an earlier, more reliable gate for the existing PS1 heads.
2. **Scherer member:** if SCH passes, it ships only after the licence decision.
3. **W12 specialists and SBNX families:** low priors (about 10 % each); confirm on fresh bases before any build.
4. **Segmentation hypothesis for the rescoring:** replay with per-interface resets or merged sessions. Cheap, CPU
   only; reuse `runs/archive_scratch_11nov/order/olib.py`. It explains the past rather than adding points.
5. **GPU-eigh speed:** only if new members need runtime headroom.

**Honest standing:** rank 3 needs about +1.3 sealed points and the field is moving. No candidate above has an
estimate near that.

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
