# State on 11 Nov 2026 (11:05 UTC): the starting point

This file supersedes docs/STATE_7NOV.md as the entry point. It is written so that a new thread needs nothing from the
chat history.

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
| Lean reference code | `submissions/ensemble/submission.py` `dacee85a…` (committed 9491882) | Reproduces all 3 uploads |

**Fast package details:**
- ZIP SHA-256 `cfd1a4b4…f43f417d`, 57,280,590 B; `submission.py` `ccbce506…`, 904 lines. It contains the same 32 data
  files as route3geb.
- Its runtime code is Codex's shared FIR bank plus CPU/GPU overlap.
- **Parity:** every stage is bitwise equal on all 2,655 public windows.
- **Speed:** A100 solver time 0.41× of route3geb; whole run 0.53× (ABBA job 3086004). Estimated sealed run 19–35 min.
- **Not committed:** its code lives only in gitignored `runs/` and in `runs/archive_scratch_11nov/fastpkg/`. It must be
  committed before or with any upload, for the audit.
- It is the base for every future upload. Upload only with your explicit go-ahead, after checking the live quota and
  phase.

## 3. Pipeline (unchanged since 7 Nov; full description in docs/STATE_7NOV.md)

**Members:**
- two tangent-space cores (core1 and a Hilbert core);
- about 35 plain nets (ShallowConvNet, FBCNet, FBLight, SincShallow);
- 30 SessionBN nets;
- the external-MI teacher (EEGNeX + IFNet pretrained on 7 MI corpora, weight 5).

**Stages after the mix:**
- rebalancing (α 1.15, W 30);
- GEB SessionEM;
- GEB TSEM (PC128, λ 0.15);
- routing v3 (returning-person PS1 heads).

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
- **Synthetic data:**
  - Round 3's "synthetic people" was a feature-space trick and came out negative.
  - Augmentations: spatial rotation was adopted (rot 0.05); frequency scaling and mirroring were tested, mirroring
    refuted; crops rejected.
  - A generative EEG model was never tried. Prior: low, since it cannot add class information it never saw.

## 5. Running when this file was written (Galvani; no agent is watching)

**Collecting results:**
- Galvani access is `ssh mlcloud-galvani`. You type the 2FA code yourself; agents must never re-authenticate.
- The ControlMaster was renewed at 10:15 UTC on 11 Nov and drops after about 12 h (around 22:15 UTC).

**Before running any scorer:** scripts hard-code the scratch path
`S=/private/tmp/claude-501/-Users-sebastianboehler-Documents-GitHub-neural-interfaces-2026/95eb4b0e-c5db-4ae9-81b3-afeeeac7221a/scratchpad`.
If macOS has purged it, restore it first with `mkdir -p $S && rsync -a runs/archive_scratch_11nov/ $S/`. Run local
scoring only through `scripts/run_capped.sh`.

**W12, pair specialists for the WORD confusions** (ledger §6; prereg `$S/w12/prereg.txt`)
- **Jobs:** smoke 3086617 OK; array 3086618 (21 tasks).
- **What it tests:**
  - SessionBN FBCNet and SessionBN ShallowConvNet nets, 3 seeds each, trained only on SUB/WORD, MI/WORD or WORD vs rest.
  - A multitask FBCNet.
  - The specialists re-split the chain's pair mass at β 0.25/0.5/0.75.
- **Gate:** nested dJ ≥ +0.5, both held-out halves ≥ 0, seen ≥ −0.3.
- **Known before results:** the 3-class FB member already splits the pairs at 74.2 / 77.5 / 72.7 BA on new people, so
  the ledger's pair-BA predictions (66 / 70 / 65) will miss low.
- **Score it:**
  ```
  bash $S/w12/fetch.sh
  NI26_MAX_GB=8 scripts/run_capped.sh $S/w12/score.log .venv/bin/python $S/w12/score_w12.py
  ```
  Add `fresh=1` only after a gate pass.

**SCH, Scherer 2015 as extra training data** (ledger §7; prereg `$S/schr/prereg.txt`)
- **Jobs:** smoke 3086621 OK; array 3086622 (24 tasks). Slurm's pessimistic start estimate was 20:18 UTC.
- **What it tests:** the SessionBN FB slot (members 6 and 9) and the raw ShallowConvNet (member 4), each retrained
  with Scherer at loss weight 0.1, 0.25, or 0.25 on SUB+WORD only. Hand-MI windows get zero loss weight but stay in
  the normalisation.
- **Control:** PROTEUS-only twins from the same code. The chain swaps the member in, against the twin swapped in.
- **Score it:**
  ```
  bash $S/schr/fetch.sh
  NI26_MAX_GB=8 scripts/run_capped.sh $S/schr/score.log .venv/bin/python $S/schr/score_schr.py NJ=2
  ```
- Research use only; see §6 for the licence.

**GPU-eigh speed probe** (ledger §8; the agent was stopped mid-way)
- **Idea:** run the tangent cores' batched eigendecompositions on the A100. Today they take 40 of 124 ms per window, and
  the CPU tangent path is still critical after the overlap.
- **Code:** `$S/gpueig/submission.py` and `code.diff` (a side CUDA stream; the CPU path is meant to stay unchanged).
- **Jobs:** A100 parity 3086722 and ABBA timing 3086723 were running. Results go to Galvani
  `/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/runs/diag/gpueig/` (`parity_3086722.log`, `abba_3086723*`).
- **Still to do:** the local CPU bitwise check (`$S/gpueig/local/`) and the summary.

To cancel any of these: `ssh mlcloud-galvani scancel 3086618 3086622 3086722 3086723`.

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

**Commits:** you said in the Codex thread not to show Codex or Claude as author. Settle the attribution line before
the next commit. Nothing has been committed since 9491882. Pending items:
- the fast package's `submission.py`;
- the doc updates (AUDIT_SBN, SEALED_UPLOADS, REVIEW_5NOV, PREDICTIONS_11NOV, Codex's PROTEUS docs, this file).

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

**Sealed cohort:** the organisers removed the cohort numbers. An earlier text said "20 participants × 6 sessions, ten
new participants", which implies roughly 57 % new people and 43 % later sessions of represented people. That is
unconfirmed. If true, returning-person sessions are a larger share than our 2-person seen proxy suggests.

**Local runs:** only through `scripts/run_capped.sh`, one at a time. The Mac shares RAM with the GPU and froze on
22 Oct.

**Planning lesson:** headroom estimates mislead. On L2D, the forecast was about 3rd and the result 13/110.

## 8. Open directions (none tested; low-to-moderate priors)

1. **Scherer member:** if SCH passes its gate, it goes into a package only after the licence decision.
2. **W12 specialists:** if one passes, confirm on fresh bases, then build on the fast package.
3. **Returning-person lever:** if later sessions are about 43 % of sealed cells, better identification and personal
   heads matter more than the dev proxy shows.
   - Routing gives an oracle +2.2–3.4 on seen, 90 % of it from P04; open-set identification is the limit.
4. **Segmentation hypothesis for the rescoring:** replay with per-interface resets or merged sessions. Cheap, CPU only;
   reuse `runs/archive_scratch_11nov/order/olib.py`.
5. **SessionBN versions of untried families** (TIDNet, CTNet) or a small recurrent net over sub-window band power: free
   2080 GPU-hours, low prior.
6. **GPU-eigh speed:** only worth adopting if new members need runtime headroom.

## 9. Where everything is

**Docs (tracked or untracked in `docs/`):**
- this file; PREDICTIONS_11NOV; SEALED_UPLOADS; REVIEW_5NOV; AUDIT_SBN; STATE_7NOV;
- Codex's PROTEUS_SOURCE_10NOV, PROTEUS_ACQUISITION_10NOV, PROTEUS_ALPHA_CONFIRMATION_10NOV and
  PROTEUS_CHANNEL_POWER_BLEND.

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
