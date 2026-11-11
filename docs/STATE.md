# State: the starting point (baseline of 11 Nov 2026, 22:30 UTC)

This is the entry point. It is written so that a new session needs nothing from any chat history. Where an older
document contradicts it, this file wins.

**Read in this order:** this file; [LEDGER.md](LEDGER.md) (what is closed, and where new runs are pre-registered);
[SEALED_UPLOADS.md](SEALED_UPLOADS.md) (upload log and procedure); [../OFFICIAL_CONTRACT.md](../OFFICIAL_CONTRACT.md).

## 1. Where we stand

- **Best sealed score:** 0.8161546323, submission 1048575, package `cand_v41_bio` (routing v4.1), 11 Nov, 1,128.6 s.
- **Rank:** 9th. Board on 11 Nov, about 19:00 UTC: 0.846 (seohee, ETRI), 0.830, 0.826, 0.825, 0.824, 0.821, 0.819,
  0.817, **0.816 (us)**, 0.815. Third place needs +1.0, first place +3.0, and the field keeps moving.
- **Deadline:** 21 Nov 2026, 22:59 UTC. **Uploads:** 7 per team, 1 per UTC day; 4 used, **3 left**.
- **Honest standing:** nothing in hand is worth +1.0. Two levers work and add up to about +0.3 to +0.6 sealed (§6).
  A search for a +3 source on 11 Nov found none (ledger index, "hard bounds").

## 2. Hard bounds (from the official code and rules)

- **Input:** one window per call, float32 `(1, 41, 480)`: 4 s at 120 Hz, starting 1 s after the cue. Output: one label
  (0 motor imagery, 1 mental subtraction, 2 word generation). No labels, no person, session or interface metadata.
- **Stream:** one participant-session; runs in order, windows in recording time; Graz and BrainHero share the session.
- **State:** the evaluator deep-copies the loaded model at every session boundary. Nothing may cross sessions.
  Inside a session everything causal is allowed (adaptation, fine-tuning, identity inference from EEG).
- **Metric:** balanced accuracy per participant × session × interface cell, then the plain mean over cells.
- **Runtime:** 60 min by rule (the worker's real timeout is 80 min; plan for 60). The package is a ZIP with
  `submission.py` plus data files; no extra Python files.
- **Worker image:** Python 3.12, torch 2.14 + CUDA, no compilers, numba or cupy. Only pure-torch speed-ups are safe.
  A native CPU replay does not prove the Linux/A100 worker runs.
- **Uploads:** each needs the user's explicit go-ahead and a live check of phase and quota. Never cancel a running
  sealed submission: it costs a slot. Codabench has a bot check; the user opens the page in Chrome.
- **Data:** public datasets are allowed if their terms permit. Scherer 2015 (CC BY-NC-ND) was cleared by the
  organisers on Discord on 11 Nov; it must be declared in the method write-up.
- **Cue-order regularity** (Graz runs follow a weakly shuffled WORD → SUB → MI cycle): not raised with the organisers;
  it goes into the final method write-up (user decision, 11 Nov).

## 3. The system (uploaded as v4.1)

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
| `cand_v41_bio` (routing v4.1) | `runs/upload/sealed_cand_v41_bio.zip` (`53fc1679…`); code `submissions/ensemble_v4/submission.py` (`45f9f2fd…`) | **Uploaded, best** |
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

**What dev numbers are worth on the sealed set:**
- Noise: 8 permuted-view null arms gave −0.22 to +0.18 J. Treat anything below +0.3 as noise.
- Small stacked gains transferred at 0.6–0.7×; v4.1 at 0.46× (+0.58 dev, +0.267 sealed).
- **Dev "seen" weights P04 and P10 equally. On the sealed set P04 probably has 3 of about 46 returning sessions**
  (if the organisers' withdrawn cohort text holds: 20 people × 6 sessions, 10 of them new). Every returning-person
  gain so far is P04's. Read returning-person levers by P10's response.
- New people are person-equal in both, so "new" transfers more directly. P04 as a new person is 54 %; everyone else
  75–90 %.

## 5. What is established about headroom

- The label-free chain (77 % of windows right on new people) is as accurate as a supervised model trained on about
  150 labelled windows of the same session in core1's features (76 %).
- True labels inside the session would add +3 J through a per-session head and +1.6 J through fine-tuned nets.
  No label-free source of labels gets past 77 %: smoothing in both directions, run-balance constraints and feeding
  improved labels back all stop there.
- The members share their errors. The 12-member mix is 0.6 points more accurate than core1 alone before the session
  stages; the best label-free combination adds about 1 point there and loses 1–2 J after the stages.
- Extra members lower the chain even when they are strong alone and temperature-calibrated (five feature or model
  families on 11 Nov). Member weights and the session stages sit at a joint optimum (29 + 8 stage arms, two Weco
  searches: nothing passes).
- P04 is a low-signal person (55–59 % with their own labels), not a bad recording.
- No shortcut in slow potentials, gaze or feedback: checked per interface and per half-window.

## 6. Open levers (measured, not built)

**TTT: fine-tune the nets inside the session on the chain's own labels (for new people).**
- Every 10 windows from window 30, a copy of each SessionBN net is trained for 20–60 full-batch steps on the windows
  so far, labels = chain output, weighted by confidence.
- Four families, three seeds each: new +0.6 to +0.7 (11 of 14 sessions gain; G1 about 0, G2 +1.6), seen −0.5.
- Runtime-feasible form (one fine-tuned net per family next to two frozen seeds): new +0.4 to +0.65, seen −0.4 to
  +0.6, six of six draws. About 5 s per net and session on a 2080 Ti; a sealed run has roughly 100 sessions.
- With true labels the same procedure gives +1.6 J: the lever is label-limited.
- Code: `studies/11nov/ttt/galv/ttt.py`, `ttt2.py`; scorers `studies/11nov/ttt/score.py`, `round.py`, `poe.py`.

**PFT: nets fine-tuned on a returning person's own public session.**
- Standalone +2 to +9 points per family across sessions; chain seen +1.4 to +2.0 with five families, all from P04.
  Only the head-and-BN-affine variant has both people ≥ 0 (+1.44: P04 +2.36, P10 +0.53).
- Ten-person within-session proxy (runs 1–3 → runs 4–6): every person gains standalone, mean +5.7 points; 7 of 10
  gain at least half of P04's.
  A standalone gain does not settle the chain value: P10 gains standalone and nothing in the chain.
- Code: `studies/11nov/ttt/galv/pft.py`, `pft_proxy.py`; scorers `pft_score.py`, `pfx_score.py`.

**A "v5" package** = v4.1 + TTT for sessions not recognised as returning + PFT nets behind the identity gate.
Estimate +0.3 to +0.6 sealed. Needs: per-person net exports, a catch-up replay of buffered windows when a person is
recognised, in-session training inside `submission.py`, and a timing replay on A100 and in the worker image
(TTT adds roughly 20–40 min; a timeout costs an upload). Not started; the user has not decided.

**Smaller, not adopted:** personal class templates as a TSEM prior (+0.19 J, P04 only); rest periods as a fourth
class in net training (+0.1 to +0.3 J); slot weight 4.5 for the personal head (+0.12).

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
| Closed directions 5–10 Nov | `docs/REVIEW_5NOV.md` |
| History (superseded; may be wrong) | `docs/archive/` (its README lists the wrong assumptions) |
| Chain replay | `replay/` |
| Study scripts of 11 Nov | `studies/11nov/` (the ledger cites them as `runs/archive_scratch_11nov/<same path>`) |
| Training, features, exports | `experiments/`, `scripts/` (only code that packages, audit recipes or the ledger still reference; the rest was removed on 11 Nov, see `docs/archive/README.md`) |
| Identity gate | `experiments/bio_gate.py`, `experiments/bio_id.py` |
| Compact evidence logs | `evidence/` (11 Nov: `evidence/wsp_bio1_11nov/`, `evidence/seq_bio_11nov/`) |
| Local only (gitignored) | `data/`, `runs/` (`runs/upload/` ZIPs, `runs/replay_data/`, `runs/ttt/` outputs, `runs/archive_scratch_11nov/` full scratch mirror, 3.5 GB) |
| Galvani outputs | `runs/diag/{ttt,rest4,aec,sbnx,w12,schr,steeg}` under the work dir |

## 9. Decisions waiting for the user

**Update, 11 Nov 22:25 UTC (ledger §34–§38; scripts `studies/12nov/`).** The user's steer: a team scores 0.846 under
the same rules, so a +3 method exists; the ceilings in §5 are ceilings of our representation, not of the data.
- **Confirmed (§36):** with its own labels a session holds 81–82 % in session-own representations (FBCSP, own
  tangent space, slow time courses) against 74.9 % in core1's population PCs. P04 reaches 74–80 % in its first
  session: **P04 is not low-signal; the population model does not cover it.** The §5 statements about the 76 % level
  and about P04 hold only inside core1's features.
- **In-session learning on the chain's labels is closed in three feature families** (§35 PC head −0.7, §38 own space
  −0.3, TTT nets +0.6). True labels would add +2 to +3 J in each. This is not where the +3 is.
- **Returning people:** own-space personal members as a tempered product in the head's slot give P04 +2 to +3 and P10
  0 (§37, exploratory), the same pattern as PFT. Five of the eight single-session people look P04-like by §36.
- **Open:** a population model that covers more kinds of people (next: the learning curve over training people and
  what a P04-like person needs from the training set); a returning-person package from PFT plus own-space members.
  No upload until the user says so.

1. Build v5 (§6) or keep the three uploads for something larger.
2. Any information on what the leading teams do (Discord). The leader gained +2.5 in one upload while its runtime
   rose fourfold, which fits in-session fine-tuning; on our chain that is worth far less.
