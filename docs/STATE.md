# State: the starting point (as of 12 Nov 2026, 03:50 UTC)

This is the entry point. It is written so that a new session needs nothing from any chat history. Where an older
document contradicts it, this file wins.

**Read in this order:** this file; [LEDGER.md](LEDGER.md) (index of what is adopted, built, open and closed, and where new
runs are pre-registered; the full records are PREDICTIONS_11NOV.md and PREDICTIONS_12NOV.md);
[SEALED_UPLOADS.md](SEALED_UPLOADS.md) (upload log and procedure); [../OFFICIAL_CONTRACT.md](../OFFICIAL_CONTRACT.md).

## 1. Where we stand

- **Best sealed score:** 0.8161546323, submission 1048575, package `cand_v41_bio` (routing v4.1), 11 Nov, 1,128.6 s.
- **Rank:** 9th. Board on 11 Nov, about 19:00 UTC: 0.846 (seohee, ETRI), 0.830, 0.826, 0.825, 0.824, 0.821, 0.819,
  0.817, **0.816 (us)**, 0.815. Third place needs +1.0, first place +3.0, and the field keeps moving.
- **Deadline:** 21 Nov 2026, 22:59 UTC. **Uploads:** 7 per team, 1 per UTC day; 4 used, **3 left**.
- **Candidate ready, not uploaded: `cand_v51_own`** (v4.1 + an own-space personal stage for recognised returning
  people). Dev +0.89 J; **sealed estimate about 0.819 (0.817–0.824)**, revised down on 12 Nov after an independent
  review: the dev gain is P04's, and P04 is 3 of about 106 sealed sessions (ledger §63).
  Checked, replayed on A100 and in the worker image, independently reviewed, timed. See §3 and `SEALED_UPLOADS.md`.
- **Honest standing:** the night of 11–12 Nov scored about 65 pre-registered arms. One lever passed (the stage above).
  Nothing for new people moved, and no +3 source is known (§5, §9).

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
- **Order (changed 10 Nov):** the organisers removed every guarantee that windows arrive chronologically within a
  session ("the same evaluation order is used for all submissions"; commit 8734b2f6). Our chain loses 3–5 points
  under a re-ordering. Ten cuts and orders were fingerprinted against our four sealed numbers and none fits, so the
  sealed order is still close to acquisition order, but nothing in the rules keeps it so (ledger §62).
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
| `cand_v51_own` (v4.1 + own-space stage; sealed estimate +0.3, range 0.05–0.8) | Galvani `runs/sealed/cand_v51_own/` (33 files; `weights.npz` `e88bc3b3…`); code `submissions/ensemble_v5/submission.py` (`34eba3b0…`) | **Ready, not staged as a ZIP, not uploaded** |
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

## 5. What is established about headroom (revised 12 Nov; ledger sections in brackets)

- **New people are limited by the number of real training people.** core1 rises +8 points per doubling (56 / 61 / 65 /
  69 % for 3 / 5 / 7 / 9 people) with no flattening; the nets flatten from 7 to 9 [§39, §48, §56]. Nothing
  substitutes: seven kinds of generated variation, seven people-efficient estimators, noise, and Scherer 2015 even on
  real shared channels are all ≤ 0 [§46b, §47, §49, §53].
- **The chain gains only from a person's own labels.** The same three own-space models are worth +1.8 fitted on the
  person and −0.5 to −3.3 fitted on others; population members lose in the mix, as a final stage and as a view of
  SessionEM [§57c, §58]. Member weights and stages sit at a joint optimum (11 Nov: 29 + 8 arms, two searches).
- **In-session learning on the chain's own labels is closed in three feature families** (nets +0.6 for new people,
  population-PC head −0.7, session-own members −0.3). With true labels each would add +2 to +3 [§35, §38, 11 Nov §31].
- **The data hold more than core1's PCs show.** With its own labels a session reaches 81–82 % in session-own
  representations against 74.9 % in the population PCs; P04 reaches 74–80 % and is not a low-signal person; 512 PCs of
  core1's features lose four points [§36, §49]. This is what the own-space stage uses for returning people.
- **Dev resolution is about ±0.3, and exploration-base gains of that size do not survive.** Three times a +0.2 to
  +0.6 on the two exploration bases was zero on the four fresh ones [§54b, §58b, §61b]. Small levers do not stack
  [§57b].
- **The leaders' margin is unexplained.** Ranks 1 and 2 plausibly share one lab (name and roster only); its published
  methods (eye-movement branches, two-branch fusion, Mixup) are worth −7 to +0.3 here [§58, §59].
- No shortcut in slow potentials, gaze or feedback, and no usable cue-order prior after the stages (11 Nov §9, §33;
  12 Nov §55, §58).

## 6. Levers (state on 12 Nov)

**Built: the own-space personal stage (package `cand_v51_own`).** For windows the v4.1 gate routes to public person j
with ramp r, the routed output is multiplied by exp(0.10 · r · mean clipped log posterior of three models fitted on j's
public sessions in j's own space: FBCSP + LDA, own tangent space + logistic regression, slow time courses + LDA).
- Dev, six bases: seen +1.79 (P04 +3.5, P10 +0.1), new 0.00, J +0.89 [§45]. Ten-person within-session proxy: +4.2 to
  +5.1 over a person-naive chain, twice a population-PC head [§41]; 12 real session pairs of P04 / P10: +10.4 against
  +5.8 [§42].
- The gain on dev is P04's. How the eight single-session returning people respond is not measurable on dev; it is the
  whole range of the sealed estimate (+0.4 to +0.9).
- Code: `submissions/ensemble_v5/submission.py`, `scripts/add_own_stage.py`, `scripts/check_own_stage.py`,
  `scripts/check_perm_stage.py`; study code `studies/12nov/ownspace.py`, `v5_replay.py`.

**Measured, not built.**
- **TTT** (in-session fine-tuning of the nets on the chain's labels): new +0.6 to +0.7 with four families, seen −0.5;
  runtime-feasible form new +0.4 to +0.65 (11 Nov §24, §31). Re-measured on 12 Nov in the form a package can run
  [§60]: the form works on single nets (+4.8 alone, trace-exposed parameters suffice), but the chain gain for new
  people is about +0.2 with a person-bootstrap interval of −0.5 to +0.9. **Not built.**
- **PFT** (nets fine-tuned on a returning person's public session): on top of the stage seen +2.3 against +1.9; P10
  −1.0 [§40, §43]. Needs per-person net exports and a catch-up replay.
- Small and not built: an eye-movement net's log posteriors in SessionEM's view (+0.28 on six bases) [§58b]; REST4
  (+0.1 to +0.3, 11 Nov §26).

**No v6 is planned:** with TTT at about +0.1 sealed for 20–40 extra minutes of runtime, v5.1 is the candidate.

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

1. **Upload `cand_v51_own` or hold.** It is the next candidate and also the only way to measure the returning branch
   on the eight single-session people. Reading rule fixed in ledger §63: ≥ +0.4 over v4.1 → remaining uploads go to
   the returning branch; ≤ +0.15 → that branch stops. The user said on 12 Nov (several times) to keep iterating before another
   upload; no go-ahead has been given. Every upload needs the user's explicit go-ahead and a live phase and quota check.
2. **The two leading accounts appear to belong to one lab** (§5). Whether to raise that with the organisers is the
   user's call.

**Rules set by the user on 12 Nov** (also in `AGENTS.md`): no gate or closure is final (check what actually ran before
citing one); cleanups and reviews must hunt silent no-ops (options parsed but not applied); the codebase exists to
optimise the final submission. Time stamps in the ledger come from `date -u`.
