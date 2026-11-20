# Red team · fundamental bugs and wrong assumptions (Fable, Fri 20 Nov 2026, 22:25–22:55 CET)

Scope: strict review, no tuning. I ran local real-data checks and read cluster outputs; **I started no cluster job**
(scope set by the user). Checks that need cluster runs are specified under "Handoff checks" for an executor.
Rules read from upstream `2026-competition` main @ `8447e181` (cloned to scratch; `git diff a66bc23 8447e18` touches no
BCI code, only pages, yaml and README text).

## Verdict in four lines

1. **CONFIRMED, largest:** the in-lab returning instrument used since 17 Nov (P04/P10, visit 1 enrolled) contains
   three supervised parts fitted on the test visits' labels. Its "90–97 %, near ceiling" is wrong. The clean number
   from the real fold packages is **P04 70.2 %, P10 89.2 %** (mean 79.7 %, against 90.6 % on the same kind of order).
   Every returning-lever verdict taken on that instrument (dose, keep-adapting R4/R4b, drift D3) has to be re-read.
2. Recognition itself is fine on genuinely new days: 16 of 16 clean returning streams recognised, right person, and
   0 of 56 new-person streams falsely recognised. The prime suspect "recognition never fires" is refuted.
3. No data-path bug: offset, labels, channel order, µV, 120 Hz, interface mapping and per-person wiring all check out.
4. Nothing here explains the 3 pp to the top by itself. What it changes: the returning half was steered by a biased
   instrument, and the clean in-lab test of the returning levers has not been run yet (H1, H5 below).

## F1 · Returning instrument leaks test-visit labels — CONFIRMED

**What ran.** The "fold package" behind Lane A and its descendants is not a fold fit. It is a copy of the uploaded
all-data package with some parts replaced by source-only fits:
`studies/17nov/js_readiness321/returning_bundle.py:12` (source `draft_base`), `:34` (`base_v66.zip`, sha `9c4d5538…`,
the uploaded cap250), `:94-97` (its own manifest: status `…built_NOT_validated`, scope
`ENGINEERING_CONDITIONAL_FIXED_PRODUCTION_PRIOR_ONLY`, `retained='ORIGINAL rest4/eye auxiliary productionNN;other
classical member weights outside core1;…'`). Consumers: `friday/ea_returning/native_inner/inputs.py:12` (`BUNDLE`),
`laneA/dose_worker.py`, `laneR4/ret_build.sbatch:12` and `laneD3/ret_setup.sbatch:23`
(`ni26_final_stack_ret_20261120/pkg_stack/{P04,P10}_hardcb`).

**Reproduced.** Against the uploaded package (`runs/sealed/final_zips/v66_cp1_cap250.zip`):

| Part, retained from the all-data fit | Role | Evidence |
|---|---|---|
| `ts2_log_glob, ts2_mu, ts2_sd, ts2_W, ts2_b` | Hilbert tangent core, logistic regression, 1/15 of the mix | arrays equal in `returning_bundle2/P04_s23/base/weights.npz`; refit in the real fold |
| `rest_s0..2.pt`, `v6_rest_G`, `v6_rest_scale` | rest member, supervised nets, 0.0625 of the mix | files byte-identical (sha `2f771a1f…` for `rest_s0.pt`) |
| `v6_cf_W, v6_cf_sw, v6_cf_thr, v6_cf_lp` | count-filter classifier | arrays equal |
| `eye_s0..2.pt`, `em_ev_g`, `net34_G`, `net34_scale` | label-free fits that saw the test windows | files and arrays equal |

The same fingerprints (`ts2_W a28f10b7…`, `v6_cf_W 670767b6…`, `rest_s0.pt 2f771a1f…`) are in
`ni26_final_stack_ret_20261120/pkg_stack/{P04,P10}_hardcb`, the packages R4, R4b (stored tapes) and D3 scored.
The shipped package was fitted on all 14 public sessions, so these parts saw the labels of P04/P10 visits 2–3.

**Numbers** (official cell metric, visit 1 enrolled, visits 2–3 streamed, full-session shuffle):

| Cell | Clean: real fold packages, real Solver, orders 71000–71003 | Contaminated: Lane A stock arm, full0/full1 | Same cell held-new (real fold) |
|---|---:|---:|---:|
| P04 visit 2 | 68.3 | 86.8 | 47.9 |
| P04 visit 3 | 72.1 | 86.3 | 46.0 |
| P10 visit 2 | 88.5 | 92.3 | 86.2 |
| P10 visit 3 | 90.0 | 97.0 | 83.0 |
| Mean | **79.7** | 90.6 | 66.0 |

Clean source: Leonardo `…/cap250/runs/production-20261120-32a94ab7/{stock,hardcb}` (`scores.json`, `pred_ret_*.npz`;
hardcb: P04 70.1, P10 89.2). Both instruments agree on held-new people (73.6–74.1 against 73.74), so the gap is
specific to the returning path. One enrolled visit is worth +22 pp for P04 and +5 pp for P10, not +40 / +12.
Limit: the two instruments are not paired (different pipelines and orders). The leak is a fact; that it explains
the whole 11 pp is PLAUSIBLE until H2 runs.

**Why it matters.** With label-leaked members in the mix, any change that moves weight to label-free adaptation or
to a one-session personal head can only lose. That is the sign every returning lever showed on this instrument:
Lane A dose (−1.7 … −9.8), R4 keep-adapting (P04 −0.5 … −1.1), D3 (P04 −1.3), and R4b takes its in-lab veto and its
threshold bracket ("P04 conf 0.75, P10 0.91") from the same stored tapes. The clean foreign-lab instruments showed
the opposite sign at a 68.6 % base (NETBCI: R5 +0.55, R1a +0.36, R2 +2.5, R4 +3.7). Clean P04 sits at 70 %.

**Also affected.** `evidence/audit_20nov/first_principles_fable.md:27,38`: "returning 89.98" and the composition check
"0.57 × 77.35 + 0.43 × 89.98 = 82.78 ≈ sealed 82.99". With 79.7 the sum is 78.4. The sealed score is still
reachable if the eight one-session people average about 91 % as returning, which is plausible (as new people they
average 80.7 % on their first visit; P04 and P10 are the two weakest at 55.0 and 68.6), but it is no longer a
confirmation of the 57/43 mix. Cards that quote the contaminated level: `laneR_onesession.md:36`,
`laneR4_keepadapt.md`, `laneR4b_dose.md`, `laneD3_drift.md`, `laneB10_returning_stress.md:4`, `HANDOVER.md:37`,
`final_stack.md`, wave_19nov lanes A/C/G/I returning columns, and the returning columns of the STATE alternates
scored through `friday/ea_returning` (FBL2, F-C20; their scorers run on the same release, not rechecked per card).

**Sealed impact.** None on the staged hardcb package: its returning delta was already measured on the clean
instrument (−0.07 pp mean, per order 0.00 / 0.00 / −0.55 / +0.26). The opening is a returning lever that was vetoed
on the wrong instrument: returning cells are about 43 % of the sealed set (unverified), so a clean in-lab gain of
+0.5 to +3 pp on them is +0.2 to +1.3 pp sealed. Sign and size are unknown until H1 and H5 run.

**Fix.** Use only the real fold packages (`production-20261120-32a94ab7/{stock,hardcb}/ret_P04|ret_P10`) as the in-lab
returning instrument. Re-score the runtime-feasible returning arms there before the Saturday decision (H1, H5).
Do not apply the Lane A based veto to R4b.

## F2 · Recognition on genuinely new days — CONFIRMED working

- Clean real fold packages (stream logs of the production run): returning streams recognised 16 of 16; first
  recognised window P04 visit 2: 29, 10, 10, 22, visit 3: 10 ×4, P10: 10 ×8; P04 has 155–179 of 188–189 windows routed.
  New-person streams: `recognised_at -1` in 56 of 56 sessions, in both arms. No false recognition.
- Per-window gate tapes of the 32 Lane A streams: recognised 32 of 32, never the wrong person. The identity gate
  names the true person on every window; its arrays were source-fitted there and it reads only the raw windows, so
  its traces are valid. The contrast gate reads posteriors that include the leaked parts (F1), so its traces are not.
- The spectral gate alone is thin on visit 2: end-of-session true-person score 0.295 (P04 v2), 0.449 (P04 v3),
  0.351 (P10 v2), 0.370 (P10 v3) against the threshold 0.26. Under run-contiguous delivery it first fires at window
  160 (P04 v2) and 106–109 (P10 v2); there the contrast gate fired at its start window 60, but only on the
  contaminated tapes, so that rescue is unproven. Under full shuffle the identity gate fires at 10–32 (clean: 10–29).
- NETBCI cross-day, foreign lab (cards B10 and Q, not rerun): 92 % of streams latch, median window 10, 86 % of
  windows routed; spectral gate alone: true accept 0.82, wrong person 0.10, impostor latch 0.15.
- Sealed evidence that routing happens: upload 5 minus upload 4 is +0.361 pp for a change that acts only on routed
  windows (`docs/SEALED_UPLOADS.md`).
- Personal path against treating the person as new, clean: P04 +22.2 pp, P10 +4.6 pp (table above). This bundles
  member training on visit 1 with the personal route; the split is lane RP's gate-suppressed control (H4).
- Sealed impact: about 0 central. Tail: a miss rate of 10–20 % on the eight one-session people (38 % of cells) times
  the route's own share of the gain (unmeasured, H4) is at most about 0.4 pp.

## F3 · Latch is irreversible, routing is per window — CONFIRMED, small

`submissions/lean_v66/submission.py:1553-1567`: the first positive ramp sets `recB` for the session and fine-tuning
stops (`v6_routed = 1`); `:1533` personalisation stops too. `:837-843`: the ramp itself is recomputed per window.
Seen on tapes: two P10 visit 2 streams latched at window 10 and then stayed mostly unrouted until window 60
(identity ramp positive on 46 % and 82 % of windows). Such a session has neither adaptation nor the personal route for up to 50 windows. For a new
person one false blip would cost the session its adaptation; none occurred in 56 dev sessions. Impact ≤ 0.2 pp.
No change advised before the deadline unless H1 shows keep-adapting helps (those arms remove the dependence).

## F4 · Data contract — CONFIRMED correct (no bug found)

| Item | Check on real data | Result |
|---|---|---|
| Window offset | Upstream FAQ and the installed `_motor_imagery_stream/config.yaml:37-38` | start 1.0 s after the cue, 4.0 s |
| Which file is the sealed offset | `proteus_windows.npz` against `…_start0.5 / 2 / 3.npz` | default equals each shifted by exactly 60 / 120 / 240 samples (max abs diff 0.0); onsets differ by −0.5 / +1 / +2 s. Default = cue + 1.0. The start files are training crops; no production script loads them |
| Labels and alignment | all 112 local BIDS `events.tsv` against the export | 2,655 of 2,655 rows: window start = cue + 1.000 s; 33024/33025/33026 → 0/1/2 (885/886/884) |
| Channel order | package `CH_NAMES` (`submission.py:172-175`) against the export; `channel_map` `:178-188`, `:1721` | same 41 names and order; runtime reorders by name |
| µV | per-channel window std | median 15.0 µV (5–95 %: 8.5–44.1) |
| 120 Hz | spectrum under fs = 120 | notch at exactly 50.0 Hz (24.2 dB against 46.1 / 45.6 dB at 49 / 51 Hz) |
| Interface in a cell | task name: `…Graz` or `…BH` | each visit holds both (about 96 + 94 windows; 2 acquisition and 4 online runs); feedback starts 3.0 s after the cue in both, so the last 2 s of every window overlap feedback |
| Can the model tell interfaces apart | leave-one-person-out classifier on label-free spectral features | Graz against BrainHero 56–57 % balanced accuracy (chance 50); acquisition against online 50 % |
| Behaviour per interface | real fold, held-new, 4 orders | Graz 73.5, BrainHero 74.1; by run type 75.4–78.9 % plain accuracy. P04/P10 visits 2–3 are 4–10 pp better on BrainHero in both the enrolled and the held-new package: a property of these visits, not of the personal path |
| Both interfaces in training | fold source rows | all non-held rows (P04 returning 2,278 = 2,655 − 377), no interface filter; only the rest class uses Graz pre-cue rows, by design |
| Per-person wiring of the shipped package | arrays against data | identity gallery rebuilt from data equals `bio_gal` to 7e-15 in `route_names` order; own-space priors equal each person's session-mean covariance (10 of 10); personal heads and template priors match their own person's contrast gallery 10 of 10 (correlation 0.86 / 0.99 on the diagonal, 0.07 / 0.10 off it) |

## Not verified

- The eight one-session public people on a later day: no labelled data exists. P04 and P10 are two people.
- Sealed logs (hidden), the RTX PRO 4500 runtime of any returning arm, and the pairing of the leak with the gap (H2).
- I did not rerun NETBCI; its numbers are quoted from the B10 and Q cards.

## Handoff checks (for an executor; register the block below in docs/LEDGER.md before scoring anything)

Common setup, Leonardo (as `scripts/overnight21/pkgcmp_submit.sh:5-13`; sbatch only, no work on the login node):
```
W=/leonardo_work/AIFAC_P02_1636/ni26; REL=$W/cap250/releases/15a639718601e959015fd533fa9106476614aa8c
SRC=$W/cap250/runs/production-20261120-32a94ab7; RUN=$W/cap250/runs/redteam-21nov; mkdir -p $RUN/logs $RUN/src
export CAP250_RELEASE=$REL CAP250_RUN=$RUN CAP250_ASSETS=$W/cap250/assets CAP250_DATA=$W/data/proteus_windows.npz \
  CAP250_PYTHON=$W/.venv312/bin/python CAP250_SITE=leonardo CAP250_REV=15a63971 CAP250_SMOKE=0 \
  CAP250_CB_STAGEA=/leonardo_work/AIFAC_P02_1636/ni26_aux_20261120/runs/cb_stage_a \
  CAP250_CB_CHECKPOINT=$W/cap250/assets/runs/cb_stage_a/cbramod.safetensors
SB=(sbatch --parsable --account=AIFAC_P02_1636 --partition=boost_usr_prod --gres=gpu:1 --chdir=$RUN)
stream() { # name pkg person sessions orders out
  "${SB[@]}" --job-name=$1 --cpus-per-task=4 --mem=16G --time=45 --output=$RUN/logs/%x-%j.out --wrap="bash $REL/cleanslate/cap250/slurm.sh \
  cleanslate/cap250/stream_fold.py --pkg $2 --held $3 --sessions ${4:-2,3} --orders ${5:-71000,71001,71002,71003} --data $CAP250_DATA --out $6"; }
```
Copy `laneR4/build_keep.py`, `laneT/` and `scripts/redteam/` from the repo to `$RUN/src/` (rsync). Score locally:
`scp` the `pred_ret_*.npz` (20 kB each) and run
`python scripts/redteam/score_ret.py data/proteus_windows.npz <ref_dir> <arm_dir> …` (smoke-tested: it reproduces
stock P04 70.20 / P10 89.22 and hardcb − stock per order 0.00 / 0.00 / −0.55 / +0.26).
One stream job is 8 sessions, about 5–10 GPU minutes. H1–H5 together are about 25 short jobs; H1 and H5 come first.

**H1 · keep-adapting arms on the clean instrument (decision-relevant).**
Build in one job: for `arm` in R0z R1a R5 R2 R4 and `who` in P04 P10:
`$CAP250_PYTHON -B $RUN/src/laneR4/build_keep.py $arm $RUN/$arm/ret_$who/pkg $SRC/hardcb/ret_$who/pkg`
(anchors are asserted once each; if one fails on the fold hardcb source, build R1a, R5, R2 on `$SRC/stock/…` and use
`$SRC/stock` as reference). Stream: `stream rt-$arm-$who $RUN/$arm/ret_$who/pkg $who 2,3 71000,71001,71002,71003 $RUN/$arm/pred_ret_$who.npz`.
Smoke first: R0z, P04, order 71000 only; its labels must equal `$SRC/hardcb/pred_ret_P04.npz` (`labels_71000_2`,
`labels_71000_3`) exactly, else stop. Guard: every keep arm's log must show refits or personalisation after the
recognised window (`[v6] … refits` > 0 for R1a/R2/R4, `[cpers] … fused` > 0 for R5/R2/R4); otherwise it is a no-op.
Expected output: `$RUN/<arm>/pred_ret_{P04,P10}.npz`; table of gains over hardcb per person and order.
Decisive: an arm with P04 gain > 0 and P10 ≥ −0.3 supports "the in-lab loss was an artefact"; P04 ≤ −0.3 on all
four orders keeps the old verdict. A supported arm that fits the runtime (R1a about 3,372 s, R5 about 3,340 s
projected; R4b's sparse or gated forms once timed) and is positive on NETBCI and ds007221g is a candidate against hardcb.

**H2 · paired leak reproduction (attribution of the 11 pp).** For `who` in P04 P10, with the uploaded package
unzipped to `$RUN/shipped`:
`python scripts/redteam/patch_weights.py $SRC/stock/ret_$who/pkg $RUN/leak_all/ret_$who/pkg --donor $RUN/shipped --take ts2_log_glob,ts2_mu,ts2_sd,ts2_W,ts2_b,v6_cf_W,v6_cf_sw,v6_cf_thr,v6_cf_lp,v6_rest_G,v6_rest_scale,em_ev_g,net34_G,net34_scale,rest_s0.pt,rest_s1.pt,rest_s2.pt,eye_s0.pt,eye_s1.pt,eye_s2.pt`
then stream as above, reference `$SRC/stock`. Optional single parts: `leak_ts2` (the five `ts2_*`), `leak_rest`
(rest files and two arrays), `leak_cf` (four `v6_cf_*`). Decisive: P04 rising by ≥ +8 pp confirms the attribution;
below +5 pp the gap has another cause and neither instrument's absolute level should be quoted until it is found.

**H3 · in-sample sanity of the clean instrument.** `stream rt-insample-P04 $SRC/stock/ret_P04/pkg P04 1 71000 $RUN/insample_P04.npz`
(visit 1 is in the fold's training rows). Decisive: ≥ 95 % confirms visit 1 was trained on; far lower means the
returning fold did not train on it and the clean numbers are too low.

**H4 · value of the route itself.** Lane RP already runs "the same fold weights with both native recognition gates
suppressed" (`evidence/wave_20nov/laneRP_returning_person.md`). Read that control instead of running a new one.
If it is missing: `patch_weights.py $SRC/stock/ret_$who/pkg $RUN/noroute/ret_$who/pkg --set bio_tau=1e9 route_tau=1e9`;
the log must show `recognised_at -1` in all 16 sessions. Decisive: `noroute` ≥ stock on both people means the
personal route adds nothing over members trained on the person, and the lever is adaptation, not routing.

**H5 · data-only dose of the personal parts (no runtime cost).** Arms from `$SRC/hardcb/ret_$who/pkg` with
`patch_weights.py … --set`: `g6` (`route_slot_g=6`), `g12` (`route_slot_g=12`), `beta02` (`own_beta=0.2`), `beta04`
(`own_beta=0.4`), `kappa40` (`v6_tpri_kappa=40`). Reference `$SRC/hardcb`. Guard: labels must differ from the
reference on at least one routed window, and one unrecognised stream (`new_P01`, order 71000, same patch) must be
label-identical to `$SRC/hardcb/pred_new_P01.npz`. Decisive: P04 gain > 0 with P10 ≥ −0.3 reverses the Lane A dose
verdict; then confirm on NI-V B-ret before packaging.

**Registration block (copy to docs/LEDGER.md before scoring).**
RT-1 · returning levers on the clean in-lab instrument (red team, 20 Nov). Mechanism: the Lane A returning packages
carry label-leaked members, which forces a negative sign on any lever that shifts weight to adaptation or to the
personal head; on fold-clean packages the sign should follow the foreign-lab instruments. Instrument: real fold
packages of `production-20261120-32a94ab7`, P04 and P10, visit 1 enrolled, visits 2–3, orders 71000–71003, real
Solver, paired per row order against the stored reference. Predictions, pp over the reference (80 % intervals):
R5 P04 +0.5 [−0.8, +2.0], P10 +0.1 [−0.6, +0.8]; R1a P04 +0.4 [−1.0, +2.0], P10 0.0 [−0.7, +0.7]; R2 P04 +1.5
[−1.0, +4.5], P10 +0.2 [−1.0, +1.5]; R4 P04 +2.5 [−1.0, +6.0], P10 +0.3 [−1.0, +1.8]; g6 P04 +1.0 [−1.5, +3.5], P10
0.0 [−1.0, +1.0]; leak_all − stock P04 +14 [+8, +20], P10 +5 [+2, +8]; in-sample visit 1 ≥ 95 %; noroute − stock
P04 −4 [−10, +2], P10 −1 [−3, +1]. Refutation: for a lever, P04 ≤ −0.3 on all four orders refutes the sign reversal
for that lever; for the leak, P04 below +5 pp refutes the attribution. Two people and four orders: intervals are over
orders per person and descriptive; P04 and P10 are reported separately and neither decides alone. Sealed weighting
(returning about 43 %, unverified) is shown next to the result, not applied.

## How the checks here were run

Local, CPU, venv of the repo. Scripts in `scripts/redteam/` (`score_ret.py`, `patch_weights.py`, and the check
scripts `check_*.py`, which take a scratch directory holding the unzipped uploaded package, the two fold
`weights.npz`, the production `pred_*.npz` and the Lane A `out/` copies). Inputs read from the clusters:
Galvani `ni26_js321_20261117_returning_bundle2`, `ni26_final_stack_ret_20261120/pkg_stack`,
`ni26_lane_a_returning_20261119/out`; Leonardo `cap250/runs/production-20261120-32a94ab7` (packages, predictions,
stream logs). Source revision at review start `7552e9ed`.
