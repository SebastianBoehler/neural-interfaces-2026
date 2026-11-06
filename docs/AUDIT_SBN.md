# Audit recipe: cand_sbn (a115 + per-session latent normalisation, arm D)

`runs/sealed/cand_sbn` = `runs/sealed/cand_nxif_a115` (cand_nxif with `rb_alpha` 1.15) where the five
`fbcnet_ea_rot05` members (nets 10–14) are replaced by FBCNets with trained-in per-session batch norm
("fb_sbn_rot05", BEETL 2021 winner trick; 5 Nov tournament arm D):

- every batch norm uses per-session statistics in training (batches of 2 × 32 same-session windows) and, at
  test time, the causal statistics of the session's windows 0..t blended with the training running
  statistics as k pseudo-windows (`experiments/session_bn.py`);
- seeds 0–5, trained on all 10 public participants, each exported twice, at test prior k 2 and k 5, as a
  session-stateful TorchScript member (`submission.py` `SJit`, key `sjit_file`, files `sbn_k{2,5}_s{0..5}.pt`);
- the k 2 group carries the replaced nets' total weight (0.125 = one v2 unit), the k 5 group adds the same
  again; nets 10–14 stay in the package at weight 0. Every other key of a115 is unchanged (rb_alpha 1.15,
  teacher weight 0.625, session EM).

`submission.py` changes (only code change since the freeze, `25d51a6e…` → `6e526e3a…`): `import copy`, class
`SJit(Jit)` (own module per session copy, `reset_state` zeroes buffers `s1`/`s2`/`n`), dispatch on
`sjit_file`, and `Ensemble.reset_state` calling a member's `reset_state` when it has one. Packages without
`sjit_file` keys behave identically (checked below).

## One command

Prerequisite: the a115 package, rebuilt by docs/AUDIT_NXIF.md (`bash scripts/audit_nxif.sh OUT`) followed by
`python scripts/set_params.py OUT OUT_a115 rb_alpha=1.15`. Then, on the Tübingen ML Cloud (Slurm, account
`gehler`; RTX 2080 Ti like the dev run):

```bash
J=$(sbatch --parsable --export=ALL,NI26_SOURCE_REV=$(git rev-parse HEAD) -o runs/sealed/cand_sbn_export_%j.log \
    scripts/cloud/export_sbn.sbatch runs/sealed/cand_nxif_a115 runs/sealed/cand_sbn)
sbatch --dependency=afterok:$J scripts/cloud/replay.sbatch runs/sealed/cand_sbn
sbatch --dependency=afterok:$J --time=02:00:00 scripts/cloud/worker_replay.sbatch runs/sealed/cand_sbn
```

`export_sbn.sbatch` runs `scripts/check_sjit.py --cuda` (SJit unit check on random nets) and
`scripts/export_sbn.py SRC OUT` (defaults `--cfg fb_sbn_rot05 --seeds 0-5 --ks 2,5 --drop 10-14`). It trains
the six nets (cached in `runs/sbn/fb_sbn_rot05-s{seed}.pt`, reused when present), checks the train-time EA
against the package's EA stream, exports and checks the traced members, writes the package and
`runs/sealed/cand_sbn_check.json` (harness = the rest of a115's streamed member mix + the members' eager
batch-causal probabilities, mixed with the package weights, then rebalancing and session EM exactly as
`scripts/set_params.py`). Backward compatibility of the code: `python scripts/check_compat.py
runs/sealed/cand_nxif runs/sealed/cand_nxif_a115 --mix`.

Training recipe (dev harness `lnorm.py` md5 `c3ddd81e`, copied into `experiments/session_bn.py`): braindecode
FBCNet on causal-EA input (k 5, shrink 1e-3, global reference G = mean covariance of the training windows),
40 epochs, AdamW lr 1e-3, weight decay 1e-3, one-cycle, batch 64 = 2 chunks of 32 windows of one session,
rotation augmentation ε 0.05, every BatchNorm replaced by SessionBN (momentum 0.1). The dev probabilities
came from the 08:20 CET copy of `lnorm.py` (md5 `d2fa4e97`), whose `swap_bn` did not copy the old batch norm's
parameters; for freshly initialised nets (ones / zeros) the two versions are identical.

The upload ZIP is the worker replica's `runs/sealed/cand_sbn_worker_<job>/submission.zip`, copied to
`runs/upload/sealed_cand_sbn.zip`. Provenance caveat: the package was built from uncommitted files on top of
`1caf03c` (`NI26_SOURCE_REV=1caf03c+sbn-uncommitted`): `submissions/ensemble/submission.py`,
`experiments/session_bn.py`, `scripts/{export_sbn,check_sjit,check_compat}.py`, `scripts/cloud/export_sbn.sbatch`
and this file must be committed for the audit.

## Reference values

- Code: `submission.py` `6e526e3a…` (25,657 B). `scripts/check_compat.py` (Mac CPU, 5 Nov): cand_nxif and a115
  give byte-identical labels and byte-identical pre-rebalancing member mixes on all 2,655 public windows with the
  new code vs their own frozen code. `scripts/check_sjit.py` (random FBCNet / ShallowConvNet, CPU and RTX 2080 Ti
  CUDA): window-by-window SJit = batch-causal SessionBN (max |dP| ≤ 1.9e-6), reset = fresh session, deepcopy
  independent, 3 state buffers per batch norm, saved state zero.
- Training (job 3059164, RTX 2080 Ti, galvani-cn114): 28–31 s per net; checkpoints `runs/sbn/fb_sbn_rot05-s{0..5}.pt`
  `6c4a7c02…`, `65356087…`, `dcf345ed…`, `4e9b8432…`, `dea58383…`, `8621b470…`.
- EA: train-time G and scale equal the package's net-10 keys exactly (relative diff 0.0), k 5 = 5, shrink 1e-3;
  the harness causal EA equals the package `InputStream` to 3.0e-8. The SJit members share a115's EA stream.
- Trace: traced batch = eager (0.0); window by window on a deepcopy ≤ 1.1e-5; CUDA ≤ 1.3e-5 (probabilities).
- Package: 47 nets, member weights sum 1.75 (nets 10–14 at 0; 12 × 0.0208333); `weights.npz` `7d78aa4a…`,
  `teacher.pt` `98f7df45…` (unchanged), `sbn_k2_s0..5.pt` `d088ff22…`, `b95aa90a…`, `b022fcc8…`, `6a39ff54…`,
  `b3d8bc1e…`, `ab13f06e…`, `sbn_k5_s0..5.pt` `d42a1bc1…`, `d80d153f…`, `d8b54493…`, `98548d45…`, `08f2bf7a…`,
  `d0e8b0fe…` (all in `runs/sealed/cand_sbn_check.json`).
- Export check (all 2,655 public windows, CPU): package = harness on 2,655 / 2,655; 14 labels differ from a115;
  in-sample 0.97510 (a115) vs 0.97512 (in-sample, not evidence).
- Replays: official A100 3059165 141.2 s, worker-image replica (v3, torch 2.14.1+cu130) 3059166 99.1 s, both on
  galvani-cn209 at the same time; files unchanged; identical scores (in-sample cell mean 0.965891; a115 0.964631,
  102.1 s / 67.7 s). Upload ZIP `runs/upload/sealed_cand_sbn.zip` `cee75861…` (10,060,252 B), byte-identical to
  both replays' `submission.zip`.
- CPU vs CUDA (job 3059450, RTX 2080 Ti node, `scripts/device_agreement.py`, member mix per window): label
  agreement 1.0, max |dP| 5.7e-7 (a115: 1.0, 4.8e-7); 94.7 vs 58.3 ms per window on CUDA, 410 vs 346 ms on CPU.
- Runtime: 1.38–1.62 × a115 (A100 replay, replica, 2080 CUDA timing), so about 19–23 min on the sealed set
  (a115/cand_nxif 835 s); well inside 60 min on the GPU worker. A CPU-only fallback would already exceed it for a115.

**Tolerance.** GPU training is not bit-deterministic: a rebuild will not hash equal. Expect label agreement
≥ 99 % with the staged package on the public windows (the SJit group is 14 % of the mixture weight).

**Dev evidence (not an audit target; 5 Nov, person-equal scorer, seeds 0–2, Δ vs R19e):** arm D J +2.43
(new +2.07, new9 +1.57, seen +2.80) vs a115 +1.65; FBCNet alone all_loso / seen_session 66.0 / 64.1 (sbn, k 2)
vs 59.1 / 58.1 (plain twin).

# Audit recipe: cand_rec_ts (REC with the 6 Nov tangent-space core)

`runs/sealed/cand_rec_ts` = `runs/sealed/cand_rec` with its tangent member (v2) replaced by the new causal
tangent-space core at weight 1.5 v2 units (`member_weights[0]` 0.125 → 0.1875, sum 1.8125); every other key
and file of cand_rec is unchanged. cand_rec = cand_sbn + session-normalised raw FBCNet (k 20), SincShallow
(k 20) and FBLight (k 5), 6 seeds each (export 3059663:
`export_sbn.py runs/sealed/cand_nxif_a115 runs/sealed/cand_rec --extra fbcnet_raw_sbn:15-19:20
--extra sincshallow_ea_ptm7r_rot05_sbn:24-28:20 --extra fblight_ea_rot05_sbn:29-33:5`).

**Decision.** Not adopted by the pre-registered rule. On the dev scorer (person-equal, J = 0.5 new + 0.5 seen,
Δ vs R19e, `decide.py`) the arm has pooled J +3.17 vs REC +2.52 and single-seed splits +0.69 ± 0.30 (6/6). The
rule (pick variant and weight on one half of the people, judge on the other) failed: on G1, the sibling
`car_ov17_sub2` at weight 1.5 beat this arm by 0.0002 (JG1 2.81398 vs 2.81375), and that pick lost 0.08 to REC on
G2. **The user adopted fs2 at weight 1.5 after this tie ("continue iterating").**

**Core** `car_ov17_fs2_t0.1_le_k2`, head `lr10` (harness `experiments/ts_core.py`, md5 `870bb047`, a verbatim
copy of the 6 Nov scratch harness that reproduces v2 exactly; standalone person-equal all_loso 69.3 vs 58.8 for
v2, seen 66.9 vs 57.7):
- common-average reference of each window (minus the channel mean), before the covariances and the slow block;
- 17 overlapping bands (Hz) (1,3) (2,4) (3,6) (4,8) (6,10) (8,12) (10,14) (12,16) (14,18) (16,20) (18,24)
  (20,28) (24,32) (28,36) (32,40) (36,45) (40,58), 4th-order Butterworth, zero-phase within the window;
- per band the full-window covariance and the covariances of the two 240-sample halves (mean removed per
  segment), fixed trace shrinkage 0.1; 51 blocks of 41 × 41, band-major (full, first half, second half);
- per block log-Euclidean recentering: training sessions by their whole-session mean; at test, window t of a
  session by the mean of windows 0..t plus the training prior (the log-Euclidean mean of all training windows)
  weighted as k 2 pseudo-windows; tangent vectors (upper triangle, √2 off-diagonal);
- the 1 Hz slow waveform (every 60th sample) centred the same way (k 2 zero windows); 44,239 features;
- StandardScaler + multinomial LogisticRegression (C 10, lbfgs, max_iter 2000), trained on all 2,655 public
  windows of the 10 people.

**submission.py** (`6e526e3a…` → `c791c2b7…`): new AlignedTangent keys `ts_car` (common-average reference)
and `ts_nsub` (full window + nsub sub-window covariances per band), default off. The nsub path is vectorised:
covariances by batched matmul, batched `torch.linalg.eigh` (CPU, float64), `_filtfilt` = scipy `sosfiltfilt`
with its zi and padding computed once (bitwise equal to `sosfiltfilt`), and a float64 slow-block sum as in
ts_core. Packages without these keys take the old code path.

## One command (cand_rec_ts)

Prerequisite: `runs/sealed/cand_rec` and its `cand_rec_harness.npz` (export above). On the Tübingen ML Cloud
(CPU for the export; A100 for the replays):

```bash
J=$(sbatch --parsable --export=ALL,NI26_SOURCE_REV=$(git rev-parse HEAD) -o runs/sealed/cand_rec_ts_export_%j.log \
    scripts/cloud/export_ts.sbatch runs/sealed/cand_rec runs/sealed/cand_rec_ts \
    --old-labels runs/sealed/cand_rec_harness.npz)
sbatch --dependency=afterok:$J scripts/cloud/replay.sbatch runs/sealed/cand_rec_ts
sbatch --dependency=afterok:$J --time=02:00:00 scripts/cloud/worker_replay.sbatch runs/sealed/cand_rec_ts
```

`scripts/export_ts.py SRC OUT` (defaults `--cfg car_ov17_fs2_t0.1_le_k2 --C 10 --weight 1.5`) trains the core
with ts_core's `Feats`/`Core` on all public windows, writes every `ts_*` key (`ts_car` 1, `ts_nsub` 2, `ts_k` 2,
`ts_shrink` 0.1, 17 `ts_bands`, 51 `ts_log_glob` blocks, scaler and LR as `ts_mu/ts_sd/ts_W/ts_b`,
`ts_slow_cutoff` 1, `ts_slow_step` 60), scales `member_weights[0]`, copies the other files and the current
submission.py, then checks: (1) the package's tangent member, streamed window by window per session, equals
ts_core's in-sample causal stream (`Core.predict_proba_stream`); (2) the package's labels equal the harness
(cand_rec's mix with the tangent member at weight 0 + core probabilities, mixed with the package weights, then
rebalancing and session EM as `scripts/set_params.py`). Backward compatibility: `python scripts/check_compat.py
PKG --mix` for cand_nxif, cand_nxif_a115, cand_sbn and cand_rec.

The upload ZIP is the worker replica's `runs/sealed/cand_rec_ts_worker_<job>/submission.zip`, copied to
`runs/upload/sealed_cand_rec_ts.zip`. Provenance caveat: built from uncommitted files on top of `1caf03c`
(`NI26_SOURCE_REV=1caf03c+rts-uncommitted`): besides the cand_sbn files above, `experiments/ts_core.py`,
`scripts/export_ts.py` and `scripts/cloud/export_ts.sbatch` must be committed for the audit. The dev decision
scripts (`decide.py`, `seeds.py`, person-equal scorer `pe_metric.py`) are session scratch, not in the repository.

## Reference values (cand_rec_ts)

- Code: `submission.py` `c791c2b7…` (28,415 B; md5 `5f4ae351`). `scripts/check_compat.py --mix` (Galvani CPU, jobs
  3060822–3060825): cand_nxif, cand_nxif_a115 (own code `25d51a6e…`), cand_sbn and cand_rec
  (`6e526e3a…`) give byte-identical labels and byte-identical pre-rebalancing member mixes on all 2,655 public
  windows with the new code. A subset test (4 sessions, 755 windows, core fitted on them) gave max |dP| 4.4e-16
  between the package member and ts_core; `_filtfilt` equals `sosfiltfilt` exactly (max |diff| 0.0).
- Training (job 3060819, galvani-cn115, CPU): core fit plus in-sample stream 820 s; 44,239 features.
- Member check: the package's tangent member equals ts_core's in-sample causal stream on all 2,655 windows
  (max |dP| 3.0e-14, labels 1.0); 32 ms per window on the CPU (single thread, cn115). Profiling (cn112): v2 14 ms,
  new core 39 ms per window (filter 10.5, three batched eigh 3 × 8, covariances 3).
- Package: only `member_weights[0]` (0.1875; sum 1.8125) and the `ts_*` keys differ from cand_rec (new: `ts_car`,
  `ts_nsub`; `ts_slow_*` unchanged); `weights.npz` `5bd1f53c…`; every other file is byte-identical to cand_rec
  (`teacher.pt` `98f7df45…`, the 30 `sbn_*.pt`).
- Export check (all 2,655 public windows, CPU): package = harness on 2,655 / 2,655; 13 labels differ from
  cand_rec; in-sample 0.97437 (cand_rec) vs 0.97848, core alone 0.99886 (in-sample, not evidence); 0.45 s per
  window for the whole package on the CPU.
- Replays: official A100 3060820 196.1 s and worker-image replica 3060821 160.5 s, both on galvani-cn202 at the
  same time (load average 15–19); files unchanged; identical scores (in-sample cell mean 0.969755; cand_rec
  0.963203, 171.4 s / 146.0 s). Upload ZIP `runs/upload/sealed_cand_rec_ts.zip` `1a4b48af…` (13,176,118 B),
  byte-identical to both replays' `submission.zip`.
- Runtime: 1.10 × cand_rec (replica) and 1.14 × (A100 replay). Scaled by cand_nxif (835 s sealed; a115 replay
  102.1 s, replica 67.7 s): about 27–33 min on the sealed set (cand_rec 23–30 min), well inside 60 min.

**Tolerance.** The core is trained on the CPU (lbfgs); a rebuild should give the same labels. The SJit members
are copied from cand_rec, not retrained.

# Audit recipe: cand_rec_ts2 (cand_rec_ts + a Hilbert tangent-space core)

`runs/sealed/cand_rec_ts2` = `runs/sealed/cand_rec_ts` plus a second tangent member, the analytic-signal
("Hilbert") core, at 0.125 (one v2 unit); the first core (`car_ov17_fs2_t0.1_le_k2`) goes back from 1.5 to
1 v2 unit (`member_weights[0]` 0.1875 → 0.125), and the new member's weight is appended as the last
`member_weights` entry (sum 1.8125 → 1.875). Every other key and file of cand_rec_ts is unchanged.

**Dev evidence (6 Nov, decide.py scorer, Δ vs REC; not an audit target).** "arm4" = the v2 slot replaced by the
mean probability of the two cores at slot weight 2.0: vs the packaged (pooled-seed) REC +1.30 J (an independent
audit found no leakage or causality bug); single-seed REC splits s0–s5 paired dJ +1.57 (sd 0.26), 6/6 seeds, both
people halves positive (G1 +1.48, G2 +1.68); the first core alone at 1.5 (= cand_rec_ts) +0.69. Dev sources, named
explicitly (ana.member's res/-then-r1/ lookup resolves each to its only copy): first core = scratch
`ts_core2/r1/car_ov17_fs2_t0.1_le_k2.npz` head `lr10` (sha256 `6fca00f4…`, byte-identical to `ts_core/res/`, made by
ts_core.py md5 `870bb047`); Hilbert core = `ts_core2/res/car_fb7_hfs2_t0.1_le_k2.npz` head `lr3` (`2436301b…`, =
Galvani `runs/diag/ts_core2/res/`, made by ts_core2.py md5 `67708c24`). Package sources: the first core is
cand_rec_ts's `ts_*` keys unchanged (`weights.npz` `5bd1f53c…`, ts_core `870bb047`, lr10, export 3060819); the
Hilbert core is trained by `export_ts2.py` with ts_core2 `67708c24` (lr3). The
mixer is a fixed-weight linear average, so arm4 = each core as its own member at weight 1: checked on the dev
probabilities (pooled REC and s0–s5, both protocols): final posteriors max |dP| ≤ 4.4e-16, 0 labels differ, dJ
identical (s0–s5 +1.74 +1.41 +1.18 +1.78 +1.47 +1.83; pooled REC +1.30 vs +0.65 for cand_rec_ts).

**Core** `car_fb7_hfs2_t0.1_le_k2`, head `lr3` (harness `experiments/ts_core2.py`, md5 `67708c24`, a verbatim copy
of the Galvani scratch harness that produced the dev probabilities; its real-valued path is ts_core's; standalone
person-equal all_loso 67.0, seen 66.6; first core 69.3 / 66.9):
- common-average reference of each window; 7 bands (1,4) (4,8) (8,13) (13,20) (20,30) (30,45) (45,58) Hz, 4th-order
  Butterworth, zero-phase within the window;
- per band the analytic signal (`scipy.signal.hilbert` over the band-passed window, complex64), then the complex
  Hermitian covariances of the full window and of the two 240-sample halves (mean removed per segment), trace
  shrinkage 0.1; 21 blocks of 41 × 41, band-major;
- log-Euclidean recentering per block as the first core (training sessions by their whole-session mean; at test the
  causal mean of windows 0..t plus the prior, the log-Euclidean mean of all training windows, as k 2 pseudo-windows);
  tangent vector per block = Re(upper triangle, √2 off-diagonal) followed by √2 · Im(strict upper triangle);
- the 1 Hz slow waveform block exactly as the first core; 35,629 features; StandardScaler + multinomial
  LogisticRegression (C 3, lbfgs, max_iter 2000) on all 2,655 public windows of the 10 people.

**submission.py** (`c791c2b7…` → `806b9511…`): AlignedTangent key `hilbert` (nsub path only: analytic signal,
complex covariances, complex `torch.linalg.eigh` in `_eig_b`, Re/Im tangent vector), and an optional second
tangent member read from `ts2_*` keys, weighted by the last `member_weights` entry (`Ensemble(tangent2=…)`, reset
with the session). Packages without these keys take the old code path; note that rebuilding cand_rec_ts with the
committed code now ships `806b9511…`, which check_compat shows to be byte-identical in labels and member mix.

## One command (cand_rec_ts2)

Prerequisite: `runs/sealed/cand_rec_ts` and its `cand_rec_ts_harness.npz` (cand_rec_ts recipe). On the Tübingen ML
Cloud (CPU for the export; A100 for the replays):

```bash
J=$(sbatch --parsable --export=ALL,NI26_SOURCE_REV=$(git rev-parse HEAD) -o runs/sealed/cand_rec_ts2_export_%j.log \
    scripts/cloud/export_ts2.sbatch runs/sealed/cand_rec_ts runs/sealed/cand_rec_ts2 \
    --src-harness runs/sealed/cand_rec_ts_harness.npz)
X=galvani-cn109,galvani-cn111,galvani-cn121,galvani-cn122
sbatch --dependency=afterok:$J --exclude=$X scripts/cloud/replay.sbatch runs/sealed/cand_rec_ts2
sbatch --dependency=afterok:$J --exclude=$X --time=02:00:00 scripts/cloud/worker_replay.sbatch runs/sealed/cand_rec_ts2
```

`scripts/export_ts2.py SRC OUT --src-harness H` (defaults `--cfg car_fb7_hfs2_t0.1_le_k2 --C 3 --w1 0.125
--w2 0.125`) trains the core with ts_core2's `Feats`/`Core`, writes the `ts2_*` keys (`ts2_hilbert` 1, `ts2_car` 1,
`ts2_nsub` 2, `ts2_k` 2, `ts2_shrink` 0.1, 7 `ts2_bands`, 21 complex `ts2_log_glob` blocks, `ts2_mu/sd/W/b`,
`ts2_slow_cutoff` 1, `ts2_slow_step` 60) and the weights, copies the other files and the current submission.py, then
checks on all public windows, streamed per session: (1) the package's second member = ts_core2's in-sample causal
stream (features and probabilities); (2) its first member = cand_rec_ts's (`P_pkg` of H); (3) its labels = the
harness (H's nets' mix + both cores with the package weights, then rebalancing and session EM as
`scripts/set_params.py`). Backward compatibility: `python scripts/check_compat.py PKG --mix` for cand_nxif,
cand_sbn, cand_rec and cand_rec_ts.

The upload ZIP is the worker replica's `runs/sealed/cand_rec_ts2_worker_<job>/submission.zip`, copied to
`runs/upload/sealed_cand_rec_ts2.zip`. Provenance caveat: built from uncommitted files on top of `1caf03c`
(`NI26_SOURCE_REV=1caf03c+rts2-uncommitted`; the export ran with `--code` = a scratch copy byte-identical to the
repository's `submissions/ensemble/submission.py` `806b9511…`): besides the cand_rec_ts files above,
`experiments/ts_core2.py`, `scripts/export_ts2.py` and `scripts/cloud/export_ts2.sbatch` must be committed. The
feature-parity and equivalence checks (`parity.py`, `equiv.py`) and the dev scorer are session scratch.

## Reference values (cand_rec_ts2)

- Code: `submission.py` `806b9511…` (29,756 B; md5 `1c4a965e`). `scripts/check_compat.py --mix` (Galvani CPU, jobs
  3061034–3061037, run with this code in place of the repository file, `runs/sealed/cand_rec_ts2_compat.json`): cand_nxif
  (own code `25d51a6e…`), cand_sbn, cand_rec (`6e526e3a…`) and cand_rec_ts (`c791c2b7…`) give byte-identical labels
  and byte-identical pre-rebalancing member mixes on all 2,655 public windows, so this one file rebuilds cand_rec_ts
  and cand_rec_ts2.
- Feature parity (Mac CPU, sessions P01|1 and P02|1, window by window vs the harness's causal features, prior = the
  two sessions' log-Euclidean mean): Hilbert core max |dF| 2.0e-7 (max |F| 278), max |dF| / feature sd 1.8e-6, slow
  block exact (the analytic signal is complex64 on both sides; float32 rounding); first core 2.0e-13 / 8.3e-13, so
  the packaged first core already matched ts_core. Smoke test (job 3061032, the export on 2 sessions): package =
  harness 379 / 379.
- Training (job 3061033, galvani-cn123, CPU): core fit 141.6 s; 35,629 features.
- Member checks (all 2,655 windows, tolerance 1e-5 on features: the harness band-passes in float32 and its Hilbert
  FFT is complex64, so batch and per-window runs differ by up to about 3e-6): second member = ts_core2's in-sample
  causal stream, max |dF| 1.2e-7 (1.6e-6 in feature sd), max |dP| 1.2e-8, label flips 0 / 2,655; 26 ms per window
  on the CPU; first member = cand_rec_ts's stream (max |dP| 0.0).
- Package: 33 files; only `member_weights` (67 entries, [0] 0.125, last 0.125, sum 1.875) and the 13 new `ts2_*`
  keys differ from cand_rec_ts (`ts2_log_glob` complex128 (21, 41, 41), `ts2_W` (3, 35629)); `weights.npz`
  `b60c4be7…`; every other file byte-identical to cand_rec_ts (`teacher.pt` `98f7df45…`, the 30 `sbn_*.pt`).
- Export check (CPU): package = harness on 2,655 / 2,655; 18 labels differ from cand_rec_ts; in-sample 0.97848
  (cand_rec_ts) vs 0.98455, core alone 0.99923 (in-sample, not evidence); 0.55 s per window for the whole package
  on the CPU (cand_rec_ts 0.45).
- Replays: official A100 3061051 211.3 s (galvani-cn212) and worker-image replica (v3, torch 2.14.1+cu130) 3061052
  173.8 s (galvani-cn208), started 30 s apart; files unchanged; identical scores (in-sample cell mean 0.982860;
  cand_rec_ts 0.969755, 196.1 s / 160.5 s). The CPU export labels give the same scores on the 755 replay windows
  (0.982860 / 0.982781 / 0.982777), as cand_rec_ts's do (0.969755). Upload ZIP `runs/upload/sealed_cand_rec_ts2.zip`
  `e9f13567…` (14,933,775 B), byte-identical to both replays' `submission.zip`.
- Runtime: 1.08 × cand_rec_ts (both replays). Scaled by cand_nxif (835 s sealed; a115 replay 102.1 s, replica
  67.7 s): about 29–36 min on the sealed set (cand_rec_ts 27–33 min), inside 60 min.

**Tolerance.** The core is trained on the CPU (lbfgs); a rebuild should give the same labels. Everything else is
copied from cand_rec_ts.

# Audit recipe: cand_rec_ts2_tsem (cand_rec_ts2 + TSEM after the session EM)

`runs/sealed/cand_rec_ts2_tsem` = `runs/sealed/cand_rec_ts2` plus a final, causal per-session stage (TSEM) after the
gaze/event session EM. Every key and file of cand_rec_ts2 is unchanged (member weights included); `weights.npz`
gains `tsem_*` keys.

**Stage.** For window t of a session, u_t = (z_t − m) @ V, where z_t is the first tangent member's scaled causal
feature vector (`(features − ts_mu) / ts_sd`, exactly what its LR sees; 44,239 dims), m the mean and V (44,239 × 128,
shipped as float32) the top 128 right singular vectors of the centred scaled whole-session training features of all
2,655 public windows (exact `scipy.linalg.svd(..., lapack_driver="gesdd")`, `svd_flip` on V). Over the session's
earlier windows the u's are standardised; per class the k = max(2, ⌊0.7 t / 3⌋) windows with the highest input
posterior among those whose input argmax is that class seed a class mean (Σ / (n + m0 20)); the pooled within-class
covariance is shrunk to its trace (0.7); window t's input posterior is multiplied by the resulting Gaussian (LDA)
posterior raised to λ t / (t + n0), λ 0.15, n0 60, from t ≥ 15 on. Seeds and multiplicand are the stage's input
posteriors (the session-EM output), not its own outputs. Reset per session; a `LinAlgError` or non-finite output
falls back to the input; λ 0 returns the input bit-identically.

**Dev evidence (6 Nov, pre-registered confirmation; not an audit target).** On base B2 = REC with the two cores
(cand_rec_ts2's dev analogue) and fresh deep-member seeds 3–5, the fixed config gave dJ +1.71 (new +2.38, seen +1.04,
new ≥ 0 for 8/10 people; pass bar dJ ≥ 1.0, new and seen ≥ 0.5, ≥ 7/10). Pooled seeds +1.46; B1 / B3 bases
+1.58 / +0.89 (s3–5). All add-ons (causal session PCA, other ranks, hedges) were rejected; this is the fixed config.
In the dev folds the basis was fitted on the fold's training people; the package fits it on all 10 (as the core).
Dev sources (session scratch `tsem_confirm/`): `bases.npz` sha256 `33158616…` (bases.py md5 `6848843c`), views
`feats_exact/` (export_exact.py md5 `631d7b2e`), batch reference `tsem.py` md5 `7367a278` (copied verbatim to
`experiments/tsem.py`), confirm.py md5 `6054a2e6`.

**submission.py** (`806b9511…` → `243c9901…`): class `TSEM` (keys `tsem_lam/rho/shrink/m0/n0/nmin`, `tsem_m`,
`tsem_V`), `AlignedTangent.proba` keeps the last scaled feature vector (`self.z`), `Ensemble(tsem=…)` collects the
views in `mixed_proba` and applies the stage after the session EM in `predict`, reset with the session. Packages
without `tsem_*` keys behave identically (check below). The class stays inside `submission.py`: the official
contract does not support additional Python files in the submission.

## One command (cand_rec_ts2_tsem)

Prerequisite: `runs/sealed/cand_rec_ts2` and its `cand_rec_ts2_harness.npz` (cand_rec_ts2 recipe).

```bash
J=$(sbatch --parsable --export=ALL,NI26_SOURCE_REV=$(git rev-parse HEAD) -o runs/sealed/cand_rec_ts2_tsem_export_%j.log \
    scripts/cloud/export_tsem.sbatch runs/sealed/cand_rec_ts2 runs/sealed/cand_rec_ts2_tsem \
    --src-harness runs/sealed/cand_rec_ts2_harness.npz)
X=galvani-cn109,galvani-cn111,galvani-cn121,galvani-cn122
sbatch --dependency=afterok:$J --exclude=$X scripts/cloud/replay.sbatch runs/sealed/cand_rec_ts2_tsem
sbatch --dependency=afterok:$J --exclude=$X --time=02:00:00 scripts/cloud/worker_replay.sbatch runs/sealed/cand_rec_ts2_tsem
```

`scripts/export_tsem.py SRC OUT --src-harness H` (defaults `--npc 128 --lam 0.15 --rho 0.7 --shrink 0.7 --m0 20
--n0 60 --nmin 15`) recomputes the first core's training features with `ts_core.Feats` and checks that a fresh
StandardScaler and the log-Euclidean priors equal the packaged `ts_mu`, `ts_sd` and `ts_log_glob` bit for bit (so
the basis is built on exactly the features and order the core was trained on), fits the basis, writes the `tsem_*`
keys, copies the other files and the current submission.py, then checks on all public windows, streamed per
session: the package's views vs ts_core's causal features projected on the same basis; its session-EM output vs the
harness (`set_params.harness` steps); its labels vs batch TSEM (`experiments/tsem.py`) on the harness posteriors and
views; then, on the first two sessions fed in batches of 64 as by the evaluator, a truncated stream (first T windows)
is bit-identical to the full stream before T, and a deep-copied, reset model equals a freshly loaded one on the
next session. Backward compatibility: `python scripts/check_compat.py PKG --mix` for cand_nxif, cand_sbn, cand_rec,
cand_rec_ts and cand_rec_ts2.

The upload ZIP is the worker replica's `runs/sealed/cand_rec_ts2_tsem_worker_<job>/submission.zip`, copied to
`runs/upload/sealed_cand_rec_ts2_tsem.zip`. Provenance caveat: built from uncommitted files on top of `1caf03c`
(`NI26_SOURCE_REV=1caf03c+tsem-uncommitted`; `--code` = a scratch copy byte-identical to the repository's
`submissions/ensemble/submission.py` `243c9901…`): besides the files of the recipes above, `experiments/tsem.py`,
`scripts/export_tsem.py` and `scripts/cloud/export_tsem.sbatch` must be committed. The stage-parity script
(`parity_tsem.py`) and the dev data are session scratch.

## Reference values (cand_rec_ts2_tsem)

- Code: `submission.py` `243c9901…` (709 lines). `scripts/check_compat.py --mix` (Galvani CPU, jobs 3061359–3061363,
  this code in place of the repository file, `runs/sealed/cand_rec_ts2_tsem_compat.json`): cand_nxif
  (own code `25d51a6e…`), cand_sbn, cand_rec (`6e526e3a…`), cand_rec_ts (`c791c2b7…`) and cand_rec_ts2 (`806b9511…`)
  give byte-identical labels and byte-identical pre-rebalancing member mixes on all 2,655 public windows, so this one
  file rebuilds all five and cand_rec_ts2_tsem.
- Stage parity on the dev streams (`parity_tsem.py`, Mac CPU): the packaged `TSEM` class, window by window and reset
  per session, vs batch `experiments/tsem.py` on all nine confirmation bases (B1/B2/B3, seed sets) and both protocols
  (final posteriors of `bases.npz`, exact-SVD PC128 views of `feats_exact/`): max |diff| 0 (bit-identical), 0 label
  flips (TSEM itself changes 153–176 of 2,655 all_loso labels); λ 0 bit-identical to the input; truncated streams
  bit-identical before T; one object reset between sessions and a deep copy taken mid-session then reset both
  bit-identical to fresh objects.
- Export (job 3061380, galvani-cn112, CPU): features 154 s; a fresh StandardScaler and the priors equal the packaged
  `ts_mu`, `ts_sd`, `ts_log_glob` bit for bit; exact SVD 31 s (rank 2,655, S1 1889.7, S128 314.3; the 128 PCs carry
  22.5 % of the variance).
- Checks (all 2,655 windows): package views vs ts_core's causal features on the packaged float32 basis max |du|
  1.4e-12 (2.5e-7 vs the float64 basis); session-EM output vs the harness 8.3e-10; final posteriors vs batch TSEM
  8.3e-10 (float32 basis) and 8.8e-10 (float64 basis); package = harness 2,655 / 2,655, 0 label flips against either
  basis. 31 labels differ from cand_rec_ts2 (all from the TSEM stage); in-sample cell mean 0.98455 → 0.98033
  (in-sample, not evidence: the cores are near-perfect on their own training windows). First two sessions in batches
  of 64: truncated stream (T 128) bit-identical before T; deep-copied, reset model = freshly loaded model on session 2
  (bit-identical); batches of 64 vs one window at a time differ by 7.8e-7 (net batch arithmetic). 0.51 s per window
  for the whole package on the CPU; the stage itself costs about 0.3 ms (view) + 0.5 ms (step at t 300) per window
  (Mac, one thread).
- Package: 33 files; only the 8 `tsem_*` keys are new (`tsem_V` float32 (44239, 128), `tsem_m` float64); every other
  key and file is byte-identical to cand_rec_ts2. `weights.npz` `9812d2f9…`.
- Replays: official A100 3061465 218.9 s (galvani-cn208) and worker-image replica 3061466 181.2 s (galvani-cn205),
  started 30 s apart; files unchanged; identical scores (in-sample cell mean 0.974922; cand_rec_ts2 0.982860); the
  CPU export labels give the same score on the 755 replay windows (12 of them changed by TSEM). Upload ZIP
  `runs/upload/sealed_cand_rec_ts2_tsem.zip` `9c31a167…` (36,119,899 B), byte-identical to both replays'
  `submission.zip`.
- Runtime: 1.04 × cand_rec_ts2 (both replays). Scaled by cand_nxif (835 s sealed; a115 102.1 s / 67.7 s): about
  30–37 min on the sealed set (cand_rec_ts2 29–36 min).

**Tolerance.** The features and the SVD are deterministic on one machine; across CPU types the basis may differ in
the last bits (LAPACK). Expect the same or near-identical labels. Everything else is copied from cand_rec_ts2.
