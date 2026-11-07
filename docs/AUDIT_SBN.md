> **Code version for these recipes (7 Nov cleanup, docs/STATE_7NOV.md).** `submissions/ensemble/submission.py` is now
> the lean file (`dacee85a…`, 781 lines; previously `786bcab0…`, 967 lines, commit `c4d600d`). It keeps only the code
> paths of the three uploaded packages, cand_nxif (docs/AUDIT_NXIF.md), cand_rec_ts2_tsem and cand_rec_ts2_tsem_route3geb.
> It reproduces all three byte for byte (route3geb reference values, end of this file). The export steps copy the
> current `submission.py` into their package. The intermediate packages (a115, cand_sbn, cand_rec, cand_rec_ts,
> cand_rec_ts2, and cand_rec_ts2_tsem256 as route3geb's basis) use only code paths that the uploaded packages
> exercise, and each export checks its package against its harness. Byte identity with their staged ZIPs was not
> re-checked. **The recipes for packages that were staged but never uploaded use commit `c4d600d`:** cand_rec_ts2_tsem256
> as an upload, routing v1 (`cand_rec_ts2_tsem_route`), v2 (`route2g` / `route2f`), route3 and route3eb. For them,
> check out `submissions/ensemble/submission.py`, `experiments/route_stream*.py` and `scripts/export_route.py` from
> `c4d600d`. The lean code refuses routing v1 / v2 keys and an early boost without gate keys.

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

# Audit recipe: cand_rec_ts2_tsem256 (hedge: TSEM with PC256, λ 0.2)

An **alternate** upload, not a replacement for cand_rec_ts2_tsem (sealed 0.8197, submission 1038262). The board keeps
our best score. It is cand_rec_ts2_tsem with the TSEM view widened to the top **256** exact-SVD PCs and **λ 0.2**.
ρ 0.7, shrink 0.7, m0 20, n0 60 and nmin 15 are unchanged. The change is data-only: the same `submission.py`
`243c9901…`, since `TSEM` takes the view width from `tsem_V`'s shape. No code change, so the compat check does not
apply.

**Dev evidence (6 Nov, `tsem_confirm/confirm.log`; picked after the confirmation, so expect some selection bias).**
Paired against the fixed config (PC128, λ 0.15) on base B2 over member seeds 3, 4 and 5: ΔJ +0.42, +0.05 and +0.42.
The mean is +0.30 (sd 0.21, 3/3 seeds). Seen is +0.53 and new is +0.07. P02, P03 and P06 are worse in every seed.
B2_s345 + hedge: dJ +2.01 over B2 (fixed +1.71). Other bases (hedge − fixed): B1_s345 +0.23, B3_s345 +0.52,
B1_pool −0.07.

## One command (cand_rec_ts2_tsem256)

Same as cand_rec_ts2_tsem, with `--npc 256 --lam 0.2` and its own output directory:

```bash
J=$(sbatch --parsable --export=ALL,NI26_SOURCE_REV=$(git rev-parse HEAD) -o runs/sealed/cand_rec_ts2_tsem256_export_%j.log \
    scripts/cloud/export_tsem.sbatch runs/sealed/cand_rec_ts2 runs/sealed/cand_rec_ts2_tsem256 \
    --src-harness runs/sealed/cand_rec_ts2_harness.npz --npc 256 --lam 0.2)
X=galvani-cn109,galvani-cn111,galvani-cn121,galvani-cn122
sbatch --dependency=afterok:$J --exclude=$X scripts/cloud/replay.sbatch runs/sealed/cand_rec_ts2_tsem256
sbatch --dependency=afterok:$J --exclude=$X --time=02:00:00 scripts/cloud/worker_replay.sbatch runs/sealed/cand_rec_ts2_tsem256
```

The upload ZIP is the replica's `runs/sealed/cand_rec_ts2_tsem256_worker_<job>/submission.zip`, copied to
`runs/upload/sealed_cand_rec_ts2_tsem256.zip`. It was built from commit `26b0f54` (`NI26_SOURCE_REV=26b0f54`). The
Galvani copies of the tracked code match it by SHA-256 (95 files). Only `scripts/diag/agent_opt.py`, which the build
does not use, differs. The stage parity uses session scratch: `hedge_build/parity_tsem256.py` (the
cand_rec_ts2_tsem parity script set to PC256 and λ 0.2). The ABBA timing uses
`runs/diag/paired_timing_tsem256.sbatch`.

## Reference values (cand_rec_ts2_tsem256)

- Stage parity (Mac CPU, `run_capped`): the packaged `TSEM` class, window by window and reset per session, against
  batch `experiments/tsem.py` on all nine confirmation bases and both protocols (exact-SVD PC256 views, λ 0.2). Max
  |diff| is 0 (bit-identical), with 0 label flips. TSEM changes 237–248 of 2,655 all_loso labels (PC128: 153–176).
  λ 0 is bit-identical to the input. Truncated streams are bit-identical before T. Reset and deep copy + reset are
  both bit-identical. On B2_s345 the packaged stage reproduces dJ +2.01 exactly.
- Export (job 3064464, galvani-cn001, CPU, 46 min). A fresh StandardScaler and the priors equal the packaged
  `ts_mu`, `ts_sd` and `ts_log_glob` bit for bit. SVD: 29 s, rank 2,655, S256 281.6. The 256 PCs carry 32.0 % of the
  variance (128: 22.5 %). `tsem_V`'s first 128 columns are byte-identical to cand_rec_ts2_tsem's, and `tsem_m` is
  identical.
- Checks on all 2,655 windows. Views against the reference: 1.4e-12 (float32 basis), 2.5e-7 (float64). Session-EM
  output against the harness: 8.3e-10. Final posteriors against batch TSEM: 8.3e-10 (f32) and 1.6e-9 (f64). The
  package matches the harness on 2,655 / 2,655 windows, with 0 label flips on either basis. 41 labels differ from
  cand_rec_ts2, all from TSEM (PC128: 31), and 23 from cand_rec_ts2_tsem. In-sample cell mean: 0.98455 →
  0.97693 (in-sample only, not evidence). The first two sessions were fed in batches of 64. The truncated stream
  (T 128) is bit-identical before T. A deep-copied, reset model equals a freshly loaded one.
- Package: 33 files. Only `tsem_V` (float32, 44,239 × 256) and `tsem_lam` (0.2) differ from cand_rec_ts2_tsem. Every
  other key and file is byte-identical, including `submission.py` `243c9901…`. `weights.npz` is `90fda086…`.
- Replays, with files unchanged in both:
  - official A100: job 3064458, 225.7 s (galvani-cn209);
  - worker-image replica: job 3064459, 188.9 s (galvani-cn229).
  The two scores are identical (in-sample cell mean 0.968190). The CPU export labels give the same score on the 755
  replay windows (18 of them changed from cand_rec_ts2). Upload ZIP `runs/upload/sealed_cand_rec_ts2_tsem256.zip`
  `3799c0e0d1d76e0321ff30cb280e20bcdfad89dc9d944b8694658b41dc50f528` (57,134,934 B). It is byte-identical to both
  replays' `submission.zip`.
- Runtime. Paired ABBA in the replica on one node (job 3064629, cn211):
  - base: 185.3 s and 188.7 s;
  - hedge: 182.8 s and 191.5 s.
  The ratio is 1.00. An earlier pair (job 3064463, cn209) gave 184.7 s and 183.5 s. The stage itself costs +1.1 ms per
  window (Mac, one thread: view 0.52 ms against 0.28; step 1.28 ms against 0.43, mean over a 500-window session).
  Per-window scaling of cand_rec_ts2_tsem's real 2,497 s gives about 42 min. The unpaired cross-node ratio (≤ 1.04),
  with up to 60 s fixed overhead removed, gives at most 44 min. Both are under the 52-min flag and the 60-min limit.

**Tolerance.** As for cand_rec_ts2_tsem: the basis may differ in the last bits across CPU types (LAPACK). Expect the
same or near-identical labels.

# Audit recipe: cand_rec_ts2_tsem_route (cand_rec_ts2_tsem + returning-person routing)

An **alternate** upload, not a replacement. `runs/sealed/cand_rec_ts2_tsem_route` is `runs/sealed/cand_rec_ts2_tsem`
(the validated input: TSEM PC128, λ 0.15, `9c31a167…`) plus a causal per-session routing stage after TSEM. Every key
and file of cand_rec_ts2_tsem is byte-identical except `submission.py`. `weights.npz` gains 12 keys (`route_*`,
`route_V_hi`).

**Stage.** The view is u_t = [TSEM view (128 PCs), (z_t − tsem_m) @ `route_V_hi`], 256 PCs in all. `route_V_hi` is
cand_rec_ts2_tsem256's `tsem_V[:, 128:]` (float32), so TSEM's own view and basis are untouched. From window t ≥ 60:
1. Probe: the class contrast of windows 0..t−1, minus the debias vector `route_b`. The u are standardised over those
   windows. Per argmax class of the TSEM outputs, the max(2, ⌊0.7 t / 3⌋) most confident windows give a class mean;
   each class mean has the mean of the three removed.
2. Score: score_p = cos(probe − `route_pop`, gallery_p − `route_pop`) for each of the 10 public people; p* is the
   argmax.
3. Blend: w = 0.25 · clip((score_p* − 0.20) / 0.10, 0, 1). If w > 0, the output is normalise(exp(log q_t + w log h_p*)).
   h_p* is person p*'s LR head on u_t, standardised over windows 0..t.

q_t is the TSEM128 output, not the TSEM256 hedge's: `route_b` was fitted on TSEM128 outputs. The stage resets per
session. An exception or non-finite output falls back to q_t, and w = 0 returns q_t. Data: `route_pkg.npz`
(SHA-256 `797ded20…`, route team, session scratch `route/pkg/build_pkg.py`):
- galleries: the per-person mean of true-label contrasts;
- heads: one LR per person (C 0.01);
- debias vector: from the B2_pool all_loso OOF B2+TSEM posteriors.

All are on the final PC256 basis (`pkg_ptr.npz` from `runs/diag/route/pkg_ptr.py` on Galvani). The streaming reference
is `experiments/route_stream.py` (md5 `6ca3878b`), a verbatim copy of `route/pkg/route_stream.py`.

**Dev evidence (pre-registered, 6 Nov; `route/prereg.txt`, `route/confirm.log`).** Against B2 + TSEM on 4 fresh bases
(B2_s012, s0, s1, s2):
- dJ +0.73, +0.86, +0.80, +0.79;
- new people 0.00: no new session was routed (0 impostor routings);
- seen +1.45 to +1.72 (P04 +2.9 to +3.5, P10 −0.3 to 0.0);
- a wrong-person-head null gives −0.14 to −0.34 (real − null +0.94 to +1.19);
- the truncation test passes.

Caveats:
- The gain comes only from returning people; on dev, almost all of it is P04.
- The pre-registered external gate failed (Stieger TAR@FAR1% 0.30, Lee 0.21).
- The closest dev impostor was P06 (new): it peaked at 0.193 against τ 0.20.

## One command (cand_rec_ts2_tsem_route)

Prerequisites:
- `runs/sealed/cand_rec_ts2_tsem` and `runs/sealed/cand_rec_ts2_tsem256` with their harnesses (recipes above);
- `route_pkg.npz` (`797ded20…`; kept as `runs/diag/rroute/route_pkg.npz` on Galvani and
  `runs/sealed/cand_rec_ts2_tsem_route_route_pkg.npz` on the Mac).

```bash
A="runs/sealed/cand_rec_ts2_tsem runs/sealed/cand_rec_ts2_tsem_route --route runs/diag/rroute/route_pkg.npz \
   --hi runs/sealed/cand_rec_ts2_tsem256 --src-harness runs/sealed/cand_rec_ts2_tsem_harness.npz \
   --hi-harness runs/sealed/cand_rec_ts2_tsem256_harness.npz"
J=$(sbatch --parsable --export=ALL,NI26_SOURCE_REV=$(git rev-parse HEAD) -o runs/sealed/cand_rec_ts2_tsem_route_export_%j.log \
    scripts/cloud/export_route.sbatch $A)
sbatch --export=ALL -o runs/sealed/cand_rec_ts2_tsem_route_wmax0_%j.log scripts/cloud/export_route.sbatch $A --wmax0-only
X=galvani-cn109,galvani-cn111,galvani-cn121,galvani-cn122
sbatch --dependency=afterok:$J --exclude=$X scripts/cloud/replay.sbatch runs/sealed/cand_rec_ts2_tsem_route
sbatch --dependency=afterok:$J --exclude=$X --time=02:00:00 scripts/cloud/worker_replay.sbatch runs/sealed/cand_rec_ts2_tsem_route
sbatch --dependency=afterok:$J --exclude=$X runs/diag/paired_timing_ab.sbatch runs/sealed/cand_rec_ts2_tsem runs/sealed/cand_rec_ts2_tsem_route
```

`scripts/export_route.py` does no fitting. It checks:
- SRC's `tsem_m` and `tsem_V` equal HI's `tsem_m` and the first 128 columns of HI's `tsem_V`.
- It writes the `route_*` keys and `route_V_hi`, copies the other files and the current `submission.py`.
- On all public windows, streamed per session:
  - the stage's input against SRC's harness;
  - its views against HI's harness;
  - its output against `route_stream.py` on the stage's own inputs, and on the harness's;
  - that every public session routes to its own person from window 60.
- On the first two sessions, fed in batches of 64: truncation and deep-copy-then-reset.

`--wmax0-only` checks a temporary copy with `route_wmax` 0 against SRC run with its own code.

The upload ZIP is the replica's `submission.zip`, copied to `runs/upload/sealed_cand_rec_ts2_tsem_route.zip`.

Provenance caveat. This was built from uncommitted files on top of `26b0f54` (`NI26_SOURCE_REV=26b0f54+route-uncommitted`).
`--code` was `runs/diag/rroute/code/submission.py`, which is byte-identical to the repository's
`submissions/ensemble/submission.py` `ab5fc0b9…`; the Galvani workspace copy is still `243c9901…`. These files must be
committed:
- `submissions/ensemble/submission.py`
- `experiments/route_stream.py`
- `scripts/export_route.py`
- `scripts/cloud/export_route.sbatch`

The route data builder and the dev data are session scratch.

## Reference values (cand_rec_ts2_tsem_route)

- Code: `submission.py` `243c9901…` → `ab5fc0b9…` (801 lines).
  - New class `Route` (plus `_softmax_rows`).
  - `Ensemble(route=…)` computes the extra view in `mixed_proba` and applies the stage after TSEM in `predict`; it
    resets with the session.
  - The stage is loaded when `route_tau` is in the weights. Loading asserts that TSEM is present and that
    128 + 128 = 256.
  - `scripts/check_compat.py --mix` (Galvani CPU, jobs 3064723–3064729, this code via `runs/diag/rts2/compat.py`,
    `runs/sealed/cand_rec_ts2_tsem_route_compat.json`) covers cand_nxif, cand_sbn, cand_rec, cand_rec_ts,
    cand_rec_ts2, cand_rec_ts2_tsem and cand_rec_ts2_tsem256, each with its own code. All seven give byte-identical
    labels and byte-identical pre-rebalancing member mixes on all 2,655 public windows. So this one file rebuilds all
    seven and cand_rec_ts2_tsem_route.
- Stage parity. The packaged stage is bit-identical to `route_stream.py` on its own inputs (max |diff| 0, 0 label
  flips). Against `route_stream.py` on the harness inputs (cand_rec_ts2_tsem `ref32`, cand_rec_ts2_tsem256 `U32`) it is
  within 8.3e-10, with 0 flips. Over all 2,655 windows:
  - stage input against SRC's harness: 8.3e-10, with labels equal to SRC's;
  - views: 1.4e-12.
- Routing (in-sample). All 14 public sessions route to their own person from window 60 and from no earlier window
  (lowest score 0.44, P04|3), so 1,815 windows are routed. The route stage changes 20 labels.
- In-sample cell mean, 0.98033 → 0.98523. This is not evidence: the heads were trained on these sessions.
- wmax 0 (job 3064719): labels and final posteriors are byte-identical to cand_rec_ts2_tsem run with its own code, and
  the stage returns its input bitwise.
- Export (job 3064718, galvani-cn123, CPU, 37 min).
  - Truncation (T 128) is bit-identical before T.
  - A deep-copied, reset model equals a freshly loaded one.
  - Batches of 64 against one window at a time differ by 6.8e-7.
- Package: 33 files. Only `submission.py` and `weights.npz` (`aa8b2ddb…`) differ from cand_rec_ts2_tsem. Every
  cand_rec_ts2_tsem key is byte-identical, and the `route_*` keys equal `route_pkg.npz`.
- Replays, with files unchanged in both:
  - official A100: job 3064720, 211.0 s (galvani-cn241);
  - worker-image replica: job 3064721, 198.0 s (galvani-cn229).

  The scores are identical (in-sample 0.982734; cand_rec_ts2_tsem 0.974922). The CPU export labels give the same
  score on the 755 replay windows (10 changed by the stage). Upload ZIP `runs/upload/sealed_cand_rec_ts2_tsem_route.zip`
  `34f60ecac0ff8a402063ba33f7803a65c08ce2cf7087083d14020ce23f7e3dd2` (57,266,420 B), byte-identical to both replays'
  `submission.zip`.
- Runtime. Paired ABBA in the replica on one node (job 3064722, cn218):
  - base: 184.0 s and 178.6 s;
  - route: 184.0 s and 181.9 s.

  The ratio is 1.01. The stage costs about 1.5 ms per window (Mac, one thread): the extra 128-column view 1.0 ms and the
  step 0.43 ms on average (at most 1.25 ms). Per-window scaling of cand_rec_ts2_tsem's real 2,497 s gives about 42 min.
  The unpaired cross-node replica ratio (1.09), with up to 60 s fixed overhead removed, gives at most 47 min. Both are
  under the 50-min flag.

**Tolerance.** The build fits nothing: it copies `route_pkg.npz` and the high half of cand_rec_ts2_tsem256's basis,
so the inputs determine it byte for byte. That basis may differ in the last bits across CPU types (see
cand_rec_ts2_tsem256).

# Audit recipe: cand_rec_ts2_tsem_route3 (cand_rec_ts2_tsem + routing v3)

An **alternate** upload. It supersedes cand_rec_ts2_tsem_route (v1) and routing v2. `runs/sealed/cand_rec_ts2_tsem_route3`
is `runs/sealed/cand_rec_ts2_tsem` (TSEM PC128, λ 0.15, `9c31a167…`) plus routing v3. Every key and file of
cand_rec_ts2_tsem is byte-identical except `submission.py`. `weights.npz` gains 16 keys: the `route_*` keys of
`route_pkg3.npz` and `route_V_hi`. v1's `route_head_W`, `route_head_c` and `route_wmax` are absent.

**Stage (class `Route3`).**
- **Shadow tail.** The ensemble's own unrouted rebalancing → SessionEM → TSEM output, i.e. exactly cand_rec_ts2_tsem.
- **Gate (v2).** It reads only the shadow outputs and the 256-PC views u (`route_V_hi` as in v1) of windows < t:
  1. robust standardisation (median / 1.4826 MAD, clipped at ±4);
  2. q70 seeds of the shadow outputs;
  3. the class contrast is projected off span{`route_pop`, `route_b`} (`route_Q`);
  4. cosine with the projected galleries.
  
  From t ≥ 60, r = clip((s* − 0.22) / 0.05, 0, 1).
- **Routed tail.** Its own rebalancing state and deep copies of the SessionEM and TSEM stages run on
  mix' = (mix − r a core1 + g r a h) / (1 − r a + g r a), where:
  - mix is the member mixture before rebalancing;
  - core1 is the first tangent member's probabilities, with mix share a = `Ensemble.w[0]` = 1/15 (= `route_slot_a`);
  - g = 3;
  - h is person p*'s PS1 head (LR, C 0.1) on u, standardised by the mean and std of windows 0..t.

  With r = 0 the routed tail equals the shadow tail bitwise. The output is the routed tail. The gate never reads it.

Data: `route_pkg3.npz` (md5 `cfb79ef7…`, SHA-256 `b4de2fe7…`; route team, session scratch `route2/build_pkg3.py`). Its
`rb_*`, `em_*` and `tsem_*` scalars equal the package's and are not written. The streaming reference is
`experiments/route_stream3.py` (`V3Tail`, md5 `419c1637`).

**Dev evidence (pre-registered, `route2/prereg3.txt`, `route2/confirm3.log`).** Against B2 + TSEM on 4 fresh bases
(B2_s012, s0, s1, s2):
- dJ +1.12, +1.06, +1.04, +0.99;
- new 0.00, with 0 impostor routings;
- seen +2.23, +2.11, +2.08, +1.98 (P04 +3.4 to +4.2, P10 0.0 to +0.7);
- a wrong-PS1 null: real − null dJ +1.63 to +1.77;
- all bars pass, including truncation.

Before the prereg, stress tests passed: constant, noise, random-posterior and weakened sessions, and artifact windows
all give 0 impostor routings. External Stieger FAR is 0.42 % (13 / 3,104; false accepts have mean r 0.085). Caveats:
- the gain comes only from returning people (dev: mostly P04);
- external genuine acceptance is low (Stieger 0.26, Lee 0.43);
- the 4 seen probe sessions were used across v1, v2 and v3.

## One command (cand_rec_ts2_tsem_route3)

Same as cand_rec_ts2_tsem_route, with `--route runs/diag/rroute3/route_pkg3.npz`, output
`runs/sealed/cand_rec_ts2_tsem_route3`, and `--off-only` in place of `--wmax0-only`. For v3, `--off-only` sets
`route_tau` to 1e9 in a scratch copy. The paired timing is `runs/diag/paired_timing_ab.sbatch runs/sealed/cand_rec_ts2_tsem
runs/sealed/cand_rec_ts2_tsem_route3`. The upload ZIP is the replica's `submission.zip`, copied to
`runs/upload/sealed_cand_rec_ts2_tsem_route3.zip`.

Provenance caveat. This was built from uncommitted files on top of `26b0f54` (`NI26_SOURCE_REV=26b0f54+route3-uncommitted`).
`--code` was `runs/diag/rroute3/code/submission.py` `3ef76b58…`. That was the repository's `submission.py` at
build time. The repository now holds the early-boost code `db3aa8b6…`, which rebuilds route3 byte-identically (see
cand_rec_ts2_tsem_route3eb). These files must be committed:
- `submissions/ensemble/submission.py`
- `experiments/route_stream.py`, `experiments/route_stream2.py`, `experiments/route_stream3.py`
- `scripts/export_route.py`
- `scripts/cloud/export_route.sbatch`

`route_pkg3.npz` is kept as `runs/diag/rroute3/route_pkg3.npz` on Galvani and
`runs/sealed/cand_rec_ts2_tsem_route3_route_pkg3.npz` on the Mac. Its builder is session scratch.

Routing v2 (`route_v2` keys: shape score, robust standardisation, log-blend of the personal head). Its packages
`runs/sealed/cand_rec_ts2_tsem_route2g` (wmax 0.5, ramp 0.05) and `…_route2f` (wmax 0.25, ramp 0.10) were built and
checked with code `2966128b…` (Galvani jobs 3068004 / 3068005), then superseded before their replays. Both are
bit-identical to `route_stream2.py` and route every public session to its own person from window 60. No v2 ZIP exists.

## Reference values (cand_rec_ts2_tsem_route3)

- Code: `submission.py` `ab5fc0b9…` (v1) → `3ef76b58…` (947 lines).
  - Class `Route` keeps v1 unchanged and adds the v2 path (`route_v2`).
  - New class `Route3`, the route team's v3 class verbatim (`route2/zipcode/submission_v3.py`, md5 `5a75474c`).
  - `Ensemble` binds `Route3` to copies of the SessionEM and TSEM stages, stores `core1` in `mixed_proba`, keeps the
    pre-rebalancing mix in `predict`, and steps `Route3` on (x, u, mix, core1, shadow output).
  - The v1 / v2 call is unchanged.
  - `check_compat --mix` (jobs 3068247–3068255, `runs/sealed/cand_rec_ts2_tsem_route3_compat.json`) covers the 7 arms
    up to cand_rec_ts2_tsem256, cand_rec_ts2_tsem_route (v1, own code `ab5fc0b9…`) and cand_rec_ts2_tsem_route2g (v2,
    own code `2966128b…`). Each gives byte-identical labels and member mixes on all 2,655 public windows. So the v1
    and v2 loader paths are kept, not fail-safed.
- Parity on all 2,655 windows (export job 3068242, galvani-cn002, CPU, 34 min):
  - The routed output and the shadow output are both bit-identical to `V3Tail` on the stage's own inputs (max |diff| 0,
    0 flips).
  - The shadow output against the cand_rec_ts2_tsem harness: 8.3e-10, with labels equal.
  - Views: 1.4e-12.
  - `route_slot_a` equals `Ensemble.w[0]`.
- Routing (in-sample). All 14 public sessions route to their own person from window 60 and from no earlier window,
  at r = 1 (lowest score 0.42, P04|3), so 1,815 windows are routed. 14 labels differ from cand_rec_ts2_tsem.
- In-sample cell mean, 0.98033 → 0.98337. This is not evidence: the PS1 heads were trained on these sessions.
- Truncation (T 128) is bit-identical before T. A deep-copied, reset model equals a freshly loaded one. Batches of 64
  against one window at a time differ by 7.1e-7.
- Routing off (`route_tau` 1e9, job 3068243): labels and final posteriors are byte-identical to cand_rec_ts2_tsem run
  with its own code. The routed output equals the shadow bitwise, and no window is routed.
- Package: 33 files. Only `submission.py` and `weights.npz` (`1d72faa3…`) differ from cand_rec_ts2_tsem. Every
  cand_rec_ts2_tsem key is byte-identical. The `route_*` keys equal `route_pkg3.npz`, and `route_V_hi` equals v1's.
- Replays, with files unchanged in both:
  - official A100: job 3068244, 225.0 s (galvani-cn220);
  - worker-image replica: job 3068245, 188.3 s (galvani-cn227).

  The scores are identical (in-sample 0.980088). The CPU export labels give the same score on the 755 replay windows
  (8 changed). Upload ZIP `runs/upload/sealed_cand_rec_ts2_tsem_route3.zip`
  `9903c1efd31a51f405b143fec7961ed2f9afeb70292c97ce07b61e137ce8cd45` (57,280,039 B), byte-identical to both replays'
  `submission.zip`.
- Runtime. Paired ABBA in the replica on one node (job 3068246, cn207):
  - base: 185.7 s and 180.6 s;
  - v3: 186.1 s and 184.0 s.

  The ratio is 1.01. Per-window scaling of cand_rec_ts2_tsem's real 2,497 s gives about 42 min. The unpaired replica
  ratio (1.04), with up to 60 s fixed overhead removed, gives at most 44 min. Both are under the 50-min flag. The second
  tail adds one SessionEM step and one TSEM step per window; the route team measured v3 at about 0.5 ms per window
  more than v2 on the Mac (`route2/pkgtest3.log`).

**Tolerance.** As for cand_rec_ts2_tsem_route. The build fits nothing, so `route_pkg3.npz` and cand_rec_ts2_tsem256's
basis determine it.

# Audit recipe: cand_rec_ts2_tsem_route3eb (route3 + early boost; PROVISIONAL)

An **alternate** upload, **provisional**: the adversarial verification of the early boost (EB) was still running at
build time. It is cand_rec_ts2_tsem_route3 with EB in SessionEM and TSEM. `weights.npz` equals route3's except:
- `em_nmin` 10 → 20;
- new keys `em_boost` 2.0, `tsem_boost` 4.0, `eb_T` 30.0.

**Stage change (4 code lines, route team `early/pkg/eb.diff`).** `SessionEM.__init__` reads `em_boost` / `eb_T`, and
`TSEM.__init__` reads `tsem_boost` / `eb_T`. The defaults 0 / 0 / 30 keep every package without these keys bitwise
unchanged (`e *= 1.0`). Each `_adapt` multiplies its output exponent by 1 + boost · clip((2T − t) / T, 0, 1): full boost
up to t = 30, linear to none at t = 60, identical from t = 60 on given the same inputs. `Route3.bind` deep-copies the
boosted stages, so the routed tail inherits EB. That is the routing+EB design of `early/combo.py`, which injects EB into
both the shadow and the routed tail.

**Dev evidence (`early/prereg.txt`, `early/confirm.log`, `early/combo.log`; config chosen post hoc in round 2).**
- **EB alone** against the shipped chain on 4 fresh bases: mean dJ +0.36 (+0.67 / +0.42 / +0.14 / +0.21). New is +0.53
  to +1.09, seen −0.50 to +0.41. All bars pass.
- **EB + routing v3** against routing v3: dJ +0.74 / +0.54 / +0.28 / +0.34. All of it is from new people (+0.53 to +1.09).
  Seen is −0.22 to +0.54; P10 is −0.5 to −1.2.
- **EB + routing** against shipped: +1.86 / +1.60 / +1.31 / +1.33.
- Routing is unaffected: 0 new-person routings, and the same 4 seen sessions route.

## One command (cand_rec_ts2_tsem_route3eb)

As route3, plus `--set em_nmin=20 --set em_boost=2.0 --set tsem_boost=4.0 --set eb_T=30.0` and output
`runs/sealed/cand_rec_ts2_tsem_route3eb`. With EB keys, the reference is `experiments/route_stream3eb.py`
(`V3TailEB`: `route_stream3.V3Tail` with the boost in both tails). `--off-only` then compares against a temporary
cand_rec_ts2_tsem + EB package without route keys. The upload ZIP is the replica's `submission.zip`, copied to
`runs/upload/sealed_cand_rec_ts2_tsem_route3eb.zip`.

Provenance. This was built from uncommitted files on top of `26b0f54` (`NI26_SOURCE_REV=26b0f54+route3eb-uncommitted`).
`--code` is `runs/diag/rroute3eb/code/submission.py` `db3aa8b6…` (md5 `1fcece8e`), byte-identical to the route team's
`early/pkg/submission_eb.py`. The repository now holds the GEB code `786bcab0…`, which rebuilds route3eb byte-identically
(see cand_rec_ts2_tsem_route3geb). Besides the route3 files, `experiments/route_stream3eb.py` must be committed.

## Reference values (cand_rec_ts2_tsem_route3eb)

- Code: `3ef76b58…` + 4 lines → `db3aa8b6…` (951 lines).
  - `check_compat --mix` (jobs 3068926–3068935, `runs/sealed/cand_rec_ts2_tsem_route3eb_compat.json`) covers 10
    packages, each with its own code: the 7 arms, route v1 (`ab5fc0b9…`), route2g (`2966128b…`) and route3
    (`3ef76b58…`, EB keys absent). All give byte-identical labels and member mixes on all 2,655 public windows
    (0 windows differ).
- Stage parity against the offline EB chain (`early/mech.py`): the route team's `early/pkgtest.py`, rerun on this exact
  file. On B2_s012, both protocols, the packaged `SessionEM` and `TSEM` classes equal the offline chain bitwise (max
  |diff| 0, 0 argmax mismatches) with EB (2, 20, 4) and with EB off. The offline chain's truncation test passes (rows
  < T unchanged for T 5 / 25 / 45 / 70).
- Package parity on all 2,655 windows (export job 3068921, galvani-cn001, CPU, 32 min):
  - Routed and shadow outputs are bit-identical to `V3TailEB` on the stage's own inputs (0 flips).
  - The routed tail's SessionEM and TSEM equal the shadow's: nmin 20, boost 2.0; TSEM boost 4.0; T 30.
  - `route_slot_a` equals `Ensemble.w[0]`.
  - Views: 1.4e-12.
  - The shadow now differs from cand_rec_ts2_tsem's harness (max 0.21), as expected, since EB changes the chain.
- Routing (in-sample). All 14 public sessions route to their own person from window 60 at r = 1 (lowest score 0.44).
  25 labels differ from cand_rec_ts2_tsem, 12 of them from the route stage.
- In-sample cell mean, 0.98033 → 0.97931. This is in-sample, not evidence.
- Truncation (T 128) is bit-identical before T. A deep-copied, reset model equals a fresh one.
- Routing off (`route_tau` 1e9, job 3068922): labels and final posteriors are byte-identical to a temporary
  cand_rec_ts2_tsem + EB package without route keys, and no window is routed. Against cand_rec_ts2_tsem itself, EB
  changes labels by design.
- Package: 33 files. Only `submission.py` and `weights.npz` (`ba521255…`) differ from route3. In the weights, only
  `em_nmin` and the 3 EB keys differ.
- Replays, with files unchanged in both:
  - official A100: job 3068923, 222.9 s (galvani-cn219);
  - worker-image replica: job 3068924, 189.8 s (galvani-cn220).

  The scores are identical (in-sample 0.973578; route3 0.980088). The CPU export labels give the same score on the 755
  replay windows (13 changed from cand_rec_ts2_tsem). Upload ZIP `runs/upload/sealed_cand_rec_ts2_tsem_route3eb.zip`
  `2f34907d54c40e661e1e09a1e6703a8652a0a80fe15e7f1844e53dc2460a1fe0` (57,280,276 B), byte-identical to both replays'
  `submission.zip`.
- Runtime. Paired ABBA in the replica on one node (job 3068925, cn220):
  - base: 182.8 s and 182.1 s;
  - EB: 187.4 s and 187.4 s.

  The ratio is 1.03. Per-window scaling of cand_rec_ts2_tsem's real 2,497 s gives about 43 min. The unpaired replica
  ratio (1.05), with up to 60 s fixed overhead removed, gives at most 45 min. Both are under 50 min.

**Tolerance.** As for route3.

# Audit recipe: cand_rec_ts2_tsem_route3geb (route3 + gated early boost)

An **alternate** upload, for staging only (uploads are on hold by user decision). It replaces route3eb as the boost
variant. It is cand_rec_ts2_tsem_route3 with the gated early boost (GEB, route team `eb2/`). `weights.npz` equals
route3eb's plus `eb_gate_lo` 0.20 and `eb_gate_hi` 0.35. Relative to route3, the changes are:
- `em_nmin` 10 → 20;
- `em_boost` 2.0, `tsem_boost` 4.0, `eb_T` 30.0;
- the 2 gate keys.

**Stage change (`eb2/pkg/geb.diff`: 19 lines added, 3 changed, against the EB code).**
- `SessionEM` keeps the running sum of its input margin (top1 − top2) over windows 0..t−1. Its gate is
  g(t) = clip((0.35 − mean) / 0.15, 0, 1); g = 1 at t = 0, and also when no gate keys are present.
- g multiplies EB's h(t) in that SessionEM. It also multiplies h(t) in the TSEM whose `gate_src` points to that
  SessionEM; `Ensemble` sets the link.
- `Route3.bind` deep-copies `(em, tsem)` as one pair, so the routed tail's TSEM reads the routed tail's own SessionEM
  gate, not the shadow's and not an ungated one. Each tail's gate therefore comes from its own SessionEM input, as in
  `eb2/combo2.py`.
- Without gate keys every gate is 1.0 and h · 1.0 = h exactly. EB packages and earlier packages are bitwise unchanged
  (compat below).

**Dev evidence (`eb2/prereg.txt`, `eb2/res/analyze.log`, `controls.log`, `combo2.log`).**
- **Pre-registered gates on the fresh-base cross-fit:** all pass. dJ +0.54 (+0.86 / +0.48 / +0.44 / +0.38), new +0.88,
  seen +0.20 (P04 +0.40, P10 +0.01), 10/10 new ≥ 0. Joint sign-flip p = 0.001. A permuted-gate null gives +0.30 on average.
- **Against EB, paired:** +0.18 (seen +0.32, P10 +0.50; p 0.094).
- **With routing v3 (`combo2.log`, mean of 4 fresh bases):** GEB + routing scores +1.80 against shipped (EB + routing
  +1.53, routing only +1.05), with new +0.88 and seen +2.73. GEB + routing against routing only: +0.75.
- The team's simulated GEB shadow equals their verifier chain to about 1e-15.

## One command (cand_rec_ts2_tsem_route3geb)

Prerequisites (recipes above): `runs/sealed/cand_rec_ts2_tsem` and `runs/sealed/cand_rec_ts2_tsem256` with their
harnesses, and `route_pkg3.npz` (SHA-256 `b4de2fe7…`: Galvani `runs/diag/rroute3/route_pkg3.npz`, Mac
`runs/sealed/cand_rec_ts2_tsem_route3_route_pkg3.npz`, archive `runs/archive_7nov/route2/pkg/route_pkg3.npz`). The
build fits nothing. With the lean code (7 Nov cleanup, below) the whole recipe is:

```bash
A="runs/sealed/cand_rec_ts2_tsem runs/sealed/cand_rec_ts2_tsem_route3geb --route runs/diag/rroute3/route_pkg3.npz \
   --hi runs/sealed/cand_rec_ts2_tsem256 --src-harness runs/sealed/cand_rec_ts2_tsem_harness.npz \
   --hi-harness runs/sealed/cand_rec_ts2_tsem256_harness.npz \
   --set em_nmin=20 --set em_boost=2.0 --set tsem_boost=4.0 --set eb_T=30.0 --set eb_gate_lo=0.20 --set eb_gate_hi=0.35"
J=$(sbatch --parsable --export=ALL,NI26_SOURCE_REV=$(git rev-parse HEAD) \
    -o runs/sealed/cand_rec_ts2_tsem_route3geb_export_%j.log scripts/cloud/export_route.sbatch $A)
sbatch --export=ALL -o runs/sealed/cand_rec_ts2_tsem_route3geb_off_%j.log scripts/cloud/export_route.sbatch $A --off-only
X=galvani-cn109,galvani-cn111,galvani-cn121,galvani-cn122
sbatch --dependency=afterok:$J --exclude=$X runs/diag/replay_bundle.sbatch runs/sealed/cand_rec_ts2_tsem_route3geb \
    runs/sealed/cand_rec_ts2_tsem
```

The reference is `experiments/route_stream3geb.py` (`V3TailGEB`: each tail's TSEM reads its own tail's SessionEM gate;
since the cleanup a self-contained file that merges the former `route_stream3.py` and `route_stream3eb.py` with
unchanged arithmetic). The export also asserts that wiring:
- the shadow TSEM reads the shadow SessionEM;
- the routed TSEM reads the routed SessionEM;
- the two SessionEMs are distinct.

`--off-only` builds a temporary copy with `route_tau` 1e9 and compares it with cand_rec_ts2_tsem + the same `--set` keys
without route keys. The A100 replays run as one job, `runs/diag/replay_bundle.sbatch PKG BASE` (Galvani scratch, not in
the repository: the official A100 replay, then the paired ABBA in the worker-image replica). Its runs 2 and 3 are the
replica replays of PKG, and run 2's `submission.zip` is copied to `runs/upload/sealed_cand_rec_ts2_tsem_route3geb.zip`.
`scripts/cloud/replay.sbatch` and `worker_replay.sbatch` (as in the recipes above) give the same two replays.

Provenance: built from uncommitted files on top of `26b0f54` (`NI26_SOURCE_REV=26b0f54+route3geb-uncommitted`). `--code`
is `runs/diag/rroute3geb/code/submission.py`, byte-identical to the route team's `eb2/pkg/submission_geb.py` (md5
`b575d3d1`) and to `submissions/ensemble/submission.py` `786bcab0…` as committed in `c4d600d` (7 Nov). The uploaded ZIP
carries `786bcab0…`; the lean file only replaces the repository copy (reference values below).

## Reference values (cand_rec_ts2_tsem_route3geb)

- Code: `db3aa8b6…` + GEB → `786bcab0…` (967 lines).
  - `check_compat --mix` (jobs 3069164–3069174, `runs/sealed/cand_rec_ts2_tsem_route3geb_compat.json`) covers 11
    packages, each with its own code: the 7 arms, route v1, route2g, route3 (GEB keys absent) and route3eb
    (`db3aa8b6…`, gate keys absent). All give byte-identical labels and member mixes on all 2,655 public windows
    (0 windows differ). So this one file rebuilds every staged package.
- Stage parity against the offline gated chain.
  - **Team test, rerun:** the route team's `eb2/pkgtest.py`, rerun on this exact file (B2_s012 and B2_s1, both
    protocols). The packaged `SessionEM` and `TSEM` equal `eb2/gm.py` bitwise, and the verifier chain to about 1e-15,
    with 0 mismatches. With the gate disabled they equal the EB classes bitwise. The routed-tail deep copy keeps the
    em↔tsem link.
  - **Reference against the offline chain** (`geb_build/ref_vs_gm.py`, session scratch): `route_stream3geb.ChainG` equals
    `gm.pipe_g` to ≤ 1.4e-15 (0 mismatches; gates ≤ 1.9e-15). That holds for the real mix (shadow; mean gate 0.61–0.73
    at t 15–59) and for a perturbed mix standing in for a routed mix' (mean gate 0.88–0.92). With EB-only keys it
    equals `early/mech.py`'s chain.
  - **Package with the gate open** (`geb_build/open_gate.py`): one public session with degraded EEG gives gate 1.0 at
    t 15–59. The packaged routed and shadow outputs are bit-identical to `V3TailGEB`.
- Package parity on all 2,655 windows (export job 3069161, galvani-cn117, CPU, 39 min):
  - Routed and shadow outputs are bit-identical to `V3TailGEB` on the stage's own inputs (max |diff| 0, 0 flips).
  - Gate wiring holds: the shadow TSEM reads the shadow SessionEM, the routed TSEM reads the routed SessionEM, and the
    two are distinct objects.
  - The routed tail's stages equal the shadow's: SessionEM nmin 20, boost 2, gate 0.20 / 0.35; TSEM boost 4; T 30.
  - `route_slot_a` equals `Ensemble.w[0]`.
  - Views: 1.4e-12.
  - The shadow differs from cand_rec_ts2_tsem's harness by at most 0.041, as expected: GEB changes the chain only
    where the gate opens.
- In-sample, the public sessions are strong, so the routed gate is almost closed (mean 0.009, max 0.53 at t 15–59).
  The open-gate path is covered by the dev and degraded-EEG checks above.
- Routing (in-sample). All 14 public sessions route to their own person from window 60 at r = 1 (lowest score 0.41).
  16 labels differ from cand_rec_ts2_tsem, 14 of them from the route stage.
- In-sample cell mean, 0.98033 → 0.98412. This is in-sample, not evidence.
- Truncation (T 128) is bit-identical before T. A deep-copied, reset model equals a fresh one. Batches of 64 against
  one window at a time differ by 7.1e-7.
- Routing off (`route_tau` 1e9, job 3069162): labels and final posteriors are byte-identical to a temporary
  cand_rec_ts2_tsem + GEB package without route keys, and no window is routed.
- Package: 33 files. Only `submission.py` and `weights.npz` (`813d8c54…`) differ from route3. In the weights, only
  `eb_gate_lo` and `eb_gate_hi` are added relative to route3eb.
- Replays (bundle job 3069163, galvani-cn219), with files unchanged:
  - official A100: 196.3 s;
  - worker-image replica, ABBA runs 2 and 3: 183.2 s and 185.4 s.

  The scores are identical (in-sample 0.981432). The CPU export labels give the same score on the 755 replay windows
  (9 changed from cand_rec_ts2_tsem). Upload ZIP `runs/upload/sealed_cand_rec_ts2_tsem_route3geb.zip`
  `74433d9bf7ce983b325c0c2b13a82da764279594a11af27547f729f8d6395ff9` (57,280,711 B). It is replica run 2's
  `submission.zip`, byte-identical to the A100 replay's and run 3's.
- Runtime. Paired ABBA in the replica on cn219:
  - base: 184.6 s and 182.5 s;
  - GEB: 183.2 s and 185.4 s.

  The ratio is 1.00. Per-window scaling of cand_rec_ts2_tsem's real 2,497 s gives about 42 min. The unpaired replica
  ratio (1.02), with up to 60 s fixed overhead removed, gives at most 43 min. Both are under 50 min.
- **Uploaded:** submission **1040847** (7 Nov), sealed **0.8240266154**, 2,687.2 s.

**Tolerance.** As for route3.

## Reference values: lean code (7 Nov cleanup)

- **Code.** `submission.py` `786bcab0…` (967 lines) → `dacee85a…` (781 lines). Removed:
  - the CBraMod member;
  - routing v1 / v2 (class `Route`);
  - the early boost without gate keys (`em_boost` now requires `eb_T`, `eb_gate_lo`, `eb_gate_hi`);
  - SessionEM without ocular-event keys;
  - rebalancing over the whole past (`rb_w` 0) and the n0 = 0 branches.

  Rebalancing became one class (`Rebalance`), shared by the shadow and the routed tail (`Route3.bind` deep-copies
  rebalancing, SessionEM and TSEM together). `experiments/route_stream3geb.py` is self-contained (`19d3118b` md5; it
  merges `route_stream3.py` / `route_stream3eb.py`, which were deleted together with `route_stream.py` /
  `route_stream2.py`). `scripts/export_route.py` is v3-only.
- **Compatibility** (`scripts/check_compat.py PKG --post --code submission.py`, Mac CPU, `run_capped`). Each run uses
  the shipped ZIP extracted, with its own code, against the lean code on all 2,655 public windows, streamed per
  session. It compares labels, the pre-rebalancing member mix and every stage's per-window output: SessionEM, TSEM,
  routing; the last of these is the final posteriors. Records are in `runs/archive_7nov/cleanup/`.
  - cand_rec_ts2_tsem_route3geb (`74433d9b…`, own code `786bcab0…`): every output byte-identical, 0 windows differ
    (1,122 s).
  - cand_rec_ts2_tsem (`9c31a167…`, own code `243c9901…`): every output byte-identical, 0 windows differ (1,008 s).
  - cand_nxif (`0a562e62…`, own code `25d51a6e…`): labels, mix and SessionEM output (= final posteriors)
    byte-identical, 0 windows differ (459 s).
- **Recipe run with the lean code** (Mac CPU, the command above with local paths; inputs SHA-256-equal to Galvani's:
  cand_rec_ts2_tsem `9812d2f9…`, tsem256 `90fda086…`, `route_pkg3.npz` `b4de2fe7…`, both harnesses).
  - **Rebuilt package:** `weights.npz` `813d8c54…`, equal to the shipped ZIP's. 32 of 33 files are byte-identical to
    the ZIP; only `submission.py` differs (the lean file).
  - **Reference and wiring:** routed and shadow outputs equal `V3TailGEB` bitwise (max |diff| 0, 0 flips); gate
    wiring true.
  - **Routing:** every public session routes to its own person from window 60.
  - **Same numbers as the original export:** 16 labels changed vs cand_rec_ts2_tsem, 14 of them by the routing
    stage; in-sample 0.98412; routed gate mean 0.009, max 0.53 at t 15–59.
  - **Streaming checks:** truncation (T 128) and deep copy + reset are bit-identical.
  - **Platform differences:** comparisons with Galvani's stored harnesses differ only at platform level (views
    5.3e-9 vs 1.4e-12 on Galvani; BLAS / LAPACK).
  - **Routing off** (`--off-only`, Mac): labels and final posteriors are byte-identical to cand_rec_ts2_tsem + GEB
    without route keys; the stage output equals its input bitwise and no window is routed (as in job 3069162).
- **Reference merge.**
  - **Galvani:** the merged `V3TailGEB` output equals the `c4d600d` reference's stored output bitwise (`ref`,
    `ref_shadow` in the Galvani harness comparison above).
  - **Mac:** the merged reference run on the stored stage inputs differs from Galvani's stored outputs by at most
    4.4e-16 (BLAS / LAPACK).
  - **Mac, same machine** (`runs/archive_7nov/cleanup/local/ref_check.py`): the merged reference and the `c4d600d`
    reference (`route_stream3geb.py` md5 `41bd2d5f` on `route_stream3.py` `419c1637` / `route_stream3eb.py`
    `c7051d10`) give bitwise-equal routed and shadow outputs on all 2,655 windows. Label flips vs Galvani: 0.
- **Recipe run on Galvani** (Linux CPU; job 3069827, galvani-cn001, 38 min). The setup was a mini-root
  `runs/diag/clean7nov/root/` holding `c4d600d` + the cleanup edits, run with the real Galvani inputs and the command
  above.
  - **Rebuilt package:** `weights.npz` `813d8c54…`; 32 of 33 files byte-identical to the shipped ZIP; only
    `submission.py` differs.
  - **Harness:** every array of the new harness is bitwise equal to the original export's
    `runs/sealed/cand_rec_ts2_tsem_route3geb_harness.npz` (job 3069161, cn117, `786bcab0…` and the old references).
    That covers the package's routed and shadow outputs (`O`, `Q`), the reference outputs (`ref`, `ref_shadow`), the
    views, the mixes, the SessionEM features and the labels.
  - **So:** on the platform of the original build, the lean code and the merged reference reproduce the
    route3geb export exactly.
  - Compat of route3geb on Galvani (job 3069795, cn002, 2,976 s, shipped ZIP extracted): every output
    byte-identical, 0 windows differ, as on the Mac. Records: `runs/diag/clean7nov/` on Galvani.
