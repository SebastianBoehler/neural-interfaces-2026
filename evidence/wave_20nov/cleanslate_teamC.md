# Clean-slate team C · compute-maximal inference (per-session test-time training)

Brief: `CLEANSLATE.md`. Angle: turn the sealed budget (3,600 s / ~20,083 windows ≈ 180 ms GPU per window on one RTX PRO
4500) into accuracy by genuine test-time training of a cross-person CNN ensemble on each session's delivered prefix,
with the adaptation gain measured separately from the starting network.

## System (code `cleanslate/teamC/`)
- `submission.py` (the package file itself; official `Solver`): members ShallowConvNet-style, EEGNet-style, FBNet
  (FIR-initialised learnable filter bank + per-band spatial filters + log-variance). Per window: causal Euclidean
  alignment (shrunk mean covariance of the delivered prefix, current window included); on a schedule (from window
  n0=16, every window up to 48, then every 4th) BN statistics re-estimated on the aligned prefix (mixed with source
  statistics, weight n/(n+8)); optional gradient TTT on the prefix (each step: batch of ≤64 prefix windows, strong
  view = shift/scale/channel-drop/noise, target = Sinkhorn-balanced ensemble posterior, + information maximisation
  + L2 anchor to source weights, Adam); output = ensemble mean posterior, optionally causal prefix Sinkhorn.
  Session state lives in `reset_state`/`__deepcopy__`; base nets are shared read-only; nothing at module level.
- `train.py --people ... --out DIR`: trains every member on causally aligned windows (each epoch, each session in a
  fresh random delivery order, every window whitened by its own prefix, exactly the test-time path), session-
  homogeneous batches, label smoothing 0.1, AdamW one-cycle. No spatial rotation by default (lane V: random
  rotations of the EA space cost 2–17 pp standalone; our `rot` 0.3 equals their harmful V2w dose).
- `stream_eval.py`: own LOPO quick-eval until the shared harness lands. Held person streamed session by session on
  `deepcopy(model)` + `reset_state()`, one window per `predict`, order k = `runs/shuffle_perms.npz` `full|k|all_loso`
  (order 0 is the order of cap250's full0 bank, so the delta vs cap250 is paired on order 0). Per-window seconds.
- `score.py`: official cell-mean BA (benchmark_utils.metrics), 28 held-new cells, people-resampled 80 % intervals
  (cell-weighted), paired delta vs cap250 stock terminal full0 (73.74, checked locally from the c25d bank).
- Runtime model: per-window GPU seconds on A100 × 2.5 → RTX PRO 4500 estimate (A100→RTX PRO 4500 ratio unmeasured;
  2.5 assumed conservative for small-kernel, latency-bound nets). Budget line: 3,600 s / 20,083 = 179 ms.

## Compute
Leonardo login nodes timed out from this machine on 20 Nov ~22:40 (all of login01/02/05/07); runs on Galvani
`a100-galvani,a100-preemptable-galvani`, release `/mnt/lustre/work/gehler/gfh527/ni26_teamC_20261120` (data sha256
2be4187f… equal to the local export). Script `cleanslate/teamC/galvani.sh`.

## C1 · base ensemble E6 and the adaptation ladder (registered before any C1 posterior was scored)
Base E6 = shallow/eegnet/fbnet × seeds 0,1, 60 epochs, rot 0. Variants on the same base (held-new, orders 0 and 1):
A0 source (causal EA only) · A1 + prefix BN · A1b + causal Sinkhorn · A2 + gradient TTT (4 steps, lr 1e-4, all
params) · A2b A2 + Sinkhorn · A2h 8 steps lr 3e-4 · A2n BN affine + head, lr 1e-3.
Predictions (BA %, mean of 2 orders): A0 56 [51, 61]; A1 − A0 +1.5 [−1, +4]; A1b − A1 +0.5 [−0.5, +1.5];
A2 − A1 +1.5 [−1, +4]; best variant − cap250 (order 0) −12 [−17, −7].
Rules: adaptation quality = A2* − A1 on the same base, per order, people-80 % interval: supported if the lower bound
> 0 on both orders, refuted if the upper bound < 0 on both, else undecided (bars are reading aids, ±0.5 instrument
noise). The starting network is judged by A1. No closure from C1: a negative A2 opens other TTT forms (anchor,
lr, steps, teacher), since the angle's ceiling is not measured by one form.

## C1x · E6 trained with extra cue-offset crops (registered with C1, before scoring)
Same members, 30 epochs, training rows ×5: the official window (start 1 s) plus the exports starting 0.5, 2, 3, 4 s
after the cue (same trials, same labels; each offset is its own alignment stream). Test input unchanged (start 1 s).
Prediction: A1(C1x) − A1(C1) +1.5 [−0.5, +3.5] (more crops ≈ more within-person variety, not more people). Same
reading rule as C1 (people-80 % interval on both orders). Jobs: Galvani 3139558 (C1), 3139559 (C1x); smoke 3139488
passed (2 epochs, code path only; A2 ≈ 62 ms/window on a 2080 Ti including Python overhead).

## C2 · heavier TTT forms on the C1 packages (registered 20 Nov ~23:30, after seeing C1 partial logs of P07/P09 only)
Seen before registering: P07/P09 order-0 accuracies (A1 73.0/68.8, A2 78.3/76.7, A2h 81.5/77.3). No cell-mean score yet.
Eval-only (`galvani.sh eval C1`, sub `c2`), orders 0,1, all ten people, against A2h (8 steps, lr 3e-4):
A3a 16 steps · A3b lr 1e-3 · A3c no anchor · A3d no information maximisation · A3e confidence ≥ 0.5 on the balanced
target · A3f update every 2nd window and dense to 64. Prediction: A3a − A2h +0.5 [−1, +2]; A3b −0.5 [−3, +1.5] (drift);
A3c 0 [−1, +1]; A3d −0.5 [−2, +1]; A3e +0.3 [−1, +1.5]; A3f +0.3 [−0.5, +1]. Same reading rule (both orders).
Runtime is part of the result: a form is only usable if its 2080 Ti ms/window ≤ 179 (the 2080 Ti is slower than the
RTX PRO 4500, so this is conservative for the sealed card).

## Resume 20 Nov 07:40 (previous agent lost to a network outage)
Galvani node cn122 failed CUDA init: C1 folds P01-P06, P08 and C1x P04, P06 never ran; C2 ran only where a C1 package
existed. Resubmitted with `--exclude=galvani-cn122` (galvani.sh `EXCL`, `DEP`): C1 3140324, C1x 3140325, C1 returning
3140326 (P04/P10 sessions 2-3, trained on all but those), C2 3140327 (P09) + 3140328 (afterok C1). Seen before C3 was
registered: C1x on 8 people (A1 63.1, A2 63.6, both orders) and C1 on P07/P09/P10 (P10: A2h −6 vs A1).

## C3 · base push on Leonardo (registered 20 Nov ~08:30, before any C3 output)
Release `/leonardo_work/AIFAC_P02_1636/ni26_teamC_20261120` (`leo.sh`). Held-new, orders 0,1, variants A0,A1,A2,A2n.
- C3e: E9 = shallow/eegnet/fbnet × seeds 0,1,2, 60 epochs, PROTEUS only (seed lever).
- C3s: E9 + Scherer 2015 (`--ext scherer2015`: 2,130 windows, 2 visits per person, already in PROTEUS label space,
  22 of 41 channels spline-interpolated) as extra training people (B5: more people is the largest measured base lever).
Predictions: A1(C3e) − A1(C1) +0.7 [−0.5, +2]; A1(C3s) − A1(C3e) +1.5 [−1, +4]; A2 − A1 on either base +1.5 [−1, +4].
Reading rule as C1 (people-80 % interval, both orders). A C3s package can no longer be scored on NI-V Scherer; the
NI-V run uses the C3e recipe (Scherer excluded) and says so.
