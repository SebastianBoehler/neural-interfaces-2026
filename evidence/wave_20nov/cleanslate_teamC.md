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
