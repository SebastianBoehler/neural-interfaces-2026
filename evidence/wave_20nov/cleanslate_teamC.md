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
Jobs: C3e Galvani 3140401 (E9, PROTEUS only); C3s Leonardo 64917408 (afterok smoke 64917399).

## C1 result (20 Nov ~09:50; all 10 people, orders 0,1; official cell-mean BA; paired = people-resampled 80 %, person = mean of its cells)
| variant | held-new | − A1 (pooled) | ms/win (2080 Ti) | returning P04/P10 |
|---|---|---|---|---|
| A0 | 61.57 | −1.88 [−2.97, −0.80] | 6 | 57.23 |
| A1 | 62.90 | — | 11 | 66.57 |
| A1b | 61.77 | −0.45 [−1.22, +0.29] | 11 | 66.57 |
| **A2** | **65.25** | **+2.97 [+1.62, +4.29]** (o0 +2.82 [1.48, 4.17], o1 +3.12 [1.50, 4.76]) | 94–102 | **69.73** (+3.16 vs A1) |
| A2b | 64.06 | +2.27 [+0.54, +3.97] | 95 | 69.49 |
| A2h | 62.01 | +0.16 [−2.16, +2.60] | 178–181 | 67.69 |
| A2n | 64.38 | +1.61 [−0.05, +3.35] | 94 | 67.90 |
- Adaptation (A2 − A1): **supported** (lower bound > 0 on both orders); predicted +1.5, measured +3.0. A1 − A0 +1.9
  (predicted +1.5). A2h, the form that looked best on P07/P09, is no better than A1 over ten people (P10 −6.3, P08 −6.8, P01 −6.4).
- vs cap250 (order 0, paired cells): A2 −8.56 [−12.36, −4.49]; predicted −12. Returning cap250 archived 92.76: −23 pp.
- C1x (cue-offset crops, 30 ep): A1 − C1 A1 −3.77 [−5.79, −1.69]: **refuted** (predicted +1.5).
- C3e (9 nets, 3 seeds): A2 − C1 A2 +0.07 [−0.43, +0.58]; seeds do not lift the base. A2 − A1 on C3e +3.02 [1.88, 4.17].
  ms/window 141–149 (2080 Ti/A100 mix) vs 94–102 for E6.
- Identity: a raw log-covariance nearest-session check finds the true person top-1 in 73 % of P04/P10 sessions at 64
  windows with no distance margin (median 9.3 vs 10.0); team A's gallery accepted 1/8 returning replays. No personal route here.

## C4 · lighter TTT around A2 (registered ~09:55 before scoring; eval-only on C1 packages, orders 0,1, + A2 on orders 2,3)
A4a 2 steps every 2nd window dense to 64 · A4b lr 5e-5 · A4c start at 8 windows · A4d anchor 0.1 · A4e no information max.
Prediction vs A2: A4a +0.3 [−0.7, +1.3]; A4b −0.5 [−1.5, +0.5]; A4c +0.2 [−0.5, +1]; A4d −0.3 [−1.2, +0.6]; A4e −0.3 [−1.5, +1].
Same reading rule. Multiple forms on the same ten people = selection; the chosen form is re-scored on orders 2–5.

## C3s result (Leonardo 64917408, 10 people, orders 0,1)
A1 59.36, A2 62.55, A2n 63.91 (mean of orders). A1 C3s − C3e **−4.28 [−5.66, −2.92]**, A2 −3.29 [−4.61, −1.94]: Scherer
2015 as extra training people **refuted** (predicted +1.5). TTT gain is larger on the weaker base: A2 − A1 +4.01 [2.55, 5.54].
Scherer has 9 people (18 visits), not 8 as written in C3.

## C5 · prefix tangent-space self-training fused with A2 (registered ~11:10, before any C5 output)
Mechanism: cap250's stage gain comes partly from classical features fitted on the session prefix with pseudo-labels (CSP,
fine-tuning: friday/cap250c18). Here: per window, log-covariance (tangent at identity) of the EA-aligned window in three
bands (8–13, 13–20, 20–30 Hz, FIR), 3 × 861 features, standardised on the prefix; on the adaptation schedule from n ≥ 32 an
L2 multinomial LR is refit (warm, Adam) on the prefix with targets = Sinkhorn-balanced ensemble posterior; output =
normalised p_ens^(1−β) · p_lr^β. Variants: A5 β 0.3, A5b β 0.5, A5c β 0.3 on A1 (no gradient TTT). Same C1 recipe retrained on
Leonardo (also a cross-cluster replicate of C1 A1/A2), orders 0,1.
Prediction: A5 − A2 +1.0 [−0.5, +2.5]; A5b − A2 +0.5 [−1.5, +2.5]; A5c − A1 +1.5 [0, +3]. Replicate: |A2(Leo) − A2(C1)| < 1.
Rule as C1. Runtime must stay ≤ 179 ms/window on the timing card.

## C5 result (Leonardo 64922136 held-new, 64922150 returning; orders 0,1)
Held-new A1 63.12, A2 64.96, A5 64.21, A5b 61.24, A5c 62.21. A5 − A2 −0.33 [−1.19, +0.49] **undecided** (predicted +1.0);
A5b − A2 −3.67 [−4.70, −2.65] **refuted**; A5c − A1 −0.92 [−1.45, −0.38] (o0 −0.20 [−1.02, 0.59], o1 −1.63) undecided.
Returning A2 70.13; A5 − A2 −0.99, A5c − A2 −4.30. Replicate: A2(Leo) − A2(C1) −0.53 [−1.29, +0.23], A1 +0.40; TTT gain
on the replicate +2.04 [+0.91, +3.18]. Reading: 2,583 tangent features fitted on ≤190 own pseudo-labels echo the ensemble
and overfit; the direction is not closed (one form).
T4 validation (coordinator, laneT4_valid.md): Scherer NI-V A2 − A1 −0.15 [−0.93, +0.62]; heavy forms refuted by drift.

## C5v2 (registered ~13:05, before scoring; eval-only on the C5 packages, orders 0,1, held-new + returning)
Per-channel band log-variance (3 × 41 = 123 features) instead of tangent space, β 0.2: A5d (on A2), A5e (A5d with L2 0.1),
A5f (on A1). Prediction: A5d − A2 +0.3 [−0.5, +1.0]; A5e − A2 +0.3 [−0.5, +1.0]; A5f − A1 +0.3 [−0.5, +1.0].

## C5v2 result (Leonardo 64924663; vs A2/A1 of the C5 pass on the same packages, separate pass)
A5d − A2 +0.54 [+0.03, +1.10] (o0 +0.48 [0.11, 0.88], o1 +0.61 [−0.12, 1.39]); **A5e − A2 +0.65 [+0.19, +1.12]** (o0 +0.81
[0.30, 1.33], o1 +0.49 [−0.06, 1.04]); A5f − A1 +0.38 [+0.05, +0.73]. All three inside the predicted range; by the both-orders
rule **undecided, consistently positive** (o1 lower bounds −0.06 to −0.23). Not yet with an in-pass control.

## C5v3 (registered ~13:45 before scoring): A2, A5e, A5g (β 0.3, L2 0.1), A5h (β 0.2, L2 1.0) in one pass, orders 0–3,
held-new (C5 packages) and returning (C5ret package). Prediction A5e − A2 +0.5 [0, +1] on 4 orders; A5g +0.4 [−0.5, +1.2];
A5h +0.3 [−0.3, +0.9]. Supported if the people-80 % lower bound pooled over 4 orders is > 0 and ≥ 3 of 4 orders have point > 0.
