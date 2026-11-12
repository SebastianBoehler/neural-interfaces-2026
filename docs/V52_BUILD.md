# v5.2 build: personal SessionBN nets on routed windows (12 Nov night)

Built and checked locally; **not uploaded**. By ledger §63 it is a candidate for upload 2 only if the v5.1b upload shows
the returning branch pays (v5.1b − v4.1 ≥ +0.4). Ledger context: §78 (+0.17 seen over the stage on six bases), §96
(+2.4 within a session, ten-person proxy), §102 (preparation).

## What v5.2 is

v5.1b (`submissions/ensemble_v5b/submission.py`, package `runs/sealed/cand_v51b_own`) plus class `PersonalNets` in
`submissions/ensemble_v52/submission.py`:

- `weights.npz` gains `pnet_member` (the 30 SessionBN members: net indices 35–64) and `pnet_files[m, j]`
  (`p_<base stem>_<person>.pt`, `route_names` order P01–P10). All 300 personal traces (pfh60, §96; traced by
  `scripts/export_pnets.py`) are loaded once at `load_model` and shared by all session copies, never run themselves.
  A file that is not the base trace's personal version fails loudly at load time.
- Every window's network input of the two SessionBN input groups is buffered in the session.
- When Route3 routes window t to person j (ramp r > 0, from either gate), session copies of j's 30 traces are made
  on first use and run on the buffered windows they have not seen (chunks of 32; StreamBN statistics are causal
  within a batch), so their state equals a run over the whole stream. Their posteriors at t replace the base
  SessionBN posteriors in the **routed tail's** mix: mix' = mix + Σ_m w_m (p_mj − p_m), with full replacement for any
  r > 0, as in the §78 dev form. The head slot, the routed rebalancing, SessionEM and TSEM, and the own-space stage
  (β 0.10) then run unchanged on mix'.
- The gate and the shadow tail never see the personal nets, so routing decisions are v5.1b's. Every window before a
  session's first routed window, and every window of an unrouted session, equals v5.1b.
- A person switch mid-session works: the new person's traces are created lazily and caught up from the buffer; a
  lagging person is caught up when routed again.
- Fallback: any failure on a window leaves v5.1b's mix for that window. A person whose traces failed mid-update is
  not used again in the session. A second stderr line per session is written next to the own-stage line:
  `[pnets] windows N routed R swapped S errors E [first: …]`.
- The 24 FBCNet/FBLight personal traces go through `SharedFIRSJit` (FIR bank fingerprint checked) and share the
  group's bank output; the 6 SincShallow traces use the generic SJit.

Builder: `scripts/build_v52.py` (copies v5.1b's files unchanged, checks every v5.1b array is byte-equal after
re-saving, requires exactly the 300 expected traces). Checker: `scripts/check_v52.py`.

## Package

`runs/sealed/cand_v52_pnets`: 333 files, 102,584,179 bytes on disk; upload ZIP 85,471,739 bytes
(`643365016775465c…`), against v5.1b's 62,308,135. `submission.py` `5e59900f77a4a201…`, `weights.npz`
`0b2375777fb139ab…`, traces (concatenated SHA-256 of the 300 sorted file hashes) `e29397ae230e6393…`.

## Checks

`scripts/check_v52.py runs/sealed/cand_v52_pnets runs/sealed/cand_v51b_own` (Mac CPU via `run_capped.sh`, 18 min;
log `runs/check_v52_local.log`, JSON `runs/sealed/cand_v52_check.json`, copied to `evidence/v52_build_12nov/`):
**V52 CHECK OK.** All 14 public sessions, 2,655 windows, one window per call, both packages:

| Check | Result |
|---|---|
| Shadow tail (gate input) equal to v5.1b on every window | yes (bitwise) |
| Routed output equal to v5.1b before each session's first routed window (window 10 everywhere) | yes (bitwise) |
| Same windows routed as v5.1b; swapped = routed; errors | 2,515 / 2,515; 0 errors |
| Personal posteriors vs the personal traces freshly loaded and run on the session's inputs from fresh input streams | max 5.4e-5 (< 1e-4) |
| Buffered inputs vs freshly recomputed inputs | 0.0 |
| Routed mix vs mix + Σ w (reference − base) | max 1.3e-6 |
| Not a no-op: routed outputs differ from v5.1b | max 0.18, 4 labels changed (in-sample accuracy 0.9917 both) |
| Gates forced off (τ = ∞, three sessions): outputs equal v5.1b, personal calls | equal, 0 calls |
| Person switching (forced schedule over five people, two sessions): posteriors / mix vs reference | 3.3e-5 / 7.5e-7 |

The 5e-5 differences are float32 batch-shape effects (the package catches up in chunks of up to 32 windows, then one
window at a time; the reference runs the session in one batch). There is no unrouted session in the public data;
the forced-off pass covers that path. Stream time in the check: v5.2 382 s, v5.1b 284 s (1.34×). Loading the 300
traces adds about 2 s.

## Official replay (Mac CPU, `scripts/replay_package.py PKG OUT cpu`)

| Package | Score (in-sample) | Files unchanged | Wall time |
|---|---|---|---|
| cand_v52_pnets (`runs/sealed/cand_v52_pnets_replay_mac`) | 0.9946208557 | yes | 117.3 s |
| cand_v51b_own, paired run right after (`runs/sealed/cand_v51b_own_replay_mac_pair`) | 0.9932767697 (= earlier) | yes | 86.4 s |

All four replay sessions printed `[pnets] … swapped = routed, errors 0`. The score is in-sample (the personal nets
were fine-tuned on these sessions) and says nothing about the sealed value.

**Runtime estimate:** 1.36× v5.1b on the Mac replay, where 95 % of windows are routed (the worst case: on routed windows
30 extra traces run, plus a one-off warm-up of about 10 windows). Sealed v4.1 ran 18.8 min and v5.1 is about 20 min, so
v5.2 is at most about 27 min if the CPU ratio carried over. On the A100 the nets are a smaller share, and new people
are not routed. Limit 60 min. Not measured on the A100 or the worker image (Leonardo/Galvani).

## Deviations and findings

1. **The personal nets do not all sit on the shipped base nets.** In the frozen layers all ten people's traces are
   identical to each other but differ from the package's shipped traces. Functional test on one session (P06):
   personal trace with the shipped output layer and BN affine copied back in, against the shipped trace:
   - FBCNet k2/k5 (12 traces) and FBLight (6): **identical** (max |Δp| < 5e-4; the spatial-bias and running-mean
     shifts cancel). These are fine-tuned from the shipped nets.
   - raw FBCNet (6) and SincShallow (6): **retrained twins**, mean |Δp| 0.011 / 0.007, max 0.09. This matches §96's
     Leonardo build, which retrained the SessionBN nets whose checkpoints were not local.
   Effect on the mix: those 12 members carry weight 0.0111 each, so a twin difference moves the mix by at most about
   0.01. The swap therefore applies personalisation plus a small base change for 12 of 30 members. Fine-tuning from
   the shipped traces' own weights would remove it, but it was not done (Galvani holds the original checkpoints).
2. Personal traces only replace members on routed windows of the routed tail. The dev form (§78) also swapped them in
   the shadow (gate input) on seen sessions; a package cannot do that before it knows the person.
3. Full replacement at any r > 0, not scaled by r (the task's definition and the §78 form).
4. `scripts/export_pnets.py`'s own per-trace check (traced vs eager personal net on the person's public sessions,
   < 1e-4) ran on Leonardo. Logs in `runs/pnets_jit_logs/pexport-*-task*.out`: 300 JSON lines for 300 distinct traces,
   maximum difference **0.0**, and all 24 tasks printed DONE.
