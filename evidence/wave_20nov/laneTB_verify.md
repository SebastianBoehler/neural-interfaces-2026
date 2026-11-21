# Lane TB-verify · adversarial check of the Team B blend (Sat 21 Nov 2026, 02:00–02:35 CET)

Registered in `docs/LEDGER.md` ("Lane TB-verify", c90a7dfb) before any score was read. Leonardo run
`cap250/runs/laneTB-verify-21nov`, code `scripts/laneTB/{verify,pkgcheck,verify_collect,basecheck}.py` (e8b814e7 + collector fix).
[Paired results](laneTB_verify.json) (sha256 `94bcf8d8…`).

**Verdict.** No leak found. Held-new replicated on 8 fresh orders. Returning is also positive but about a third of the
card's number: the four-order +1.82 was helped by order 71121 (+3.05). Hard-only gives the same gains as hardcb.

## 1 · Leakage (checked in code and on-cluster files, not taken from the card)
- Team B training reads only `D.train_rows(people, first_only)` (`cleanslate/teamB/train.py:fit`). There is no external bank
  (`CSB_EXT` unset; `pack["ext"] == ""`), no experts and no stored normalisation. EA whitening and SFN are causal per session.
  BatchNorm running statistics come from the training rows only.
- Per-fold rosters, recomputed from each `tb/<fold>/weights.pt` (`audit_<fold>.json`), match the training-log counts:
  held-new folds 2,465–2,468 windows with 9 people, the held person absent (new_P04 2,089 and new_P10 2,078: neither
  person's 3 sessions are present). ret_P04 has 2,278 windows and ret_P10 2,270. Each includes **only visit 1** of the
  returning person and all other people in full. `run.py:scored` asserts the same roster before scoring.
- Base fold packages `production-20261120-32a94ab7/hardcb/*`: all 14 F1 leak arrays (`ts2_*`, `v6_rest_*`, `v6_cf_*`, `em_ev_g`,
  `net34_*`) differ from the uploaded cap250 in all 12 folds (168/168), as do `rest_s0.pt` and `eye_s0.pt`. Uploaded
  hashes are `2f771a1f`/`567afb36`; ret_P04 has `f3bbcea5`/`2a06a4bd`. This is the clean instrument from redteam F1.
- Code and selection: the scored code is the repo's code (identical SHA-256 on Leonardo). Doses 15/25 % were registered.
  nine25 was picked from 4 arms on these 10 people, which is selection on the same people, not a leak.

## 2 · Replication: 8 fresh orders 71124–71131
Each order × fold × arm ran as a separate job, 96/96 COMPLETED. Smoke checks: on both bases, the base arm equals a stored tape
(hardcb Lane TB 71120, hard R1a-bases 71000; P01/1, 190 windows), nine25 changed 10 labels, and v6 ran on every window.
Paired 80 % intervals resample people, then orders, with official cell weights. 57/43* is the UNVERIFIED sealed mix.

| Contrast | Held-new Δ pp [80%] | Returning Δ [80%] | P04 / P10 | J | 57/43* |
|---|---|---|---|---|---|
| hardcb nine15 − hardcb (8 fresh) | +0.71 [+0.41, +0.98] | +0.74 [+0.43, +1.07] | +1.00 / +0.49 | +0.73 | +0.66 |
| hardcb nine25 − hardcb (8 fresh) | **+0.85 [+0.18, +1.39]** | **+0.61 [+0.33, +0.88]** | +0.60 / +0.62 | +0.73 | +0.75 |
| pooled nine25, 12 orders 71120–71131 | +0.85 [+0.24, +1.36] | +1.01 [+0.69, +1.36] | +1.11 / +0.92 | +0.93 | +0.90 |
| hard nine15 − hard (8 fresh) | +0.77 [+0.43, +1.09] | +0.47 [+0.20, +0.75] | +0.53 / +0.41 | +0.62 | +0.63 |
| hard nine25 − hard (8 fresh) | +0.97 [+0.31, +1.54] | +0.59 [+0.30, +0.89] | +0.43 / +0.74 | +0.78 | +0.85 |
| hard − hardcb (bases) | −0.07 [−0.19, +0.04] | +0.09 [−0.09, +0.33] | +0.27 / −0.09 | +0.01 | −0.05 |
| hard nine25 − hardcb | +0.90 [+0.22, +1.46] | +0.68 [+0.39, +0.94] | +0.70 / +0.65 | +0.79 | +0.80 |

- Held-new nine25 was positive on every order (+0.50 to +1.08). It is two-person heavy: P04 +2.63 and P05 +1.84, while P02 −0.56,
  P03 −0.89 and P09 −0.08. nine15 has the same J (+0.73) with a tighter interval and only P02 −0.02 below zero. The two
  doses are not separable here. nine25 leads on the unverified 57/43 point; nine15 is the lower-variance choice.
- Rule outcome. Held-new is **replicated** for both doses (lower bound > 0, mean ≥ +0.3). Returning is **replicated** in sign
  (lower bound > 0, P04 and P10 both > 0), but the predicted +1.2 came in at +0.61. Order 71122 was −0.39. With two people,
  the returning interval understates person-level uncertainty. Hard is within ±0.3 of hardcb, as predicted.

## 3 · External NI-V Scherer ([card](laneTB_verify_niv.md), jobs 64998390/64998410/64998411)
Team B is Scherer-free: PROTEUS only, `ext ''`. Comparison: hardcbB + nine25 vs hardcbB, 9 people, paired 80 %.
- B-new, full-session: **+0.01 [−0.29, +0.33]**. Within: −0.19 [−0.71, +0.34].
- B-ret, full: **−0.23 [−0.74, +0.27]**. Within: −0.10 [−0.86, +0.66].
- No veto: the −0.3 bar was not crossed. The in-lab gain also does not show up out of lab; every interval contains 0. NI-V magnitude is
  weak evidence (22/41 channels interpolated, base 48.6 %).
- Blend-zero smoke: 120/120 labels identical. Blend on changed 12/120.

## 4 · Hard-only base and budget
Team B gains as much on hard-only as on hardcb (table). Arithmetic from measured A100 parts gives 2,555 + 20,083 × (11.51 ms hard
actors, cb_speed.md + 7.82 ms Team B nine, laneTB timing) ≈ **2,943 s** for hard + nine. That leaves about 650 s against 3,600 s.
The RTX PRO 4500 timing is unmeasured. Limit: Team B re-embeds the whole prefix every 4th window, so its cost grows with
session length. It was measured on 189-window sessions; public sessions are 187–196 windows (from the fold roster counts).

## 5 · Package `final_stack_hardcb_safe_team_b_nine25.zip` (sha `af40e375…`; pkgcheck 64998513, GPU, order 71124)
- Inventory: 44 parent entries are byte-identical, `submission.py` changed, and `team_b.pt` (sha `40ccb568…`) was added. The
  source diff is only the `TeamBlendStack` class (`TEAM_B_MIX = 0.25`, members 3, signed SHA), the `_TEAM_B` code namespace
  (no mutable state), and the factory line. The dead-session guard `not self.dead` is present, as in hardcb_safe.
- Blend zero (`TEAM_B_MIX = 0.0` copy) against hardcb_safe on P01/1 and P04/2: 0 labels changed, max posterior Δ **0.0**.
- Blend on: the posterior moves by up to 0.14/0.12, so the weight is applied. **0 labels changed** on these two streams. Both are
  in the all-data package's own training set, where the base is already near-perfect. The fold packages, which use the same
  `build.revised` code, changed labels (smoke: 10 on P01/1). The job's label-change assert therefore failed by design of the
  streams, not the package.
- Session isolation: `fresh_model` deep-copies and calls `reset_state`, which resets the Team B encoder. The template's encoder was still
  empty after streaming, and P04/2 run alone matched its run after P01/1. A Team B failure injected at window 40 sent the
  session to the fallback (`dead`, 190 valid labels), with no crash.
- Team B bank roster: P01–P10 all sessions, first_only [], 3 × shallow/slow/FBC, 1,200 steps, no experts or external bank.
