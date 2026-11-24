# Cross-review · Opus on Codex (strategy_codex, bugs_codex) and Fable (design_fable) · 19 Nov 2026

Read-only. Main at 09e6627c (includes the Friday merge a7db5818). "Verified" = I re-read the cited file:line and redid
the arithmetic. Old ledger = `git show a5843492:docs/LEDGER.md`.

## 1. Codex claims checked against the repo

| Claim | Verdict | Evidence |
|---|---|---|
| EA correction sizes: within0 Δadv −0.374 [−0.714,+0.050]; full0 common-source +0.301 [+0.004,+0.731]; full0 Sinc-only −0.222; cap250 absolute drop 0.15–0.67 | **Verified** | `friday/causal_replay/within0-overnight-queue.md:61-64`, `runs0_replay.md:47-50`, `complete-sinc-result-18nov.md:3-8`. 78.324−73.739=4.585 and 2.546−0.667=1.879 also check. |
| Count-filter latches: rec/runs0 14/14 at 40, within0 14/14 at 40–80, full0 4/14 at 40 | **Unverifiable from files** (Codex's own sweep; no receipt in repo). Plausible but higher than the repo's measured rate | Old ledger §228c (:5997-6003): full shuffle 7/56 false latches (12.5 %), gain +0.03/−0.02, "cost ≈ 0 in sum". 4/14 = 29 % on one order is p≈0.09 under 12.5 %; and the ZIP's thresholds are one all-public fit (`lean/tools/fit_cf.py:33-43`), so a public replay is in-sample. 308 changed full0 rows is verified (`runs0_replay.md:52`). |
| Dev→sealed decomposition 76.104 → 80.648 (56.6/43.4, P04/P10 20/80) → 2.339 pp unexplained | **Arithmetic verified**, inputs verified (`evidence/next_wave_16nov/joint450.md:3-11`) | It is a sensitivity, not a decomposition: equal P04/P10 gives 78.19, and sealed returning people are mostly the 8 one-session people never seen as "returning" in dev. The 2.3 pp residual is unidentified, not "training data". |
| Sealed/dev transfer ratios 0.41–0.94, cap250 0.91 | **Verified** | `docs/SEALED_UPLOADS.md:195-200,305,431`; 3.573/3.80, 0.546/0.75, 0.267/0.58, 0.361/0.89, 1.011/1.11. |
| Runtime: 2,554.9 s = 70.97 %, 1,045.08 s slack, v5.1b 2,313 s slack, plan ≤3,155 s | **Verified** | 3600−2554.92; `friday/DECISION_18NOV.md:64`. |
| RTX switch + raised limit (organiser), 4,800 s participant-sourced | **Verified** | `OFFICIAL_CONTRACT.md:71-77`, `DECISION_18NOV.md:64-69`. Missed implication: cap250's 2,554.9 s was uploaded after the 07:50 switch, so it is most likely already an RTX receipt (cap250/v5.1b sealed ratio 1.99 sits in the RTX-proxy band 1.97–2.06, not the A100 band 1.67–1.84). The 1,045 s slack is then real-worker slack, not A100 slack to be scaled down. |
| "+1.5–2.3 pp" diverse-ensemble ceiling, "anchored by +2.285 pp oracle" | **Unsupported** | The +2.285 is a label-aware pick among four fixed outputs with zero regressions (`DECISION_18NOV.md:51`). Every realizable selector measured so far recovered ≈0 of such oracles: F-C19 oracle +1.2/+1.8 vs causal rule −0.15/+0.03; F-C13 fixed 24 and broke 22 (`friday/OVERNIGHT_TOP3.md:17`); learned selector −0.053, simple comparator +0.208 then −0.020 (`RETROSPECTIVE_18NOV.md:13`). Codex's own budget prediction (+0.65) is the honest number; the headline range is not. |
| "+1.5–2.0 pp" exchangeable prefix adaptation | **Unsupported, and partly a re-form of existing stages** | TSEM already is "per-session Gaussian class templates … seeded by confident outputs" and SessionEM is per-session class templates (old STATE :365-366). Codex cites no measurement for the size. The repo's direct evidence on this family is negative: §252 classic personal on chain labels −0.98 (old ledger :7153-7163), §253 confidence-selected −0.01 (:7467), §259 offline balanced clustering purity 48 % vs chain 79 % (:7488-7492). |
| F1–F4 38.8–43.9 %, F9 48.6 %, F5 −3.3/−4.0 | Verified | `friday/RETROSPECTIVE_18NOV.md:7-13`. |

## 2. Codex bug findings ranked by likely sealed cost (expected, not worst case)

1. **#2 Hardware/graph drift (A100 vs RTX: 154/4,260 decisions, comparative gain −0.188 [−0.565,+0.172]).** Already
   paid by cap250 on sealed; it is the only finding with a measured, non-trivial size on the real worker class, and it
   grows with adaptive state (refits amplify early differences). It also decides whether any +0.2 pp candidate is real.
   Cheap probe: same ZIP with the TorchScript profiling executor off / fp32 matmul pinned, A/A on the RTX proxy.
2. **#3 Recognition latch retires FT and CSP for the rest of the session.** Verified at
   `submissions/lean_v66/submission.py:1552-1553` (`recB` never clears) and `:1354-1357` (CSP stops). FT+CSP are worth
   +1.65 pp on corrected within0 (`friday/cap250c18/result-19nov.md:5-13`), so a false accept on a new person costs
   ~1.5 pp on that session. At the gate's 1.9 % false-accept rate (old STATE :368) expected sealed cost is ~0.02 pp;
   it only matters if sealed false accepts are far above public. Lane B2 tests it.
3. **CountFilter under full shuffle (Codex's "highest-impact gap").** Repo already measured it: full shuffle ≈0
   (+0.03/−0.02) with 12.5 % false latches costing ≈0 (old ledger §228c). Expected sealed cost ≤0.1 pp either way;
   Codex ranks it first without citing §228c. Lane B1 predicts +0.15; I expect |Δ| < 0.1 on full orders.
4. **#5 CSP two-class fit puts log(1e−6)×0.10 ≈ ×0.25 on the absent class** — verified (`:1333-1366`); real collapse
   amplifier, but needs a ≥90-window prefix whose chain argmax never hits one class: rare except P04-like people.
5. **#1 dev refit/count instrument ≠ ZIP** (AdamW vs Adam, fold-specific vs all-public count fit, cap at 250 vs refit at
   250). Costs no sealed score directly; costs decision quality for any FT/count change scored on the dev bank.
6. #4 fallback masking and #6 montage/NaN sanitising: no evidence of sealed occurrence; correctness hygiene only.

## 3. What Codex missed

- **Force_Best and failed-run rules** (`docs/SEALED_UPLOADS.md:18-20`): the board keeps the best score and "a failed run
  uses no quota" (unverified for timeouts). The last slot is an option, so the selection rule is "highest expected
  upside that is worker-safe", not "smallest-risk hard-only +0.2". Codex's plan optimises the wrong objective for slot 7
  (Fable gets this right, design_fable §0.5).
- **Realizable/oracle ratio ≈ 0** across F-C13, F-C19, D1, learned selectors — the single most important number for both
  of Codex's step changes, never stated.
- **The adaptation family's existing negative screens** (§252/§253/§259) and that TSEM/SessionEM already occupy the
  "soft class-conditional prototype" slot.
- **Scheme dependence of cap250 itself**: corrected advantage +2.55 within0 vs +0.67 full0 (`OVERNIGHT_TOP3.md:15,21`);
  F-C18's +1.65 FT+CSP value is within0-only. If sealed is full shuffle, the adaptation stages are worth less than the
  within0 receipts show — Codex's bank screens should run full orders first (Fable D3).
- RTX receipt point above (slack is real-worker slack).

## 4. Fable checks (for the adjudication)

- 74–78 % chain pseudo-label purity: verified (old ledger :7153, §252, r15 chain, within-session test runs). 88 % at
  n≥100 for +1: verified (:7467). Caveat: derived for a from-scratch classic FB-TS-LR personal model on hard labels.
- Error budget: uses returning ≈90 %; joint450's 20/80 returning is 86.6 → new-person sealed ≈80.2 %, not 78. The
  conclusion (new people carry ~2/3 of the error) survives; the "+5 pp on new people for 0.86" is ±1.
- "Label-free adaptation cannot": too strong as stated. F-C18 shows label-free FT+CSP adds +1.65 pp [+0.67,+2.50]
  inside cap250 (within0, reused people, one order). Right in the narrow sense that matters: what is left after
  TSEM/SessionEM/FT/CSP is bounded by the same chain purity, and nothing has raised purity since.

## 5. Adjudication

- **Ensemble (+1.5–2.3).** Codex's ceiling is an oracle number; realizable fraction measured so far ≈0. Fable does not
  dispute ensembling — its D1 *is* a diverse member. The dispute is whether a new member exists whose errors are
  independent enough; no current bank shows it. Codex's own +0.65 is the defensible prior.
- **Class-aware prefix adaptation (+1.5–2.0).** Fable is right on evidence: three direct negative screens, an existing
  TSEM stage of the same form, unchanged purity. Codex is right that F-C18 proves adaptation is not zero — but that gain
  is already in cap250. Both agree on the lever: only a more independent teacher raises purity (Fable D2 = Codex #2 with
  a second teacher). So the adaptation idea is downstream of the member idea, not a parallel step change.
- **Decisive result tonight (CPU, 1–2 h, existing corrected banks):** the **offline whole-session transductive oracle**
  that Codex itself proposes as #2's screen — refit a source-anchored, shrinkage-regularised soft class-conditional
  Gaussian model on *all* windows of each session with cap250's own posteriors (no gold), score vs cap250 on corrected
  full0 and within0, report pseudo-label purity per session fifth. It upper-bounds any causal prefix form in that space.
  **< +0.5 pp on full0 → Fable right, drop #2. ≥ +1.5 pp → Codex's ceiling is real, build the causal form.** Pre-register
  before running. Run the same oracle once more with chain∩best-Lane-C-member agreement labels: that one number tells
  whether D1+D2 (Fable) lifts purity toward 88 %.
