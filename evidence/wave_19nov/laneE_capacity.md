# Lane E · Unused runtime as variance reduction (capacity arms on a copy of cap250)

Registered Thursday 19 Nov 2026 ~13:30 CET BEFORE any arm was scored. Source: strategy_claude.md candidate C (FT-max),
bugs_codex.md §4 (cap250 used 2,555 of 3,600 s; neural time guard reads 1e9; FT capped at 250 windows; Eye/Rest 3+3
seeds; 45 of 65 neural terms active). Hypothesis: runtime cap250 leaves unused converts to accuracy through variance
reduction (more independent seeds of members that already help), without changing any rule or stage.

## Code and arms (all built from the signed cap250 archive `9c4d5538…ea3b` by `lane_e/make_arms.py`)

Copy `submissions/lane_e_capacity/submission.py` (cap250 bundle `37c666a8…fb69f` + one key-gated change; the original
is untouched). Change: key `v6_ft_extra` (n_slots, R-1) adds fine-tuned replicas per family; each replica is
fine-tuned on its own input stream (generator seed + r) and the family posterior moves by the MEAN of the replicas'
(fine-tuned − frozen) changes. Without the key the code path is cap250's expression. Everything else is keys:

| Arm | Change vs cap250 | Mechanism |
|---|---|---|
| E0 | none, `v6_ft_session_s` 1e9 → 150 | identity / guard null; must equal cap250 bit for bit on every window |
| E-a | Eye 3→6 seeds, Rest 3→6 seeds (`eye_s3-5`, `rest_s3-5`, trained 19 Nov on Galvani with the shipped recipe `scripts/train_v6_members.py`, all ten people; jobs below) | average of more seeds of two members that carry the +1.4 new-person gain: lower seed variance of the eye and rest posteriors |
| E-b | FT replicas 4→8 (seed-1 member of each family: nets 48, 36, 54, 60; delta averaged with seed 0) | the FT change of a family (the only sealed-confirmed mechanism, +1.01) is the output of a 5-step Adam refit from one seed; averaging two refits halves its seed noise |
| E-c | 10 of the 20 zero-weight nets (nets 10–19, FBC) set to the raw weight 0.025 of nets 0–9 (55 of 65 active; ≈ 21 % of the mixture mass after normalising) | more members; nets 24–33 (sinc / FBC variants, exports this bundle cannot run: `build_net` raises) stay off, so "all 65" is not buildable without porting code |
| E-d | E-a + E-b | |
| guard | `v6_ft_session_s` 1e9 → 150 s in every arm. It is cap250's own per-session limit (refit seconds from the session's second refit, reset with the session; nothing crosses a session; V6_BUILD.md §6.2i). Past it no new refit, the last fine-tuned nets keep predicting | a long or slow session degrades gracefully instead of timing out |

Limits of the guard, stated now: it bounds a session, not the run (V6_BUILD: 106 sessions × a limit is not a run bound), and
it makes outputs depend on machine speed only on a session that exceeds it. Cap250 measured 5.9 s (190 windows) and
23.6 s (378 windows) of refit seconds per session on an A100, so 150 s binds at ≳ 3× the E-d worst case.

## Predictions (pp of balanced accuracy, arm − cap250; 80 % prior in brackets)

Evidence used: aux430 (3 → 9 eye/rest seeds: +0.175 held-new dev, interval −0.04…+0.40); GPU run-to-run posterior
spread 0.034; FT effect +1.38 new dev / +1.01 sealed but only +0.04…+0.49 on nine Scherer people; the instrument's null
(cap250 vs cap250_seed1, NI-V) sets the noise floor.

| Arm | FULL-SYSTEM Scherer, full-session shuffle, 8 fresh orders (primary) | Within-run (secondary) | Added A100 time per unrecognised 190-window session | Projected sealed runtime (A100, 106 sessions, 57 % unrecognised) |
|---|---|---|---|---|
| E0 | 0.000 exactly (bitwise) | 0.000 | ≤ +0.1 s | 2,555 s |
| E-a | +0.08 (−0.25, +0.40) | +0.05 (−0.30, +0.40) | +0.3 s (+0.1, +1.0) | +60 s (2,615) |
| E-b | +0.10 (−0.25, +0.45) | +0.08 (−0.30, +0.45) | +6 s (+3, +12) | +360 s (2,915; 80 %: 2,750–3,300) |
| E-c | −0.05 (−0.45, +0.30) | −0.05 (−0.45, +0.30) | +1.0 s (+0.3, +3) | +60 s (2,620) |
| E-d | +0.15 (−0.25, +0.55) | +0.10 (−0.30, +0.50) | +6.5 s (+3.5, +13) | +400 s (2,955; 80 %: 2,800–3,350) |

Held-new P01–P10 (FULL-SYSTEM DEV): not scored for E-b/E-c/E-d (needs 10 leave-one-person-out fits per arm and the 15-job
corrected chain per order, which does not exist for these arms). For E-a the historical chain exists for the eye half (eye
LOPO seeds 0–8 in every order) and for both halves in within0/within1 (rest seeds 0–8); the partial E-a run on it is
NOT run (the historical chain's pinned inputs and ~15 asserts would need re-registration; no time before the VPN switch).
Held-new evidence for E-a is therefore the already measured aux430 bracket (3 → 9 seeds, +0.175 [−0.04, +0.40]), not a new
score; E-b/E-c/E-d have no held-new number.

## Rules (stated before scoring)

- Instrument: NI-V (evidence/wave_19nov/NIV.md owner; `niv/score_candidate.sh` conventions): Scherer 9 people ×
  2 visits, every member re-streamed in delivery order, full-session shuffle seeds 71000–71007 (headline) and within-run
  (secondary), paired against the exact cap250 run (tag `cap250`, job 3136444), people-and-order resampled 80 %
  intervals. The NI-V cohort-B scale: Scherer is external (no held-new training exclusion issue: no Scherer row is in any
  fit). Report new-person only; Lane E arms are not returning-person levers (FT stops at recognition).
- An arm is **supported** if the paired 80 % lower bound > 0 on the headline AND the within-run sign is not negative AND
  the projected A100 runtime ≤ 3,300 s. **Refuted** if the 80 % upper bound ≤ 0. Otherwise **undecided** (the instrument's
  resolution is about ±0.5; no arm is closed on this cohort alone, and held-new dev is not available for most arms).
- E0 ≠ cap250 on any label, or an `[v6]` error count > 0, or a guard hit (`over` > 0) in a scored stream, voids the run.
- Runtime: warmed, synchronised, ABBA per fixture on A100 (and RTX 2080 Ti), fixtures of 120 / 190 / 378 Scherer windows
  forced unrecognised; added seconds per fixture vs cap250; projection = 2,555 s + 106 × 0.57 × Δ(190) (central) and
  + 106 × Δ(190) (all unrecognised). The 3,300 s ceiling applies to the central A100 projection; RTX shown beside it.
- Guard demonstration: E-d with `v6_ft_session_s` = 2 s on the 378-window fixture must show refits stop (`over` > 0),
  0 errors, every window labelled, wall time below the unguarded E-d, and labels still agreeing with it on ≥ 90 % of
  windows (a degraded session is still a decoder, not a crash).

## Jobs and what was actually run (19 Nov, all on Galvani; release dirs ni26_lane_e_capacity_20261119 and ni26_niv_v1_20261119)

Pre-submission checks (Mac CPU, Scherer windows, 45 unrecognised windows, `lane_e/smoke.py`): E0 == cap250 bit for bit
(labels and posteriors, guard key 150 included, 3 refits); E-b fine-tunes 8 nets (ft_s 24 vs 12, 1 label differs of 45, no
errors); E-a (6+6) and E-c load and run (0 labels differ of 45, outputs move 0.06–0.09); guard demonstration, E-d with
`v6_ft_session_s` = 2 s, 100 windows: 2 refits then `over` 6, 0 errors, all windows labelled, 42.5 s vs 200 s unguarded,
labels agree 99/100. CPU only: not a runtime estimate (CPU refits are ~10× an A100's; CPU wall: E-a +0 %, E-c +5 %, E-b +55 %).
Finding: "all 65 neural terms active" is not buildable: nets 24–33 are sinc/FBC exports this bundle cannot run (`build_net`
raises), so E-c is 55 of 65 (nets 10–19). Extra Eye/Rest seeds 3–5 trained with the shipped recipe on all ten people
(Galvani 3136617: eye train acc 1.0; rest in-sample .990–.994, as shipped .990–.992). The rest input-stream Gram differs from
the shipped one by 3e-10 of 2,269 (summation order), accepted at rtol 1e-9; the package keeps the shipped one.

| What | Job |
|---|---|
| Eye/Rest seeds 3–5 | 3136617 (members/ in ni26_lane_e_capacity_20261119) |
| Build E0 Ea Eb Ec Ed Ed_g2 from the signed archive (sha `9c4d5538…`) | 3136812 done; zips pinned in zips.sha256 and the NI-V inputs.sha256 |
| Timing A100 / RTX 2080 Ti (cap250, E0, Ea, Eb, Ec, Ed, Ed_g2; F120/F190/F378, ABBA, warmed) | 3136813 / 3136814 → out/timing_*.json |
| NI-V B-new Ea / Eb / Ec / Ed (36 tasks each, full + within, seeds 71000–71007) | 3136815 / 3136817 / 3136819 / 3136821 |
| On-cluster collectors vs cap250 (3136444), null cap250_seed1 (3136445) | 3136816 / 3136818 / 3136820 / 3136822 → report_laneE_<arm>_vs_cap250.json |
| E0 identity run (person 0) + bitwise check vs cap250 | 3136823 → 3136824 → out/null_laneE_E0_vs_cap250.txt |

Leonardo variant ready, not submitted (`lane_e/leo_release.sh`; A100, re-scores cap250 there as the same-hardware reference
and uses E0 as the null). Scorer: NI-V (`niv/*`, as on 19 Nov 13:00), not the corrected Friday scorer. Cohort A (held-new
P01–P10) not scored. No result yet: the verdicts above are open until the collectors write.
