# Lane FT-labels · v6_ft_labels flipped in the real fold packages (audit item 2)

Registered in `docs/LEDGER.md` ("Lane FT-labels", with amendment and smoke result) before any wave was scored. Scripts `scripts/overnight21/ftl_*`. Run dir (Leonardo, account AIFAC_P02_1636) `/leonardo_work/AIFAC_P02_1636/ni26/cap250/runs/laneFT-labels`.

## Setup facts (found while smoking)
- First smoke 64986849 (hardcb_safe, flag only) **failed loudly**: `HardWaveEnsemble.__init__` raises "expected the original cap250 raw FBCNet layout" when `v6r` is set. A weights-only flip is not a valid hardcb arm; the Opus audit's "the switch already exists" holds for stock only.
- Arms: S stock + flag; H1 hardcb_safe + flag with only that clause removed (the label chain reads the stock raw mix); H2 H1 + a teacher that reads the hardcb mix (hard pair + CBraMod proposal, cached per window). `submissions/*/submission.py` untouched; patched copies live in the run dir.
- Prep job 64987062 (all source flags asserted 0, copies 1, 1,046 other arrays bit-identical). Smoke 64987128/33/37/49: S/H1/H2 change 5/5/4 of 190 labels (P01, 71000), H2 vs H1 3, T0 (flag 0) 0.

## Test A result (Sat 21 Nov, ~00:10 CET; full tables `laneFT_labels/result_flip.md|json`, collector `scripts/overnight21/ftl_collect.py`)
Jobs: prep 64987062; smoke 64987128 (H1) / 64987133 (H2) / 64987137 (T0) / 64987149 (S); waves 64987160–64987777 (216 streams + 18 scorers; scorers S 64987416 / 64987551 / 64987592 / 64987646 / 64987690 / 64987745, H1 64987451 / 64987564 / 64987609 / 64987661 / 64987706 / 64987759, H2 64987523 / 64987579 / 64987629 / 64987675 / 64987722 / 64987777 for orders 71000, 71004, 71008, 71012, 71016, 71020), all 237 jobs COMPLETED. Orders 71000–71023 (24, six chunks of four), ten held-new people + returning P04/P10, flipped arm minus its unflipped reference (production stock/hardcb for 71000–71003, overnight stock/hardcb_safe for 71004+).
Intervals are paired 80 %, 10,000 draws, seed 0, with people carrying their official cell counts (2 cells each, P04 and P10 6 each; the mean over orders then equals the official cell mean, asserted); "nested" also resamples orders.

| Contrast (new people, pp) | mean | people-only 80 % | people-nested 80 % | orders > 0 | P04 / P10 as new | returning P04 / P10 |
|---|---:|---|---|---:|---|---|
| S: stock + flag − stock | **+0.022** | [−0.10, +0.16] | [−0.12, +0.18] | 11/24 | +0.20 / −0.39 | +0.03 / 0.00 |
| H1: hardcb_safe + flag (stock-mix teacher) − hardcb | **−0.044** | [−0.16, +0.10] | [−0.19, +0.13] | 11/24 | −0.03 / −0.48 | −0.01 / 0.00 |
| H2: hardcb_safe + flag + hardcb-mix teacher − hardcb | **+0.059** | [−0.10, +0.22] | [−0.12, +0.24] | 13/24 | +0.35 / −0.42 | +0.03 / 0.00 |
| H2 − H1 | +0.104 | [+0.01, +0.19] | [−0.02, +0.22] | | | |

- Registered rule: all three **undecided** (upper bounds +0.18 / +0.13 / +0.24 are above 0, means below the +0.2 support bar). Nothing is refuted or supported. Predictions: S −0.10 (seen +0.02, inside), H1 −0.10 (seen −0.04, inside), H2 +0.05 (seen +0.06, inside); returning ~0 (seen ~0).
- The flag reaches the code path: 0.5–5 % of delivered decisions change per new-person fold (P04 4.3 %, P05 5.2 %, P10 2.3 %, P08 0.8 %), no fold with 0 changed. Returning P10 is exactly 0.00 in all three arms: P10 is recognised before FT starts at window 20, so recognised sessions never read the label chain (the P04 deltas, ≤ +0.03, come from the unrecognised prefix). The reading of the 0.2/0.8 mix is therefore ≈ 0.
- J (UNVERIFIED 0.57 new + 0.43 × [0.2 P04 + 0.8 P10], other returning people 0): S +0.015, H1 −0.026, H2 +0.037.
- Per-order spread is ±0.3 (S −0.41…+0.67); the H1-vs-H2 gap (+0.10) is the only contrast whose people-only interval clears 0, and it is the teacher choice (hardcb mix vs stock-mix), not the flag. Audit item 2's 0 to +0.5 prior is not confirmed: on the real fold packages the plausible gain is about +0.1 at best (H2), inside the noise.
- Not closed. Untested: FT labels from the *terminal* of the live chain (after rebalance/EM/TSEM/count, i.e. the actual final prediction) rather than the pre-rebalance v6 chain; label-source effects on returning visits after recognition; any change of FT dose. H2 is new code (cached hard/CBraMod proposal); a candidate would need a hardcb ZIP build, parity and runtime qualification, for an expected +0.06 (J +0.04).
- Side note for the overnight collector (`pkgcmp_collect.py`): its people bootstrap resamples ten people with equal weights, while the official metric gives P04/P10 six cells each and the other eight two; its headline cell mean is unaffected, the intervals are slightly mis-weighted. Fixed here, not there.
