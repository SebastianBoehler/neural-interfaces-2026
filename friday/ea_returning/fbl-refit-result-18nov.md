# Returning FBLight age-20 refit

Galvani CPU job `3131487` passed D-EArF in 48 seconds and completed at 21:24:45 CET on 18 November. In the two fresh arms, causal delivery changed all 20 confidence weights used by the real cap250 FBLight refit. The largest change was 0.03840345. The 20 hard targets stayed the same. The fitted parameters differed, and the eight warm predictions changed by up to 0.00176645 in probability. Their class decisions stayed the same. The registered activity and confidence-effect predictions passed. [Archived route and fresh inputs](bridge-result-18nov.md)

## Fold and frozen actor

The pinned `P10_s23` seed-0 checkpoint comes from the original §281 FBLight guard. Its source mask contains 2,270 rows, including 192 P10 visit-1 rows and excluding all 385 P10 visit-2/3 rows. The original source recipe and log report the matching fold. Recalculating the saved alignment prior and scale from those source rows reproduced the prior exactly. The checkpoint stores weights, alignment and scale. Its format omits the original training-row hash.

| Session | First global row | Rows | Max eager difference from frozen QF | Max one-window difference from frozen QF |
| --- | ---: | ---: | ---: | ---: |
| P10 visit 2 | 1238 | 186 | 7.64e−7 | 7.13e−7 |
| P10 visit 3 | 1326 | 199 | 9.94e−7 | 1.014e−6 |

All 385 frozen decisions matched the authentic `full0` archive. The exported batch trace matched the eager actor exactly. Native single-window runs preserved the first 29 outputs exactly on independent prefixes and after streaming the other visit. PyTorch was `2.14.1+cu126` on Galvani CPU.

## Live refit

The smoke executes the `FineTune` class and `strong` augmentation directly from the authenticated cap250 ZIP, with the held-fold FBLight trace replacing the shipped full-source trace in memory. The stock Adam schedule and augmentation seed were shared. Each run started from the same frozen bytes and fresh state.

| Input arm | Visit 2 refit ages | Visit 2 warm ages | Max parameter update | Visit 3 refits |
| --- | --- | --- | ---: | ---: |
| Original §281 archive | 20 | 20–27 | 0.005020274 | 0 |
| D-EAr0 archived-input zero | 20 | 20–27 | 0.005020274 | 0 |
| Fresh recording | 20 | 20–27 | 0.005020274 | 0 |
| Fresh causal | 20 | 20–27 | 0.005020170 | 0 |

The two archive-input warm arrays matched bit for bit. On visit 2, each run made five Adam steps from the 20 earlier aligned windows and stopped at the externally verified recognition age 28. On visit 3, each stopped at recognition age 10 and emitted no adapted outputs. Reversing visits and arm order preserved the exact outputs, actions, confidence weights and generator states. In a separate check, changing the current input at age 20 changed that window's output and preserved the entire fit update and augmentation trace exactly. The packaged ensemble's transition and outer route were outside this single-family smoke.

## Complete returning replay requirements

The historical exact comparator still needs a verified Sinc seed-0 source checkpoint. The prior §281 candidate rebuild failed the frozen parity guard, and the §321 fresh Sinc checkpoint cannot establish historical identity. The EA and raw FBCNet peers need their own `full0` checkpoint-to-QF proof before a four-family replay. Their checkpoint and frozen banks exist. The earlier §321 FB comparison on `within0` exceeded the strict 1e−5 bound on CPU and A100. No complete returning `r15c250` warm or final archive was found in the historical QF/LPC banks.

A separate matched producer would need to export the live contrast and identity routing tuples for every window from each fresh shadow. D-EAr0 preserves routed owners and latch ages. The package's `RouteB` also needs the per-window strengths and the matching fold's personal head, TSEM prior and own-space transforms. For both arms it must run from the first window after consuming all four live family deltas with the original weighted mixer, matched eye and rest inputs, count filtering, and CSP. The existing §321 fold-injection guard is a separate engineering control that retained parts of the production fit. No complete returning replay or accuracy was computed here.

## Receipt

The Galvani release is `/mnt/lustre/work/gehler/gfh527/ni26_friday_ea_returning_fbl_v1_20261118/`.

| Artifact | SHA-256 |
| --- | --- |
| `fbl_source.sha256` | `44624592acd62cc1cf45c3e7d4725f0191a13ebe7a6c55b8c34cb4e80df266da` |
| `fbl_inputs.sha256` | `3ad404f970bb7dcfe087700025dfd17a86a666ae9246e5c3f7f9b470a1866b48` |
| `fbl_progress.json` | `7c64650470c70c3efd4a5c488464f4516ac6d30cdf253975adfbae6a17a6efc9` |
| `fbl_progress.npz` | `be35506efaa8ca9558a146e50fb7fc71b1b8187202ad909864cbcf29a1ffde5a` |

The NPZ contains session-local global indices and per-arm posteriors, with NaN outside the adapted windows. An independent reader verified the bank hash, every active-output mask and exact archive equality. No held gold was opened.
