# F6–F9 waveform results · 18 November 2026

All 160 Ferranti production fits and all four Galvani held evaluations completed by 15:08 CET. Results use ten source-excluded people, eight pinned orders and two seeds (460 and 1460). The exact historical cap250 reference for all terminal tests is 76.1039% full-system development balanced accuracy. P04/P10 here are held-new roles.

| Wave | Whole-decoder balanced accuracy | Paired effects in percentage points, seeds 460 / 1460 |
| --- | --- | --- |
| F6 PhysioNet motor pretraining | Authentic 43.319% / 43.898%, shuffled 42.786% / 42.131% | Authentic minus shuffled +0.533 [80% −0.565,+1.889] / +1.767 [+0.491,+3.120]. Motor recall contrast −8.977 / −10.657 pp. The whole-score gain and motor effect were undecided under the two-seed rule. |
| F7 supervised raw waveform | 47.097% / 47.277%, F3 chronological 42.145% / 42.855% | Raw supervised minus F3 +4.952 [+2.154,+7.285] / +4.421 [+1.718,+6.860]. The improvement held at both seeds. |
| F7 masked latent training | 46.542% / 47.124% | Masked minus supervised −0.555 [−2.094,+0.770] / −0.153 [−1.237,+1.028]. Undecided. |
| F8 source covariance style | Full covariance 47.282% / 46.272%, matched gains 46.667% / 46.753% | Full covariance minus matched gains +0.615 [−0.047,+1.297] / −0.482 [−1.258,+0.289]. Full covariance minus F7 supervised +0.186 / −1.005, both intervals spanning zero. Undecided. |
| F9 causal normalization | Fixed 48.070% / 48.591%, learned 48.297% / 48.136% | Learned minus fixed +0.227 [−0.028,+0.490] / −0.455 [−0.678,−0.177]. Fixed minus F7 supervised +0.973 [+0.078,+1.939] / +1.314 [−0.056,+2.771]. Both contrasts were undecided across seeds. |

Both F9 normalization arms beat their matched blank and same-age donor controls at both seeds. Fixed genuine minus blank was +1.118 and +1.946 pp. Learned genuine minus donor was +1.128 and +0.853 pp. Most of the gain accrued from window 41 onward. The exact first-window contrast was zero. The learned prior counts ranged from 15.825 to 21.874 against fixed 20. This is evidence of useful session history in these standalone decoders.

Every complete-waveform terminal insertion remained below stock. The F6 authentic changes were −0.430 / −1.045 pp. F7 supervised changes were −3.141 / −3.358 pp. F8 full covariance changes were −3.048 / −3.186 pp. F9 fixed changes were −2.862 / −3.133 pp. No new package is supported by these results.

Galvani scored authenticated actor, source and order manifests and exact zero-change controls. Actual comparison jobs completed F7 at 15:02:50, F6 at 15:04:47, F8 by 15:04:51 and F9 at 15:07:54 CET. Raw artifacts are in Galvani releases `ni26_friday_wave_{motor,raw,style,norm}_20261118`. [F8 detail and artifact hashes](style_score_result.md) · [Ledger](../../docs/LEDGER.md)
