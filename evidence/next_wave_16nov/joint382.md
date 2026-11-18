# Joint training of the A6 waveform model · §382

On Galvani seed 382, training the spatial convolution and classifier alongside the learned session normalization changed held accuracy by −0.206 percentage points relative to training only the normalization layer. The 80% paired interval spans −0.924 to +0.528 pp. The jointly trained decoder retained a substantial benefit from the correct session history.

| Complete waveform decoder or reference | Official accuracy (%) |
| --- | ---: |
| Current full stack, JS | 81.236 |
| Current frozen stack | 81.216 |
| Original A6 with causal k20 | 60.785 |
| K · joint training with explicit k20 | 60.234 |
| F · learned normalization, frozen convolution and classifier | 59.911 |
| J · learned normalization, jointly trained convolution and classifier | 59.705 |
| J · current window with blank previous history | 53.813 |
| J · current window with source-donor history | 52.938 |
| Original population normalization | 53.830 |

| Contrast | Change (pp) | Paired 80% interval | People above zero |
| --- | ---: | ---: | ---: |
| J − F | −0.206 | −0.924 to +0.528 | 2/4 |
| J − K | −0.530 | −1.139 to +0.086 | 1/4 |
| J − original A6 k20 | −1.080 | −1.940 to −0.145 | 1/4 |
| K − original A6 k20 | −0.551 | −1.050 to −0.071 | 1/4 |
| J − population | +5.875 | +2.576 to +9.108 | 3/4 |
| J − J blank | +5.892 | +3.411 to +8.403 | 4/4 |
| J − J donor | +6.767 | +3.772 to +9.769 | 3/4 |
| J − frozen stack | −21.511 | −23.449 to −19.511 | 0/4 |
| J − JS | −21.531 | −23.568 to −19.447 | 0/4 |

**Interpretation.** J retained 84.5% of the original k20 benefit over population normalization. Its history effect exceeded the registered three-point prediction. Joint training missed the registered positive J−F criterion on this seed. The native k20 model also lost 0.551 pp after this CE refit. The highest observed standalone A6 score still came from the original k20 checkpoint. §381 C/D/DC scored 59.911/59.530/59.623%; all three paired J comparisons include zero. The four held people have been examined in prior studies. Results address this training recipe and population. P04 and P10 were training people here.

**Training.** All arms used the original six A6 source people, seed 382, 120 identical shuffled 48-window episodes and weighted cross-entropy. F and J started from identical §381 D weights; K retained the explicit k20 algorithm. The analytic filterbank stayed fixed. J's spatial convolution, normalization and classifier had positive initial maximum absolute gradients 1.6752/0.0844/0.2359. All three changed parameters. F updated just the normalization. No arm consumed a second waveform model or reference posterior at inference. [Source records](joint382_fits.json) · [Training kernel](../../studies/18nov/joint382/kernel.py)

**Validation.** P01/P02/P03/P09 received eight pinned within, full or run-shuffled deliveries. Each person's original static and causal posterior matched the historical CPU replay exactly. Maximum discrepancies in the independent learned one-step and prefix checks were 9.54×10⁻⁷ and 2.03×10⁻⁶. The scorer verified source exclusion and digests, then used the official interface/class metric and paired people-and-order intervals. Zero-change replay returned OK. [Prediction commitments](joint382_before_labels.json) · [Complete contrasts](joint382_result.json)

Galvani A100 smoke 3123330 and paired array 3123331 consumed 152 allocated GPU-seconds. Portable export 3123335 failed during Braindecode property introspection. Export 3123357 reused the native forward on its constituent modules and matched native source logits exactly, with maximum gradient difference 2.51×10⁻⁶. Both export attempts together consumed 39 A100-seconds. CPU2 actors 3123361/3123362 took 108–109 seconds each; scorer 3123363 took 119 seconds. Source revision b0cf5ed. The independently seeded Ferranti run uses the same actor and reports its own comparison.

## Independent Ferranti seed

All three H100 fits completed with seed 1382. The same Galvani CPU scorer evaluated the same four people and eight orders and reproduced the historical controls. J scored 58.125%, F 59.550% and K 58.356%. J minus F was −1.425 pp [paired 80% −2.207, −0.623]. J minus K was −0.231 [−1.272, +0.817]. Real history beat blank by +5.293 [+2.664, +7.882] and donor history by +6.892 [+3.965, +9.786]. J trailed original A6 k20 by 2.660 pp and the frozen full stack by 23.091 pp. Both seeds favor the frozen-body and explicit controls on point estimates. The four-person sample is shared across seeds. Seed and accelerator changed together. [Ferranti fits](joint382_h100_fits.json) · [Held predictions audit](joint382_h100_before_labels.json) · [Paired scores](joint382_h100_result.json)
