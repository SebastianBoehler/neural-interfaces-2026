# F-C21gn · Run the FBLight selector on a live native worker

The executable prototype calls the exact stock Solver for each EEG window. It captures the actual native outer mix and recognition state, then advances an independent copy of RouteB, CountFilter and CausalPers. The alternate state starts at age zero on every session. It uses the real six-member FBLight posterior including the native warm correction. Both tails evolve independently, and the chosen emission feeds neither tail or the protected upstream. Stock is emitted through age 19 and again starting on the actual recognition frame. [Registered test and repeat guard](../../../docs/LEDGER.md#f-c21gn--live-shared-upstream-conformance-and-paired-runtime)

| Galvani release | CPU delayed P10 visit 2 | RTX G and normal recognition | RTX paired timing |
| --- | --- | --- | --- |
| `ni26_friday_c21gn_live_v1_20261119` | `3134394` passed | `3134395` hit independent bitwise prefix guard | `3134396` blocked |
| `ni26_friday_c21gn_live_v2_20261119` | `3134422` | `3134423` | `3134424` after both proofs |

V1 source SHA-256 `d7ba21ee105d020860ea9ec596f1963289671c1619ccaa8fb52c5c059eaf0bdd`. V2 `9b81b180ae71209a93807c01ac46c9cc5387dd2d6abaaf4e899c9fc29cc909d7`.

The exact CPU delayed-recognition proof passed at 06:47:50 CET. Both native stock and FBL2 terminal arrays matched the authenticated P10 visit-2 archive. The selector accepted ages 20–27 and emitted stock at age 28 itself. Twenty original optimizer steps, eight warm FBLight outputs and the 29-frame prefix all matched. This fixture uses the conditional returning source-fold package. V1 CPU receipt `b4afa82acc15247d5ff6b0130be49ca6132a0ed73505196ff94be5fe78ffc7c9`.

V1 RTX matched the independent native tails bitwise when they consumed the same live frames. The separate fresh truncated G prefix stopped at an original stock-stage maximum difference of 5.164e−9. The first difference occurred at age 20, when native warm fitting starts. V2 preserves that failure and separately measures independent numeric repeatability against stock-only repeats. Same-input comparisons and the deliberately delayed CPU checkpoint keep bitwise equality.

## Signed conformance and runtime

V2 passed on 19 November at 06:53:13 CET. The RTX G worker selected FBL2 on exactly 100 of 120 rows, changing four decisions before labels. The independent alternate branch used 100 actual adapted-family outputs and made five native CSP refits with 30 fusions. RTX P10 visit 3 recognized at age 10 and emitted stock on all 31 rows. Both entire same-input independent mirrored tails and same-input truncated prefix were bitwise. P10's independent RTX prefix was also exact. The independent G prefix preserved the first 20 frames exactly and all tested decisions and work. Its maximum stage difference was 6.356e−9. The untouched stock repeat differed by 7.943e−8, above the registered 2e−8 prior and inside the predeclared 2e−7 repeat guard. The v1 independent bitwise failure stands.

| Warm paired fixture | Original stock runs (s) | Full dual selector runs (s) | Mean ratio | Worst matched ratio |
| --- | --- | --- | ---: | ---: |
| Scherer G visit 1, 120 windows | 27.207 / 27.225 | 32.339 / 32.235 | 1.18634× | 1.18863× |
| PROTEUS P10 visit 3, 31 windows | 4.760 / 4.601 | 5.529 / 5.533 | 1.18170× | 1.20254× |

The registered runtime condition required both means ≤1.12× and worst ratios ≤1.15×. Both fixtures missed it despite passing native conformance. Measurements are on the pinned RTX proxy fixtures. Timing preserves the separate conditional CPU delayed result and provides no accuracy or official worker-limit claim.

| Receipt | Galvani job | SHA-256 |
| --- | --- | --- |
| `delayed.json` | `3134422` | `188321bb1271740fb36381058d6a9153e5d77ee7707905912a98e15c52d71ac3` |
| `smoke.json` | `3134423` | `d61af7d446602d8c643b4e2bf8ddde7041ae59028d6bf63a8d45d4141f3d31fc` |
| `timing_0.json` | `3134424_0` | `81e6252bc0c43c1712ddb7acf14a369452ea45f179d9f5ba581936e8b51d1f2b` |
| `timing_1.json` | `3134424_1` | `02365ba24ad8cf2dc87f996d99b3e33a994f522755ad8b810830b184c54e22f5` |

## F-C21gt · Stop the alternate branch when recognition becomes permanent

`RetiringSelector` keeps every alternate state through the frame before the live sticky recognition latch. On the recognition frame it stops both the extra FBLight mix and the alternate tail. The original stock Solver continues, and each new session starts with the full alternate again. The source class is in [live_gate.py](live_gate.py), SHA-256 `6af1cb7229dae99834e15a1564b931d9f8610503722a510b10d245119dea3b12`. [Registered test](../../../docs/LEDGER.md#f-c21gt--retire-the-alternate-tail-on-sticky-native-recognition)

The actual conditional P10 visit-2 replay stopped after 28 alternate states, selected only ages 20–27, preserved the original 20 warm steps and matched every full-dual and archived output bitwise. RTX P10 visit 3 advanced the alternate through age 9, retired on age 10 and emitted 31 exact stock decisions. RTX G never recognized. It retained all 120 alternate states, 100 eligible and 100 warm outputs, and five alternate native CSP refits with 30 fusions. Every selected vector matched the complete same-input dual reference. All first-20 and same-input prefix checks were bitwise. The separate G fresh repeat preserved exact decisions and had at most 2.098e−8 output movement versus 7.539e−8 measured within independent stock.

| Warm paired fixture | Exact stock (s) | Retired selector (s) | Retired / stock mean | Worst pairing | Verdict |
| --- | --- | --- | ---: | ---: | --- |
| Scherer G visit 1, 120 rows | 26.503 / 26.541 | 31.353 / 31.300 | 1.18115× | 1.18299× | Missed both thresholds |
| PROTEUS P10 visit 3, 31 rows | 4.705 / 4.692 | 5.091 / 5.063 | 1.08055× | 1.08215× | Passed both thresholds |

The symmetric P10 block also measured full dual at 5.803 / 5.650 seconds. The retired path was 11.34% faster than full dual. G stayed above the original limits because recognition never fires on that fixture. The combined timing requirement still failed. Both fixture results concern the exact RTX proxy only.

Immutable release `ni26_friday_c21gt_retired_v1_20261119`, source SHA-256 `714add28327fb564cf2bb7b00b7f5420b58402d15ab74a95fc9b9ca5668de96c`. The final timing block completed at 07:06:39 CET on 19 November.

| Receipt | Galvani job | SHA-256 |
| --- | --- | --- |
| `delayed.json` | `3134440` | `168cbdbac953b708cbe1b0e898cb260cc6280490ac29bcfe42e3293d36ad9671` |
| `smoke.json` | `3134441` | `cb019632465844fba246682c6041daa19f78d65b1be0a49d7b88a908eb22188d` |
| `timing_0.json` | `3134442_0` | `26ba7ffb14871ae1e91a745bf0ba84590ecf4995b63536ebf738436a782cafec` |
| `timing_1.json` | `3134442_1` | `307c0d71fac55d8f0f2aada057abe1843f6d3ff5ee00246324b27bedcc5ea3aa` |
