# F36c · Crossfade-trained raw pupils in corrected cap250

Completed 19 November 2026 at 10:46:58 CET. Same-trial training lost to the exact A100 hard pupil on both corrected deliveries, with both paired upper bounds below zero. Cross-trial training lost to hard and same-trial on full0. Its within0 contrasts remain undecided. Neither treatment met the registered support rule. [Protocol](prereg-19nov.md)

| Full-system DEV contrast | Within0, pp [paired 80%] | Full0, pp [paired 80%] |
|---|---:|---:|
| Exact hard − stock | +0.425 [+0.157, +0.725] | +0.374 [+0.223, +0.525] |
| Same-trial − stock | +0.192 [−0.188, +0.610] | +0.252 [+0.123, +0.383] |
| Same-trial − exact hard | −0.233 [−0.444, −0.011] | −0.121 [−0.227, −0.012] |
| Cross-trial − stock | +0.301 [−0.042, +0.696] | −0.125 [−0.448, +0.215] |
| Cross-trial − exact hard | −0.124 [−0.299, +0.078] | −0.498 [−0.717, −0.266] |
| Cross-trial − same-trial | +0.109 [−0.048, +0.305] | −0.377 [−0.617, −0.127] |

Stock scored 78.3241% within0 and 73.7390% full0. Exact hard scored 78.7492% and 74.1125%. Same-trial scored 78.5162% and 73.9910%. Cross-trial scored 78.6252% and 73.6142%. The 4,000 paired bootstrap draws resampled the ten reused public people within each order. The F36 fits excluded each scored person. Other cap250 components and calibration retained their original scope.

Relative to stock, same-trial net correct motor/subtraction/word decisions were +1/+3/+1 within0 and +4/−5/+8 full0. Cross-trial gave −1/+8/+1 and +1/−5/+1. Same-trial P04/P10 net changes were +5/−5 and +1/+2. Cross-trial gave +4/−4 and −4/+2. The same-trial standalone points over hard of +0.389/+0.444 pp did not transfer through this fixed raw slot. [Standalone F36](../fbc323_crossfade/result-19nov.md)

Fresh recording-order `replay/check.py` matched all six original bases. The 20 independent pupil replays `3135713` reproduced signed F36 decisions and bitwise F-C22h hard posteriors in both orders. The 168 first-window and 168 interleaved-reset checks passed, with maximum posterior prefix drift 1.034e−6. P01 pupil smoke `3135705` and 379-row native full0 pilot `3135714` passed. Both complete native jobs `3135715/3135716` matched stock/identity and prior hard across all stages. Each arm executed 126 CSP refits and 1,395 fusions from strictly past data. Job `3135717` sealed the decisions before labels.

| Artifact in release `ni26_friday_f36c_held_20261119` | SHA-256 |
|---|---|
| Pupil manifest | `577fddd5a31ddc6248e7e8ca4679acb8a948f543f726ee06c071c1693c61db83` |
| Whole manifest | `f346e2cb9d1de08a93c631fd46be50aa978cb129ae15ac208dcb4a7979f844a1` |
| Native smoke | `e1fde8a149921cb72fce1e8ccef9d3043d6c27321758e1bade612a2118e7b3ba` |
| Within0 bank | `9b204e4bca9f2fef47880469e1344509011cfb1f481a7897745be0e331f9e773` |
| Full0 bank | `4532260e50a3565cff240a33ee173eecd4cda98a5baf8e66eff91c20b58b2499` |
| Blind lock | `7f63e999fc322276c1d8120e7adbf8965e140847e6f2ca422610ee36c893171d` |
| Official score | `4552919148d10ad3af6896249c2658bd2d75b786585993cc93937c40423d4643` |
