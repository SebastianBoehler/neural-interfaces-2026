# Data inventory for the data-scale front (15 Nov; from docs only, clusters and file sizes not checked)
Target: PROTEUS, 41 ch, 120 Hz model-facing, classes MI (non-dominant hand) / SUB / WORD; 10 dev people; CC BY 4.0. Public data allowed if its terms permit (STATE:55).
| Corpus | People / sessions | Classes | Channels | Licence | Tried |
|---|---|---|---|---|---|
| Scherer 2015 (BNCI 004-2015) | 9 patients, 2 days, 2,130 windows | MI/SUB/WORD, the same 3 | 19 real, 22 spline-interpolated; no band below 4 Hz | CC BY-NC-ND, cleared 11 Nov; declare in write-up | §7 +1-3 alone, chain 0; §18 teacher +3.3 alone, -1.0 chain; §21 -0.9..-1.1; **§53 native 19 channels -3.2..-3.7**; §191 pooled SessionBN negative; §264 within 46.6 = cross 45.2 |
| Shin 2017A | 29 / 3 | MI L/R | n/s | n/s | teacher; §264 within 63.8 |
| Shin 2017B | 29 / 3 | arithmetic vs rest | n/s | n/s | teacher, cog1 nothing; §264 within 77.5 |
| PhysionetMI | 109 / 1 | L/R/feet | n/s | n/s | teacher; §216 |
| Schirrmeister HGD | 14 / 1 | L/R/feet | n/s | n/s | teacher |
| Lee 2019 | 54 / 2 | L/R | n/s | n/s | teacher |
| Cho 2017 | n/s | MI | n/s | n/s | teacher |
| Dreyer 2023 | 87 | MI | n/s | n/s | pretraining only; §189 collapsed |
| Shin 2018 WG | 26 | word generation vs rest | n/s | n/s | raw zips only on Galvani, need user 2FA; cog2 never ran |
| ds007554, Zyma 2019 | 24 / 36 | MI+SUB separate / SUB only | n/s | n/s | named in §189 only |
| Stieger 2021, BNCI2014_001 | n/s | MI | n/s | n/s | not tried / smoke only |
Classes beyond MI exist only in Scherer (SUB+WORD), Shin 2017B (SUB), Shin 2018WG (WORD, blocked), ds007554, Zyma (SUB). Docs name no BNCI H2020 mental-task set and no Graz corpus beyond PROTEUS's own runs.
Reading: the native-channel Scherer subset was tested in §53 (worse, not better, than the interpolated export), so Scherer is hard for reasons beyond interpolation (patients, device); within-corpus 46.6 shows weak signal in itself. Open: Shin 2018 WG needs the user's 2FA copy.
