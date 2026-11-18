# §428 Scherer montage reconstruction

PROTEUS uses M1 as the recording reference and FpZ as ground. The Scherer authors specify left mastoid reference and right mastoid ground. The two releases support the same intended reference location. Physical placement, contact impedance and amplifier transfer functions can still differ. Neither release provides both mastoid potentials to reconstruct that difference. [PROTEUS release](https://www.nemar.org/dataset/nm000290), [Scherer acquisition](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0123727), [Scherer file description, pp. 2–3](https://lampx.tugraz.at/~bci/database/004-2015/description.pdf).

The original exporter passes its three positional mapping arguments and leaves the optional re-reference unset. Its 30 native channel names follow the published order, using modern T7/T8 for historical T3/T4. The participant A MAT file supplies a numeric matrix without channel labels. Across the two montages 19 names coincide. The exporter estimates the other 22 model inputs using spherical splines. Earlier common-mode and reference experiments evaluated trained PROTEUS views, while this intervention changes only the 22 inferred Scherer inputs and evaluates both exact archives.

## Raw waveform checks

We recomputed the original 2,130 windows from nine people's MAT recordings. The largest difference from the frozen 41-lead export was 0.00000191 µV. All 19 measured target leads were bitwise identical in the alternative input. The replacement averages the three nearest measured sites using inverse squared Cartesian distance on the fixed standard montage. The weighted average preserves a spatially constant signal.

For each of the 19 overlapping sites, we removed that measurement and separately interpolated it from the other 29 native channels. Averaged equally over sites and people, held-electrode RMS error measured 2.556 µV for the original spline and 2.493 µV for the local average. The prediction favoring the spline missed. The positions of the 22 missing leads differ from the positions that can be checked.

## Execution

Galvani study directory is `ni26_montage428_20261118`. Real raw smoke 3124902, eight remaining preparations 3124919, lane setup 3124920, worker smoke 3124921, nine paired archive producers 3124922 and collector 3124923. The scorer reproduced the §411b control difference (+0.03968 pp) and returned an exact zero with a [0,0] interval on identical real records. SHA-256 of the alternative input is `3a4b41c4ad5d4a7634915c17f76473e78d6bec4641a266bbbaeaa45d84638a3c`. Original exporter SHA-256 is `bf0495c89e0ce80c911d26a511762ebb6125f25fbd93c3438352e459d58d9ad7`.

## Exact archive results

Changing the 22 reconstructed inputs reduced the external cap250 advantage by 1.128 pp [80% paired people/day/order interval −2.034, −0.245]. The registered prediction of a point change below 1 pp missed. The interval spans the −1 pp boundary, so the registered equivalence test remains undecided. Five of nine people shifted toward v5.1b.

| External Scherer cohort | Original spline, v5.1b / cap250 | Local average, v5.1b / cap250 | Change in cap250 minus v5.1b, pp [80%] |
|---|---:|---:|---:|
| Preassigned five | 50.786 / 49.964% | 49.405 / 47.988% | −0.595 [−1.732, +0.560] |
| Reserve four | 46.042 / 47.158% | 44.821 / 44.144% | −1.793 [−3.065, −0.469] |
| All nine | 48.677 / 48.717% | 47.368 / 46.280% | −1.128 [−2.034, −0.245] |

The waveform difference at the 22 inferred leads was 2.000 µV RMS across all samples. Of 4,260 paired deliveries, v5.1b changed 580 decisions and cap250 changed 605. All metadata arrays were identical across inputs. Body and ground differences remain physically possible, but the observed change follows directly from the spatial interpolation intervention.

All scheduled jobs completed and the production replay printed OK. The nine participants, two orders and two days were exposed in previous experiments. PROTEUS records all 41 model-facing positions, so these Scherer scores qualify the external instrument. They do not measure a change to the candidate on native PROTEUS data. [Paired metrics and held-site checks](montage428_result.json)
