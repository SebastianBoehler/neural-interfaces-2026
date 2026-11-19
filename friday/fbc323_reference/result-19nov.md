# F26 · Common-average referencing in the raw S1C whole decoder

19 November 2026, 05:33:58 CET · Whole-decoder dev · Ten fit-excluded PROTEUS participants and eight delivery orders

Common-average referencing (CAR) removed the common offset as designed. The held accuracy change versus identity was **−0.351 pp** at seed 0 and **+0.673 pp** at seed 1. CAR beat the energy-matched artificial geometry on both point estimates. All paired intervals straddled zero. This form stays undecided across seeds. [Preregistered test](prereg-19nov.md)

| Bounded whole balanced accuracy | Seed 0 | Seed 1 |
|---|---:|---:|
| Exact hard CE | 59.1613% | 58.8654% |
| Common-average reference | 58.8102% | 59.5385% |
| Norm-matched artificial reference | 58.5183% | 59.1576% |

| Paired contrast, pp [80% interval] | Seed 0 | Seed 1 |
|---|---:|---:|
| CAR vs hard CE | −0.351 [−1.077, +0.395] | +0.673 [−0.248, +1.955] |
| CAR vs artificial reference | +0.292 [−0.931, +1.834] | +0.381 [−1.173, +2.542] |
| Artificial reference vs hard CE | −0.643 [−2.056, +0.447] | +0.292 [−0.816, +1.215] |
| CAR vs hard CE, ordinary inference | −0.141 [−0.876, +0.566] | +0.472 [−0.368, +1.659] |

The bounded CAR changes against hard CE were +0.117/−1.360 pp for P04 and −1.174/−0.691 pp for P10 at seeds 0/1. Both were fitted as held-new people. F26 has no measured whole-cap250 or external score.

The paired model fits preserved the original native optimizer, 347,903 parameters, both seeded tapes and all fit exclusions. Both active input maps ran on every update. The shared norm probe matched energies to 1.19e−7 relative on the P01 source smoke. The fixed common-mode perturbation moved CAR by at most 4.77e−7 normalized units and moved the sham by 0.271. All hard checkpoint bytes and same-mode held decisions matched the F16 archive. The held collector authenticated 672 first-window checks, 1,344 interleaved resets and maximum prefix drift 1.934e−6 before the gold score.

Ferranti `511455` and `511456`, and Galvani `3134002–3134004`, completed. The independently authenticated held result used `ni26_friday_s1c_reference_20261119`.

| Artifact | SHA-256 |
|---|---|
| Actor manifest | `b93711fe591771ffdd51dff50f4b07a7a21d3e1500635ebdfaecfabde818c2d2` |
| Held scorer manifest | `fdc72508bb8439512b015698da05e5c8d435fd14310ccfddfb87a57a3b80a023` |
| Blind decision manifest | `74f368d1ebac831197de465299578b7935296c9942867d48dee0678429ba934c` |
| Result | `bed4bf0abc95f1c75613c40b6caefab545e4d1f1e14c96ca598d5caf5a3fc3eb` |
