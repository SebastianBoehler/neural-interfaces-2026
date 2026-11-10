# EEG signal follow-ups

All fits and scoring use the 1,323 session 1 windows from P01, P02, P03,
P05, P06, P07 and P08. Leave one of the frozen A/B/C source participant
groups out of each fit. P04, P09 and P10 are reserved. Repeat the four
previous auxiliary bootstraps (400–700, step 100) and four fresh ones
(800–1100, step 100). All arms share the draws and the three archived
overlap seeds. Replace the 10% power vote and require exact source
comparator parity before fitting any auxiliary.

## Geometry and frequency

Keep all five band power features. Add real Hann coherency from the 12
mirrored electrode pairs or two predetermined right-electrode rotations.
The rotations preserve the number of features and use each electrode
once. Test all five bands, 8–20 Hz, and 20–45 Hz. The latter two each
use two bands of coherency. The controls test pair assignment while
preserving the channel set and feature count. They do not match physical
distance. The real component can include shared sources or artifacts.

## Waveform direction

Common-average reference, center each channel and divide by its window RMS.
Bound extremes with tanh, then compute second and third increment moments at
3, 8, 16 and 30 samples. Reversing a trial preserves the second moment and
flips the third moment exactly. Compare the third moment with a fixed
random reversal per trial and a second-moment-only control. Run both with
power alone and with the existing Hann real view. This tests information
that depends on the waveform's direction. It does not establish its origin.

Require +0.8 points overall, +0.5 BrainHero, nonnegative Graz, +0.3
against each specified control, and positive overall change in six of
eight auxiliary seed averages before advancing. No held-out scoring
or Codabench submission is scheduled.
