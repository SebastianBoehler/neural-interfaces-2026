# Spectral estimation and signed cross-channel phase

The source-only cross-view result did not beat shuffled pairing. Test whether
direct spectral estimation can recover complementary information with a linear
head. Average four orthogonal DPSS tapers (NW 2.5) versus one Hann taper with the
same five frequency bands and power/complex-coherency feature dimensions.
Separately compare power plus imaginary coherency to power plus real coherency
and to power alone. Imaginary coherency measures the lagged component of sensor
association and does not establish physiological causality.

Use four-second official windows with common-average rereferencing. Use the
previously specified 12 symmetric electrode pairs. Every arm uses the same
bootstrap rows, model regularization, three participant-disjoint source folds
and overlap seeds 0/100/200. The four auxiliary seeds are 0/100/200/300.
For each arm, test adding a 10% vote to fixed 10%, and replacing the original
10% power vote. Only P01/P02/P03/P05/P06/P07/P08 session 1 enter feature
extraction, fitting, or scoring. P04/P09/P10 remain reserved. Exact fixed 10%
source comparator parity is required for every overlap seed.

Per contrast and mode, advance only for mean +0.8 percentage points overall,
+0.5 BrainHero, no Graz loss, +0.3 over each specified control, and positive
overall results for at least three of four auxiliary seeds. These are source
screens only. No protected scoring, model packaging, or upload is queued.

- Thomson 1982: https://doi.org/10.1109/PROC.1982.12433
- Nolte et al. 2004: https://pubmed.ncbi.nlm.nih.gov/15351371/
