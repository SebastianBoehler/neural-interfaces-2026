# §372 and §372b · Learning from held-person proxy mistakes

Error weighting reduced the accuracy of this two-person waveform-member recipe. Across the six train/test splits the error-weighted member reduced fused accuracy by 0.332 percentage points versus ordinary weighting. The same weight values shuffled within source cell and class outperformed error weighting by 0.420 points. Both comparisons use a six-person A6 anchor; the complete deployment chain and genuine returning sessions require separate tests.

| Four-person diagnostic, equal-cell accuracy | Fused accuracy | Member alone | Rescued anchor mistakes | Lost correct anchor decisions |
|---|---:|---:|---:|---:|
| A6 anchor | 60.785% | | | |
| Ordinary weights | 60.695% | 41.474% | 4.118 pp | 4.208 pp |
| Error weights | 60.364% | 40.196% | 3.844 pp | 4.265 pp |
| Confidence weights | 60.767% | 41.498% | 3.664 pp | 3.682 pp |
| Shuffled weights | 60.784% | 42.083% | 3.834 pp | 3.836 pp |

The paired 80% people/order intervals for error minus ordinary, confidence and shuffled were [−0.635,−0.010], [−0.759,−0.035] and [−0.739,−0.096] pp. P01 improved on all three comparisons. P02, P03 and P09 declined. Error weighting reduced standalone accuracy by 1.278 pp [−2.123,−0.432]. This misses the preregistered 1 pp standalone match. The four people had prior public exposure. Intervals condition on their reused source pairs. [Rotation results](complement372b_result.json)

A6 fit P04/P05/P06/P07/P08/P10. Its bank verified frozen native checkpoint identity, causal prefixes, pinned orders, resets and unchanged parameters. For each pair among the remaining four people, we trained the member on that pair alone. The unchanged 75% A6 + 25% member fusion ran on the two held people in eight orders. Each person's estimate averages the three training pairs that exclude them. We then weighted the official cell accuracies equally across the three ordering schemes. P04 and P10 supplied body training data and cannot serve as held-person checks here. [Bank and lineage](causal_prefix357.md) · [Producer](../../studies/17nov/complement372/pilot372.py) · [Matrix collector](../../studies/17nov/complement372/rotation/matrix.py)

The initial split trained on P02/P03 and tested P01/P09. Error weighting changed fusion by −0.219 pp [−0.477,+0.0001] versus ordinary. The matrix preregistration preceded inspection of that score. The original three pairs of held member outputs were preserved bitwise. Array 3122965 trained the 21 remaining models and job 3122966 scored the matrix. Job 3122959 consumed 47 s × 4 CPUs. The matrix consumed 221 s × 4 CPUs and 23 s × 2 CPUs, or 0.258 CPU core-hours. Both studies used zero GPU. [Initial results](complement372_result.json) · [Preregistrations](../../docs/LEDGER.md)

The immutable Galvani input directories are under `/mnt/lustre/work/gehler/gfh527`. The initial study `ni26_complement372_v1_20261117` has source manifest `ec76c2a91b24fdd4d26c42b28d0a75403619d4adcd87241538b4f8b7b9bad0a1`. The matrix `ni26_complement372b_v1_20261117` has manifest `f6a60404f3b1ab785e9a98d1b5dc38f257f127828b997889171d200b816ddfe6`. Initial and matrix jobs started from project revision `dd4e8be` with these physically copied study files. Each matrix fold records its training and held people, matched batch hashes, parameter identities and a prediction hash before scoring. The frozen bank pins their row masks.
