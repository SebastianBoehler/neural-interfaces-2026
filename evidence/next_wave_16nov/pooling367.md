# §367 — Fixed pooling of frozen and JS

The two complete decoder branches evolved independently on each delivered window.
The final readout averages their class probabilities or their log probabilities.
It fits no parameters and sends no result back into either branch. The verified
archive supplies the same 2,655 windows, six bases and eight shuffled orders.
The scorer checked the pre/post-personalization score domains and wrote predictions
and the endpoint proof before reading task labels. Every agreement label stayed fixed.

| New-person DEV | Frozen | JS | Arithmetic pool | Geometric pool |
| --- | ---: | ---: | ---: | ---: |
| Six-base official measure | 76.2177% | 76.4410% | 76.3733% | 76.3765% |
| B2_pool | 76.2793% | 76.5517% | 76.3890% | 76.3862% |

Geometric minus JS was −0.0645 pp [80% official −0.1243,+0.0107] across six bases.
The corresponding people/order delta was −0.0416 pp [−0.1105,+0.0334].
B2_pool was −0.1655 pp [−0.3365,+0.0179] and the people/order delta was −0.1701
[−0.3547,+0.0343]. B2_pool P04/P10 were −0.1558/−0.1522 pp in their new-person folds.
Arithmetic minus JS was −0.0677 pp [−0.1243,+0.0063] across six bases and −0.1628 pp
[−0.3370,+0.0211] on B2_pool (new-person P04/P10 −0.1847/−0.1250 pp).

The geometric and arithmetic pools changed 2,019 and 2,021 of the 4,021 disagreement
decisions versus JS, respectively. They chose a third label 16 and 15 times. The
point estimates favored JS, and the paired intervals leave both contrasts undecided.
This was reuse of exposed public development data. No genuine returning score or
dual-branch worker time was measured.

Job 3122820 completed in 116 s on two CPUs and used no GPU. Source `c7bce76`.
Galvani result `ni26_pooling367_v1_20261117/result.json`, SHA-256
`a32292d3c725b36cfffd29c4fbefeba58864c56cd177c71e85f47ebdc93dc825`.
Pre-score proof SHA-256 `bddc59ec93107312d32690d0fc7f8ca1c534e7be9a564a654c9e0867e83301cc`.
