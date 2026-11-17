# §377 Can source-only rewards select the refit time?

§375 found that choosing between two refit schedules using the eventual
whole-session gold score had a 0.1567 pp advantage over choosing by the next
ten gold labels on public new-person DEV. Both schedules have six refits
and use the same total prefix lengths. This experiment learns that one choice
from other people's labeled source sessions, then makes it at delivery index
100 using the preceding 100 upstream probabilities. [§375 trajectories](../../studies/17nov/refit375/trajectory.py) · [§375 diagnostic](refit375_result.json)

P01/P02/P03/P09 were excluded from the six-person A6 fit. For each outer
person, the other three supply ridge targets from the two CP schedules using
A6 for both pseudo-labeling and the prior. These proxy trajectories have
no count filter. The genuine target trajectories use the frozen B2_pool
upstream, its count filter and the corresponding §375 schedule pair. Eight
orders per person describe four independent people. There is one session
per held person. Features, scaling and fitted action rules use exactly the
roles and weights in [the registration](../../docs/LEDGER.md).
[Source and target trajectories](../../studies/18nov/refit377/produce.py) · [Ridge and metrics](../../studies/18nov/refit377/fit.py)

The smoke verifies source resets by repeating a 131-window prefix, including
the divergent refits, from a fresh stage. The full matrix verifies both
schedules' actual refit times and matching fusion work, the pre-action prefix,
participant/session rows and delivery order. The target fixed labels must
match both native and frozen controls. Each fold writes its selected actions
to the shared proof before evaluation on held labels. The bootstrap resamples
four people and orders within each order scheme; every scheme has equal mass.

**Galvani.** `ni26_refit377_v1_20261118`; source manifest
`33a51bd87b83ebb673cb8dbcceb7756e0787c6ccb0dec9a3412292b3e2c9eff2`.
CPU smoke 3123083 passed. Fold array 3123084 and collector 3123085
depend on complete predecessors.

## Held results

All jobs completed. The smoke took 8 s, the four producers took 13, 13, 13
and 24 s, and the collector took 104 s. Total allocation was 350 CPU core-seconds.
The historical replay printed OK and the returning-stage inactivity guard passed.

| Held four-person equal-cell BA | Fixed | Delay | Short ridge | Full ridge | Short intercept | Full intercept |
|---|---:|---:|---:|---:|---:|---:|
| A6 proxy + CP | 61.391% | 61.362% | 61.347% | 61.242% | 61.391% | 61.263% |
| Frozen B2_pool + CP | 81.216% | 81.139% | 81.271% | 81.271% | 81.216% | 81.271% |

On the genuine complete target, full versus short reward changed 13 of
32 actions. Every per-person/order balanced accuracy was identical. The
full-intercept control also matched full ridge on every target score.
Full minus fixed gained +0.055 pp [80% paired people/order 0,+0.153],
all from P02. On the held proxy, full minus short was −0.106 pp
[−0.260,0] and full minus fixed was −0.149 [−0.313,−0.017].

The measured target effect of horizon was zero despite active decisions. The
positive oracle horizon difference did not translate into an advantage for
this learned policy. The four-person result misses both advancement criteria.
No returning-person or sealed score was measured. [Complete contrasts](refit377_result.json)

Galvani action proof `cd9d3acf89f69b3e2c7551babcf0a8ea0a7104b9032f4bcf1e996c071b1bebfb`;
result `3ca10cff97361a48dd225e3aeb02843384f4c9fa0f95a8f8bff29e63a453ad85`.
