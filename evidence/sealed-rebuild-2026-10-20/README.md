# Sealed package rebuild evidence (20 October 2026)

`scripts/build_sealed.py scripts/sealed_r6nf.json OUT` rebuilds the
recommended sealed package `ensemble_all_r6nf` (no external data) from
scratch on all ten PROTEUS participants (2,655 windows, data sha256
`2be4187f…`), in one run instead of the hand-run chain. Source rev
`5f9b9ab` plus the then-uncommitted script, config and pins; official
submodule `a66bc23`; torch 2.14.1, braindecode 1.8.1, exca 0.5.29,
neuralset/neuralfetch/neuraltrain/neuralbench 0.4.0, numpy 2.5.4,
scipy 1.18.1, scikit-learn 1.9.1 on both machines.

| Build | Training | Harness agreement | Official replay (cell mean) | weights sha256 |
|---|---|---:|---|---|
| reference `ensemble_all_r6nf` | mixed (A100 / local / MPS) | 100% | 0.997312, Mac CPU | `2739b263…` |
| MPS rebuild | M4 Max MPS, 59 min | 100% | 0.998656, Mac CPU, files unchanged | `6c93ac0c…` |
| A100 rebuild | Galvani 2982358, 3.4 min | 100% | 0.998656, A100 node (`COMPET_DEVICE=cuda`), files unchanged, 98 s | `9d055406…` |

Leakage: these packages are trained on the warm-up test participants, so
the replay scores say nothing about sealed accuracy.

Window-level comparison, both packages streamed per session (alignment and
windowed rebalancing as in `submission.py`):

| Pair | Official test (755) | All windows (2,655) | Mean / max abs prob diff (all) |
|---|---|---|---|
| MPS rebuild vs reference | 754 agree (+0.13 pt cell mean) | 2,654 agree (+0.04 pt) | 0.0049 / 0.068 |
| A100 rebuild vs reference | 754 agree (+0.13 pt) | 2,654 agree (+0.04 pt) | 0.0029 / 0.034 |
| MPS vs A100 rebuild | 755 agree | 2,655 agree | 0.0050 / 0.069 |

- v2 weights match the reference to 2e-12 (deterministic).
- The MPS rebuild reproduces the five unaligned FBCNets (net15–19, the
  round-6 members originally trained on MPS) **bit for bit**; the other
  nets differ because they were first trained on other devices (round-1
  ShallowConvNets on an A100, round-3 aligned FBCNets locally).
- Exact bit-equality is not expected across devices; the audit tolerance
  is ±2σ of the sealed score, and both rebuilds differ from the reference
  in one training window out of 2,655.
- Not shown: agreement on held-out data (all windows are training
  windows here), Scherer package `r6f` (needs external data per member).

`rebuild.json`: both check files (environment, per-member fit time,
harness scores, comparison with weight differences per net), both replay
summaries and the MPS-vs-A100 comparison.
