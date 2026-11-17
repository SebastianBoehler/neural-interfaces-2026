# §380 Multiscale whole EEG decoder

This is an EEG-only, source-supervised comparison motivated by November work
on combining temporal signals with spectral information. All three networks
start from the same parameters, use the same source-only scale, and train on
the exact §379 episodes. In each fold the nine training people supply labels,
and the tenth person supplies EEG only until scoring. Each inference session
begins from a 20-window explicit normalization prior. [Recent papers and transfer limits](recent_methods_18nov.md) · [§379 harness](../../studies/18nov/wave379/train.py)

Each model learns 24 direct features and a second spatial projection feeding
24 short and 24 long filters. Short and long kernels span 33 and 97 samples,
respectively, centered on the same 112 positions. All treatments have the
same learned parameters, filters and gating widths.

- **RAW** learns signed, unconstrained short/long time-domain filters.
- **SPEC** learns filter envelopes for fixed quadrature carriers, then
  computes log-amplitude before mixing short and long features at each position.
- **PERM** trains the same spectral network with short features permuted
  within each quarter of a window. The permutation preserves values in each
  quarter while breaking alignment with the long branch.

SPEC−RAW tests the combined constraints and amplitude readout. SPEC−PERM
tests whether within-quarter time alignment helps this implementation. The
three complete models use the original official metric and interval code on
the same held rows as frozen cap250 and JS. [Model](../../studies/18nov/scale380/decoder.py) · [Training](../../studies/18nov/scale380/pilot.py) · [Scorer](../../studies/18nov/scale380/score.py)

The first source-only smoke 3123279 failed on a Python name collision before
held inference. The fixed source manifest is
`4253861dd55c70fe89961b1a191c94c7d7ce94d719135b1bb12d84745744f4a6`
in Galvani `ni26_scale380_v2_20261118`. Its smoke, ten A100 fold producers
and CPU collector are 3123282, 3123283 and 3123284. The decision rule and
compute bounds are in [the ledger](../../docs/LEDGER.md).

## Results

All 30 fitted checkpoints passed source, parameter, active-branch, order,
reset, prefix and finite-value guards. The original replay printed OK. The
independent CPU step-by-step audit reproduced every one of the 63,720 A100
decisions, across all ten held people, eight orders and three arms.

| Whole decoder | Official equal-cell new-person DEV |
|---|---:|
| RAW | 44.835% |
| SPEC | 44.738% |
| PERM | 43.338% |
| Frozen cap250 | 76.279% |
| JS | 76.552% |

SPEC−PERM was +1.400 pp [paired people/order 80% interval +0.720,+2.171],
positive for 7/10 people. SPEC−RAW was −0.097 pp [−1.028,+0.793], positive
for 5/10; P04 +2.309 pp and P10 −1.632 pp as new people. SPEC−JS was
−31.814 pp [−37.384,−26.390], negative for all ten. The aligned branch
beat its disrupted counterpart. Its advantage over the signed raw decoder
remains undecided. Both complete systems are far below the common strong
controls. The early 32-window supplementary estimate covers only three of
28 cells; the later stratum covers all 28. No rank, returning or deployment
claim follows. [Full results](scale380_result.json) · [Independent CPU audit](scale380_cpu_audit.json)

V1 failed smoke and v2 successful smoke used 12 and seven A100-seconds.
The ten producer allocations used 211 A100-seconds together. The total was
230 A100-seconds (0.064 A100-hours). Primary scorer 3123284 used 128 s
on two CPU cores and audit 3123300 used 62 s on two CPU cores.

For context, the original §379 explicit-normalization model scored 44.149%
on the identical ten-person/eight-order DEV matrix. RAW and SPEC are +0.686
and +0.589 pp above that historical architecture, respectively. These are
post-hoc differences of means; neither is a registered attribution to spectral
features. The matched SPEC−RAW comparison above remains the primary test.
[§379 results](wave379_result.json)
