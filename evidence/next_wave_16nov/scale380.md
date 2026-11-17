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
