# §302 neutral-information control

Registered in `docs/LEDGER.md` at commit `9800549` before implementation.
The ENU slot has weight 0.5 and a uniform three-class posterior on every row.
It measures mix dilution, not an EEG model. It uses the §302 full-chain and
arm-specific count filter and causal personalisation, over eight delivery orders
and six bases. It is not a submission candidate.

Private Galvani workspace: `/mnt/lustre/work/gehler/gfh527/ni26_s302_neutral_20261116/neutral`.
Its inherited source hashes passed. The SHA-256 of `neutral-src.sha256` is
`5a5ae387ad446a4c33518e500e819025e5263314ee5493516661528140cedb7d`.
The seven reused zero orders, eight E05 orders, both within0 identity controls,
and control gates were copied privately and verified against the §302 workspace.
Their hash manifest `neutral-reuse.sha256` is
`710d75ff232a5c3cf40332dc81c1cacf7d02b5aca398ca212c87dbba4cc2d755`.

- §302 treatment array 3114513 and complete scorer 3114514 completed 0:0.
- Replay 3114866 completed 0:0 and printed `OK`.
- Fresh-source zero and ENU smoke 3114865 completed 0:0. Zero compares its entire chain,
  gate, and all six personalisation outputs with the original §302 zero.
- Release 3114876 completed 0:0 after checking eight zero orders, both
  within0 identities, source hashes and the ENU gate and six CP outputs.
  Array 3115011 runs the remaining seven ENU cells at %6; scorer 3115012
  depends on that array. Inspect their states before claiming completion.
  All scoring remains exploratory.
- Local transform tests pass (two tests). Returning-person package routing has
  not been validated by this new-person study.
