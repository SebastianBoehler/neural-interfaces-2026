# Lane FT-labels · v6_ft_labels flipped in the real fold packages (audit item 2)

Registered in `docs/LEDGER.md` ("Lane FT-labels", with amendment and smoke result) before any wave was scored. Scripts `scripts/overnight21/ftl_*`. Run dir (Leonardo, account AIFAC_P02_1636) `/leonardo_work/AIFAC_P02_1636/ni26/cap250/runs/laneFT-labels`.

## Setup facts (found while smoking)
- First smoke 64986849 (hardcb_safe, flag only) **failed loudly**: `HardWaveEnsemble.__init__` raises "expected the original cap250 raw FBCNet layout" when `v6r` is set. A weights-only flip is not a valid hardcb arm; the Opus audit's "the switch already exists" holds for stock only.
- Arms: S stock + flag; H1 hardcb_safe + flag with only that clause removed (the label chain reads the stock raw mix); H2 H1 + a teacher that reads the hardcb mix (hard pair + CBraMod proposal, cached per window). `submissions/*/submission.py` untouched; patched copies live in the run dir.
- Prep job 64987062 (all source flags asserted 0, copies 1, 1,046 other arrays bit-identical). Smoke 64987128/33/37/49: S/H1/H2 change 5/5/4 of 190 labels (P01, 71000), H2 vs H1 3, T0 (flag 0) 0.
