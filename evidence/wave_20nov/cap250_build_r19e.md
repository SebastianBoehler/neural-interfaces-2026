# cap250 build: R19e ordinary families (20 Nov 2026)

Code: `cleanslate/cap250/families/{shallow_ea,shallow_raw,fbc_ea_fixed,fbc_raw_fixed,fbc_ea_fir,fbc_raw_fir,sinc_ea,fbl_ea}.py`
(thin wrappers) over `_r19e.py` (reads `scripts/sealed_r19e.json` group, fits `deep.AlignedNet` on source rows in file order,
exports with the `build_sealed.py` exporters as `net<I>_*`, diagnostics/fit.json). Launcher `_r19e_leo.sh` (Leonardo, worker.py
receipts), `_r19e_slurm.sh` (Galvani overflow, smoke queued then cancelled unrun), `_r19e_compare.py` (vs archive). Commit 437367d0.
Release dir `$WORK/ni26/cap250/r19e_rel` on Leonardo (AIFAC_P02_1636, boost_usr_prod A100, array task = family, seeds sequential).

| Run | Job | Result |
| --- | --- | --- |
| smoke (CAP250_SMOKE=1, all ten) | 64931320 | 34/34 fits PASS, receipts written |
| repro (all ten, full recipe) | 64931322 | 34/34 PASS, 3:30 max per family array task |
| fold (P02..P10, held P01) | 64931324 | 34/34 PASS, SPEC file set verified by worker.py |
| rerun of shallow_ea, fbc_ea_fixed (all ten) | 64932275 | run-to-run check |

Reproduction vs archived `runs/sealed/cand_v51b_own/weights.npz` (max |diff| over each net's arrays; keys identical, G and scale bitwise equal):
- BITWISE: fbc_raw_fixed (5 seeds), fbc_raw_fir (2 seeds).
- Not bitwise, trained weights differ 0.03-0.27: shallow_ea, shallow_raw, fbc_ea_fixed (0.06-0.10), fbc_ea_fir (0.04), sinc_ea (0.07-0.23), fbl_ea (0.03-0.04).
- Sources: (a) shallow nets are GPU run-to-run nondeterministic on Leonardo (shallow_ea rerun differs from repro by 0.08-0.09, same as vs archive);
  (b) fbc_ea_fixed is run-to-run bitwise on Leonardo yet differs from the archive, so EA-input platform differences (CPU eigh/BLAS vs the original
  Galvani/A100 build) are the likely cause for EA families; not isolated further (original ran on a different cluster). Tolerance claim: none; decision parity needs the package-level fresh-train comparison.
- Caveat: mean/G/scale are computed on source-only rows (batch of 512 covariances), they came out bitwise equal.

GPU time (A100, summed fit seconds; Slurm elapsed incl. ~14 s start-up per fit is 0.43 GPU-h): all ten 802 s = 0.223 GPU-h; fold 736 s = 0.205 GPU-h
per fold for all 34 fits (within the SPEC range for the summed families).
