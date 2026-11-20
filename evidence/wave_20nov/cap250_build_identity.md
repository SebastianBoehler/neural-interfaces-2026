# cap250 build card: hilbert and identity families

Code: `cleanslate/cap250/families/{hilbert,identity}.py`, helpers `_id_own.py` (own-space fit on explicit source rows, with a
check that `studies/12nov/ownspace.py` constants are unchanged) and `_id_parity.py` (archive parity, exposure test).
Commit b05149ca (code), run revision 642f1e34. Site: Leonardo boost_usr_prod (CPU work on 1 GPU slot, 8 cpus), numpy 2.4.6.
The external two-covariance calibrator is read from `runs/replay_data/bio_model_psd_d64.npz` and is not refitted. Its arrays
match the archive bitwise (bio_sd/V/B/W/S), so the external-corpus provenance is that of the archive.

## 1. All-ten parity against `runs/sealed/cand_v51b_own/weights.npz`, through `worker.py`

| Family | Job | Run time | Result |
| --- | --- | ---: | --- |
| hilbert (13 `ts2_*` keys) | 64931281 | 155 s fit, 162 s job | 8 keys bitwise; W, b, mu, sd, log_glob differ at most 9e-11 absolute (relative 1.9e-9) |
| identity (24 `bio_*`, `own_*` keys) | 64931291 | 73 s fit, 78 s job | 12 keys bitwise incl. bio_B/S/V/W/m/sd; all others at most 4e-11 absolute (relative 6e-12) |

Reason for the nonzero rest: different BLAS/numpy build (archive made on the Mac), LR lbfgs and eigh summation order. No key
missing or extra, shapes and dtypes equal. Not bitwise; stated tolerance 1e-8 relative.

## 2. Folds (job 64931952 hilbert, 64931955 identity, scoped tag G)

- Nine-person fold P01..P09 and returning setting (P01..P10, `--first-session-only P04,P10`) run end to end through
  `worker.py` with receipts. Identity per-person axis follows the sorted roster ((9,64) gal, (9,3,7749) own_t_W; returning
  bio_m = all ones, i.e. P04 and P10 galleries are one session).
- Exposure test: refit with every non-source row (held person; P04/P10 sessions after the first) replaced by noise and random
  labels, 577 and 762 rows. Outputs are bitwise identical for both families in both settings.

## 3. Runtime (one seed, one fit, 8 cpus)

hilbert 92-156 s, identity 67-73 s; well inside the 35 and 40 minute allowances in `components.py`.

## Notes

- Neither family uses the nested B2 posteriors: `route_inputs.b2_fits` already excludes identity and hilbert has no route
  dependence. Only geometry's `route_b` needs the nested refits (lead's family).
- bio_tau .26, ramp .05, t0 10, own_beta .1 are the frozen archive scalars.
