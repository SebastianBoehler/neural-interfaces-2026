# Package review: lean_v62g (14 Nov) — DRAFT, checks still running

Subject: `lean/src62g` -> `submissions/lean_v62g/submission.py` (sha256 37da0689...96ea5f), packages
`runs/sealed/cand_v62_r2g`, `r1g`, `r2gL`, `r1gL`. Read-only review; nothing in the package was run on data.

## Part 1, correctness and robustness (most dangerous first) — draft

1. **`step62` has no error protection at all** (`lean/src62g/v6chain.py:127-163`; bundle: `Ensemble6.step62`).
   The `try/except` that gives "an error falls back to v5.1b for the rest of the session" exists only in `step61`
   (`v6chain.py:184-200`), which runs only for `v6_routed = 0`. All four candidates have `v6_routed` 1 or 2, so they
   run `step62`: `ft.relabel`, `eye.step`, `rest.step`, `tail6r.step`, `ft.step` (every refit), `mix6`, an explicit
   `raise FloatingPointError("v6.2 mix not finite")` (line 155) and `routeB.stepx` are all uncaught. The ingestion
   program sets `BENCHOPT_DEBUG=true`, so any exception ends the evaluation. `out_v5` is already computed at line 140.
2. Budget clock silently off when `FT_CLOCK["t0"] is None` (`finetune.py:130`).
3. `FineTune.reset_state` (4 x `torch.jit.load` on the device) runs outside any protection, twice per session.
4. Stress cases of `lean/tools/record.py` (stress, extreme, faults, long = 577 windows, switch, lapse) have no
   recorded run on any v6 package.

## Part 2, efficiency — draft

1. Own-space stage computed twice per window (`route.own` and `routeB.own`, identical state).
2. FIR bank outputs recomputed in fine-tuning although the NetBank already has them per window.
3. v5.1b's routed tail duplicates the shadow tail until the first routed window.
4. `FineTune.reset_state` twice per session; loads even when fine-tuning can no longer run.

(details follow in the final version)
