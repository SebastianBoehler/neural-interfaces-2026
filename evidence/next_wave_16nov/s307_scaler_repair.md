# §307 target scaler repair

- Original CPU pilot 3116015 failed the independent §305 anchor check on
  P01/within0: maximum probability difference 0.0005567537386616528, zero
  non-tied label changes. No target score was computed.
- Both pilots used the same ten source-cache hashes and pinned permutations.
  The original §307 target re-fitted a scaler from native features, while §305
  used a separately pinned source-only reference scaler. The fitted native
  objectives were 0.0015985028346066684 and 0.0015985033501317325 respectively.
  This discrepancy establishes an input mismatch, not its complete causal
  contribution to the posterior difference.
- Commit `a86aecd` loads the exact §305 scaler after checking its hash, fold and
  source rows. It independently fits the native head. It does not copy §305
  coefficients or relax the original eight-order 1e-5 probability guard.
- Local narrow scaler, adaptation and person-ID checks: five passed using
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q` on their three test
  modules. The real-data no-score P01 pilot is Galvani job 3116319, in
  `ni26_s307_meta_20261116/target_cpu_scaler_v3`; `target-src.sha256` hashes to
  `9259f24780f829248a58d419f085939e98edd9d0b3e37cfbee5d3959ea91289f`.
  The original failed `target_cpu_v2` remains unchanged. Pilot 3116319 also
  failed its unchanged guard: maximum difference 0.000338051805592432 and zero
  non-tied label changes on P01/within0. That failed pilot released no
  production or chain job.
- The original §305 native fit used one OMP, MKL and OpenBLAS thread. Both
  §307 pilots used four OMP and MKL threads and one OpenBLAS thread. Separate
  no-score pilot 3116329 uses the exact §305 thread settings in snapshot
  `target_cpu_scaler_v4`, whose `target-src.sha256` hashes to
  `6f731d44f3df762ef4311d9e5493b215feff7db41c3f2cd927a6969ecd3ef466`.
  It completed 0:0 in 3:20. The maximum probability difference across all
  eight orders was 2.60e-14; no non-tied labels changed. Zero-update, session
  reset and prefix checks passed. No held labels or score were used.
- The production snapshot is `target_cpu_scaler_v5`, with target-source lock
  `be25ead75b9deaffc7d1026e22e2af2963a95ccbb82aef396ebc93eab2b053b6`.
  It uses the same target source, scaler and single-thread environment as the
  passed pilot. Its copied P01 output is byte-identical and passed the complete
  independent guard again. Release 3116400 queued P02–P10 as array 3116401
  (maximum two concurrent) and complete assembly 3116402. The separate
  `chain_scaler_v3` snapshot is locked by `s307-chain.sha256`, SHA-256
  `8258291e94e8d1ab498a1288d446a963761d2fb8e487147d3117672e3fb15c91`.
  Its dispatcher checks capacity before submitting any chain matrix. The old
  pending release 3116016 was cancelled after the corrected release was queued.
  Completion, full-chain scoring, returning-person validation and package
  compatibility remain unproven.
