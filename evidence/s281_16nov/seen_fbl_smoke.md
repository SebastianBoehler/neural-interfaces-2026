# §281 seen-session FBLight consumer smoke

The FBLight consumer calls the shipped `FineTune` implementation with a fresh session instance. It checks the explicit base/order, route trace hash and recognition mask, pinned permutation, session positions, inner shadow labels, posterior prefixes, package settings, packaged FineTune source and shipped member bytes. It stops at the first recognised window. Its refit callback accepts only available prefixes through t=250; the package handles the cap and keeps the last model afterward. Unrecognised windows with no fine-tuned output and all recognised windows are marked NaN.

The real-window implementation smoke passed on CPU, one thread, within0/B2_s0, P10 session 2. Recognition began at window 60; refits and prefix relabels occurred at 20, 30, 40 and 50. The largest weight update was 0.0193874836. A separate source comparison confirmed that the workspace InputStream and eig_apply definitions match the pinned bundle. Four unit tests passed: input provenance, fresh-session isolation, recognition stopping, missing prefix, and the t=250/t=260 boundary. The boundary test uses a fake FineTune scheduler; the real-window smoke ends at recognition before t=250.

```sh
.venv/bin/python -m unittest discover -s studies/14nov/newmember -p 'test_s281_seen_fbl.py'
scripts/run_capped.sh runs/s281_seen_fbl_smoke.log .venv/bin/python -I studies/14nov/newmember/s281_seen_fbl.py smoke
```

The smoke used the same package and prefix hashes recorded in `ft_implementation_parity.md`. Source SHA-256 after adding a post-smoke path-validation guard: `s281_seen_fbl.py` is `e98ead7010d450b17d80a5e183ffc867d703f01962c6e90dd50d57fda2044fa3`; its unit tests are `fbef59eb00c7ff9157bf5357fafd579fa4a02ce56bd639edb7585dc5db08de60`. The final unit tests include that guard.

The shipped model was trained on public participants. Its outputs are an implementation smoke only; the CLI enables no held-out dev run and writes no prediction artifact. A usable dev run still needs a validated fold-specific FBLight trace and matching causal inputs, with model-to-fold provenance and archived frozen-output equality. This smoke does not validate another family, GPU, t=250 behavior of the actual package on real windows, or the full routed system. No accuracy was computed.
