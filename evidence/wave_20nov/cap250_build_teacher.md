# cap250 build: teacher family (20 Nov)

Code: `cleanslate/cap250/families/teacher.py` (+ `_teacher_lib.py`, `_teacher_compare.py`). Runs on Leonardo (boost_usr_prod, 1 GPU, 8 CPU), staged under `$WORK/ni26_teacher`.

- Provenance: 24 trunks (`eegnex`/`ifnet` `_ea-m7r-s0..11`) are checked at fit time against `scripts/sealed_nxif_inputs.sha256` and their metadata (the seven MI corpora, no UDA key, nothing PROTEUS); mismatch raises. They never saw PROTEUS people.
- Fold-fitted parts: EA reference G, scale (mean shrunk window covariance over source rows, k 5, shrink 1e-3), causal EA stream, balanced LR C .01 on 38 seed-averaged head log-probs, TorchScript with the map.
- Smoke (CAP250_SMOKE=1, 600 rows, 1 seed per arch): job 64931055, 31 s.
- (1) All ten, seed 0: job 64931127, 10 m 39 s. net34 align/k/G/scale/shrink equal the archive bitwise (max abs diff 0). `teacher.pt` vs archived (sha 98f7df45) on all 2,655 windows with the archive EA stream: max abs logit 0.0227, mean 0.0017, argmax differs on 2 of 2,655. Not bitwise: the head features come from CUDA (A100) rather than the original CPU run, and the LR refit amplifies that; the archived trunks were also the pinned ones. Trace vs eager 0.0, batch 1 about 5e-7, CUDA vs eager 1e-4.
- (2) Nine-person fold (P02..P10, P04 first session only, 2,088 rows): job 64931128, 4 m 50 s, SPEC files `weights.npz`, `teacher.pt`, `diagnostics/`. Trace checks pass.
- (3) GPU time: about 0.18 GPU-h (all ten) and 0.08 GPU-h (fold), wall including trunk hashing, load and compare; inside the SPEC .25-.75 range's lower end or below. External trunk pretraining is separate (about 6.6 GPU-h once, audit_nxif).
- Not done: the receipt path through `worker.py` (families are run directly here), and assembly of the part into a package.
