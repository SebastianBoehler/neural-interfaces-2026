# Exact active runtime map for an isolated native port

Read from immutable Galvani `ni26_js321_20261117_inputreuse2/shared_src` on17November. This is a source map, not a completed native implementation or proof. Frozen production ZIP SHA9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed remains unchanged. The JS variant and conditional returning bridge are experimental; user-requested native port must state which stack it reproduces.

`solver` constructs `ReuseEnsemble` → `EnsembleJS` → original `Ensemble6`. Modules in concatenation order: numerics,tangent,inputs,nets,stages,gates,personal,routing,ensemble,extras,finetune,countfilter,cpers,v6chain,adapter_shared,adapter_inputreuse,solver. Original baseline omits the last two adapters. `RecognitionEnsemble` (§350) is a separately registered private wrapper only.

Actual external inference dependencies: NumPy; Torch/TorchScript/autograd/functional/optimizer; SciPy signal butter,sosfiltfilt,sosfilt,sosfilt_zi,hilbert,savgol_filter; SciPy linalg.eigh, spatial.distance.pdist, special.logsumexp. Standard library copy/io/hashlib/pathlib/sys/time/os/concurrent.futures. `benchmark_utils.base_solver.CompetSolver` is the required interface. CausalPers.__init__ also imports sklearn LinearDiscriminantAnalysis and pyriemann Covariances inside the constructor; these are active runtime dependencies despite being absent from the top-level import AST. Native OAS/CSP/LDA must replace them. A C++ runtime with native PyTorch operators is distinct from the original Python orchestration and from previous custom C++ probability-bank tooling.

| Runtime boundary | Native parity obligation |
|---|---|
| inputs/tangent | Exact41×480 tensor, float64 covariance/shrinkage/eigendecomposition, causal G+k current prefix, raw prior5 versus latentBN2; reset all per-session accumulators. |
| nets/ensemble |45 active +20 inactive original member slots; preserve actual Shallow/FBC/teacher/SessionBN/JIT classes, precision, memory layout, six-seed order and original nonzero weights. |
| extras | Rest/eye models have separate per-session state; eye features enter nonlinear SessionEM. No module-level mutable state crosses sessions. |
| finetune | Per-family trainable copies and optimizer/RNG, original shadow/relabel labels and input histories, exact trigger/cap/session-time accounting; source weights immutable. |
| v6chain/routing | Original inner15 computes gates and FT labels; recB latches before outer mix6. Inner, RouteB, label/shadow tails and OwnStage are separate evolving state. Recognition does not imply final output equals inner output. |
| countfilter/cpers | Count-filter latch and CP six-checkpoint fusion/refit accumulators, prefix/gate stopping, class-degeneracy checks; require actual late FT/CP activity rather than only final labels. |
| copy/reset | `fresh_model` deepcopies and resets every original and auxiliary buffer, Torch generator, stream covariance, personal heads/tails and clock. Independently interleaved instances must match isolated prefixes; original weight/buffer hashes remain fixed. |
| JS input reuse | Read exactly original current `nets.last_inputs[35]` after bank/FT, once per window; original bank/cache unchanged. Six auxiliary tails still evolve even when §350 chooses baseline outer recipe. |

Training port scope is broader: source-only fold masks, augmentation/sampling/RNG, EA and global scale fitting, original architecture/init, optimizer/schedule, SessionBN export, PCA/tangent logistic heads, source galleries/teacher taskmaps and serialization. Preserve frozen production exposure separately from genuinely source-excluded refits; do not call a custom C++ stack original PyTorch reference.
