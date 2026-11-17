# Native training and submission pipeline

This is the full C++ port authorized on 17 November. It is under construction;
the existing submission and research jobs remain unchanged.

The target is C++ ownership of data preparation, source fitting, neural training,
window preprocessing, inference, all session stages, routing, fine-tuning,
count filtering and causal personalization. LibTorch supplies optimized native
tensor/autograd kernels. Native linear optimization must not call SciPy.

The competition requires a Python `Solver` entry point. That adapter will only
load the native runtime, pass the input pointer and return its labels. It must
contain no preprocessing, model, adaptation or training logic.

Existing trained TorchScript files may be read by the C++ runtime as migration
artifacts. This is native execution, but is not evidence that their training
recipes or architecture definitions have been ported. Each is tracked separately.
The one-time archive converter is tooling, not part of training or inference.

## Qualification

Use frozen v6.6cap250 (ZIP SHA-256
`9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed`)
as the prediction reference. [§353 in the ledger](../docs/LEDGER.md) records
the exact scope and result of each comparison.

1. Run the existing real-input canary with both implementations. Require
   matching labels, final scores within the registered bound, matching
   adaptation events and identical reset and interleaved-copy behavior.
2. Run the actual Python adapter and native libraries in the Linux worker
   image replica with an A100. Use complete source streams and pinned shuffled
   orders. Cover natural recognition, active fine-tuning and personalization
   and the long-session limits. Keep the frozen ZIP as a separate reference arm.
3. Measure paired cold load and full warm replay. Record host RSS, live and
   reserved GPU memory, and refit and cap counts. Compare equal work first, then
   retain the production wall-clock budget and report any changed work.
   Use the [60-minute one-A100 limit](https://neural-interfaces26.github.io/rules.html).

Use the same model inputs, weights, metric and delivery order in each pair.
Offline training parity and training throughput each require a matched source
fit with held-stream prediction checks. Worker timing measures the cost of the
completed inference artifact. New learning rules and compute allocation
policies need their own accuracy and cost comparisons.
