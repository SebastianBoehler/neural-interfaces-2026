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

Acceptance requires real-data numerical parity, exact output labels, active
fine-tuning/personalization paths, session reset/deep-copy isolation and clean
Linux worker/A100 compatibility. Performance comparisons must include cold load,
warm complete inference, training work, CPU/GPU time and peak memory. Component
speedups do not establish complete submission speed or a leaderboard gain.
