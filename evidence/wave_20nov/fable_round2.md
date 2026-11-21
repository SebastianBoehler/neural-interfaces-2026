# Fable Round 2 · 21 November 2026

Specs `ni26-cleanroom@6a608db`. [Preregistration](../../docs/LEDGER.md#fable-round-2--21-november-registered-before-diagnostic-scoring) · [Dispatch](DISPATCH_21NOV.md)

**Step 0.** EyeNet derives two horizontal and two vertical raw EEG differences, smooths them with a 25-point cubic Savitzky–Golay filter, and normalises by running session-wide trace mean and variance. The filter preserves constants. Running centring retains each window's offset from the delivered prefix, apart from clipping. The EyeNet TorchScripts consume the traces and supply three log posterior columns to SessionEM. SessionEM also pools unfiltered frontal differences into time bins and adds low-pass differentiated event rates. Both paths already see slow relative offsets. P1's exact six-channel contrast and responsibility update still need the residual check. The delivered EEG itself has the official 0.1–75 Hz preprocessing.

**Step 1.** Pending Slurm smoke and pinned fold-tape diagnostics. Orders 71124–71131 are reserved for this diagnostic, with eight held-out confirmation orders kept separate.
