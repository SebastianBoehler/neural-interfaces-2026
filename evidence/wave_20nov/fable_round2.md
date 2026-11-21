# Fable Round 2 · 21 November 2026

Specs `ni26-cleanroom@6a608db`. [Preregistration](../../docs/LEDGER.md#fable-round-2--21-november-registered-before-diagnostic-scoring) · [Dispatch](DISPATCH_21NOV.md)

**Step 0.** EyeNet computes left-minus-right contrasts over F5/AF3/F3 vs F6/AF4/F4 and FC5/C5 vs FC6/C6, plus two vertical contrasts, smooths them with a 25-point cubic Savitzky–Golay filter, and normalises by running session-wide trace mean and variance. The filter preserves constants. Running centring retains each window's offset from the delivered prefix, apart from clipping. The EyeNet TorchScripts consume the traces and supply three log posterior columns to SessionEM. SessionEM directly pools the first horizontal contrast and an AF-to-parietal contrast into 16 unfiltered bin means and adds low-pass differentiated event rates. Both paths already see slow relative offsets. P1's exact six-channel contrast and responsibility update still need the residual check. The delivered EEG itself has the official 0.1–75 Hz preprocessing.

**04:28 CET interim.** Production folds, full shuffles 71124–71131. v3 replay returned OK. A100 smoke matched 190 pinned lead labels, 190 blend-zero labels and the standalone Team B encoder posteriors against live taps.

- **P1 stopped.** Residual AUC 0.543 held-new (2,019 confused deliveries) and 0.513 returning (464), below 0.60. Correct-pair AUC 0.691 and 0.659.
- **P2 stopped.** Identity won all 144 session/order oracles. The arbiter would falsely swap P05/1 at t=80 on order 71129.
- **P3 pending.** The Leonardo encoder array and decile collector are running.
