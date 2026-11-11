Goal: edit the weights of an already fitted linear EEG decoder (function edit_head) so that the full decoding chain is
more accurate on people the decoder never saw. The metric g1_chain_ba is a balanced accuracy in percent (higher is
better); the unedited head (return W) scores 84.4964. Gains of +0.3 or more are meaningful; differences below 0.1 are noise.

Hard rules:
- Only numpy. Deterministic. No file, network or environment access. Use only the four arguments.
- Keep the signature and return a finite (3, D) float array. One call must finish in under 2 seconds
  (D = 44,239; the function is called about 50 times per evaluation).
- The overall scale of the returned weights does not matter (it is renormalised); the direction and the relative
  scaling of features do.

What is known (from experiments already run):
- The fitted head is strong. These simple edits all LOST 0.1-0.7: scaling blocks by their held-out/in-sample margin
  ratio, a random block rescaling, shrinking each weight by its jackknife stability (mean^2 / (mean^2 + var) over
  `inner`), smoothing across neighbouring bands, tying the two half-window blocks, magnitude pruning, averaging `inner`.
- A learned per-block gain (ridge towards 1) was neutral; it always raised the slow block (about 1.5x).
- Transferable signal concentrates in 10-14 Hz and 3-8 Hz; bands above 24 Hz hold 22 % of the weight energy but only
  6 % of the held-out margin; channel-power entries transfer about 2.7x better per unit of weight than channel-pair
  entries. Yet removing weak blocks hurts: the weights are a joint solution.

Ideas worth trying (structure rather than uniform shrinkage):
- Low-rank or tensor structure: reshape the tangent part to (3 classes, 17 bands, 3 windows, 861 entries) or the 861
  entries back to symmetric 41x41 matrices and denoise with a truncated SVD / rank-k spatial filters per band.
- Project W onto the subspace that is consistent across the `inner` heads (e.g. keep the top principal directions of
  the stacked inner weights), or move W along the mean inner direction by a small step.
- Class-contrast-specific edits (the three rows are centred; think in contrasts MI-SUB, MI-WORD, SUB-WORD).
- Combine small, well-motivated changes; prefer edits that change few degrees of freedom.
