"""Track 2 BCI decoding: streaming ensemble (lean source of the package cand_v51b_own, same outputs).

One 4 s window (41 channels, 120 Hz) per call, three classes; all state is per session, causal and label-free.

Members, mixed with fixed weights (weights.npz ``member_weights``):
  core1 and the Hilbert core (tangent)  filter-bank covariances recentred per session, logistic regression;
  nets (nets, inputs)                   ShallowConvNet and FBCNet exports, the external teacher and SessionBN
                                        TorchScript nets, on windows whitened per session or globally scaled.
Stages on the mix (stages): rebalancing -> SessionEM -> TSEM, with a gated early boost.
Returning people (gates, routing, personal): two gates recognise a public person; that person's head then enters
the mix, a second tail runs on the routed mix, and the own-space stage reweights its output.

Modules in bundle order: numerics, tangent, inputs, nets, stages, gates, personal, routing, ensemble, solver.
"""
