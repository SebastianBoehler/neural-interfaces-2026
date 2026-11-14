"""Track 2 BCI decoding: streaming ensemble, v6 (v5.1b = cand_v51b_own unchanged, plus a second chain for unrecognised
sessions: in-session fine-tuning of one SessionBN trace per family, the eye-net view and the rest-class member).

One 4 s window (41 channels, 120 Hz) per call, three classes; all state is per session, causal and label-free.

Members, mixed with fixed weights (weights.npz ``member_weights``):
  core1 and the Hilbert core (tangent)  filter-bank covariances recentred per session, logistic regression;
  nets (nets, inputs)                   ShallowConvNet and FBCNet exports, the external teacher and SessionBN
                                        TorchScript nets, on windows whitened per session or globally scaled.
Stages on the mix (stages): rebalancing -> SessionEM -> TSEM, with a gated early boost.
Returning people (gates, routing, personal): two gates recognise a public person; that person's head then enters
the mix, a second tail runs on the routed mix, and the own-space stage reweights its output.

v6.1 (13 Nov): the fine-tuning labels come from the v6 chain without fine-tuning (a third tail on the mix plus the rest
member, with the eye-net view), key v6_ft_labels = 1; schedule s15 (start window 20, lr 1e-3) by the v6_ft_* keys.
v6.2 (13 Nov night), key v6_routed: 0 = v6.1; 1 = v6.2-safe (rest member and eye view also on recognised sessions, in
the main chain the gates read); 2 = v6.2 (also fine-tuning on recognised sessions, labelled by the inner v5.1b's routed
output). The main chain's personal head enters at core1's share of the new mix.
v6.2g (14 Nov): fine-tuning cost guards. No refit once a session's prefix exceeds v6_ft_cap windows (the last
fine-tuned nets keep predicting); no refit and frozen nets once the process has run v6_ft_budget_s seconds since
load_model (a module-level clock). Inert below both thresholds.
v6.2h (14 Nov): step62 guards the additions as v6.1's chain does. An exception (counted in the [v6] line's errors,
one stderr line) keeps v5.1b's output for that window and the rest of the session; recognition bookkeeping continues.
v6 (extras, finetune, v6chain): while the gate has routed no window of the session, a second chain on the mix with
the fine-tuned families' changes and the rest member, rebalancing -> SessionEM (view + eye-net columns) -> TSEM, gives
the output; from the first routed window on, the output is v5.1b's. The gates never see the second chain.

Modules in bundle order: numerics, tangent, inputs, nets, stages, gates, personal, routing, ensemble, extras,
finetune, v6chain, solver.
"""
