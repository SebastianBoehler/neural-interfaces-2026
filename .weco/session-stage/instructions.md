Goal: improve run_session, a causal, label-free session-adaptation stage for a 3-class EEG decoder. The metric
g1_new_ba is the balanced accuracy (percent, higher is better) of the final argmax on people the ensemble never saw;
the current code scores 84.4964. A gain of +0.3 or more is meaningful; differences below 0.15 are noise.

Hard rules:
- Keep the signature run_session(P, X, F) -> (n, 3) finite array; numpy only; deterministic; no file, network or
  environment access.
- CAUSAL: output row t may use only rows 0..t of P, X and F. This is tested (the session is rerun on a prefix and the
  rows must match to 1e-9); a violation scores 0.
- One evaluation runs the function on about 60 sessions of 190 windows; keep one session under about 0.15 s.

What the inputs are: P = ensemble probabilities per window (under-confident: mean top probability 0.54 at 67 % accuracy;
it carries a session-specific class bias). X = 128 principal components of covariance features, where class
differences are person-specific (a template fitted on the session's own true labels would add several points; with
pseudo-labels the current stage gets about half of that). F = ocular / slow-potential features.

Protocol structure you may exploit: a session is a sequence of runs of 30-36 windows; inside each run the three classes
occur exactly equally often in shuffled order (run boundaries are not given; typical lengths are 36, 30-36, 30, 30, 30, 30).

What the current stages contribute (removing one costs): rebalancing by the last-30-window class share (large, about
5 points), the EM on F (about 2), the template stage on X (about 2-4), the early boost (about 0.3).

Already tried by hand without gain (do not just repeat): changing alpha / window of the rebalancing (30 is right;
longer or cumulative is worse), a count-based without-replacement prior using the stage's own soft counts, Dawid-Skene
style reliability, recursive or margin-ranked or Sinkhorn seeds, class-specific seed quotas, shrinkage schedules,
restricting seeds to recent windows, a second template stage on other views, pair-specific covariances.

Ideas of a different kind that are untested: exact or particle inference over the hidden run structure and the
remaining class counts (a proper sequence model instead of a share heuristic); a joint probabilistic model that
iterates class templates and labels over the whole past at every step (full EM / variational, with the balanced-run
prior inside the E-step); using both views X and F in one model; transductive re-estimation where earlier windows'
labels are re-inferred at each step to improve the templates used for the current window (only the current row's
output must be causal; internal re-labelling of the past is allowed); better use of uncertainty when selecting seeds.
