# Scoring and research audit · 18 November 2026

Scope: current research queue, scoring validity and research assumptions. Read the current state,
ledger, official contract, Wednesday handoff, Friday queue and the code paths below. No accuracy
experiment, new Slurm job, package modification or upload was performed for this audit.

## 1. High priority: the historical shuffled baseline is only partially re-streamed

**Confirmed protocol mismatch; accuracy impact remains unmeasured.**
[shufcombo.py](../../studies/13nov/shufcombo.py:114) always leaves the transformer and shallow EA
members in recording order. Its `STRICT` assertion checks `REC_LEFT[:-1]`, explicitly excluding
those members. [aux430/run.py](../../studies/18nov/aux430/run.py:52) executes this setup. The
three-seed stock then feeds [stocks456.py](../../studies/18nov/matched456/stocks456.py:29), including
the Friday terminal-insertion comparisons. Zero-change identity authenticates that historical
calculation; it cannot establish that every input was available at its delivery position.

Euclidean alignment (EA) uses a running covariance of the windows seen so far. Permuting an
already aligned recording-order prediction does not recompute that covariance. A shuffled first
window can therefore use earlier recording windows that have not been delivered yet.

The label-blind [real-EEG reproduction](../../studies/18nov/validity_audit/causal_ea.py) uses the actual
`AlignedNet._aligned` implementation, pinned full0 order, P01 session 1, and a prior from 24 P02
windows. The first delivered global row is 94. Recorded-order alignment includes 94 unavailable
earlier windows. Its aligned input differs from immediate alignment by a maximum 6.584965.
Immediate alignment exactly matches the first window of a correctly re-streamed sequence.
This is an input-level demonstration, not a measured error in the deployed weights or score.

**Decision correction:** label these results *historical full-system DEV, partial delivery replay*.
They remain useful conditional comparisons. Neither their positive deltas nor their negative
terminal insertions qualify an exact causal package or close a model family. The staged ZIP itself
streams its inputs; this finding does not show a submission-rule violation in that ZIP.

**Next validation work:** rebuild these missing member streams from the same source-excluded fits,
then rerun both complete branches, pseudo-label production, downstream adaptation and routing on
the pinned deliveries. Reproduce recording-order controls first. Commit the new decision banks
before scoring. Do not splice corrected members into final cached scores: their changed predictions
can change later pseudo-labels and fitted stages. Keep the old source snapshots as historical evidence.
No new corrected gain or ranking claim is available yet.

## 2. Fixed: training cache can silently change the requested experiment

[AlignedNet.fit](../../experiments/deep.py:264) used a shared cache without the concrete model class,
SessionNet's `group`, `causal`, `k_train`, subclass augmentation settings, or `cotrain`. With caching
enabled (`adabn` is not `None`), an earlier plain fit could satisfy a SessionNet request without
training SessionBN. Changing co-training could also return the old fit before its training path ran.

Real-EEG regression checks reproduced both failures. The repair limits reuse to the concrete
`AlignedNet` class and includes `cotrain` in its key. Subclasses run their own training implementation.
Inference-only `adabn` changes still share the base fit and return separate network copies.

**Also fixed: accepted sample weights could be ignored.** `fit` declared support for nonuniform
`w` on all datasets, but `_fit` used weighted loss only when the external-data `tag` was present.
The ordinary mean-cross-entropy path now rejects those unsupported weights before fitting or cache
lookup. A separate real-EEG regression reproduced the missing rejection before the repair. This
does not introduce a new weighted training recipe or change the external-data loss.

**Impact limit:** recent standalone Friday waveform actors do not use this cache. Normal SessionNet
fits with `adabn=None` already bypass it. No completed campaign score has been identified as affected.
The fix is for subsequent source snapshots; existing cluster releases and frozen ZIPs are unchanged.

## 3. Repeated validation is being mistaken for additional independent evidence

The ten PROTEUS people are reused across hundreds of choices. Two training seeds test fit variability;
they do not add people. New delivery orders on Scherer test order sensitivity on the same people.
The Scherer cohort has also informed mechanism selection, montage/crop checks and package comparisons.
Writing each new decision bank before reading labels prevents within-run feedback, but does not remove
selection based on earlier runs. People/order bootstrap intervals describe the fixed tested comparison;
they do not correct for choosing that comparison after many outcomes.

**Decision correction:** retain all measured effects and mark these cohorts as reused development
or external transfer instruments. Reserve “independent confirmation” for people untouched by the
decision being tested, with upstream training exposure separately documented. If no untouched
cohort is available, report that limit. Fresh orders cannot restore participant independence.
Use two-person returning intervals descriptively; the assumed 20/80 weights are a sensitivity model,
not a measured private population. Do not translate a public gain directly into a sealed score.

## 4. Research control: a terminal blend tests only that integration

The Friday [wave scorer](../../../neural-interfaces-2026-friday/friday/wave/score.py:139) adds a fixed
0.20 centered, clipped logit correction after the historical stock. That is a valid test of this
terminal rule. It does not test a replacement member's interaction with the original earlier stages,
or establish a ceiling for waveform learning. Conversely, a whole decoder's genuine-minus-blank gain
only establishes that its own history helps that decoder; it does not establish extra system accuracy.

The current whole-decoder lane and F12 spectral-initialization comparison can continue on their stated
scope. Before claiming progress toward a replacement, include a strong raw/FBCNet control with the
same held people, source exclusions and evaluation path. Historical 59–64% anchors from different
training recipes are references, not matched controls. Check source fit and held error before spending
another wave on terminal blending of a substantially weaker decoder. Preserve small consistent system
positives as undecided when their intervals span zero; do not turn this audit into a new hard gate.

## Queue decision

At approximately 16:55 CET, live `squeue` returned no jobs for this account on either Galvani or Ferranti.
Later accounting showed the cited Friday comparisons completed, including the repaired F-C6 smoke.
This is a timestamped observation, not proof of a stalled heartbeat or a request to cancel other work.
F12 and full-worker F-C5 were listed as preparation work in the 16:51 Friday handoff. A second
live check around 17:08 CET found F12 arrays 3128355/3128356 and score jobs 3128357–3128360
pending on Galvani; Ferranti still returned no live jobs. Those jobs were left unchanged.

Prioritize correction of the shuffled baseline and qualification of the real worker path before more
small terminal-policy sweeps. Keep coherent whole-decoder exploration separate. Every new score still
needs its ledger registration; this audit does not authorize an upload or establish a new winner.

## Verification

Source starting point: `9dce65f`, followed by the local changes described above. Training-cache
regressions use seed 0, 24 real public EEG windows, one CPU epoch and no evaluation score. They
exercise actual plain and SessionBN fits; only co-training dispatch is intercepted to avoid loading
unrelated external corpora. Run with `scripts/run_capped.sh LOG .venv/bin/python experiments/test_fit_cache.py`.
The EA reproduction is also run through `scripts/run_capped.sh` and reads no task labels.

- `replay/check.py`: all six references matched and printed `OK`. This guards the v3 recording-order
  replay only; it does not resolve the partial shuffled replay above.
- Capped CPU real-EEG checks: both cache regressions and the ignored-weight regression failed before
  repair. The inference-only cache-sharing control passed throughout. All four pass after repair.
- Capped label-blind EA reproduction: passed, with pinned data/order hashes and no labels opened.
- No accuracy was scored by these checks. No artifact under `submissions/` or the official submodule changed.
