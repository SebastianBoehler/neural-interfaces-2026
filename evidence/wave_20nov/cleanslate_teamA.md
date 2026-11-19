# Clean-slate Team A

## A1 · registered before scored replay

One pretrained full-depth encoder per package: REVE-base, CBraMod, or MANAS-2 (Pratibha-2). Each trains all backbone
parameters and the same channel-preserving four-patch readout. External stage alternates supervised corpus-specific
heads, with a shared task head for corpora with at least two known task targets. Rest/baseline is auxiliary class 3.
Fine-tune on permitted PROTEUS people only, with equal source cell/class mass. Virtual subjects mix covariance colors
from source sessions with trial amplitude, sensor and temporal variation. External fits use only external donors.

Deployment uses one forward per window. The source-regularized Euclidean map refreshes every 8 arrivals. Every 12
arrivals from 24, prefix feature normalization and a short self-training update use balanced soft labels and entropy.
The uniform prior is soft and session-local. No true interface or count is supplied. An open-set EEG fingerprint
routes to a first-public-visit personal logistic head following two consecutive margin/radius checks. Recognition and
adaptation reset each session.

- Seed 0, batch 24, 1,152 external updates, 20 target epochs. Candidate comparisons use identical rows, optimizer
  schedule and augmentation recipes. Lane H's seven corpora and Shin 2018 are required. Scherer joins the PROTEUS
  bank. B6 eligible stages ds007221 Graz, ds007221 LR and NETBCI enter if their full exports are staged, re-registering
  the actual roster before a scored fit. For Scherer NI-V, train a distinct bank with **all Scherer rows excluded**.
  The public-fit bank is also eligible for production, since sealed evaluation is on PROTEUS.
- Score the real exported Solver with the common harness. Held-new P01–P10, all 28 cells, orders 71000–71001
  first, then 71000–71007. Returning P04/P10 visits 2 and 3, with only visit 1 enrolled and available for training for
  the scored person. NI-V new: all 9 Scherer people, two visits, published Scherer export and canonical orders.
- Expected whole-system official cell BA (80% prior interval) REVE **72 [64,80]**, CBraMod **69 [61,77]**,
  MANAS **71 [63,79]**. Returning REVE **87 [77,96]**, CBraMod **85 [74,95]**, MANAS **86 [75,96]**.
  NI-V new predictions respectively **49 [40,58]**, **47 [38,56]**, **49 [40,58]**.
  Prediction against historic 73.74 is REVE −1.74 pp, CBraMod −4.74 pp, MANAS −2.74 pp. Use common same-order cap250 for conclusions.
  A paired 80% upper endpoint ≤0 refutes superiority on that instrument. A lower endpoint >0 supports it.
  Otherwise undecided. Choose the strongest whole candidate on 2 dev orders, then report selection and evaluation overlap.
- Verify checkpoint identities, real encoder updates, first-window/duplicate/reset/interleaved identity, fixed delivery and labels
  in 0..2. Measure warmed one-window worker timing. Report total elapsed against 3,600 s only when the hardware is identified.

### First dispatch

The initial matched REVE/CBraMod dev wave uses the seven Lane H corpora, Shin 2018 and Scherer. The Leonardo
MANAS wave also uses the complete 14-person ds007221 Graz and 19-person NETBCI exports. B6's remaining exports were still being staged
when smoke testing began. These are separate roster conditions for model comparisons. The full B6 wave uses
only complete staged shards. All waves use a weak virtual-subject dose (0.03–0.09 on half of batches), per Lane V's
spatial-transform results. Ferranti excluded due to its estimated 22 Nov queue start.

Initial smokes 3139280/3139281 authenticated both pretrained checkpoints and stopped on Dreyer's legacy classless
`windows.npz`. The corrected loader prefers the labeled `windows_*.npz` shards used by Lane H.

- REVE real-data smoke 3139361 passed checkpoint load, gradient updates, single-label output, prefix, interleaving
  and reset. Galvani external 3139362, LOPO/returning 3139363, collector 3139364.
- CBraMod smoke 3139365 reached the gradient guard. Braindecode declares its pretraining mask token fixed and the
  classification path never supplies a mask. The corrected guard requires every active parameter to be trainable
  and exempts exactly that constant. The first dependent wave was canceled by Slurm.
- MANAS Leonardo initial smoke 64886063 stopped on a nonexistent Python module. The staged standalone 3.12
  interpreter is used by smoke 64886267, external 64886269, LOPO/returning 64886270 and collector 64886271.

### A1 first score · REVE

The complete 3139363/3139364 wave scored 40.95% held-new and 45.45% returning. The archived cap250 cell means
are 73.74% and 92.76%. The held-new cell delta is −32.79 pp. The person-equal contrast is −34.68 pp, nested
80% interval [−38.82, −30.21]. The returning contrast is −47.31 pp [−49.86, −44.89] on two people. The
reference uses historical delivery orders. P04 and P10 scored 40.15% and 50.75% after averaging two new orders.
REVE A1 superiority is refuted on both instruments. The unverified 57/43 cohort proxy gives −40.08 pp
[−42.75, −37.40]; using 20/80 within returning gives −39.56 pp. See Galvani `a1-reve/compare_cap250.json`.

CBraMod's repaired smoke 3139396 passed. Its external job 3139397, fit array 3139398 and collector 3139399 follow.
MANAS smoke 64886267 passed and external job 64886269 completed in 201.6 s. Its fold array 64886270 started.
REVE worker-image smoke 3139483 passed. Independent NI-V banks omit all Scherer training windows.
REVE jobs 3139489–3139493 and MANAS jobs 64887075–64887081 cover that fit and eight-order replay.

## A2 · registered before scoring

One full-depth MANAS-2 encoder with a trainable input projection from seven EEG band powers per patch, from the
recorded channel and its common-average reference. Classify via the mean and variability of transformed tokens.
The target loss adds 0.12 times supervised contrast across different source people with the same task. The EEG identity
gallery uses band-specific scalp coherence. Once EEG recognition occurs, switch to the personal head and update it on
the same delivered prefix. Keep A1's weak virtual subjects, Euclidean alignment, source sample balance, seed, epochs,
external update budget and causal soft assignment. Each call uses one encoder and one classifier.

A2 uses the full B6 roster (ds007221 Graz 14 + Graz-paradigm left/right 48, NETBCI 19) alongside the first nine corpora.
For NI-V exclude Scherer and keep the B6 roster. Run common held-new and returning folds on all eight orders.
Predict A2 minus MANAS A1 +5 pp [−2,+12] held-new and +8 pp [−5,+21] returning, and +2 pp [−6,+10] NI-V.
A nested 80% upper endpoint ≤0 refutes improvement. A lower endpoint >0 supports improvement. The mixed change
covers representation, routing and B6 supervision together. It cannot attribute a gain to one component. Select using
the official held-new cell mean with returning and NI-V reported alongside. Compare against the archived cap250 bank
and use the exact production package for runtime timing.

### A1 MANAS and A2 dispatch

MANAS A1 scored **59.23% held-new** (59.14%, 59.32%) and **69.84% returning** (70.43%, 69.26%) in
Leonardo 64886270/64886271. The base pipeline is still short of cap250. Unlike REVE/CBraMod A1, this
fit also includes the ds007221 Graz and NETBCI exports. Keep the roster difference when comparing encoders.

A2 smoke 64887439 passed real-data gradient, prefix/reset, immutable weight and interleaved-session checks.
A2 external job 64887573 verified 14 ds007221 Graz, 48 ds007221 LR and 19 NETBCI people. It loaded 3,584,
6,144 and 2,432 sampled rows from those corpora. Eight-order fit/replay array 64887574 and collector 64887575.
Scherer-excluded A2 NI-V jobs 64887576, 64887578–64887581.

### Matched B6 control · registered before scoring

Run MANAS A1 again with A2's exact complete B6 roster, retaining the A1 temporal and routing path. Compare all
eight orders against A2 on held-new, returning and NI-V. Before scoring, predict B6 A1 minus first MANAS A1
+0.5 pp [−2,+3] held-new and +0.3 pp [−3,+4] returning. Predict A2 minus B6 A1 +4.5 pp [−2,+11] held-new,
+7.7 pp [−5,+20] returning and +2 pp [−6,+10] NI-V. Nonpositive nested 80% upper bound refutes improvement;
positive lower bound supports it. Choose the complete B6 package with the strongest official held-new cell mean.
The initial A1 runs are screens with different coverage; the two B6 packages control the corpus list.

CBraMod A1 reached **50.38% held-new** and **57.72% returning** in 3139398/3139399. The person-equal
deltas against archived cap250 were −24.34 pp [−28.00, −20.52] and −35.04 pp [−44.15, −25.97].
The shared official scorer verified all 28 held-new and 8 returning cells per order. Historical v3
`replay/check.py` printed OK on Galvani 3139584. The v3 guard covers recording order; Team A separately
checks the exported model, reset, and shuffled delivery.

A2 production 64887783 fit all ten public people plus the 12-corpus roster. ZIP SHA-256
`a292f815bf4153ab5d7fc5b54bd4424965b583c548d5a8d78cab4173dffee78f`.
Leonardo 64887784 timed this package on one A100-SXM-64GB, PyTorch 2.14.1+cu126. Total 223.52 s
including load and a 48-window warm-up, with 20,083 timed calls in 106 sessions, 410 MiB peak CUDA reserved.
The timing input repeats public sessions (76 recognized). This is an A100 development measurement.
Matched B6 A1 smoke 64887772 passed. External/fit/score 64887773–64887775, Scherer-excluded NI-V
64887777–64887782, release 64888133 and runtime 64888134. Recognition and zero-change checks 64888137/64888138.

### Eight-order selection

| System | Held-new official cell BA | Returning official cell BA | NI-V new cell BA |
|---|---:|---:|---:|
| MANAS A1, initial roster | 59.23% (2 orders) | 69.84% (2 orders) | 41.67% |
| MANAS A1, complete B6 roster | **58.79%** | **69.31%** | 42.10% |
| MANAS A2, complete B6 roster | 57.28% | 69.30% | 42.59% |

The complete B6 A1 fit is the selected system by the preregistered held-new cell metric. Configure it as
the default training path. A2's cell delta against the matched B6 A1 is −1.51 pp held-new and −0.01 pp returning.
A2's contrasts to archived cap250 are −17.23 pp [−20.31, −14.04] held-new and −23.47 pp [−33.53, −13.48]
returning (nested 80% intervals, equal people). Most of the P04 deficit remains. In the exact A2 replay,
the open-set route rejected both P04 visits on both checked orders.

Selected production release 64888133 completed on all ten PROTEUS people. Archive SHA-256
`4bd61c969408980e00eff3ac501ed53bef593eb0e3be04d0e20a4216ebd0fb87`.
Leonardo 64888134 measured 213.27 s on A100-SXM-64GB for the 20,083-call timing workload with load and warm-up.
This repeats public sessions and includes 83 recognized sessions.

Against archived cap250, the selected fit's official cell differences are **−14.95 pp held-new** and
**−23.46 pp returning**. Equal-person contrasts and nested 80% intervals are −15.25 pp [−17.84, −12.60]
and −23.46 pp [−33.04, −13.94]. The reference is paired by person. Its delivery orders differ.
Selected returning visit averages are P04 56.73% and P10 81.89%. The unverified sealed composition proxy is
−18.70 pp [−23.06, −14.36] with equal returning people, or −16.35 pp with the 20/80 P04/P10 proxy.

The matched A2 minus A1 nested contrast is −1.98 pp [−2.94, −1.03] held-new and −0.01 pp [−0.54, +0.57]
returning. The held-new improvement is refuted for this form. The returning direction is undecided.
NI-V is +0.49 pp by official cell mean. Both NI-V fits omit all Scherer training data.

The selected A1's exact repeat reproduced every held decision and pinned row order in the first two orders
(64888138). The EEG route accepted P10 at arrival 60 in one of eight returning replays and rejected all
28 held-new replays. Seven returning replays used the general classifier throughout. This weak
cross-visit recognition limits the returning score despite a trained personal head.
