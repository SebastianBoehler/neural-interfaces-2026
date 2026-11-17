# §354 DEFT-equation transfer on strong raw FBC — prospective first wave

Accuracy lane independent of native port. No score exists at registration.
[DEFT paper](https://arxiv.org/html/2610.14497v1) and
[author implementation](https://github.com/wythedee/DEFT/tree/875b95fe729c07b8b20b69c1104651863f984682/CBraMod/disentangle_v7)
were read first. DEFT separates content/style, not explicit spectral bands: four
Gaussian static/dynamic passes, gated counterpart subtraction/precision fusion,
temporal constraints, subject contrast and FiLM reconstruction. We transfer those
mechanisms to raw FBC's intact9frequency×4time×32spatial tokens. This is neither
an EFM reproduction nor292C's small residual adapter on a weak CBraMod baseline.
No author source/checkpoint is copied/imported; independent implementation.

**Fixed controls:** O unchanged strongest raw FBC SessionBN/original sampler;
A identical raw FBC with paper-motivated two-person sampler; M active CE-only
MLP capacity control; L MLP content/style factorizer with the SAME full losses;
G recursive Gaussian factorizer/full losses. M/L/G add47,026/47,024/47,075active
parameters (≤.11% difference, no dummy padding). Original backbone/head initialized
identically; constrained1152→3 head unchanged. Main contrastG−L; alsoG−O/G−M,
L−M,M−A,A−O. Every model trained whole, scored as INDIVIDUAL-MODEL pilot.

**Fixed source/training:** heldP02/P03/P09 separately, each nine OTHER public
people, seed0,40epochs, canonical original step count per fold, AdamW.001/wd.001/
original OneCycle. No HPO/checkpoint selection; final40-epoch weights fixed before
query transformation/labels. Native input/scaling/FIR/SessionBN machinery unchanged.
A/M/L/G have identical64-row plans: two different source people,32rows each from
one source session, no replacement within a chunk. O keeps original sampler,
including occasional final32-row batch; A controls this exposure/sampler change.
All nine source people/rows must be visited across40epochs. Only selected source
labels loaded; query labels−1 during fit. Original class cache computes pointwise
covariances of all EEG, but source G/scale/weights use ONLY fit rows; no claim that
held EEG was never pointwise transformed. No test labels or query statistics fit.

**Fixed architecture/loss transfer:** latentD32,16 transition bases, positive diagonal
precisions, initial obslogP4/priorlogP−4 and gate−2. Four posterior Gaussian passes
follow paper equations; dynamic transition/precision are data dependent. Identity
transition banks get small.001 distinct perturbations and generator.01 noise so
mixture learning cannot remain trapped in identical-bank symmetry. This registered
initialization differs from released code; paper does not prescribe bank initialization.
Frequency residual reinjection preserves the native token grid. Learned priors reset
for every EEG window; no factorizer state crosses trials/sessions. Dimension-normalized
paper variance hinge margin.01, stylevariance, subject SupCon τ.07 and feature-FiLM
MSE weights.1/.01/.2/.02; unweighted original CE weight1. This registered transfer
uses paper variance ordering, not the released code's MPD-ratio alternative.
Width/optimizer/pretraining differences mean no claim to reproduce published gains.

**Numerical predictions BEFORE any score:** mean three-person primary scheme-equal
8-order BA: O/A/M63%[58,69], L64%[58,70], G65%[59,71]. G−L+1pp[−2,+4],
G−O+2pp[−3,+5]. Same official equal-cell scorer and eight pinned within0–3/full0–1/
runs0–1 orders; primary equal weighting across the three order schemes matches336/338,
uniform8 supplement separate. Resample people AND orders for paired80% intervals.
All three exposed before this study; no blind/generalization/sealed/rank claim.
Intervals spanningzero or effects within±.5pp stay UNDECIDED. Clear adverse paired
results refute this exact transfer, never DEFT/factorization as a family.

**Guards before gold:** source/held exclusion and original source hashes; one-epoch
O training BITWISE to original reference; identical backbone initialization and
paired A/M/L/G plans/step counts; active body/head/auxiliary gradients and genuine
data-dependent Gaussian mixture; finite outputs, all source rows/people visited;
8-order coverage/indices; prefix/no-future, deepcopy/reset/interleaved-session and
unchanged inference weights/cache. Entire15cell cohort/banks/guards required BEFORE
collector opens any query labels. Replay check must printOK before chain claims.
Any source/no-op/guard failure invalidates affected arm; no fallback/partial scoring.

**Consumer/follow-up:** positiveG−L≥.5 withG−O≥0, or consistent smaller positives,
justifies independent people/seeds and a prospectively fixed current-system consumer
using existing full-system machinery. Useful addition must be measured through the
current stages/routing; a coherent whole alternative must score whole. No pilot gain
is called FULL-SYSTEM DEV. Full10person/current consumer not yet run or budgeted;
no automatic promotion. No second architecture before first-wave evidence justifies it.

**Bounded first wave:** one real-source CPU2/24GiB5min smoke (≤600coresec), then
15parallel source-person cells CPU2/24GiB11min/array%8 (≤19,800coresec), one existing-
metric collector CPU1/5min (≤300): max20,700coresec=5.75CPUh/0GPU, within6CPUh.
Account≤32running/272submitted with28reserved; preserve unrelated jobs. Fresh
immutable ni26_factor354_v1_20261117 source/import/shape/preflight before submission.
Source producer expected5–9min based strong source6A273.68s/CPU6, source9 larger;
11min caps are NOT ETAs. No tolerance relaxation/automatic failure retry/extra fits.
Frozen packages and official submodule unchanged. Source in studies/17nov/factorization354.

## Source-smoke actual and prospective first-wave scope correction

3121748 PASS:112.86s internal, original oneepochO BITWISE, identical backbone
initialization, matched A/M/L/G plans, expected active parameter counts. All four
auxiliary terms active; Gaussian generator gradient5.68594e−5 and per-batch mixture
variation.00397394. No query forward/labels/accuracy. Measured per-epoch fit times
O17.122/A17.481/M18.039/L18.132/G18.560s predict G~742s BEFORE inference; the
original11min cap is therefore too short. No production fit or gold preceded this
runtime finding. Go past the original15cell cohort BEFORE scores for that reason:
FIRST production cohort is only O/L/G×P02/P03/P09,9cells, exact original dispatch
indices0,3,4,5,8,9,10,13,14. A/M remain smoke-tested but unscored; do not claim their
planned contrasts. PrimaryG−L still matches capacity/objective/sampler/steps;
G−O is the joint factorizer/objective/sampler transfer versus unchanged strongraw.
All9cells/8orders/guards must finish before gold; no selected/partial scoring.

Override Slurm array to those exact9indices/%8 and capCPU2/18min percell. Maximum
19,440production+600smoke+300collector=20,340coresec=5.65CPUh, below6. No epoch/step/
mathematical/guard weakening, no retry/extension. Source snapshot unchanged from
passing smoke. There were40canonical steps per epoch in smoke and40productionepochs.
Expected producer13–15min plus queue; cap18min is not an ETA. Nine-fold firstwave
scope refers9cells over3heldpeople, NOT9independent people; no family closure.

Pending-only resource correction BEFORE those fits start: smoke Slurmpeak
1,793,256KiB across all five architectures, with fixed source/batch shapes and no
epoch-dependent tensor cache. Predicted production peak≤4GiB including larger
held batches/deepcopy. Lower only still-pending cells5,8,9,10,13,14 from24→8GiB
to improve backfill; keep running0/3/4 unchanged, CPU2/40epochs/18min/array%8 and
all math/source/order/score bounds unchanged. If8GiB proves insufficient it is a
resource failure, not a scored model, and no automatic rerun. Scheduler actual
memory requests/peak, not legacySBATCH memory comment, own the resource record.
Pinned permutation archive SHA6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f.

Collector guard audit BEFORE any gold: producer smoke records positive auxiliary
losses and total/generator gradients, but did not independently test each weighted
auxiliary's graph. Fresh collectorv2 adds a source-only scratch gradient check on
actual source EEG/strongO learned tokens with independently initializedL/G. Each
weighted style/contrast/reconstruction term must have finite nonzero gradients;
positive ordering hinge must too; an already satisfied zero hinge is recorded
as such, not miscalled a no-op. No optimizer/update/extra fit/query labels. This
checks implementation paths, not diagnostic task accuracy or convergence benefit.
Canonical permutation file must match6046bf5b…430f. Existing official scorer and
bootstrap semantics unchanged. Cancel only old pending collector3121852, preserve
its snapshot; fresh immutable collectorv2 after all9fits. Slurm may allocate2CPU
for1requested, so reserve collector600 rather than300coresec: worst20,640<21,600.

Actual submissions: smoke3121748 COMPLETE123s×CPU2=246allocatedcoresec;
production3121766 exact9indices above, source manifest3d662db2…e4f3c. Original
collector3121852 cancelled PENDING/no labels; guardedcollectorv2 **3121898** waits
afterok all9producer cells. Live metadata after pending RAM reduction showed8
RUNNING/%8 and1pending. Early actual batch-step peaks O1,459,932/L2,018,444/
G2,160,972KiB;8GiB covers≥2×observed and predicted≤4GiB full inference peak.
Source workflows use one Python process plus shell, direct Tensor batches and
in-process deepcopy; no process-worker/DataLoader pool. Account actuals/late
inference peaks will be reported before claiming a resource result.
