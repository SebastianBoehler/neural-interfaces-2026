# §320 forward-only, source-crossfitted branch reliability

Prepared17Nov2026. Research exposure: inherited project instructions, STATE/LEDGER, previous§317
research and coordinator brief. No independent-history or timed-research-block claim. This is a
learning/inference mechanism on the existing raw encoder, not an independent representation.
Source/data supervision remain unchanged; no external dataset is used by this lane.

## Primary source and actual code

[Li etal. BFT](https://arxiv.org/abs/2601.07556) was submitted12Jan2026 and revised17Aug2026(v3).
It is not a publication from the last few days. Its fixed feature-mask branches permit shared
encoder inference; learned branch reliability changes the weighted prediction. Its examples use
binary motor imagery and drowsiness regression, not PROTEUS MI/subtraction/word generation.
Masking can remove task information here; label preservation is a hypothesis, not an assumption.

Reviewed author code at commit`603e30dc4534144194fe88d4bec38fdd67f5d079`:

| Actual implementation | Exact primary code pointer | Consequence for this test |
|---|---|---|
| Fixed contiguous masks, surviving features unscaled | [tl/bft.py131–154](https://github.com/sylyoung/DeepTransferEEG/blob/603e30dc4534144194fe88d4bec38fdd67f5d079/tl/bft.py#L131) | Our bank explicitly specifies inverted4/3 scaling and includes identity |
| Task CE and predicted reliability averaged across batch | [tl/bft.py241–276](https://github.com/sylyoung/DeepTransferEEG/blob/603e30dc4534144194fe88d4bec38fdd67f5d079/tl/bft.py#L241), [losspredictor.py227–252](https://github.com/sylyoung/DeepTransferEEG/blob/603e30dc4534144194fe88d4bec38fdd67f5d079/paper/BFT/code/BFT-classify/models/losspredictor.py#L227) | We supervise each row from source-held-person predictions |
| Mapping loss coefficient defaults0; sorter term multiplied0 | [sodeep.py101–126](https://github.com/sylyoung/DeepTransferEEG/blob/603e30dc4534144194fe88d4bec38fdd67f5d079/paper/BFT/code/sodeep/sodeep.py#L101) | We use explicit pairwise logistic ranking, not nominal SoDeep reproduction |
| Reliability is averaged over previously seen rows | [dropout.py150–157](https://github.com/sylyoung/DeepTransferEEG/blob/603e30dc4534144194fe88d4bec38fdd67f5d079/paper/BFT/code/BFT-classify/dropout.py#L150) | Our probe is strictly current-row/stateless |
| Actual output temperature.25 | [dropout.py163–166](https://github.com/sylyoung/DeepTransferEEG/blob/603e30dc4534144194fe88d4bec38fdd67f5d079/paper/BFT/code/BFT-classify/dropout.py#L163) | Primary controls all use1; calibration changes only postaggregation confidence |
| Weighted path adds BN adaptation; uniform path stays eval | [dropout.py179–186](https://github.com/sylyoung/DeepTransferEEG/blob/603e30dc4534144194fe88d4bec38fdd67f5d079/paper/BFT/code/BFT-classify/dropout.py#L179) | Every arm uses the same frozen populationBN |

The release trains its source model on `src_data/src_label` (`train_pre_model.py73–86`) and loads
that model before ranker fitting on the same named source pool (`train_loss_model.py77–104`);
there is no person-crossfit loop inside that ranker call. Existing local code also confirms
`sesstype/desc.py` uses firstK session descriptors, `gated2/mask.py14–22` uses fixed confidence
thresholds, `repair/transform.py16–19` applies temperature, and `deep.py421–425` already has optional
uniform circular-shift prediction averaging. None of those snippets learns source-person-crossfitted
per-row mask reliability. This is a scoped audit, not a claim all reliability methods were absent.

These are observations of this public revision. They do not establish which code produced the paper's
reported experiments. Code licenseMIT; file SHA-256 values in `reliability320_primary_sources.json`.
No upstream code is executed or substantial source copied; our implementation is BFT-inspired.

## Existing coverage

| Earlier mechanism | What actually ran | Why this test differs |
|---|---|---|
| §236 session-type rule | First40window descriptors choose stage constants; nested heldpeople−.28pp on older v6f, sixorders | Learned current-row branch ranking, no session descriptor/state |
| §296 confidence gating | Fixed confidence threshold decides whether to add EEGNet, with own downstreamstages | No learned threshold gate/new-family slot: same encoder's fixed view bank |
| Fine-tuning augmentations | Changes training/input/pseudo-label adaptation with sessionstate | No inference learning; model/ranker buffers remain frozen |
| §302 softening | Existing member's probabilities transformed by fixed temperature; T2partly repairs harm | Uniform/ranked views have matchedT1; calibrated controls isolate confidence effects |
| §317 continued external supervision | JointWG auxiliary loss, separate running source-locked matrix | No external labels or change to task encoder objective here |

## Concrete bounded source pilot

OuterP01 and inner-validationP02 are excluded from every encoder fit, normalization, ranker and
calibration fit. SourcesP03–P10 supply eight person-crossfit folds: query one person, train the raw
encoder on the other seven. Each uses fixed10epochs chosen prospectively before§320scores; this choice
is informed by§317's earlier9/10/10source-selected epochs. Final raw encoder fits all eight sources
for10epochs. P02 is a separate evaluation person for this run but has previous campaign exposure.

The backbone preserves FBCNet body, the41channel means and246logpowers, source-only scalar input
normalization and source-only moment standardization. Unused reconstruction head/frozen-mode option
are omitted from this standalone copy; the task forward representation is unchanged. PopulationBN
freezes for every source-query and deployment forward. Strong-control validity requires unchanged
head sourceBA≥68.60% (within5pp of§317's73.598% source control); a failing control blocks expansion.

Bank: identity plus four disjoint contiguous body-feature masks. Masked surviving body features are
scaled4/3. All287mean/power skip features remain exact copies. Bank bytes/hash are fixed across
source ranker training/deploy, with no resampling and no waveform/amplitude augmentation. Mean
branch logits mathematically equal identity logits for this affine head; check this to expose scaling
or skip corruption. The different mean of softmax probabilities is explicitly the uniform comparator.

A shared17→16tanh→1 ranker consumes branch/identity probabilities, their absolute difference,
entropy/max/margin and fixedbranch identity. It uses posterior descriptors rather than raw latent
coordinates, since independently trained crossfit encoders need not align their latent coordinates.
Pairwise logistic supervision uses per-row negative-label-log-probability branch ordering, minimum
lossgap.01, person×class-balanced query weights,400AdamWsteps lr.001/wd.01. No source-in-sample
prediction enters ranker supervision. This differs from the paper's feature-input/sorting objective.

Controls: H unchanged head; U uniform same-bank aggregation; S constant global weights from mean
source-crossfit branchCE; R current-row learned weights. Primary alltemperature1. HC/UC/SC/RC apply
only postaggregation temperature from fixedgrid{.5,1,2,4}, independently selected on source-crossfit
class-balancedNLL, never P02. Calibrated argmax must equal its T1arm, separating whole accuracy from
confidence effects that might matter in the current chain. No temperature or mask selection onP02.

Real-source smoke first: one1epoch seven-person fit, disjoint sourceperson query, small rankergradient
check; identity, zero-mask/samepath, zero-ranker/uniform, nonzero masks, unmaskedskip, populationBN
immutability, deterministic bank, interleaved-source reset and prefix/batch invariance. Then the same
oneallocation runs the eight10epoch crossfits and final8person encoder. All smoke failures stop it.

Prior: R−U sourceP02BA+.5pp, broad80% prior[−2,+3]; R−S+.2[−1,+2]; H73%[68,77]. Future whole
R−U+.5[−1,+2], full-current-systemR−U+.15[−.6,+.9]. These are priors, not confidence intervals.
ControlH<68.60%, nonfinite/nonzero-gradient/source/BN/identity failure blocks expansion. Rworse than
U by>3pp or source weightSD<1e−4 requires diagnosis before expansion; neither closes the family.
Otherwise the source evidence guides whether a full study is justified; no automatic large matrix.

Any later whole test needs a separate concrete preregistration before outer scores, ten people,
source-crossfitted ranker/calibration splits, two seeds, all8pinnedorders and complete-matrix source
hash/label exclusion/checkpoint/order guards. Compare H/U/S/R whole and against76.218% incumbent;
report people/order intervals. Any component claim requires same fixed ensembles and weights,
current-chain unchanged control/zero/member-use/replay/order/routing/6CP guards, calibratedmatched
arms and complete outer producers first. No outer scorer or full consumer is released by this pilot.

Cost anchor: §31740epoch two-update arm~297–313s on6CPU, §312single-update sourceFBCNet213s.
Nine10epoch single-update fits estimate6–12min plus I/O/ranker; request one6CPU/16GB/30min Galvani
allocation, explicitCPU only. Account≤32running/272active and28reservedslots. Record actual cost.

## Prepared and queued

Single allocation3116903 is queued in
`/mnt/lustre/work/gehler/gfh527/ni26_reliability320_20261117_pilot1`.
Source manifestSHA`204c09593294d269a528f67d0640b59c560177fae043090ae7b1aa5f2b276726`.
Python compile and Bash syntax checks passed; no real source smoke or320accuracy result yet.
The job runs smoke first and stops on failure before the full pilot. Before execution its requested
wall time was reduced30→20min (resource-only amendment) to improve backfill; scientific source remains
immutable. At03:22CET the scheduler estimated03:49CET start; this is not a guaranteed start/completion.
Account snapshot before submission14running/90pending. No further job or automatic outer release.
