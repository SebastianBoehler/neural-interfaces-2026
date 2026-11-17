# External cognitive supervision: retain its own task during raw-encoder learning

Prepared 17 November 2026. Status: research/design and read-only data audit; no fit,
held-person score, download, package change or job submission. Source revision at inspection:
`62a3bdaf335ba3d1a5e4d17eb448139b0f4b2ef9` (working tree also contains other agents' changes).
Applied scientific-discovery skill. Parent chat was not inherited, but the task brief and mandatory
STATE → LEDGER → OFFICIAL_CONTRACT expose incumbent results/history. This is not historical isolation.

## Decision

Test **joint PROTEUS classification plus a separate Shin2018 WG/baseline head**, rather than
pooling external WG into PROTEUS WORD, mapping teacher outputs, or repeating unlabeled §312 pretraining.
This is the explicitly unrun form at §278, now specified against an information-preserving raw encoder.
The scientific intervention changes supervision and when it reaches the encoder, not just its final head.
Keep Scherer out of the first test: its clinical population, missing low frequencies and data terms add
three avoidable confounds. This does not close Scherer supervision.

## What actually exists and what already ran

| Mechanism | Actual coverage/context | Remaining distinction |
|---|---|---|
| Scherer pooled labeled rows | §7 FBCNet/raw Shallow, paired control, older chain: nets +1–3pp alone, nested chain −.23pp; §191 four SessionBN families + FT/eye/rest, within0/1 chain −.23/−.51pp | Does not test task-private external losses during training of a strong whole encoder |
| Scherer external teacher | §18 EEGNeX+IFNet, corpus-private pretraining blocks, frozen heads mapped to PROTEUS: alone +3.3pp; chain −.98 J; §132 session normalization failed to rescue it | Corpus-private heads already exist. Merely renaming them is not a new mechanism |
| Shin cognitive teacher | §148 cog1 used subtraction/rest only; rebuilt MI control reproduced; §149 unified block dropped corpus heads and was weak | Neither is joint PROTEUS+WG supervised encoder learning |
| Shin WG as PROTEUS WORD | §278 26 people,16-native-channel tangent decoder, baseline dropped after centering: −6.03pp at w=.3; offset correction −1.81pp | Real source signal (66.7% WG/baseline), incompatible class geometry; separate loss avoids forced class equivalence |
| External/in-domain SSL | §147 SimCLR FBLight output projection; §277 EEGNet SimCLR/reconstruction26-channel probe and30k-step FT, weak/negative in those forms | Task labels provide different information; no claim SSL broadly closed |
| Raw rest reconstruction | §312 FBCNet/Shallow native raw rest, source-only masked moment reconstruction + task FT | Concurrent lane; do not reuse or alter its mutable source files |

Inspected executable paths: `experiments/pretrain.py`, `scripts/export_external.py`,
`scripts/export_moabb.py`, `studies/15nov/shin278/export_shin.py`, `studies/14nov/shin/shin278.py`,
and §312 encoder/training files. The old pretrainer is shared trunk/per-corpus CE followed by transfer;
it does **not** retain external losses while PROTEUS task fine-tuning runs. Its `--unified` save path
explicitly slices the shared head to three outputs; this explains the recorded lost-head limitation.
§191's original `~m` was caught as a no-op before production; the actual wave used unmasked weight1.

Read-only Galvani audit is recorded in `external_metadata_17nov.json` and
`shin2018_native_metadata_17nov.json`. Important correction: `data/shin2018wg/windows_1-13.npz`
is an older partial export, **not** the full corpus. `data/shin2018/windows.npz` has26 people,
3120 windows, balanced1560/1560, baseline=0/WG=1, trial/block/window identifiers and16 recorded
PROTEUS channels. The old partial file instead labels S16=0/S32=1. Do not mix those label orders.
Shin2017B has3480 windows/29people but35of41 interpolated channels, so exclude it initially.
Scherer has2130 selected windows/9people,22of41 interpolated. The existing nine raw MAT files
actually retain3550 trials,710 each WORD/SUB/NAV/HAND/FEET; exported three-class windows discard
1420 NAV/FEET trials. Raw hashes/session counts are captured without reading target scores.

## Primary evidence and transfer limits

- [Shin et al., Scientific Data,13 Feb2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5810421/):
  simultaneous EEG/NIRS,26healthy participants,WG/baseline plus other cognitive tasks. Only EEG
  is available at inference here. Its WG contrast is not MI/SUB/WORD; block/cue effects can dominate.
  [Official MATLAB scripts](https://github.com/JaeyoungShin/simultaneous_EEG-NIRS) disclose that
  their tutorial omits EOG rejection. No published EEG/NIRS result is a forecast of PROTEUS accuracy.
- [Domain Separation Networks,22 Aug2016 preprint; NIPS2016](https://arxiv.org/abs/1608.06019):
  shared/private representations motivate separating acquisition-specific information. Its image
  experiments do not establish this EEG transfer. The paper's official TensorFlow path is
  `tensorflow/models/domain_adaptation`; the current fetched path was unavailable. No reproduction claimed.
- [Original BNCI description,April2015](https://bnci-horizon-2020.eu/database/data-sets/004-2015/description.pdf)
  confirms five labels,cue at trial+3s,30electrodes,256Hz and clinical participants. Its acquisition
  specification says0.5–100Hz; our ledger reports effective source high-pass near4Hz. Do not equate
  nominal acquisition filters with the observed downloaded signal spectrum.
- [Scherer publication,18 May2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0123727)
  has an article CC-BY statement. [Current source catalog](https://www.nemar.org/dataset/nm000144)
  and MOABB list **dataset CC-BY-NC-ND4.0**. Article licensing and organizer competition clearance
  do not resolve dataset redistribution rights. No Scherer artifact would ship from this proposal.
- [Official Shin repository](https://depositonce.tu-berlin.de/items/version/24) confirms dataset identity,
  public files and2017 deposit dates. Existing export code calls it CC0, but the retrieved repository
  HTML did not expose that license field. Recover exact item/file terms before new production training
  or redistribution; do not convert this audit into a claim that terms were independently verified.

## Concrete experiment card

**Question:** Does retained, task-private external WG supervision teach a useful person-general raw
representation without injecting a false external-WORD class prior?

Copy the proven FBCNet raw encoder/control recipe into a new source-locked workspace. Preserve all41
PROTEUS channels, source-only scalar normalization, channel means and power skip, optimizer, seeds,
PROTEUS minibatch sequence and selected epochs. Add a separate binary auxiliary head on the same
encoder representation; its logits never enter the PROTEUS readout or inference. Feed Shin through
its verified41-channel export initially, with a distinct source-fitted scalar; explicitly log its25
interpolated channels. Do not claim this removes the interpolation confound. A native-channel adapter
would be a later separately matched design, not another option in the first matrix.

Fixed loss: PROTEUS balanced3-class CE +.25×Shin balanced2-class CE, one equally sized Shin minibatch
per PROTEUS minibatch, sampling people first and then source trials; use one of the two windows per
trial per draw. Auxiliary gradients update the shared encoder throughout supervised training.
Separate binary head avoids demanding that WG/rest separate along PROTEUS WORD/MI/SUB geometry.

Matched arms: **A** PROTEUS-only extra-step control; **B** identical joint architecture and sampled
external inputs with auxiliary gradient stopped at the encoder; **C** active joint auxiliary gradient.
B/C have identical parameters, batches, floating work and PROTEUS updates. Keep normalization
statistics separate by source, or use frozen/no-running-stat normalization identically in B/C;
otherwise B still changes the shared encoder through BatchNorm buffers and is not a no-transfer control.
This implementation choice requires a new guard and fresh baseline; do not silently substitute it
for §312's original model. No auxiliary source BatchNorm state is used for PROTEUS inference.

**Next test:** one source-only fold: exclude outerP01 completely; train onP03–P10, validate onP02,
seed0,40epochs maximum. On the Shin side withhold people1–5 for auxiliary validation and sample21
others. Validation never selects usingP01. Before any training, assert source terms, exact classes,
26people/3120windows, native mask and grouped trial identity. Before any accuracy release, show:
B/C equal initialization and PROTEUS rows, auxiliary gradientC>0/B=0, only authorized encoder
parameters/buffers move, finite normalized predictions and session reset/prefix identity. Log CE and
auxiliary validation discrimination so a zero-effect loss or broken labels cannot masquerade as failure.

Prediction: C−B source-validationBA +1.0pp with broad80% prior[−2,+4]; auxiliary held-source BA65%
[55,75]. No claim of confidence from a single source validation person. If valid fitting gives
auxiliaryBA≤52%, first inspect source labels/optimization; it is not a transfer refutation. If C's
PROTEUS validation drops>3pp with learned auxiliary signal, do not expand that coefficient blindly.
If numerically valid, full evaluation is ten held people×two seeds×three arms, all eight orders,
outer labels opened only after checkpoint choices freeze. Predict whole C−B +.7pp[−1,+3]; intervals
resample people and orders. A negative interval refutes this form; small or zero-crossing effects stay
undecided. Compare complete raw decoders to the measured76.218% current system, not merely to each
other. A component claim additionally requires chain/routing/stages that actually consume its features.

Cost anchor: §312 source guard measured FBCNet213.3s,Shallow155.3s and exact strong source fit286.8s
on its CPU allocation. Those are related fits, not this architecture's measured runtime. At one equal
external forward/backward per PROTEUS update, conservatively allow2–3×FBCNet time per active fit:
about7–11min CPU,20–35min for the three-arm pilot plus I/O.60fits imply7–11CPU allocation-hours;
concurrency2 gives ideal3.5–5.5h, conditional on source pilot and actual slots. Warm paired pilot costs
must replace this extrapolation before release. Use Galvani only; account≤32running/272submitted,
reserve coordinator slots, no production release here.

## Follow-up: source-only §317 pilot

The coordinator authorized the bounded source-only pilot. §317 is now registered in the ledger;
its precise protocol supersedes the earlier draft controls above. Owner HTML directly fetched
with curl exposes license elements omitted by the browser extraction: historical version24 isCC0;
current item `eb49d988-33d4-4eac-9ed3-36dcd711f9d2` isCC-BY4.0. Both identify doi10.14279/depositonce-5830.
Exact excerpts and HTML hashes are `shin_terms_17nov.json`; use Shin et al. attribution.

Code `studies/17nov/aux317/` implements separate main and auxiliary optimizer steps. A gets extra
PROTEUS updates; B computes full auxiliary backward but discards only auxiliary encoder gradients;
C retains them with fixed .25weight. An independent main-only twin checks B's total encoder update
bitwise, including parameters/buffers; auxiliary head gradients remain active. Every auxiliary forward
uses body.eval, verifies every body buffer bitwise and restores the prior body mode. A shared,
independent main minibatch generator preserves source exposure across all three arms.

Read-only raw VP001/word1.vhdr inspection confirms1000Hz acquisition, EEG channels plusHEOG/VEOG,
BrainVision µV units and S16/S32 markers. MNE returns volts, and the existing exporter multiplies1e6
after120Hz resampling, so the cached model-facing values areµV. Standard-name matching leaves the
16recorded official channels listed in the metadata;25are interpolated, not native observations.

One-epoch CPU real-source smoke3116563 submitted in separate source-locked workspace
`/mnt/lustre/work/gehler/gfh527/ni26_aux317_20261117_smoke1`,6CPU/20GB/8minutes. Local Python compile
and Bash syntax checks passed. A40epoch pilot may follow only after the smoke guards pass;
no full held-person matrix is authorized and no outer accuracy is opened.

Smoke3116563 completed0:0; all numerical guards passed, including independent total main-only/B
encoder-update identity and auxiliary BN invariance. One-epoch elapsed times A8.89s/B8.61s/C8.32s;
B effective auxiliary encoder gradient0, C maximum squared-gradient sum3127.26. Source-only one-epoch
BA is a debugging diagnostic, not selected evidence of efficacy. Full40epoch bounded pilot3116570
submitted to separate `ni26_aux317_20261117_pilot1` workspace after capacity check15running/87pending.
Linear extrapolation is about17min for all three arms; allocation45min remains conservative. The
same source manifest is pinned; no full matrix or chain score is queued.

## Full study continuation

Source pilot3116570 completed0:0, actual02:24:04–02:39:25CET on17Nov (15m21s).
Selected sourceP02BA A73.506%,B73.598%,C72.859%; selectedepochs9/10/10. C−B−.739pp
stays within the declared3pp harm bound; A is.529pp below prior strong raw74.035% control.
Selected auxiliaryvalidationBA B56.67/C58.0% exceeds the declared52% optimization concern.
All registered numerical/source/BN/no-transfer guards pass. This is one source person's result,
not held-person efficacy evidence or a current-system gain.

Following the user's unattended-work authorization, the concrete60foldCPU study and private315
consumer were preregistered in§317. Repaired nested-producer smoke3116739 passed after a missing
benchmark_utils dependency blocked firstsnapshot3116724 beforetraining. Failedsnapshotmatrix1
is preserved; immutablematrix2contains the unchanged official submodule link and metric hashes.
Afterokcontroller3116804 releases24/24/12fold waves with≤4allocations, accountcapacity/reservation
checks, exactcomplete60fold wholecollector, then privatelyforked315consumer. See `aux317_handoff_17nov.md`.
No outer accuracy or partialfold result has been opened. Consumer additions retain existing
views/routing/stages and use the same fixed two-seed arithmetic-probability ensemble in sixCPbases.
