# §342 prospective spatial readout, 17 November 2026

Scope: INDIVIDUAL-MODEL source-held instrument only; no FULL-SYSTEM DEV or SEALED result.
Prior §329 retained1600 averaged all41 channels, retaining four temporal slots. Contextual tokens may already
encode spatial information; this is an untested readout axis, not a proven implementation defect.

Freeze exact cached default MIT Atom-L a28e97f7bd73bd7ff55f3b5d680f48807c335cf3134d1281228a061056aaf52b,
425 strict keys, matched random seed329, MATH SDPA/TF32off, 41×480/120Hz→800/200Hz polyphase /100.
Readout layouts: pooled400=all164 mean; temporal1600=41-channel mean; spatial16400=four-time mean;
full65600=all41×4×400. No raw-factor concatenation/new neural fitting/new assets.
Final source P04/P05/P06/P07/P08/P10 only; held P09 was previously exposed, not blind confirmation.
P01/P02/P03 labels and inference excluded. Inner train P04–P08, fixed P10 selection instrument.
For each eight prior/layout arms, fit source-only StandardScaler then divide by sqrt(D/400).
Select C in {.001,.01,.1} by minimum P10 official cell-balanced NLL, ties choose smaller C.
Refit six source people; persist all32 fits, scalers, source quality, selected C, provenance and posterior banks,
then fittedCPU/A100 sourceprobe errors≤1e−3 probability and zero nontied changes BEFORE reading P09 labels.
No P09 hyperparameter optimization or winner expansion; report all eight arms and all contrasts.

Numeric priors: spatial−pooled +3 INDIVIDUAL-MODEL percentage points [−4,+10]; full−spatial +0 [−3,+4];
pretrained−matchedrandom +1 [−3,+5]. These are predictions, not measured gains.
A negative spatial contrast refutes this specific linear retained-channel advantage; a near-zero one leaves
broader mechanisms undecided. One held person cannot close a family or establish population transfer.

Engineering stage: CPU real-data loading/token-order/425keys/input/reset/gradient/source exclusion and small
independent C-aware objective/gradient oracle; then ONE A100 compatibility/cost smoke, no held metrics.
Pinned309GPU lib027df0434ed7096b6110d0c500bbd83c796a473d21426f3953dbc245b11b394e only.
Independent objective meanCE+||W||²/(2*C*n), unpenalized bias; original LBFGS limits unchanged.
Initial budget ≤.5 A100h+2CPUcoreh, actual capacity≤32 running/244 active. Abort on guards or budget failure;
engineering failure is not an EEG-family ceiling. Source snapshots immutable; data/checkpoints stay remote.
