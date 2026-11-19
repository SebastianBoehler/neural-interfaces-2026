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
