Current: native source COMPONENT parity PASS3120312. Maximum CP error2.41229259e−12<1e−8; OwnStage active/no bypass, reset/deepcopy/non-tied labels asserted. Whole-stack remains unqualified (separate neural mismatch). Details native353_routing_parity.json.

Current: component-only capture3120062 COMPLETE65s,1.306GiBRSS, unknown10CPrefits/125fusions/0singleclass/0errors. Native numeric comparison awaits root build. Full-NN215 timeout remains failed.

# Native353 routing and emitted-label stages

Prepared17November2026 under the existing engineering-only native353 budget. No accuracy scoring, new training fit or frozen package edit. Implements original v6.6 ContrastGate, IdentityGate, OwnStage, Route/RouteB, CountFilter and CausalPers using LibTorch and exported fixed SciPy SOS coefficients.

Exact Python source hashes match the immutable Galvani inputreuse2/source66 copy:
- gates.py abcf5db200d9f07f88d2fa56ff88ac781c78880e67f8a28bf8734c4c112ba3c0
- personal.py 6c4f8ca166b2cb24a95b6900c3bd674298630b029d64432907c63a1e45afbe1b
- routing.py f767dd227b9592d64a19118be73b18fcd48567de000c7555c5af515d6b51cca6
- countfilter.py 232a003561af247508f4387dc58f35b9d8275411d95719d0edfcf2fd28107a91
- cpers.py 2aae3dc07624b7786d26020aa99ce8488e28a0484582e228fddf20afe756e53e

The installed sklearn discriminant_analysis._cov/_solve_lstsq/predict_proba and covariance._shrunk_covariance._oas/ledoit_wolf_shrinkage source were inspected directly on Galvani. pyriemann's OAS wrapper calls sklearn OAS on transposed channel/time data. Native code uses centered biased OAS, one-versus-rest generalized-eigenvector CSP, per-class standardization/Ledoit–Wolf shrinkage, prior-weighted covariance and least-squares LDA. Binary softmax of two discriminants matches sklearn's logistic discriminant difference. The packaged CausalPers holds one latest CSP/LDA model, not six checkpoint models. Numerical equivalence remains to be tested; no claim follows from source similarity.

API: routing.h contains optional GateDecision and Route(z,core1_share,outer); step uses actual gates; step_given uses the unchanged inner gate decisions for RouteB. OwnStage observes every window. set_core1_share and set_prior support actual outer normalization/TSEM prior. Output stages expose counters, latch/stopped and error/dead state. CausalPers returns unnormalized log scores once fused, exactly as the original emitted-label path. Deep-copy constructors clone all mutable tensors/history; fixed Params tensors remain shared read-only. Original stage-error bypasses remain, with explicit counters and validation rejecting any bypass.

Local C++20 LibTorch syntax checks PASS for all five implementation modules and source test. C++17 fails the installed Torch headers, not the new code. Local routing_source linking PASS; numerical parity is still pending.

Source reference3119621 TIMED OUT at its original20min limit (14:36:54CET), without a complete fixture/proof. Earlier known80 completedrec10/70OwnStages/noerrors, but the final archive was written only after both cases; no numeric parity is established. Failure is preserved, no extension/resubmit. Its original scope was in ni26_native353_routing_reference2 (2CPU16GiB20min). It attempted actual original active5/draft_base P01session1 known80 and forced-unrecognized215 windows, with no task labels or metrics. Source test compares route outputs, both gates, count-filter posterior and CP log scores at≤1e−8; tests deepcopy at17/204, reset-prefix replay, no silent stage errors, and actual late CP activity. Expected CP refits derive from source t0=20/k20 (10 by215), with125 fusions;20 is the separate FT refit count. Pending3119620 was canceled beforestart to correct that mistaken fixture assertion; old source retained. No model/scientific parameter or tolerance changed.

§350 remains separate: source guard3119571 is running, no new policy gold/cohort released. Its single continuous tail policy is not part of frozen v6.6 and is not silently included in the native target.

Prepared UNRUN `native/tools/routing_component_fixture.py` removes full neural inference only from the validation instrument: actual frozen core1/core2 outputs form the same half-mixture used by the root core/tail fixture, then actual Python Route/Own/Count/CP process real P01source windows. This is explicit component-only parity, not whole-model reference. It checkpoints each complete case and prints20-window progress. No allocation until revised shared353 budget; combined346 neural/RouteB capture remains separate preparation.

Inspected host and clean worker versions agree: sklearn1.9.1, scipy1.18.1, pyriemann0.12.
