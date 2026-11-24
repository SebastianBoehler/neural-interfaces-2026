# Experiment lanes

The lane IDs match the original experiment records. See the [preregistration ledger](../../docs/LEDGER.md)
and [run evidence](../../evidence/) for hypotheses, results, and limits.

The source below is archived with its original script arguments and cluster paths. To run a historical command,
check out its recorded source revision in a separate worktree. The last release revision with all lanes at their
original paths is [`7fdd4ecb`](https://github.com/SebastianBoehler/neural-interfaces-2026/tree/7fdd4ecb629cbd4c6b187ea0f2d536a33a50c67e).

```sh
git worktree add --detach ../ni26-experiment-replay 7fdd4ecb629cbd4c6b187ea0f2d536a33a50c67e
```

The cap250 builder imports the CBraMod trainer and the test-time training patch from their new locations.

Use the [commit map](../../docs/HISTORY.md) if a run cites a revision from before the public history rewrite.
You will also need the inputs and environment named in that run's record.

## Model families and pretraining

| Original path | Source |
|---|---|
| `laneB4/` | [Family pretraining and routing](model-families/family-pretraining-and-routing/) |
| `laneB5/` | [Learning curves](model-families/learning-curves/) |
| `laneB6/` | [Extra participant pretraining](model-families/extra-participant-pretraining/) |
| `laneB7/` | [Seed robustness](model-families/seed-robustness/) |
| `laneC/` | [REVE encoder](model-families/reve-encoder/) |
| `laneC2/` | [CBraMod encoder](model-families/cbramod-encoder/) |
| `laneC3/` | [Pratibha encoder](model-families/pratibha-encoder/) |
| `laneC4/` | [BrainCodes and NeuroRVQ](model-families/braincodes-and-neurorvq/) |
| `laneH/` | [Pooled-corpus FBCNet](model-families/pooled-corpus-fbcnet/) |
| `laneS/` | [Shin transfer](model-families/shin-transfer/) |
| `laneTB/` | [Team b encoder blend](model-families/team-b-encoder-blend/) |
| `laneV/` | [Virtual participants](model-families/virtual-participants/) |
| `lane_e/` | [Ensemble capacity](model-families/ensemble-capacity/) |

## Session adaptation

| Original path | Source |
|---|---|
| `laneB1/` | [Refit and member combinations](session-adaptation/refit-and-member-combinations/) |
| `laneG1/` | [Class-coupled SessionEM](session-adaptation/class-coupled-session-em/) |
| `laneI/` | [SessionBN shrinkage](session-adaptation/sessionbn-shrinkage/) |
| `laneS2/` | [Session stage ablation](session-adaptation/session-stage-ablation/) |
| `laneT/` | [Test time training](session-adaptation/test-time-training/) |

## Returning participants and recognition

| Original path | Source |
|---|---|
| `laneA/` | [Returning adaptation dose](returning-participants/returning-adaptation-dose/) |
| `laneB10/` | [Returning stress test](returning-participants/returning-stress-test/) |
| `laneD3/` | [Visit drift](returning-participants/visit-drift/) |
| `laneG/` | [Personal models](returning-participants/personal-models/) |
| `laneQ/` | [Cross day recognition](returning-participants/cross-day-recognition/) |
| `laneR/` | [One session enrolment oracles](returning-participants/one-session-enrolment-oracles/) |
| `laneR4/` | [Midstream enrolment](returning-participants/midstream-enrolment/) |
| `laneR4b/` | [Recognised adaptation gates](returning-participants/recognised-adaptation-gates/) |
| `laneRP/` | [Open set personal experts](returning-participants/open-set-personal-experts/) |
| `moonM2/` | [Nearest session experts](returning-participants/nearest-session-experts/) |

## Ensemble mixing and posterior correction

| Original path | Source |
|---|---|
| `laneC2v/` | [Calibrated expert votes](mixing-and-correction/calibrated-expert-votes/) |
| `laneK/` | [Posterior correction](mixing-and-correction/posterior-correction/) |
| `laneK2/` | [Feature and prior corrections](mixing-and-correction/feature-and-prior-corrections/) |
| `laneLock/` | [Sparse and class specific mixers](mixing-and-correction/sparse-and-class-specific-mixers/) |
| `laneP/` | [Dispersion correction](mixing-and-correction/dispersion-correction/) |
| `laneP2/` | [Packaged dispersion correction](mixing-and-correction/packaged-dispersion-correction/) |
| `laneR1/` | [Nested mixer reweighting](mixing-and-correction/nested-mixer-reweighting/) |
| `laneW/` | [Widened posterior corrections](mixing-and-correction/widened-posterior-corrections/) |
| `laneY/` | [Residual rotation and error recovery](mixing-and-correction/residual-rotation-and-error-recovery/) |
| `revisit/` | [Clean slate mixer inputs](mixing-and-correction/clean-slate-mixer-inputs/) |

## Evaluation and profiling

| Original path | Source |
|---|---|
| `laneN/` | [Final candidate scoring](evaluation/final-candidate-scoring/) |
| `laneO/` | [Member proposal screen](evaluation/member-proposal-screen/) |
| `laneZ/` | [Chain profiling](evaluation/chain-profiling/) |
| `niv/` | [Independent visit validation](evaluation/independent-visit-validation/) |
