# Returning FBCNet seed-0 peers on full0

The P10 FBCNet with Euclidean alignment (EA) passed frozen parity and the real cap250 refit. The raw FBCNet checkpoint failed frozen parity, with 15 differing decisions across the two visits. The raw mismatch was already present between the pdeep and ttt7 recording-order archives. The raw refit remained gated. [FBLight receipts](fbl-refit-result-18nov.md) · [Returning inputs](bridge-result-18nov.md)

## Frozen actor qualification

Galvani tasks `3131571_0` and `3131571_1` loaded the original pdeep P10 seed-0 checkpoints. The exact original `ttt5.make` and base-only `ttt7.stream` definitions built and executed the eager actors. Both matched the recorded fold's 2,270 source rows and recovered their covariance priors and scales exactly. The masks retained 192 own visit-1 rows and excluded 385 held rows. The old checkpoint formats omit the original training-row hash. The EA actor used input prior 5 and latent prior 2. The raw actor used scale-only input and latent prior 20. Both streamed the verified P10 full0 sessions starting on global rows 1238 and 1326, and passed prefix and reset checks.

| Actor / visit | Eager max probability difference from QF | Single-window max difference | Decision differences | Frozen gate |
| --- | ---: | ---: | ---: | --- |
| EA visit 2 | 5.01e−6 | 5.75e−6 | 0/186 | Passed |
| EA visit 3 | 5.25e−6 | 7.88e−6 | 0/199 | Passed |
| Raw visit 2 | 0.07455731 | 0.07456180 | 6/186 | Failed |
| Raw visit 3 | 0.11781397 | 0.11780367 | 9/199 | Failed |

Exported batch and eager frozen predictions were bitwise identical for each actor. Raw one-window-to-eager differences peaked at 2.84e−5 with identical decisions. The much larger gap was between the restored actor and the archived QF bank.

## EA refit

Task `3131571_0` completed at 21:41:11 CET on 18 November, after 47 seconds. It used the actual cap250 `FineTune` and augmentation definitions, member 35 and one shared frozen trace. The original §281 and D-EAr0 archived controls produced identical warm outputs, optimizer actions and generator states. All four arms ran five Adam steps at age 20 on P10 visit 2 and emitted eight adapted predictions before recognition at age 28. Visit 3 stopped at recognition age 10 with zero refits. Reversed arm/session order and the current-input perturbation checks passed.

Fresh causal delivery changed all 20 confidence weights, with maximum change 0.03840345, and kept all 20 hard targets. The fitted parameters differed. Maximum warm probability change was 0.00299204, with zero changed decisions across the eight adapted outputs. This qualifies a single family without held-label scoring.

## Raw checkpoint blocker

Task `3131571_1` stopped after 22 seconds at 21:40:46 CET when the raw frozen parity check failed. No raw warm output exists. The preregistered recording control compared authenticated pdeep k20 and ttt7 seed-0 artifacts in their shared recording order. It found maxima of 0.07557163 and 0.12525490 and decision counts of 7 and 6 for visits 2 and 3, respectively. The discrepancy therefore existed before full0 and before this replay's trace export. The original fit detail responsible for the gap remains unresolved.

The pdeep submission script used the Galvani RTX 2080 partition. Original §199 puts the separate QF training on Leonardo. The locally available QF NPZs contain probabilities and indices. They carry no checkpoint, generating checkout or software-environment fingerprint. Both known Leonardo login endpoints rejected the current keys. As a result, the original generating backend could not be authenticated for the authorized single retrain. The recorded disagreement falls outside the registered precision-only A100 diagnostic. The current evidence does not support a historical raw replacement.

## Receipts

Galvani release `/mnt/lustre/work/gehler/gfh527/ni26_friday_ea_returning_peers_v1_20261118/`. An independent reader verified the EA output bank hash and every finite-output mask, and confirmed the absence of a raw warm bank.

| Artifact | SHA-256 |
| --- | --- |
| `peer_source.sha256` | `1e95d0a379c58343339c0f89f50a46f4f864ed6abb81b89347d7d0953ea55129` |
| `peer_inputs.sha256` | `3916dbf419bf4e30d25ab118a987a19dde24cc00d8c4533e5550e7fd36cddaf9` |
| `fb-cpu/frozen.json` | `b8eef61154a97459176aae582e4a3beaa5b53c5db1a37c852c51c4bb3fb5aa01` |
| `fb-cpu/warm.json` | `65f45af32b27f7f143b258bf492de0244b341b8f5a26890c8c69761caeb541c8` |
| `fb-cpu/warm.npz` | `5d691cf2742292121bffb1893b863b7f2233b69f217b8cfdf22376249e4b6ecb` |
| `fbraw-cpu/frozen.json` | `6c128d77695a2475d211c03e71ac83129ea0d0245e2f57517752d8277da4394f` |
