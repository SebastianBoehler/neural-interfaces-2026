# Fable Round 6 Bet 2 · adaptation diet

[Preregistration](../../docs/LEDGER.md#fable-round-6-bet-2--adaptation-diet-registered-before-scoring) · [Dispatch](DISPATCH_21NOV.md) · [Last Push scorer](last_push_21nov.md) · [em_half reference](bottleneck_review_21nov.md#recognition-gated-sessionem)

## Production enrollment

**The main stack enrolls P04 and P10 from all three public visits. MANAS A1 uses visit 1 for each person's gallery and personal head.** The exact em_half archive is `833a28721b0de0fa34f94de0328284ca9643cbaccc2dd343926f19c254949a4d`.

- Main spectral recognition reads the whole public export and groups by person and visit. The gallery averages the visit means and records their count. Both people have `bio_m=3` in the exact archive. The public row counts for visits 1–3 are P04 189/189/188 and P10 192/186/199. [Builder](../../scripts/add_bio_gate.py#L25) · [Gallery fit](../../experiments/bio_gate.py#L66)
- Main contrast recognition averages the per-visit class contrasts. The PS1 fit concatenates the standardized rows from each person's visits. The own-space builder enumerates every visit for the person and fits three heads against those labeled rows. The TSEM prior fit also averages all visits by default. [Contrast and PS1 fit](../../replay/rlib3.py#L23) · [Own-space source rows](../../scripts/add_own_stage.py#L16) · [Own-space head fits](../../studies/12nov/ownspace.py#L62) · [TSEM prior fit](../../lean/tools/fit_tpri.py#L24)
- The v6 build checks that all v5.1b arrays survive. The hard and fused builders copy those payloads, and the FP16 and em_half builders change the Solver source. The 40 `bio_*`, `route_*` and `own_*` arrays match the original local v5.1b fit exactly, and the contrast gallery and PS1 arrays match the signed Route3 fit export `b4de2fe7…`. Direct ZIP reads give the same `weights.npz` SHA-256 `7cf13696190983cd454880d6ecb8c049902a12ec694c1a24ff74cd4013a688a2` in cap250, fused nine25, fused ba_six, FP16 ba_six and em_half. [Route3 fit provenance](../../docs/AUDIT_SBN.md#L829) · [v6 build](../../lean/tools/build6.py#L108) · [Hard builder](../../friday/cap250_hard_package/build_hard.py#L71) · [Fused builder](../../submissions/final_stack/patch_ba_six_fc.py#L38) · [Source-only archive verifier](../../submissions/final_stack/build_cb_speed.py#L45)
- MANAS A1 trains the shared encoder on all public visits, then the gallery and personal head use exactly the rows with `visit == "1"`. The fused builder copies the signed MANAS weights verbatim. In em_half, `team_a.pt` SHA-256 is `b7728c22de2827f7a8747145fb5f53e5c93fbaab3f903ddfbc59124661e3467b`. Recomputing the shipped P04/P10 gallery vectors from public EEG matched visit 1 within 3.0e−7 / 3.6e−7 maximum absolute error. Pooling all visits instead differs by 0.163 / 0.191. [Production invocation](../../cleanslate/teamA/run.sbatch#L71) · [Source selection](../../cleanslate/teamA/data.py#L55) · [Fit and bank](../../cleanslate/teamA/train.py#L79) · [Visit-1 gallery and head](../../cleanslate/teamA/gallery.py#L26) · [Archive copy](../../submissions/final_stack/patch_ba_six_fc.py#L50)

## Causal fold replay

The zero arm skips the output-tail SessionEM and TSEM from the recognition delivery. The quarter arm keeps both fits and applies a quarter of each stage's log-likelihood correction. Each arm keeps Rebalance, personal RouteB and own-space. Both pass the RouteB posterior directly to native CSP and MANAS after the latch. The upstream recognition and fine-tune inputs come from the live capture.

Each order block has 21,240 held-new and 6,096 returning decisions. The collector checks the 28 held-new and eight returning cells on official class-balanced weights and pairs 20,000 person/order bootstrap draws. The 50/50 new/returning split is the joint score, J. The same people appear in both order blocks. The nine active slices cover the canonical 32 returning streams per block and the sole fit false recognition. The collector uses the canonical unchanged labels on other streams.

Local v3 replay printed `OK` at 16:33 CET. Leonardo run `cap250/runs/fable-r6-diet-21nov` stages `09c150d5`. Smoke 65040561 gates fit array 65040562 (five tasks, at most three concurrent). The remaining four active slices are held for the frozen fit receipt. Canonical recognition manifest SHA-256 `c4bbc41808d296aa4e2ee18151bf9680ed46f465b83462e94a6be632bed3dd3e`.

The exact-source Mac CPU mechanics smoke passed at 16:48 CET on 56 shuffled public P04/1 windows. All arms latched on delivery 11. Before the latch, every terminal vector and decision matched bitwise. CountFilter consumed 56/10/10 windows for em_half/zero/quarter. Each of SessionEM and TSEM consumed 56/10/56, with both quarter-weight fits active. Rebalance, CSP, Team B and MANAS consumed all 56. Maximum post-latch terminal effect was 0.11007 for zero and 0.06816 for quarter. The smoke used all-data production weights and computed no accuracy.

The local staged-source import check exposed a missing `laneR4` dependency. The Leonardo release received that directory from the same commit before solver execution. The pending smoke time limit was reduced to 12 minutes. Slurm started it at 16:48:55 CET.

Leonardo smoke observed at 16:55 CET matched all 189 pinned native and em_half decisions on P04/2 (71800), with recognition at delivery 41. It matched each live patched Solver's labels over the first 84 windows, its RouteB and terminal scores within 2e−5, and its exact truncated-tape output. The check exercised Rebalance, both quarter-weight EM fits, the count bypass, native CSP and MANAS.

### Fit orders 71800–71807

em_half BA 74.561% held-new, 81.168% returning and 77.864% at 50/50. Deltas are in percentage points, with paired 80% intervals in brackets. P04 and P10 intervals resample orders for the named person.

| EM weight | Held-new Δ | Returning Δ | P04 Δ | P10 Δ | 50/50 Δ | 50/50 BA | Changed labels new / ret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -0.005 [-0.012, +0.000] | +0.485 [-0.044, +0.991] | +0.049 [-0.451, +0.543] | +0.922 [+0.507, +1.364] | +0.240 [-0.023, +0.494] | 78.104% | 13 / 365 |
| 0.25 | -0.009 [-0.024, +0.000] | +0.345 [-0.163, +0.813] | -0.035 [-0.591, +0.523] | +0.725 [+0.437, +1.021] | +0.168 [-0.083, +0.397] | 78.032% | 10 / 263 |

On order 71802, P10 visit 1 latched as P03 at delivery 40. The zero and quarter arms changed 13 and 10 decisions on that stream.

Fit receipt SHA-256 `d07119b843f345bf9822324660a5bbce05058bce30af2681b509e46dd001513c`.

All five fit tasks completed at 17:14 CET. The fit receipt was copied and SHA-256 checked on Leonardo before releasing held array 65043064 (four tasks, at most three concurrent).
