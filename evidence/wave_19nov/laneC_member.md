# Lane C · Held-person recovery from one diverse neural member

Registered 19 November 2026 before Lane C scoring. Population candidate: ten leave-one-person-out (LOPO) fits and one all-ten deployment fit. Each LOPO fit excludes all public rows and statistics of its held person. Use eligible external data with explicit task and montage handling and causal session normalization. Prefer a locally staged pretrained EEG backbone with auditable inputs and checkpoint provenance. If unavailable, use the strongest existing scalable raw decoder and retain a distinct training recipe. No cross-session test state. Fixed seed 20261119.

## Fast bank screen

Use authenticated corrected cap250 held-new predictions and candidate banks on matching recording rows. Prefer byte-identical delivered row orders. Keep order-mismatched and canonical feature banks in a separate descriptive screen. Use official participant/session/interface cell balanced accuracy (BA), with complete classes per scored cell. For each member report ordinary BA, weighted share of cap250 misses repaired at posterior ≥0.60, weighted repair mass, potential confident regressions, and repair mass unique relative to the other members on the same coverage. Report the union's zero-regression ceiling both at ≥0.60 and without a confidence threshold. Partial-fold banks contribute on their own complete cells and on a matched-coverage union. Use exact ranked probabilities or a documented softmax for logits.

Predict a confident union ceiling ≥3.0 pp, and a best individual confident ceiling ≥1.5 pp. Stop adding members if the unrestricted union ceiling is <0.50 pp on authenticated comparable coverage. A finite pool bounds correct-label selection from that pool and order, with stock mistakes known. It does not bound a model that has not been measured, nor arbitrary soft mixtures that produce a label outside the individual argmaxes.

## Newly trained member

Causal normalization at both train and query time. Use source-only or corpus-specific nuisance/label supervision for external classes without a verified three-way mapping. Heavy augmentation uses fixed source-only time shift, sensor dropout and additive noise. Select the recipe before looking at held scores. Score standalone on the same rows and eight full-session permutations as cap250. Give the member fixed 1/6 of one existing population family slot in the upstream mixer and rerun its consumers with each member in delivery order. Zero mass must reproduce the baseline and the pinned reference order must match its frozen row identities. Test prefix and reset parity. Compare against exact corrected cap250 and record any baseline substitutions imposed by the replay. Report paired people-and-order 80% intervals, P04 and P10 separately, and the sealed-mix estimate only with a valid returning experiment. Measure added warmed paired wall time on A100 and RTX.

Mechanism: external task supervision plus a different neural representation should repair some errors shared by cap250's existing members. Predict FULL-SYSTEM DEV held-new ≥+1.00 pp over cap250 and candidate INDIVIDUAL-MODEL held-new ≥60.0% BA. A full-system paired 80% upper bound below +1.00 pp refutes the step-change claim for this member and dose. A nonpositive upper bound refutes benefit for this form. Retain uncertainty when the interval crosses the decision point. Report actual source, training jobs and availability separately from the prediction.

## D1 smoke amendment · before training or scoring

Use the Fable D1 design. A starts at the pinned REVE-Base checkpoint and fits PROTEUS alone. B adds shared external supervised pretraining without Scherer. C adds Scherer during that same stage. Compare P02, P05, P08 and P10 held new in full0 on seeds 0 and 1. The returning B/seed 0 fit uses all ten people but excludes P04/P10 visits 2–3, then evaluates those visits. External training uses native recorded channels and corpus-masked MI, SUB, WORD and REST logits. MI-only corpora have a subtype loss so their data exercise a nonzero objective. Use curated HAND rows from Scherer, the first crop on Shin2018, and the motor-event prefix on PhysioNet.

Use one fixed schedule across smoke arms and no held-test selection. Predict A 55% [51,59], B−A +5 pp [1,9], C−B +1 pp [−1,4], and returning P10 80% [70,88] / P04 55% [45,65]. Launch full D1 only if pooled B improves A by ≥4 pp INDIVIDUAL-MODEL across the two paired seeds and its binary error correlation with corrected cap250 is ≤0.50. Stop this form if B−A <2 pp on both seeds. Otherwise mark it undecided. Report FBC and stack correlations where both are available. Full D1 uses the selected pooled arm with 3–5 frozen seeds, new and returning fits. Keep C out of Scherer evaluation.

## Execution and fast-screen receipt

The primary screen used the corrected Friday F-C20/F25c stock and pinned deliveries. The replay guard printed OK. Confidence is maximum posterior ≥0.60. The Friday CSP-active rows are log scores and are normalized with softmax. ATOM-L §330 is stateless. The recorded LaBraM and CBraMod banks belong to a different delivery order and only inform the descriptive screen.

| Bank | Coverage | full0 confident / unrestricted pp | within0 confident / unrestricted pp |
| --- | --- | ---: | ---: |
| Friday waveform stack variants | 10 people, 2,655 rows | 0.04 / 1.49 | 0.00 / 0.95 |
| A6 / J / K | 4 people, 757 rows | 2.11 / 6.08 | 1.45 / 4.89 |
| ATOM-L frozen and fine-tuned | 2 people, 378 rows | 6.13 / 7.72 | 5.35 / 6.67 |
| Friday variants + ATOM-L + A6/J/K | same 2 people, 378 rows | 6.93 / 9.84 | 5.87 / 8.00 |
| §379 independent waveform | 10 people, 2,655 rows | unavailable / 12.27 | unavailable / 10.03 |
| §280 historical REVE | 10 people, 2,655 rows | unavailable | 6.38 / 9.73 |
| LaBraM (recording order) | 8 people, 1,900 rows | 5.04 / 8.35 | 4.38 / 6.64 |
| CBraMod (recording order) | 10 people, 2,655 rows | 7.48 / 8.54 | 6.22 / 7.09 |

ATOM-L fine-tuned repaired 23.65% of weighted full0 cap250 misses on its two held people at ≥0.60 (4.51 pp). Its potential confident regressions were 36.43 pp. The label-only waveform union exceeds the stop threshold across all ten people. Its standalone arms scored 41–44% on full0, so this high ceiling can coexist with a poor practical mix.

Run `laneC/oracle.py` for the confidence, regressions and unique repairs per member. Run `laneC/oracle_history.py` for the historical foundation models and the strictly matched label-only waveform banks. Reports retain source or upstream receipt hashes.

D1 used the staged offline REVE-Base checkpoint (SHA-256 `8ecc650619598748286c2457f81f5c6bd12e8bb59db44f7b02af1955c44de8fe`). Galvani A100 held all required sources, so it was selected for the smoke before the VPN switch. Ferranti's transfer could not stage the full source set in time. Oracle 3136671 completed. Canary 3136672 rejected Shin2018’s stored class order before training. The corrected loader verifies exact class names and maps codes by name. Canary 3136742 then caught an incorrect expectation about the Scherer manifest. The archive contains 710 HAND, 710 SUB and 710 WORD rows. The new guard checks the raw-to-export mapping for every row. Canary 3136807 gates 3136808–3136810, and 3136811 writes the smoke decision. See [queue](QUEUED.md). The PROTEUS control retains the same architecture and training schedule as both pooled arms. It uses attention pooling, two causal source permutations and augmentation, so it is a matched D1 control and does not aim for exact §280 reproduction.

Local CPU canary loaded the pinned REVE weights, aligned real P02 windows using 2,466 source and 189 held rows, and matched batched and one-window logits. This checks the source split and input path. The submitted Galvani canary also exercises supervised steps on each corpus. The live Galvani retry is recorded in the queue.

If the smoke collector returns `launch`, [the Leonardo launcher](../../scripts/cloud/leo_lane_c.sh) stages three seeds and 36 fits through Slurm and collects eight full-session standalone orders. It requires the staged checkpoint, position file, external exports and smoke receipt on Leonardo. [The full0 consumer](../../laneC/full_replay.py) prepares a corrected Friday mixer test with one-sixth of the raw family replaced. Eight-order cap250 integration and A100/RTX paired runtime still require a qualified candidate and matching held-fold instruments.
