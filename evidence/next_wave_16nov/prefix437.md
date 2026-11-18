# Learned reliability between frozen A6 endpoints (§437)

Two H100 fits shared the original A6 classifier and every labeled source episode between their conditional and constant arms. The native endpoint uses causal k20 normalization. The other endpoint uses population normalization. A single sigmoid gate combines their logits. It sees prefix maturity and the strictly previous windows' average probability disagreement and confidence difference. Each seed saw 120 episodes of 48 windows. Every six episodes contained one per source person. The body had seen the six calibration people. The four evaluation people had appeared in earlier research but were excluded from these weights.

| INDIVIDUAL-MODEL accuracy (%) unless stated | 437 | 1437 |
| --- | ---: | ---: |
| gate | 60.720 | 60.700 |
| constant | 60.721 | 60.667 |
| fixed | 60.127 | 60.127 |
| A6 | 60.785 | 60.785 |
| population | 53.830 | 53.830 |
| donor | 60.709 | 60.722 |
| frozen | 81.216 | 81.216 |

The frozen row is a complete decoder reference. Four held people contributed two interface cells each on eight pinned shuffled orders. The summaries give each shuffle scheme equal weight. Intervals resample people and orders.

| Percentage points, paired 80% interval | 437 | 1437 |
| --- | ---: | ---: |
| Gate − constant | -0.001 [-0.101, +0.101] | +0.033 [-0.024, +0.101] |
| Gate − donor | +0.011 [+0.000, +0.033] | -0.022 [-0.045, +0.000] |
| Gate − A6 | -0.065 [-0.152, +0.022] | -0.085 [-0.184, +0.004] |
| Gate − fixed | +0.593 [+0.176, +1.060] | +0.573 [+0.174, +1.014] |
| Gate − population | +6.890 [+4.111, +9.692] | +6.870 [+4.102, +9.645] |
| Gate − frozen | -20.496 [-21.811, -19.158] | -20.516 [-21.843, -19.174] |

Seed 437, gate minus constant: P01 +0.000, P02 -0.040, P03 -0.136, P09 +0.174 pp. Leaving one person out gives a scheme-balanced range of -0.059 to +0.044 pp.

Seed 1437, gate minus constant: P01 +0.000, P02 +0.052, P03 -0.043, P09 +0.121 pp. Leaving one person out gives a scheme-balanced range of +0.003 to +0.058 pp.

The fitted constant gave native normalization 98.08% and 97.76% weight. Replacing the held gate descriptors with equally mature descriptors from P10 source session 3 changed one out of 6,056 decisions per seed. The learned gate changed 14 and 16 decisions against its respective learned constant. This tests a nearly saturated gate on four people. It does not estimate the ceiling of learned session memory.

Source commit fbff189. Ferranti smoke 509754 and array 509755 completed using the checksum-locked source bundle. Galvani smoke 3125300 and array 3125301 used CPU slots and reconstructed both archived endpoints on all 32 person/order pairs. Replay in scorer 3125302 printed OK. Held actors used a new session calculation per person and order. No model parameter changed during evaluation. The source audit records the paired schedule, frozen body digest and learned weights. The actor receipts retain gate ranges, source and output hashes. Raw leave_one_out in the JSON weights orders equally. The leave-one-person ranges above use scheme-balanced effects.

[Source fit receipts](prefix437_fits.json) · [Held prediction receipts](prefix437_before_labels.json) · [Paired scores](prefix437_result.json)
