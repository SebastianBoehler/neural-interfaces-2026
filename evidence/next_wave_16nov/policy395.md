# Learned session calibration on source-held waveform decisions · §395

Using the correct session history improved waveform decoding relative to a matched-length donor history on both seeds. The learned correction changed accuracy relative to the original causal k20 model by **+0.055 and −0.011 pp**. Both paired intervals span zero. The two-seed ensemble gained 0.367 pp [−0.044,+0.882]. The registered evidence threshold for improvement over native normalization was unmet.

| Whole decoder, ten held people | Seed 0 (%) | Seed 1 (%) | Probability ensemble (%) |
| --- | ---: | ---: | ---: |
| Population normalization | 54.449 | 54.640 | 54.727 |
| Native causal k20 | 58.197 | 58.689 | 58.782 |
| Fixed analytic adjustment | 58.408 | 58.825 | 58.977 |
| Learned curvature-aware adjustment | 58.251 | 58.678 | 59.149 |
| Learned mean-only adjustment | 58.997 | 58.732 | 59.471 |
| Learned adjustment, donor history | 56.958 | 56.533 | 57.214 |

The two complete decoder references scored 76.279% (frozen) and 76.552% (JS). The ensemble learned−donor difference was +1.936 pp [1.060,3.117], and learned−mean-only was −0.322 [−0.660,+0.112]. For the two individual body seeds, learned−donor was +1.293 [0.248,2.782] and +2.145 [1.317,3.318]. Learned−fixed was −0.157 [−0.433,+0.128] and −0.147 [−0.386,+0.055]. Seed 0 favored the simpler mean-only update by 0.745 pp; seed 1's comparison included zero. Person/order uncertainty and reuse of the public people limit selection. [All contrasts and person effects](policy395_result.json)

**How the experiment isolates transfer.** Ten cyclic groups each excluded four people. For two seeds, each group trained a new six-person FBCNet for 40 epochs. Four groups exclude any chosen target person. Together they provide 12 other person/body pairs covering all nine source people. A given source query therefore comes from a body that saw neither its person nor the eventual target. Policy training uses labeled source queries and only unlabeled, strictly earlier same-session support windows. A held target uses its original nine-source native k20 model. Each source person contributes eight sampled queries per policy update. The method estimates a two-dimensional class-bias correction from the accumulated class marginal and its local softmax sensitivity. Adam learns the gain and damping on the source-query objective for 400 updates. The mean-only comparison uses scalar sensitivity and the same trainable parameters and query schedule. [Body split and training](../../studies/18nov/policy395/bodies.py) · [Policy objective](../../studies/18nov/policy395/calibration.py)

**Verification.** The native two-seed outputs matched the registered §339 k20 and static posterior streams bit-for-bit on every target and order. Independent NumPy single-window execution reproduced 169,920 adjusted outputs (same decisions, maximum probability difference 1.06×10⁻¹⁵). All zero-history logit corrections were exactly zero; renormalizing an archived probability changed it by at most 2.22×10⁻¹⁶. The verified predictions were committed before requesting held-role metrics. Body smoke/array 3123742/3123743 consumed 765 allocated A100-seconds. Policy smoke 3123785, ten CPU actors 3123786, scorer 3123787 and independent audit 3123799 all completed. Source revisions 42df143 and 5094081. [Committed streams](policy395_before_labels.json) · [Independent audit](policy395_audit.json)
