# Independent waveform decoder (§379)

Ten person-held fits compared learned memory, explicit causal normalization and a static waveform decoder. Three models used the identical initialization and 360 source episodes in each fold. No teacher, incumbent prediction or held labels entered training. The scorer reproduced the full archived B2_pool endpoints (76.2793% frozen, 76.5517% JS). Replay verification printed OK.

| Complete decoder on all 28 cells | Official cell-balanced accuracy | Memory minus reference, pp (80% paired people/order) |
| --- | ---: | ---: |
| Learned memory | 41.692% | — |
| Explicit causal normalization | 44.149% | −2.457 [−4.092,−0.868] |
| Matched static decoder | 44.120% | −2.427 [−4.079,−0.695] |
| Reset memory on each call, same weights | 41.237% | +0.456 [−1.081,+2.564] |
| Replace earlier memory with matched-length source history, same weights | 43.034% | −1.342 [−2.633,+0.079] |
| Frozen B2_pool | 76.279% | −34.587 [−40.286,−29.255] |
| JS B2_pool | 76.552% | −34.859 [−40.636,−29.434] |

Explicit normalization versus static was +0.029 pp [−1.436,+1.737]. This setting fails to establish a benefit of the target operation. Learned memory also misses both registered promotion conditions. It trailed explicit normalization on seven people. For P04 and P10 as new people, memory minus normalization was −0.722 and −3.572 pp. The 32nd and later delivery indices retained all 28 complete cells, with a memory-minus-normalization effect of −2.323 pp [−4.114,−0.623]. Only three early cells (P02 and P04) contained all three classes across every order.

A separate CPU implementation reconstructed every training batch hash and replayed all five prediction arms one window at a time. All 30 checkpoint hashes and all 106,200 decisions matched. Smoke 3123197 and array 3123198 used 171 allocated A100-seconds in total. CPU collector 3123199 took 101 seconds. Results concern one seed on exposed public people, with no genuine returning or sealed score. [Score](wave379_result.json) · [Before-score proof](wave379_before_score.json) · [Independent audit](wave379_cpu_audit.json) · [Independent recurrence](../../studies/18nov/wave379/audit379.py)

The scale probe found the learned gain farther from the causal variance correction than the blank population gain (64.4% larger mean squared error). On the one-order source-fit check, the memory model reached 72.11% on the nine people it trained on and 43.66% on its held person, averaged over folds. Static scored 70.36%/45.88%. The memory model fit the training people better and transferred worse in this comparison. These descriptive means weight folds equally and use within0 only. [Scale probe](wave379_gain.json) · [Source-fit check](wave379_fit.json) · [Class recall](wave379_classes.json)
