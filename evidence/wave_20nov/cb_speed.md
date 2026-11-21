# CBraMod speed · 21 November

Both derived ZIPs retain the original hardcb assets and load-time self-test. The only changed entry is `submission.py`. The builder verifies the pinned parents, ZIP inventory, CRC and exact patched source. [Builder](../../submissions/final_stack/build_cb_speed.py) · [Parent hashes](final_r1a.md)

| Archive in `runs/sealed/final_zips/` | SHA-256 |
|---|---|
| `final_stack_hardcb_safe_st_cbgraph.zip` | `6ba4092d1f07e036cfb7e19f306eba993cfdea5b6d9cff779d03c32e847a39f5` |
| `final_stack_hardcb_safe_st_r1a_cbgraph.zip` | `7029bb25e0fac075f94c6e1982e98d08d2910e615f7666127e5fbbf6446b43cc` |

The three seeds share a whitening prior and use different fitted encoder and head weights. Keep the original NumPy float64 covariance and whitening, interpolate and scale the shared input once, then execute the three seed forwards. The CUDA graph replays those operations with a fixed device input, asynchronously staged from pinned memory, followed by one posterior transfer. The NumPy mean still uses float64. Capture and buffers belong to each session. Reset and deepcopy discard graph state. Capture failure warns and switches that session to eager execution. [Member code](../../submissions/final_stack/cb_graph.py.inc)

Local `replay/check.py` passed all six v3 references. Both exact ZIPs passed constructor self-test and matched their parent on the eight-window P01/1 prefix in recording order and shuffle 71000 on Mac CPU. The full A100 run uses Leonardo Slurm under `/leonardo_work/AIFAC_P02_1636/ni26/cap250/runs/cb-speed-21nov`. The terminal-label array compares each exact ZIP to its own parent through the official `fresh_model` and the full Solver. Galvani rejected noninteractive authentication. [Registration](../../docs/LEDGER.md#cbramod-cuda-graph--21-november) · [Slurm jobs](DISPATCH_21NOV.md)
