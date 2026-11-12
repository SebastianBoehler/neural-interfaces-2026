# Study scripts of the night of 11–12 Nov

One script per pre-registered section of [docs/PREDICTIONS_12NOV.md](../../docs/PREDICTIONS_12NOV.md). Local scripts
run through `scripts/run_capped.sh` (one job at a time); the Galvani ones carry the cluster's work-directory path.
Logs are `runs/<name>_12nov.log` locally and `runs/diag/{virt,netaug,ocunet,ttt}/` on Galvani (not tracked).

| Script | Section | What it does |
|---|---|---|
| `anatomy.py`, `slow_xsubj.py` | §34 | per-session and per-position accuracy of the chain; slow-potential decoding within and across people |
| `ladder.py` | §35 | causal in-session logistic head on true or chain labels |
| `repceil.py` | §36 | within-session ceilings in representations outside core1's PCs |
| `ownspace.py` | §37 on | **the own-space personal members** (class `Personal`); the package builder imports it |
| `own2.py`, `memsweep.py`, `dose2.py` | §44 | configurable members and their sweep |
| `xsess.py`, `xsess_poe.py` | §37 | own-space members across sessions, in the head's slot |
| `ownttt.py` | §38 | session-own members refitted inside the session |
| `popcurve.py`, `chainslope.py` | §39, §48 | accuracy against the number of training people, core alone and after the stages |
| `add.py`, `dose.py` | §40, §43 | personal nets with the stage; dose-response and fresh-base confirmation |
| `psel.py`, `xpair.py` | §41, §42 | ten-person within-session proxy; twelve real session pairs |
| `v5_replay.py` | §45 | **the dev replay of the package form** (stage where the real gate routes) |
| `oracle_gate.py` | §46a | perfect recognition from window 0 |
| `rotaug.py`, `virt.py` | §46b, §47 | virtual people for core1 (seven generators) |
| `peff.py` | §49 | people-efficient estimators and noise |
| `upw_core.py`, `upw_chain.py` | §50 | up-weighted personal core |
| `netaug.py`, `netaug_score.py`, `netpc_score.py` | §51, §56, §57a | virtual people, mixing, weight averaging and the people curve for the SessionBN nets |
| `sch19.py` | §53 | Scherer 2015 on the 19 real shared channels |
| `tsemk.py` | §54 | TSEM in more or fewer components |
| `seqcal.py` | §55 | cue-order prior with a learned fusion |
| `bundle.py`, `popstage.py` | §57b, §57c | small returning-person levers with the stage; population members as a final stage |
| `ocu_probe.py`, `emview.py`, `corefuse.py`, `ocunet.py`, `ocu_chain.py` | §58 | the eye-movement lead: probe, SessionEM view, feature fusion, learned branch, chain integration |
| `chainbag.py` | §61 | averaging complete chains or stage settings |
| `ttt3.py` | §60 (open) | in-session fine-tuning in the form a package can run |

Shared pattern: `v5_replay.py` up to its `res = []` line is a reusable head that loads the v4.1 replay (`route4.py`)
and defines `staged()`; `oracle_gate.py`, `bundle.py` and `upw_chain.py` exec it. Do not define a function named
`members` after exec'ing it (it shadows the head's own; that broke one run).
