# §281 returning tail and FBLight seed 1

Inventory: the two P04/P10 FBLight fold checkpoints exist for all six seeds. The eight required orders have all 48 FBLight QF frozen streams. The first seed per ensemble base is 0 for B2_pool, B2_s012 and B2_s0; 1 for B2_s1; 2 for B2_s2; and 3 for B2_s345. Thus four checkpoint seeds supply FBLight fine-tuning, while all six are needed for the underlying pooled frozen members. The saved seen-session route trace covers within0 only. Base-specific within0 prefix files exist for B2_pool, B2_s0 and B2_s1; unindexed old files cannot be used. Other order/base combinations fail closed.

The adapter now supports within0/B2_s1 as well as B2_pool and B2_s0. Seed-1 QF frozen checks covered all four held-out sessions, with zero changed labels: P04/2 max 1.08349036e-6, P04/3 1.30062726e-6, P10/2 7.64019346e-7, and P10/3 7.93721069e-7. The separate B2_s1 prefix producer checked eight refits against the actual package EM/TSEM stage and reapplication with zero maximum difference, plus nontrivial future/session causality checks. Its P10/2 adapter smoke passed with trace-versus-eager difference zero, four refits at 20/30/40/50, recognition at 60 and max update 0.0196182467. No accuracy or treatment score was computed and no prediction artifact was written.

Commands:

```sh
scripts/run_capped.sh runs/s281_seen_fbl_checkpoint_s1.log .venv/bin/python -I studies/14nov/newmember/s281_seen_fbl_checkpoint.py 1
scripts/run_capped.sh runs/s281_seen_relabel_within0_B2_s1.log .venv/bin/python -I studies/14nov/newmember/s281_seen_relabel.py within0 B2_s1
scripts/run_capped.sh runs/s281_seen_fbl_dev_s1_smoke.log .venv/bin/python -I studies/14nov/newmember/s281_seen_fbl_dev.py smoke-s1
.venv/bin/python -m unittest discover -s studies/14nov/newmember -p 'test_s281_seen_fbl*.py'
```

Seven consumer/adapter tests passed. Source SHA-256: `s281_seen_fbl_dev.py` `f781e60a50425007c31bb61c24f141cef42dfcf89f2c1bcecac86411b98f31a2`; `s281_seen_fbl_checkpoint.py` `6b422e8d7b09f296db4552e8f53abdb1481def6fe1549acc17de7294c7c46871`; `test_s281_seen_fbl_dev.py` `67a3b1f99206dfbfe8ce635b9fc217d025f89055e25308dcca53c8275c7e9511`. The B2_s1 prefix SHA-256 is `c9db3ea74112cafced69b19e114fc4a6a4b2815b99728ea1788196b6dcae7e91`. Checkpoint source limitations from `seen_fbl_dev.md` remain: saved weights do not contain an original training-index hash.

`studies/14nov/newmember/proxy_s.py` handles only all_loso and is an oracle proxy, not the P04/P10 returning path. `studies/14nov/pview/chain.py:system` builds an ER/eye/rest routed tail and separate inner shadow, but does not insert the package r15 pre-recognition member deltas. The §281 returning-session pre-registration explicitly keeps the original inner recognition gates unchanged and inserts the new slots in the routed output tail. RouteB must receive that same gate ramp and process its whole session, including pre-recognition state, personal head, TSEM prior and own-space stage. A candidate that adds slots to the inner shadow and changes recognition is a separate deployment-faithfulness question and needs its own registration. The full control still needs all four FT families, per-base prefixes and recognition traces, and per-window agreement against actual package behavior plus zero-weight/pinned-order and recognition-boundary checks. The ten-person routed proxy cannot establish this P04/P10 result.
