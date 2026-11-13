# v6 build (13 Nov): v5.1b plus a second chain for unrecognised sessions

Built for the user's upload decision of 13 Nov ~19:00 UTC. **Nothing uploaded**; an upload needs the user's go-ahead.
Design and evidence: `docs/V6_DESIGN.md`, ledger §152, §160, §169–§173.

## What the package does

`submissions/lean_v6/submission.py` is bundled from `lean/src6` (`lean/tools/bundle.py --v6`; `lean/src` and the pinned
`lean_v5` are untouched). The package is built by `lean/tools/build6.py` from v5.1b's data files plus the members below.

- **v5.1b runs unchanged on every window** (`Ensemble.step`: mix, gated shadow tail, gates, routed tail, own-space stage).
- **While no window of the session has been routed**, a second chain gives the output:
  - its mix is v5.1b's mix with each fine-tuned family's change, plus the rest member: mix' = (15/16)·mix_FT +
    (1/16)·rest (replay units: rest weight 1 against 15);
  - mix_FT moves each family's posterior in slots 5–8 (raw FBCNet, fb at k 2, SincShallow, FBLight) by (fine-tuned −
    frozen) of its seed-0 trace, clipped at 1e-6 and renormalised, as in `pkg.py`; slot 9 (fb at k 5) stays frozen;
  - the tail is v5.1b's gated unrouted tail (rebalancing → SessionEM with the early boost and margin gate → TSEM), with
    SessionEM's view widened by the eye net's three log posteriors (48 → 51 columns).
- **From the first window either gate routes**, the output is v5.1b's routed tail, and the second chain stops for the
  session. The gates never see the second chain.
- **Fine-tuning, gC5 delta1** (`FineTune`): runs directly on the shipped TorchScript traces, one per family (seed 0).
  - Each session's copies are loaded from the files' bytes; a deep-copied TorchScript module cannot be optimised.
  - Warm start; from window 30, every 10 windows, a new Adam (lr 5e-4) with 5 steps on windows 0..t−1. Each step: a
    clean forward without grad, a forward on GPU-drawn strong augmentation, weighted CE on the package's own earlier
    outputs (argmax, weight = max), plus KL consistency.
  - Then one pass over the prefix rebuilds the StreamBN state, and later windows are predicted window by window.
  - The refits run with `torch.jit.optimized_execution(False)`: the profiling executor otherwise re-specialises for
    every new prefix length (a timing run at 190 windows went 2.63× → 1.76×).
  - `v6_ft_on = 0` switches fine-tuning off; everything else is identical. That gives the second package.
- **Eye net** (`EyeNet`): `studies/12nov/ocunet.py` arm nomix, three seeds trained on all ten dev people, traced. Per
  window: 7 derived ocular traces with the session's running standardisation, then the mean posterior → log, clipped at
  1e-3.
- **Rest member** (`RestMember`): `studies/11nov/rest4/rest4.py` sh r1, three seeds trained on all ten people with the
  pre-cue rest windows. StreamBN traces at test prior k 2 on their own input stream; softmax over the first three
  outputs; mean of seeds. Training checks: `evidence/v6_build_13nov/train_v6_members.jsonl`.
- **Logging:** one stderr line per session: `[v6] windows N v6 W recognised_at R refits F steps S dtheta D swapped X errors E`.
  An error stops the second chain for the session, and the windows that follow get v5.1b's output.

## Packages and ZIPs

| Package | ZIP (SHA-256) | Bytes |
|---|---|---|
| `cand_v6f_gc5` (fine-tuning on) | `a5db140470ce15d841ca8db0cb74303cda73e07e389a0ded52b64e69e99cbba2` | 66,738,355 |
| `cand_v6f_noft` (fine-tuning off) | `05c0e3861a891f75fcdb18309fc8bda09581247f67f7a84613097983092e1564` | 66,738,354 |
| v5.1b as uploaded (pairing reference) | `4a0bc427f2c9c862ee779db5766badeae93f783446561f4dc8abb43589933bef` | 62,308,135 |

Each ZIP was built once by the official local replay (`runs/sealed/final_zips/`). That same file was copied to
Leonardo and Galvani (SHA-256 verified there), and every check below ran on it or on a directory unzipped from it.

## Checks on the final ZIPs

Upload ZIP (fine-tuning on) `a5db1404…cbba2`, fallback `05c0e386…e1564`, v5.1b `4a0bc427…3bef`.

| Check | Where | Ran on | Result |
|---|---|---|---|
| Official local replay | Mac CPU, `scripts/replay_package.py` | a5db1404 / 05c0e386 (the replays built these ZIPs) | 0.9932767697 both (= v5.1b), files unchanged, 89.4 / 87.5 s; `[v6] … errors 0` per session |
| Official A100 replay series | Leonardo 64381214, the copied ZIPs | 4a0bc427, a5db1404, 05c0e386 (ZIP sha256 in every replay.json) | warm-up 286.9 s, then v5.1b 101.4, gc5 104.4, noft 101.3, gc5 106.1, v5.1b 107.9 s; all 0.9932767697, files unchanged |
| Worker-image replay | Galvani 3096606, `runs/cloud/ni26_worker.sif` (torch 2.14.1+cu130, braindecode 1.8.1), the copied ZIPs | 4a0bc427 ×2, a5db1404, 05c0e386 | v5.1b 120.4 / 117.4 s, gc5 118.0, noft 118.2; all 0.9932767697, files unchanged, `[v6] errors 0`. The replay sessions are recognised at window 10, so fine-tuning never ran here (next row) |
| Fine-tuning inside the worker image | Galvani 3096632, gates forced off, unzipped from a5db1404 / 05c0e386 | a5db1404, 05c0e386 | OK: fine-tuning ran in the worker image (16 / 35 refits at 190 / 378 windows, 320 / 700 steps, errors 0). Paired against v5.1b in the same image: gc5 ×1.702 / ×2.026 (sealed 30.0 / 34.0 min at 57 %, **36.5 / 43.5 min at 100 %**); noft ×1.123 / ×1.128 (23.0 / 23.0, **24.1 / 24.2 min**) |
| Recognised path, no-op check | Mac CPU, `scripts/check_v6.py`, 14 sessions, unzipped from each ZIP | a5db1404, 05c0e386 | OK. Inner v5.1b equals lean_v5 (= v5.1b, `lean/PARITY.md`) on every window. The output equals v5.1b from the first routed window (window 10 in all 14 sessions) bitwise. Windows 0–9 emit the v6 chain: max \|Δp\| against v5.1b 0.019–0.034 per session, 0 labels changed |
| Unrecognised path, against references | same, gates forced off, P01\|1 and P02\|1 | a5db1404, 05c0e386 | OK. Every window differs from v5.1b (max 0.37; labels changed 5 and 2). SessionEM view 51 columns against 48; rest weight 1/16. Against references outside lean/src6: eye 1.3e-6, rest 2.9e-7, fine-tuning 1.5e-5 (ttt7 form, same labels and device RNG), mix 0, tail 0. gc5: fine-tuning on 160 of 190 windows, errors 0; noft: no refit |
| Same on the A100 | Leonardo 64381218, `check_v6.py --device cuda`, 3 sessions (gc5) and 1 (noft) | a5db1404, 05c0e386 | OK (64384525, then 64386760). Recognised path on the A100: same result as on CPU. References with the dev shapes (eye over the whole session in one batch, fine-tuning with the default fusing executor and batch predictions) differ on the GPU by eye 1.4e-3 (log space) and FT 2.1e-3, beyond 1e-5 / 1e-4. Decided with the package's shapes and executor (`--ref-shapes package`): **eye 0.0, rest 9.9e-6, FT 8.9e-6, mix 3e-17, tail 0, V6 CHECK OK** for both ZIPs. So the differences are GPU float32 numerics of batch shape and kernel fusion, not logic; a dev-form run of the same fine-tuning on the GPU moves posteriors by up to 2e-3 |
| Does fine-tuning act (recognition off, all 14 sessions) | Leonardo 64383738, `studies/13nov/v6_ft_effect.py` | a5db1404, 05c0e386 | Fine-tuning acts. Recording order, 2,655 windows, errors 0. On the windows where the fine-tuned posteriors are swapped in (156–169 per session), gc5 and noft differ by max \|Δp\| 0.03–0.13. Labels changed against v5.1b: gc5 37, noft 38; gc5 against noft: 1. Mean session balanced accuracy: v5.1b 0.981, noft 0.9947, gc5 0.9943. In-sample (the members saw these sessions), so this shows only that the parts act |
| Shuffled delivery, package itself | Leonardo 64386481 / 64386484, `v6_ft_effect.py` with the orders within\|0 and full\|0 (`runs/shuffle_perms_fixed.npz`, dumped by the fixed shuffle.py; equal to the morning's dump), all 14 sessions, gates off | a5db1404, 05c0e386 | Runs without errors; FT active on all 2,235 eligible windows (max \|gc5 − noft\| 0.15 / 0.11). In-sample mean session balanced accuracy, within: v5.1b 0.982, noft 0.997, gc5 0.997; full: 0.970 / 0.992 / 0.991. Per session: gc5 about 25.9 s and noft about 17.0 s per 190-window session in every order (= recording order). The first session in a fresh process pays a one-off about 65 s with FT (about 10 s without) for CUDA / TorchScript / autograd warm-up, which the warm timing below does not include; that adds about 1.1 min once to a sealed run |
| Shuffled delivery, checks | Leonardo 64386482 / 64386485 (recognised path, 14 sessions) and 64387452 / 64387453 (references with `--ref-shapes package`, 2 sessions), orders within\|0 and full\|0 | a5db1404 | Recognised path in both orders: every session recognised at window 10, output equal to v5.1b from there (14 of 14, bitwise), inner v5.1b equal, errors 0; before recognition the v6 chain differs by 0.020–0.039 (labels changed 1 / 0). References in delivery order: within eye 0.0, rest 1.4e-5, FT 7.6e-6, mix 3e-17, tail 0; full eye 0.0, rest 1.4e-5, FT 1.6e-6, mix 6e-17, tail 0; **V6 CHECK OK** in both orders |
| Worst-case run time | Leonardo 64381217, A100, warm, paired (v5.1b, v6, v6, v5.1b), gates forced off on the whole session, unzipped from the ZIPs | a5db1404, 05c0e386 | gc5: ×1.763 (190 windows), ×2.061 (378) → sealed 30.8 / 34.4 min at 57 % unrecognised sessions, **37.8 / 44.2 min at 100 %**, margin to 60 min ≥ 15.8 min. noft: ×1.153 / ×1.156 → 23.3 / 23.4 and **24.7 / 24.8 min** (margin ≈ 35 min). Sealed = v5.1b's real-worker 1,287 s × the ratio, plus the one-off first-session warm-up (about 1.1 min with FT, 0.2 min without): **worst case about 45.3 min (gc5) and 25 min (noft)** |

## Not validated

- **The switch at the first routed window.** Dev never exercised it, because no new-person session is routed in dev.
  The package's behaviour is checked (output = v5.1b from that window, bitwise); its value is not.
- **FT labels in delivery order.** The package uses v5.1b's own causal outputs in the order windows arrive. Dev used
  chain_final (v5.1b's chain in recording order); the shuffled-protocol runs used chain_final labels too. Being
  checked in dev (§178); its shuffled-order numbers wait for the §180 rescore (shuffle.py bug, fixed in 31870cb).
- **The gate on the inner v5.1b.** The package's gate reads v5.1b's unchanged shadow; dev V6F's gate read the changed
  shadow. Equivalent for new people in dev (none routed), and being checked in dev (§178, GATE_P0).
- **Members trained on all ten people.** The dev numbers come from LOSO members; the packaged eye net and rest net
  (and the shipped SessionBN nets) have seen every public person. In-sample checks here (accuracy about 99 %) cannot
  show the gain.
- **Eye-net standardisation order.** The eye net was trained on traces standardised in recording order; the package
  standardises in delivery order.
- **Order independence of the package itself** is now tested on the public sessions in the within|0 and full|0
  orders (rows above), not on other orders or on unseen people.
- **Sealed run time** assumes the Codabench worker runs like a Leonardo A100. The worker-image replay on Galvani is
  about 15 % slower than Leonardo for every package, v5.1b included.
- **Fine-tuning numerics on the worker.** The worker image (CUDA 13 build) runs fine-tuning without errors at the same
  cost as Leonardo; its posteriors were not compared with a reference there (the GPU reference check ran on Leonardo).
