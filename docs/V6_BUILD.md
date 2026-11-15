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
- **FT labels in delivery order: checked in dev** (ledger §178–§184, commits 46af254, 623602c, 3773060). gC5 rerun
  with v5.1b's delivery-order outputs as labels; all 144 tasks' label hashes verified. Gain over v5.1b, new people:
  within 0 +1.70, within 1 +1.33 to +1.39, full 0 +0.93 to +1.18, runs 0 +1.34 to +1.51 (the lower values include the GE
  event rates in delivery order). The delivery-order labels cost 0.2–0.4 against recording-order labels, and every
  order still passes the bar.
- **The gate on the inner v5.1b: checked in dev** (same sections). Against the gate on the combination's own chain,
  |Δ| ≤ 0.07 for gC5 and the fallback on within 0/1 and runs 0 (corrected orders).
- **Members re-streamed in delivery order (dev):** the Hilbert member and the teacher land at about 16:15–17:00 UTC.
  The ea transformer and shallow slots (1.5 of 15 mix weight) stay in recording order in dev and are not checked.
- **Members trained on all ten people.** The dev numbers come from LOSO members; the packaged eye net and rest net
  (and the shipped SessionBN nets) have seen every public person. In-sample checks here (accuracy about 99 %) cannot
  show the gain.
- **Eye-net standardisation order.** The eye net was trained on traces standardised in recording order; the package
  standardises in delivery order. Dev's ocushuf standardises training and test sessions in delivery order, so it does not
  test this exact combination: open.
- **Order independence of the package itself** is now tested on the public sessions in the within|0 and full|0
  orders (rows above), not on other orders or on unseen people.
- **Sealed run time** assumes the Codabench worker runs like a Leonardo A100. The worker-image replay on Galvani is
  about 15 % slower than Leonardo for every package, v5.1b included.
- **Run-to-run reproducibility on the GPU.** Fine-tuning on the GPU is not deterministic. Streaming the same session
  twice through the same v6f package on one A100 (Leonardo 64421312, 4 sessions): fine-tuned posteriors differ by up to
  0.011 (P01|1; the other three about 1e-6), outputs by up to 3.6e-4, and no label differs. A sealed run is therefore
  not bit-reproducible; the effect on labels was nil in this test.
- **Run-to-run spread in scores** (Leonardo 64422185): 5 repeats of all 14 public sessions, recognition off, on one
  A100. The dev score (mean session balanced accuracy) is identical across repeats for v6f (0.99434) and v6.1
  (0.99359); no session's balanced accuracy varies. The posterior-level non-determinism did not reach the labels on
  these in-sample sessions; on unseen people it may move a few windows.
- **Fine-tuning numerics on the worker.** The worker image (CUDA 13 build) runs fine-tuning without errors at the same
  cost as Leonardo; its posteriors were not compared with a reference there (the GPU reference check ran on Leonardo).

## v6.1 (13 Nov evening): labels from the v6 chain, schedule s15

Prepared next to v6f for the user's choice. The staged v6f ZIP `a5db1404…` stays untouched and remains the default.
Code: `lean/src61` → `submissions/lean_v61/submission.py` (`bundle.py --v61`); `lean/src6` and `lean_v6` are unchanged.
Package: `runs/sealed/cand_v61_s15` (`build6.py --v61 --labels v6r --t0 20 --lr 1e-3`). **ZIP SHA-256
`2ca06dc085a0a3a6eeebaac4e89113dd7793cc38b5425e68cb1bc50cd85dfe32`, 66,738,606 bytes.**

Two changes against v6f, both on the fine-tuning of unrecognised sessions (ledger §198):
- **Labels** from the v6 chain without fine-tuning instead of the inner v5.1b. A third tail (`TailX`) runs on
  (15/16)·mix_v5.1b + (1/16)·rest, with the eye-net columns in SessionEM's view. It is v5.1b's gated tail with its own
  state, takes no FT change, and the gate never sees it. Its causal outputs (argmax, weight = max) are the labels
  (key `v6_ft_labels` = 1).
- **Schedule s15:** start window 20 instead of 30, Adam lr 1e-3 instead of 5e-4 (5 steps every 10 windows, warm start,
  GPU-drawn augmentation, as gC5).

Dev in the package's form (explore track, Leonardo, six bases; the label chain is the same computation as dev's V6r
under the package gate): new people over v5.1b, within orders 0–7, **v6.1 +1.49 [+1.14, +1.84]** (without P10 +1.49,
8 of 8 orders ≥ +0.5), against v6f +1.25 [+0.94, +1.59]. Returning people: unchanged against v5.1b (as v6f).

| Check on the v6.1 ZIP | Where | Result |
|---|---|---|
| Official local replay | Mac CPU (built the ZIP) | 0.9932767697, files unchanged, 98.1 s |
| Official A100 replay series | Leonardo 64419124, copied ZIP | warm-up 230.3 s; v5.1b 103.1, v6.1 102.9, v6.1 107.2, v5.1b 101.6 s; all 0.9932767697, files unchanged, ZIP sha256 = `2ca06dc0…` |
| Recognised path, references (CPU) | Mac, `check_v6.py`, unzipped | **OK**. 14/14 sessions bitwise v5.1b from window 10. References: eye 1.3e-6, rest 2.9e-7, FT 1.2e-5, mix 3e-18, tail 0, **label chain 0.0** |
| Same on the A100, package shapes | Leonardo 64419125 | Recognised path 14/14 bitwise. Recording order: eye 1.3e-9, rest 9.9e-6, label 0, mix 3e-17, tail 0, but **FT 1.02e-4 against the 1e-4 tolerance** (see the next row). within\|0: FT 8.5e-6, OK; full\|0: FT 4.9e-5, OK |
| Run-to-run spread of FT on the A100 | Leonardo 64421057, the same session streamed twice through the same package | P01\|1: fine-tuned posteriors differ by up to **0.034**, outputs by 1.3e-3, labels 0; P02\|1: 3.8e-6 / 1.9e-7. GPU training is not deterministic (atomic adds in backward passes), and the refits amplify it. So 1.02e-4 against the reference is inside the package's own spread, not a logic difference; the CPU check (deterministic) matches to 1.2e-5. A sealed run of a fine-tuning package is not bit-reproducible |
| FT acts, shuffled delivery | Leonardo 64419332, all 14 sessions, gates off, recording / within\|0 / full\|0 | Errors 0; FT on every window from 20. max \|v6.1 − noft\| on FT windows 0.14 / 0.20 / 0.17. Labels vs v5.1b: 35 / 36 / 57 (noft 38 / 42 / 60). In-sample mean BA v5.1b / noft / v6.1: 0.981 / 0.995 / 0.994; 0.982 / 0.997 / 0.995; 0.970 / 0.992 / 0.991 |
| Worst-case run time | Leonardo 64418532, A100, warm, paired, gates off | ×1.809 (190 windows, 17 refits), ×2.082 (378, 36 refits) → 31.3 / 34.7 min at 57 %, **38.8 / 44.7 min at 100 %**, plus about 1.1 min one-off warm-up: **about 45.8 min worst case** (v6f about 45.3) |
| FT in the worker image | Galvani 3099908 (node cn235; the first try, 3099513 on cn228, hit a sticky CUDA hardware error, `cudaErrorContained`), unzipped from 2ca06dc0 | **OK**. Fine-tuning ran in the worker image: 17 / 36 refits, 340 / 720 steps, errors 0. Paired with v5.1b on that node (v5.1b is slower there: 21.5 s per 190 windows): ×1.426 / ×1.698 → 30.6 / 36.4 min at 100 % unrecognised |

## v6.2 (13–14 Nov night): the additions also on recognised sessions, gate-unchanged (ledger §199, §201)

Prepared for the user's choice next to v6f and v6.1, which stay untouched. Code: `lean/src62` → `submissions/lean_v62`
(`bundle.py --v62`), one bundle with the switch `v6_routed`: 0 = v6.1, 1 = **v6.2-safe**, 2 = **v6.2**.
- v5.1b runs unchanged on every window: its mix, its shadow (GE view unchanged) and its gates, which decide
  recognition. The routed set and the first routed window are v5.1b's by construction.
- The output is `RouteB`: lean's `Route.step` code unchanged, with the gate objects replaced by stubs that return v5.1b's
  own decisions of the window. It runs from window 0 on mix_B = (15/16)·(v5.1b mix with the FT delta1 change of slots
  5–8) + (1/16)·rest. Its personal head enters at core1's share of the new mix (w0·15/16 = 1/16), and its routed tail
  carries the eye-net view (SessionEM 48 → 51 columns). Then the own-space stage.
- Before recognition r = 0, so RouteB is exactly v6.1's second chain.
- Fine-tuning: s15 (start window 20, every 10, 5 steps, lr 1e-3). Before recognition it uses v6.1's labels. At
  recognition every stored label is replaced by v5.1b's routed output of that window, which then labels every later
  window. v6.2 keeps fine-tuning after recognition; v6.2-safe stops it there and keeps the rest member and the eye view.
- Deviation from dev (`retv6.py`, GATE_V51B=1), confirmed by the explore track:
  - on windows 0–9 of returning sessions the package emits v6.1's chain, where dev emits v5.1b's shadow;
  - a session recognised after window 20 has had refits on v6.1 labels before recognition (none in the public data,
    all recognised at window 10).

| Package | ZIP SHA-256 | Bytes | Official local replay |
|---|---|---|---|
| v6.2 switch off (`cand_v62_r0`, check only) | `8ee5600eab15243af57953a875a5e0f5b2edf3b4c3bbbf47a8a7cb4b0006b38f` | 66,739,734 | 0.9932767697 (= v5.1b), files unchanged, 90.6 s |
| **v6.2-safe** (`cand_v62_r1`) | `f965ffc1c29abb0db9b52c3149ec26063dc5346f57311b78edd8156564214823` | 66,739,731 | **0.9945788530**, files unchanged, 100.9 s |
| **v6.2** (`cand_v62_r2`) | `5d4bb3a01a526ae9ac03d92410828667c1b79e57fc99525af391a83913263161` | 66,739,739 | **0.9945788530**, files unchanged, 892.7 s on the Mac CPU (fine-tuning on every session: 17 refits per session, errors 0) |

Dev in the package's form (explore track; returning people P04 / P10 against routed v5.1b, within 0–7): v6.2 (ERF s15)
**+0.89** (P10 > 0 on 8 of 8, min +0.10); v6.2-safe (ER) **+0.65**, positive on all 12 orders. Under full/runs, ER is
positive on all four orders; ERF loses only P04 on full 0 (−0.89). New people as v6.1 (+1.49 within 0–7).

| Check (v6.2) | Where | Result |
|---|---|---|
| (a) Switch off = v6.1, gates as shipped | Mac CPU, `v6_compare.py`, all 14 sessions, unzipped from `8ee5600e` vs `2ca06dc0` | **bitwise equal** on 2,655 windows (max 0.0) |
| (a) Switch off = v6.1, gates forced off | Leonardo CPU 64428710 hit its 2 h limit before printing anything; rerun per session `r0o*` | pending |
| (a) Routed set and first routed window = v5.1b | construction: recognition is v5.1b's own gate decision; every check session recognised at window 10 in both | by construction |
| v6.2 / v6.2-safe = v6.1 before recognition | Mac CPU smoke | bitwise on windows 0–9 |
| (b) Routed path against references | `scripts/check_v62.py` (v5.1b's own shadow, which its gates read; v5 tree's Route with the eye columns in its tail; mix formula, eye, FT with the relabelled labels); CPU and A100 with package shapes, recording / within\|0 / full\|0 | First A100 runs (64428711, 64428986) failed on the *shadow* reference only (0.34–0.41): it was built on mix_B with the eye columns, whereas the gate-unchanged v6.2 has no chain-B shadow. Routed outputs were exact (0.0), FT ≤ 8.1e-6. Reference fixed in 3f111dd; reruns on the unzipped ZIPs **pass**: CPU 64429469 (r1) / 64429473 (r2); A100 64429514 (r1) / 64429516 (r2) in recording, within\|0 and full\|0 |
| Worst-case run time | Leonardo 64424648, A100, on the package directories (ZIP rerun pending) | v6.2: gates off ×1.923 / ×2.233, as shipped ×1.918 / ×2.168 → **worst about 49.0 min** with warm-up, expected (43 % recognised) about 42 / 48 min. v6.2-safe: gates off ×1.894 / ×2.205, as shipped ×1.316 / ×1.261 → **worst about 48.4 min**, expected about 31 / 40 min |

## v6.2g: fine-tuning cost guards (14 Nov)

v6f, v6.1 and v6.2 have **no guard** on fine-tuning cost, which grows with the square of session length. The sealed
sessions' lengths are unknown, so v6.2g adds two guards (`lean/src62g` → `submissions/lean_v62g`; v6.2's bundle and
ZIPs untouched):
- **per-session cap** `v6_ft_cap` = 400: no refit starts once the prefix exceeds 400 windows, and the last fine-tuned nets
  keep predicting at batch-1 cost. Inert for sessions up to 400 windows; our longest check session has 378.
- **process wall-clock budget** `v6_ft_budget_s` = 2700 (45 min): a module-level clock set at `load_model`, untouched by
  the evaluator's per-session deep copy. Once exceeded, no refit starts and every later window uses the frozen nets
  (no FT swap); one stderr line per process says so.

The budget gives a tolerance. With the frozen path at about 1.3× v5.1b (v6.2) or 1.15× (v6.2-safe, unrecognised) and
v5.1b's real 21.45 min, the run still finishes inside 60 min:
- v6.2: if the fine-tuning path runs at up to 4.5× v5.1b (2.2× measured), or the whole worker is up to 1.48× slower
  than the A100 measurements;
- v6.2-safe: up to 5.35×, or 1.56×.
With 3000 s the limits would be 3.6× / 1.40×. In the expected case v6.2 (about 42–48 min) trips the budget near the
end: about the last 7 % of the stream, some 7 of 106 sessions, runs without fine-tuning. v6.2-safe trips only in its
worst case.

Smoke (Mac CPU, one session, 75 windows):
- cap 50: bitwise equal to the unguarded run through window 59, then no refit (2 skipped), errors 0;
- budget ≈ 0: bitwise equal to the fine-tuning-off package on all windows, stderr line printed once.

v6.2g ZIPs (official local replay, Mac CPU; `files_unchanged` true, errors 0):

| Package | ZIP SHA-256 | Replay |
|---|---|---|
| **v6.2g** (`cand_v62_r2g`, routed 2, cap 400, budget 2700) | `90a6247033c10036e16761b24058ca01e80d9925b9b632a322bc56434be21a35` | 0.9945788530, 892.6 s (the budget did not trip) |
| **v6.2g-safe** (`cand_v62_r1g`) | `df5d4058c40b716cc675c230241dc229e3040e7052c46fbbba15786c7db4579e` | 0.9945788530 |

| Check (v6.2g) | Where | Result |
|---|---|---|
| Identity to v6.2-safe, gates as shipped | Leonardo CPU 64429234, `v6_compare.py`, 14 sessions | **bitwise equal**, 2,655 windows (no FT after recognition at window 10, so no clock involved) |
| Identity to v6.2 / v6.2-safe, other modes | first wave (64429271/73, 64429235) | **void as identity runs**: `--fresh-clock` looked the module up in `sys.modules`, where `record.load_model` never puts it, so nothing restarted (a silent no-op, fixed in ac66fd3 with `record.ft_clock_restart`, which fails loudly). Each process then tripped the real 2700 s budget in session 2 (P02, window 131 / 141). Before the trip, bitwise equal (P01 whole, P02 to the trip window), and errors 0 after it. Rerun with the fixed restart, one Leonardo CPU job per session (wave 3): v6.2g = v6.2 **bitwise** on all 14 sessions (2,655 windows) with gates as shipped and forced off, v6.2g-safe = v6.2-safe **bitwise** with gates forced off; capped 0, frozen 0, clock never stopped, errors 0 |
| Budget trigger at window 0 = FT off | 64430193 / 64430195, `cand_v62_r2g_noft` vs `cand_v62_r2g_b0`, 14 sessions | **bitwise equal**, 2,655 windows, gates as shipped and forced off |
| Budget trigger mid-session | the natural trips above (window 131 / 141, errors 0); `--budget 20` rerun with the fixed restart: `bmid` | `bmid` (fixed restart, P01 and P02): trips after 1 and 2 refits, the remaining 169 / 158 windows run on frozen nets, errors 0. The natural trips above: bitwise equal to v6.2 before the trip window |
| Cap trigger (cap 50) | 64430196 (gates off), `--from 60` | P01, P02, P03 first differ **exactly at window 60** (the first skipped refit), 13 refits skipped, errors 0. Rerun per session with the fixed restart (`c50o*`): **all 14 sessions first differ at window 60**, 13–14 refits skipped, errors 0 |
| 800-window session, cap against no cap | A100, `v6_time.py --long` (64429573/75, 64430106/07) | v6.2 **×2.84–2.87** (78 refits), v6.2g **×1.77–1.78** (39 refits, 39 capped), errors 0. Peak live CUDA memory 14.1 GB without the cap, 7.2 GB with it (6.6 GB at 378 windows, v5.1b 0.06 GB). The allocator reserves about 67 GB of the 68 GB device and logs OOM-and-retry warnings (cache flushes, not failures). A worker GPU with 40 GB holds the live peak |
| Worker image, GPU | Galvani 3100010 (galvani-cn209), image torch 2.14.1+cu130, cuda True | every candidate replay passed inside the image: v6.2g 0.9945788530 in 185.9 s, v6.2g-safe 140.4 s, v6.2 187.9 s, v6.2-safe 139.4 s (files unchanged; v5.1b 0.9932767697 in 122.7 s). The job shows FAILED only because the closing v5.1b replay reused the first one's output folder (`FileExistsError`, replay_package.py line 42, `exist_ok=False`); worker_replay.sbatch now names folders by position and SHA-checks ZIPs |

## v6.2gL: the L label switch on v6.2g (14 Nov, ledger §206 build)

On unrecognised sessions v6.2gL fine-tunes on the inner v5.1b's output (L, v6f's label) instead of the v6 chain's (R,
v6.1). This needs no new code: `v6_ft_labels` = 0 in v6.2g's bundle. `cand_v62_r2gL` (ERF) and `cand_v62_r1gL` (ER) were
built with `build6.py --v62g --labels v51b --t0 20 --lr 1e-3 --budget 2700 --routed 2|1`. Every array but `v6_ft_labels` and
every other file is byte-identical to `cand_v62_r2g` / `r1g`, so at R the package *is* v6.2g and the guard checks carry
over. Checks: `check_v6.py --skip-recognised` (gates forced off; FT reference with the inner v5.1b's outputs as labels,
the v6f path) on the CPU and on the A100 with package shapes; `check_v62.py`; official local replay; worker image.

| Check (v6.2gL) | Where | Result |
|---|---|---|
| ZIPs, official local replay (Mac CPU) | `v62_r2gL.zip` `76441ee29602a5084379cfb00a36cf0533a8410cb1f6a35f96f27227d5772c1a`; `v62_r1gL.zip` `bb40a2fb4d773cbc46b43867081f17c4ace7915573494534962e79f2c7a1990f` | both 0.9945788530, files unchanged, 860.9 s / 100.7 s |
| Same as v6.2g but the key | `build6.py` output against `cand_v62_r2g` / `r1g` | every other array and file byte-identical |
| `check_v6.py` (L label path = v6f's, gates off), A100, package shapes, 3 orders, on the unzipped ZIPs | 64430676 (r2gL), 64430822 (r1gL) | eye 0, rest ≤ 1.7e-5, mix6 ≤ 6e-17, tail 0 everywhere. FT: r2gL 8e-6 to 1.3e-5 (pass). r1gL full 7.9e-6 (pass), recording 2.3e-4 and within 2.6e-4 (miss at 1e-4). r1gL's FT on this path is the same computation as r2gL's, so the spread is A100 run-to-run non-determinism; the CPU check is the deterministic one |
| `check_v62.py`, A100, 3 orders | same jobs | r1gL pass in all 3. r2gL recording pass (FT 1.2e-5); within 3.6e-4, full 2.9e-4 on FT only (eye, mix, shadow, routed exact) |
| `check_v62.py` CPU | 64430821 (r1gL) | **pass**. r2gL: the first run (64430675) was voided by the budget trip (the clock bug above); the per-session rerun **passes** on P01, P02 and P03 (eye ≤ 6.4e-7, mix ≤ 7e-18, shadow 0, routed 0, FT ≤ 1.1e-5) |
| `check_v6.py` CPU, on the unzipped ZIPs | first runs voided by the budget trip; rerun per session | r2gL and r1gL **pass** on P01, P02 and P03 (eye ≤ 6.4e-7, rest ≤ 3.6e-7, mix6 ≤ 1.4e-17, tail 0, FT ≤ 1.2e-5) |
| Run time on the ZIP, A100 | 64430676 / 64430822 | r2gL gates off ×1.85 / ×2.13 (190 / 378 windows), as shipped ×1.85 / ×2.10; r1gL gates off ×1.85 / ×2.13, as shipped ×1.29 / ×1.30. Errors 0, capped 0, frozen 0 |
| Worker image, GPU | Galvani 3100013 (galvani-cn203) | v6.2gL 0.9945788530 in 180.4 s, v6.2gL-safe 136.6 s, files unchanged (v5.1b 119.8 s); FAILED only by the same closing-folder collision |

## v6.2h: the step62 exception guard (14 Nov, ledger §206 build addendum)

**The v6.2, v6.2g and v6.2gL ZIPs are unsafe.** `step62` ran the v6 additions (eye, rest, label tail, fine-tuning,
mix, RouteB) with no exception guard; v6.1's `step61` has one. In the worker image on Leonardo's CPU with an unusable GPU
driver (ni26-push, 64443845), Adam's CUDA graph-capture check inside a refit raised, and the official ingestion
failed.

v6.2h (`lean/src62h` → `submissions/lean_v62h`, `submission.py` `ccf16a94…`; `build6.py --v62h`, v6.2g's keys) puts
everything after v5.1b's output and the recognition bookkeeping under try/except:
- an error is counted (errors in the [v6] line, the first error's text) and prints one stderr line;
- that window and the rest of the session emit v5.1b's output; relabel is inside the guard, recognition outside it;
- a non-finite output is also an error.
With no error, the computation is v6.2g's. `cand_v62_r2hL`, `r2h`, `r1hL`, `r1h` have data files byte-identical to
the g / gL counterparts.

| Check (v6.2h) | Where | Result |
|---|---|---|
| Forced exception at window 30 (`studies/14nov/v62h_raise.py`; ft.step, the worker's crash site, and eye.step; gates as shipped and off; P01–P03; all four packages) | Leonardo CPU, 27 jobs | **27/27 pass**: the run completes; errors 1; windows 0–29 bitwise equal to the unforced run; windows ≥ 30 bitwise equal to the inner v5.1b and to the v5.1b package |
| Identity to v6.2g / v6.2gL, 14 sessions (r2: gates as shipped and off; r1: off) | Leonardo CPU, 84 jobs `h*` | **84/84 bitwise equal** (2,655 windows per mode and package) |
| `v62_r2hL.zip` | `e22f67e71c297aef9fd98e4ae86c4ea934aacc0f03670d4900eef02874c28098` | local replay 0.9945788530, files unchanged, 882.5 s; worker image on CPU (ni26-push) running |
| `v62_r2h.zip` | `3b13ff96ecd4020d0522e1facb06e3f05a7d1b672a9058e991e02e6b12a142c4` | 0.9945788530, files unchanged, 854.4 s |
| `v62_r1hL.zip` | `2f53c3f950b32dd4f67f2dfe40de47289444a46ed2f0235de7aab69f7b5c3b8b` | 0.9945788530, files unchanged, 110.6 s |
| `v62_r1h.zip` | `9bf9f6e46f6191dcbd00c224fcaf00ea4c51d9de44d068d7ac25e000d422bce5` | 0.9945788530, files unchanged, 109.7 s |
| CPU `check_v6` / `check_v62` on the unzipped h ZIPs (P01–P03) | Leonardo CPU, `v6r*h*`, `v62r*h*` | **24/24 pass**: eye ≤ 6.4e-7, rest ≤ 3.6e-7, mix ≤ 2.8e-17, shadow / routed / tail / label 0, FT ≤ 1.2e-5 |

Worker image on a GPU for v6.2h: Galvani 3100144 (galvani-cn208, torch 2.14.1+cu130, cuda True; SHA-checked) **passed**:
v6.2hL ERF 0.9945788530 in 166.1 s, v6.2h ERF 166.6 s, v6.2hL ER 122.0 s, v6.2h ER 122.5 s, files unchanged, no errors
(v5.1b 106.5 s / 100.4 s). The g / gL ZIPs passed the same image on a GPU (3100010 / 3100013): its torch ran v6.2's routed path with
fine-tuning without error. The worker image on CPU is ni26-push's (Leonardo, image software only). A100 timing
and memory carry over from v6.2g / v6.2gL (identical computation without errors).

## v6.2i: no state across sessions (14 Nov)

**v6.2g and v6.2h break the official rule** that "state, windows, statistics, and parameter updates must not cross the
session boundary" (`scientific_faq_bci.md`). Their module-level `FT_CLOCK` (v6.2g's process budget) switches
fine-tuning off for every later session and makes outputs depend on worker speed. Neither is to be uploaded.

v6.2i rev2 (`lean/src62i` → `submissions/lean_v62i`, `submission.py` `ba2cf3a7…`, 7043cec; `build6.py --v62i --cap 400
--session-s 120`), after the performance review and the package review (`evidence/review_14nov/`):
- **Per-session limit instead of the clock.** `v6_ft_session_s` = 120 s caps this session's refit seconds
  (`FineTune.ft_sec`, on the model, reset with the session), counted from the session's second refit. A fresh
  process's first refit carries a one-off CUDA / kernel warm-up of about 65 s. Past the limit, no new refit in the
  session; the last fine-tuned nets keep predicting. The 400-window cap stays.
  - A100 refit seconds per session (Leonardo 64448969, synchronised): 5.85 (190 windows, 17 refits), 23.6 (378, 36;
    longest 1.18), 27.5 (800 capped at 400, 39). So 120 s binds only at ≥ 4.4× slower refits for the longest sessions
    and about 20× at 190 windows: it guards against pathological sessions, not a slow worker.
  - **Protection against a slow worker now comes from slack alone.** All-unrecognised ratios, from the 1287 s × ratio +
    1.1 min record: ×1.86 at 190 windows (sealed mean 189.5) ≈ 41 min, ×2.13 at 378 ≈ 47 min. A uniform worker slowdown of
    about **1.46×** (190-window ratio) to **1.28×** (378-window ratio) against the A100 measurements exceeds 60 minutes.
    The real protection is a cheaper refit path (track B).
  - No per-session limit can bound the run without cutting normal sessions: binding in all 106 sessions, refits alone
    could take 106 × 120 s ≈ 3.5 h.
  - **CPU note:** a CPU refit is about 10× slower than an A100's, so the limit binds on CPUs. On the Mac CPU the last
    4 refits of P01 are skipped (window 150 on). The CPU replays (647 s against 882 s for v6.2hL) therefore run a
    shorter schedule than the GPU. CPU identity and reference checks lift it explicitly (`--session-s-override 1e9`);
    the trigger package tests the limit itself.
- **Exception handler**: no CUDA call; its counting and stderr print sit in their own try/except.
- **Keys**: `check_package` refuses out-of-range switches and guards (tested: cap 0, session_s 0, routed 3, every 0).
  `build6.py` has no guard defaults.
- From the performance review: fine-tuning copies load lazily; `self.X` stops at the cap; `empty_cache` at
  `FineTune.reset_state`. **Not** in v6.2i: `cudnn.deterministic` (process-wide, changes v5.1b's convolutions; track B).
- **Cross-session audit:** `FT_CLOCK` was the only module-level mutable state that can change predictions.
  - Fine-tuning reloads its copies and reseeds its generators per session; eye, rest and net streams reset.
  - Module level has only a thread pool and constants.
  - TorchScript's profiling executor specialises / fuses after warm-up runs: process-global but data-independent (as in
    the uploaded v5.1b).
- Check scripts: `check_v6` / `check_v62` compare through a NaN-safe `upd()`, and a NaN or inf difference fails. Forced:
  `--inject-nan ft` on v6.2iL gives ft = inf and the check exits 1.

| Package (v6.2i rev2) | ZIP SHA-256 | Local replay (Mac CPU; the limit binds on CPU) |
|---|---|---|
| v6.2iL ERF `cand_v62_r2iL` | `023d29c8dab3f64f824398d0e392f8abe1c62a63f0f7a2e77f09e2e5883ef0d6` | 0.9945788530, files unchanged, 647.1 s |
| v6.2i ERF `cand_v62_r2i` | `b7d9878b10c8414be4dd09b9c9f7e3dcefd4ac425c1f217302ccbcd23f6ec8a4` | 0.9945788530, 641.7 s |
| v6.2iL ER `cand_v62_r1iL` | `0131fc8c844cc6275462e373d06dfb3bc6d30114186b0fa6e073c548ee3a21c3` | 0.9945788530, 108.2 s |
| v6.2i ER `cand_v62_r1i` | `23884186ec75e88cbf47685aff1272ad3b1de2c9674c0604949086ef3e2b01b9` | 0.9945788530, 103.3 s |

| Check (v6.2i rev2) | Where | Result |
|---|---|---|
| Per-session limit trigger (`cand_v62_r2iL_s0`, limit ≈ 0: refits at 20 and 30, the next skipped) | Galvani CPU, `v6_compare --from 40`, 14 sessions, gates as shipped and off | **all 14 first differ at window 40**, both modes; 15–16 refits skipped; errors 0 |
| Forced exception at window 30 (ft / eye, both gate modes, P01–P03, all four packages) | Galvani CPU, 27 jobs | **27/27 pass** |
| `record.py` stress, extreme, faults, long (577), switch, lapse; v6.2iL ERF and ER | Galvani CPU, 12 jobs | **all complete**, outputs finite, labels in {0, 1, 2}. The extreme case (float32-edge windows) raises once per run: v6.2's mix is not finite; the guard counts it, and the session continues on v5.1b |
| Identity to v6.2h / v6.2hL, limit lifted, 14 sessions (r2 both gate modes, r1 off) | Galvani CPU `K*` (84) | pending |
| `check_v6` / `check_v62` on the unzipped ZIPs, limit lifted (P01–P03) | Galvani CPU `Q*` (24) | pending |
| A100 memory share 40/68 (`set_per_process_memory_fraction`), 190 / 378 / 800 windows, v6.2iL ERF | Galvani 3100458, which is an **A100-PCIE-40GB**, so the share was about 23.5 GB | **pass**: errors 0; peak live 3.26 / 6.64 / 7.17 GB, peak reserved ≤ 24.7 GB; ratios ×1.73 / ×1.99 / ×1.65 (capped) against v5.1b on that card |
| Worker image on a GPU (v5.1b, the four, v5.1b; SHA-checked) | Galvani 3100604 | pending |

## v6.2iu: the time limit off (14 Nov, upload variant)

Finalists are audited for reproducibility, and an audit rerun on other hardware must give the same labels. v6.2i rev2's
per-session time limit makes outputs depend on machine speed (it binds on CPUs), and it protects nothing the 400-window
cap does not already bound. v6.2iu sets `v6_ft_session_s` = 1e9 s, so it never binds: same code (`submission.py`
`ba2cf3a7…`), same cap, every other file and array identical to rev2. **Run-time protection is the 400-window cap plus
slack; outputs do not depend on timing.** The deterministic work bound (fixed-size refit batches, constant cost per
window) belongs to track B. Packages `cand_v62_r1iLu` (**v6.2iLu ER, the upload candidate**, ledger §214), `r2iLu`,
`r1iu`, `r2iu`; ZIPs `v62_*u.zip`.

**Sealed run-time estimate, v6.2iLu ER.**
- Formula: 1287 s (v5.1b's real sealed run, 21.45 min) × ratio + 1.1 min warm-up.
- Ratios against v5.1b on the A100: unrecognised sessions (fine-tuned) ×1.85 at 190 windows and ×2.13 at 378
  (Leonardo SXM); recognised sessions (no fine-tuning) ×1.29 and ×1.30. Track record of this estimate: −7 % to +6 %.
- **Expected** (about 43 % of sessions recognised): **≈ 36 min** (190-window ratios) to **≈ 39 min** (378). **Worst
  case** (no session recognised): **≈ 41 min** to **≈ 47 min**.
- **Slowdown at which 60 min is exceeded** (uniform, against the A100 measurements): about **1.53–1.68×** expected,
  **1.28–1.46×** worst case.

| Package (v6.2iu) | ZIP SHA-256 | Local replay (Mac CPU, full schedule) | Worker image, GPU (Galvani) |
|---|---|---|---|
| **v6.2iLu ER** `cand_v62_r1iLu` | `d0dd9934a77ceabb283b1c94e011c6d3b1c6ce4606480b0353408587a5df6fd3` | 0.9945788530, files unchanged, 109.5 s | 3100822 **pass**: 0.9945788530 in 136.3 s (v5.1b 100.5 / 116.2 s) |
| v6.2iLu ERF `cand_v62_r2iLu` | `fbfe93102c54d03456ee340e9a35c118515b1f3a5bd6dcccd865cf37b56812a3` | 0.9945788530, 909.4 s | 3100857 pending |
| v6.2iu ER `cand_v62_r1iu` | `5e023a34a2060563ddd08f61c0c4717bd132860ca73770367078c3fcd67395f3` | 0.9945788530, 102.3 s | 3100864 pending |
| v6.2iu ERF `cand_v62_r2iu` | `e74aab581dc1663f9bcf74833da4eac60c4c5fc44c1ce7a3ebb40ab235c10053` | 0.9945788530, 977.3 s | 3100919 pending |

The rev2 ZIPs passed the GPU worker image (Galvani 3100604): v6.2iL ERF 163.9 s, v6.2i ERF 164.8 s, v6.2iL ER 119.3 s, v6.2i
ER 118.7 s, all 0.9945788530. Pending at 08:35 UTC on Galvani (141 jobs):
- identity of v6.2iu to rev2 (`U*`) and of rev2 to v6.2h (`K*`, 14 of 21 read so far, all pass);
- the stress cases on the r1iLu / r2iLu ZIPs (`W*`);
- the ZIP reference checks (`Q*`);
- the r1iLu 40 GB-share run (3100821).

### Check table: v6.2iLu ER, the upload candidate (14 Nov 12:30 UTC)

`cand_v62_r1iLu`. ZIP `v62_r1iLu.zip` SHA-256 `d0dd9934a77ceabb283b1c94e011c6d3b1c6ce4606480b0353408587a5df6fd3`;
`submission.py` `ba2cf3a7…` (lean/src62i rev2).
Keys:
- fine-tuning labels L (the inner v5.1b's output); `v6_routed` 1 (no fine-tuning once recognised);
- s15 schedule (t0 20, every 10, 5 steps, lr 1e-3);
- cap 400; `v6_ft_session_s` 1e9 (time limit off).

| # | Check | Where | Result |
|---|---|---|---|
| 1 | Build: data identical to rev2 except `v6_ft_session_s` (120 → 1e9); rev2 identical to v6.2hL except the guard key; code reviewed (package review, `evidence/review_14nov/`) | Mac, build6 asserts | pass |
| 2 | Key ranges refused at load (cap 0, session_s 0, routed 3, every 0); build6 has no guard defaults | Mac | pass |
| 3 | Official local replay of the ZIP | Mac CPU | 0.9945788530, files unchanged, 109.5 s |
| 4 | Worker image on a GPU (the ZIP, SHA-checked) | Galvani 3100822, A100-PCIE-40GB, torch 2.14.1+cu130 | 0.9945788530 in 136.3 s (v5.1b 100.5 / 116.2 s), no errors |
| 5 | Identity to rev2 (rev2's limit lifted), 14 sessions, gates forced off | Galvani CPU `Ur1iLuo*` | 14/14 bitwise |
| 6 | Identity chain rev2 → v6.2hL (14 sessions, gates off) | Galvani CPU `Kr1iLo*` | 14/14 bitwise |
| 7 | `check_v6` on the unzipped ZIP (eye, rest, FT from the L labels, mix6, tail against references outside the package), P01–P03, gates off | Galvani CPU `Z6r1iLu*` | pass: eye ≤ 6.4e-7, rest ≤ 3.6e-7, FT ≤ 1.2e-5 (tolerance 1e-4), mix6 ≤ 2.8e-17, tail 0 |
| 8 | `check_v62` on the unzipped ZIP (routed path, gates as shipped), P01–P03 | Galvani CPU `Z62r1iLu*` | pass: routed 0, shadow 0, mix 0, eye ≤ 6.4e-7 |
| 9 | Forced exception at window 30 on the unzipped ZIP (eye as shipped, FT gates off), P01–P03 | Galvani CPU `ZX*` | 6/6: completes, errors 1, windows 0–29 bitwise as the unforced run, 30 on bitwise v5.1b (inner and package) |
| 10 | `record.py` stress / extreme / faults / long (577) / switch / lapse on the unzipped ZIP | Galvani CPU `W*` | 6/6 complete, outputs finite, labels in {0, 1, 2}; extreme: one caught error (v6.2 mix not finite) |
| 11 | Memory: 40/68 share of a 40 GB A100 (about 23.5 GB) | Galvani 3100821 | errors 0; peak live 3.26 / 6.64 / 7.17 GB (190 / 378 / 800), reserved ≤ 24.7 GB |
| 12 | Run time on that card | Galvani 3100821 | gates off ×1.73 / ×1.96 (190 / 378), as shipped ×1.24 / ×1.25, 800 windows capped ×1.65 |
| 13 | Run time on an A100 SXM | Leonardo, on v6.2h / v6.2g (identical computation) | gates off ×1.85 / ×2.13, as shipped ×1.29 / ×1.30 |
| 14 | Sealed estimate (1287 s × ratio + 1.1 min) | from 12–13 | expected ≈ 36–39 min, worst ≈ 41–47 min; 60 min at a uniform slowdown of 1.53–1.68× (expected) / 1.28–1.46× (worst) |
| 15 | Cross-session state | code audit | none that changes predictions (no module-level state; fine-tuning reloads and reseeds per session) |
| 16 | NaN-safe comparisons in the check scripts; forced NaN must fail | Mac | pass (ft = inf, exit 1) |

Not validated:
- The real Codabench worker. Only a replica of its image was run, on Galvani's A100-PCIE-40GB; the worker's GPU type,
  driver and CPU count are unknown. The run-time estimate's track record is −7 % … +6 %.
- **Reproducibility of fine-tuned labels on a GPU.** cuDNN is not set to deterministic, so fine-tuning on a GPU differs
  run to run (the A100 FT reference spread reached 2–4e-4 in posteriors). An audit rerun on a GPU may change labels in
  unrecognised sessions; on a CPU it is bitwise reproducible. Deterministic cuDNN is in track B.
- The A100 package-shape reference checks (`--ref-shapes package`) were last run on v6.2g / v6.2gL / v6.2hL, not on this
  ZIP. The code path is identical, and the CPU checks above are on this ZIP.
- A poisoned CUDA context (a sticky CUDA error) also hits v5.1b's nets; the exception guard does not cover it.
- There is no timeout on the core threads.
- The sealed data: session lengths, recognition share and delivery order are unknown.

## v6.2c: the run-level count filter on v6.2iLu ER (14 Nov, ledger §228c package)

`lean/src62c` → `submissions/lean_v62c` (`submission.py` `7a4ec021…`); `cand_v62c_r1iLu_off` / `_on`; parameters
`runs/countfilter/cf_params.npz` (`lean/tools/fit_cf.py`, ten-person fit). Form and fit as in the ledger entry.

| Check (v6.2c) | Where | Result |
|---|---|---|
| Reference smoke: P01, recording, gates off, against the dev code pinned at 3a10ba8 (`studies/14nov/cf_ref`) with the same parameters | Mac CPU | latch 40 in both, outputs **bitwise** equal, 0 labels changed; **4.4 ms per window** for the filter |
| Official local replay, filter on: `v62c_r1iLu_on.zip` `6ccc0e14de9e01a75f393435a4c33c953eee469d20232f62c6e21edf356fadb8` | Mac CPU | 0.9945788530, files unchanged, 125.2 s |
| Official local replay, filter off: `v62c_r1iLu_off.zip` `0fc5e5cc9bb5cf4b2702d85ab6896b166de4a1c79c25124f17ca341427710c79` | Mac CPU | 0.9945788530, files unchanged, 111.1 s |
| Worker image on a GPU (on, off) | Galvani 3101653 (cn241) | **pass**: on 0.9945788530 in 141.7 s, off 0.9945788530 in 135.7 s, files unchanged (v5.1b 120.0 / 115.7 s) |
| Identity with the filter off, `v6_compare` to v6.2iLu ER, 14 sessions, gates as shipped / off | Leonardo CPU `cfA*`, `cfAo*` | **28/28 bitwise** |
| Identity with the filter off through `predict()`, `record.py` per case (14 public, 4 unknown, switch, lapse, long, stress, extreme, faults, perm, missing) | Leonardo `cfRoff*` against Galvani's v6.2iLu recordings | labels equal everywhere; posteriors differ by ≤ 6.9e-9 (3.2e-6 on unknown sessions, where fine-tuning amplifies) because the two machines' CPUs round differently. **Same machine (`cfRref*`, Leonardo): 23/23 cases bitwise** |
| Reference against the pinned dev code, gates off: recording, within 0 / 1, full 0 / 1 (14 sessions each) | Leonardo `cfB*` | **70/70: latch window equal and outputs bitwise equal in every session**; latched 48 / 70; 12 labels changed in all; filter cost 8.3 ms per window on average (max 13.4) on Leonardo's CPU, 4.4 on the Mac |
| Same, within 2 / full 2 | Leonardo | **void**: `shuffle_perms_fixed.npz` holds seeds 0–1 per scheme only (KeyError, no run). Replaced by runs 0 / 1 (`cfBruns*`): **28/28 bitwise, latch equal** |
| Forced exception in the filter at window 20 (before the latch) and 60 (after it), P01–P03 | Leonardo `cfC*` | **6/6**: errors 1, unfiltered from the error on, the reference before it |
| `record.py` with the filter on: unknown (4), switch, lapse, long (577), stress, extreme, faults, perm, missing | Leonardo `cfRon*` | **9/9 complete**, outputs finite, labels in {0, 1, 2}, no filter error |

**Known gap, fixed in the next revision:** the per-session `[count-filter] … latched_at … errors` line never reaches
the evaluation log. The evaluator drops each session's model copy without calling `reset_state`, and `CountFilter`
has no `__del__` (`Route` and `Ensemble6` report from theirs). Only logging is affected. The fix (a `__del__` that
reports) changes `submission.py`, so it waits for the revision that adds ni26-push's self-check, together with a rebuild
and recheck.

## v6.4: r15 and the count filter behind their keys (14 Nov, ledger §242 package)

`lean/src64` → `submissions/lean_v64` (`submission.py` `6ab43e4b…`). On v6.2iLu ER's keys; `v6_ft_relabel` (r15,
before recognition only, ni26-push's definition) and `v6_cf_on` (the count filter, now reporting from `__del__`). E_pca4
is not packaged (no package equivalent of dev's per-fold-net view).

| ZIP (v6.4, r1iLu keys) | SHA-256 | Local replay (Mac CPU) |
|---|---|---|
| `v64_r1iLu_all` (filter + r15) | `10b00efba4a8efd865d9bbeec0ebc300469c1e398b72dda3a5359803e2dc4fdc` | 0.9945788530, files unchanged, 106.2 s; one `[count-filter] … latched_at 40 errors 0` line per session in the evaluation log |
| `v64_r1iLu_r15` | `f182c6e7b555cc0add8e119b94ceef690b05abf0bae0060135c06a6167f1b0ff` | 0.9945788530, 100.3 s |
| `v64_r1iLu_cf` | `7c1eedcc8e6fde97f7804e5b4bdd0c796c4909f952c25007700a13f5c5f56a5d` | 0.9945788530, 102.2 s; filter log lines present |
| `v64_r1iLu_off` | `656a5825b3338870dd964245ebfdc2bb96593d8d9087130f600e38390f4fccc7` | 0.9945788530, 99.2 s |

The public replay sessions are recognised at window 10, before the first refit (window 20), so the replay never
exercises r15; the reference checks run with recognition off.

| Check (v6.4) | Where | Result |
|---|---|---|
| r15 reference smoke, P01 recording, gates off | Mac CPU | 17/17 refits relabelled; max difference 0.0 against relabel.py's formulas recomputed from the stored stage inputs; 0.13 ms per relabelling |
| All-off identity smoke, P01, gates off | Mac CPU | bitwise equal to v6.2iLu ER |
| Identity all-off to v6.2iLu ER, 14 sessions, gates as shipped / off | Leonardo `v4A*`, `v4Ao*` | **28/28 bitwise** |
| r15 reference, recording, within 0 / 1, full 0 / 1, runs 0 / 1 × 14 sessions, gates off | Leonardo `v4R*` | **98/98**: max difference 0.0 in every session; 1,680 refits relabelled, 2,885 labels changed against the as-emitted ones (r15 acts); 0.29 ms per relabelling on average |
| Filter reference with r15 on (recording, within 0 × 14); r15 reference with the filter on (recording × 14) | Leonardo `v4F*`, `v4Q*` | **42/42** |
| Forced exception in the relabelling (refit at window 30, P01–P03) | Leonardo `v4X*` | **3/3**: errors 1, v5.1b from 30 on, the reference before it |
| record.py stress / extreme / faults / long / switch / lapse / perm / missing / unknown, all on | Leonardo `v4S*` | **9/9 complete**, outputs finite, labels in {0, 1, 2}; extreme: one caught error (v6.2 mix not finite at window 60) |
| Paired A100 timing, all against off (gates off / as shipped), and against v5.1b | Galvani 3101756 | **void**: the timing tool's refit wrapper took one argument while v6.4 calls `refit(t, relab_fn)`. It raised at the first refit, the package's guard fell back to v5.1b (err 1, 0 refits), and the ratios (×1.0) measured v5.1b. Tool fixed (pass-through, and an assertion that no v6 error occurs in a timed run); rerun on the candidate ZIP: Galvani 3102619 |
| Worker image on a GPU (all, r15, cf, off) | Galvani 3101926 (cn225) | **pass**: all 146.1 s, r15 138.9, cf 146.7, off 139.0, all 0.9945788530, files unchanged, no traceback (v5.1b 122.6 / 117.8 s) |

## v6.5: T, the personal TSEM template prior, and the upload candidate (14 Nov, ledger §245 package)

`lean/src65` → `submissions/lean_v65` (`submission.py` `dba089f1…`, 01e5288); v6.4's bundle and ZIPs untouched.
- **T** (`v6_tpri_on`): route4's personal TSEM template prior, in RouteB's routed tail only. On a routed window with
  ramp r and person j, TSEM's class means are (Σ + 20 r · pri[j]) / (n + m0 + 20 r), with pview's denominator fix. v5.1b's
  Route, the gates and the shadow are unchanged; rows before the first routed window are unchanged.
- **pri** (`lean/tools/fit_tpri.py`, `runs/tpri/tpri.npz`): for each of the ten people, the mean over **all** their public
  sessions (P04 and P10 three each, the others one) of the class means of rlib3.stdz of the package's own core1 TSEM
  coordinates. **Dev built pri from session 1 of P04 / P10 only (seen_session); dev cannot measure this difference.**
- A window-exact comparison with pview's dev outputs is not possible: dev's chain uses seen_session members and core1
  trained on session 1 of P04 / P10, while the package's are trained on all ten people. The reference therefore
  recomputes dev's formulas from the package's own inputs.

| ZIP (v6.5, r1iLu keys) | SHA-256 | Local replay (Mac CPU) |
|---|---|---|
| **`v65_r1iLu_cand`: filter + r15 + T, the upload candidate** | `b015bb272f5f8b054ed533649678c371930b133e39bc5324c3ae7ecf95d8190c` | 0.9945788530, files unchanged, 104.8 s; 4 `[count-filter]` lines |
| `v65_r1iLu_T` (T only) | `1300079c3c8aa71002dea2dbd9c2c0638fd7521f4ef84fee530b76aa69025a5a` | 0.9945788530, 100.3 s |
| `v65_r1iLu_off` | `44ddbd50dc7ee646d096ade689706202a5607e7e84c381488dd51f0171826345` | 0.9945788530, 100.2 s |

| Check (v6.5) | Where | Result |
|---|---|---|
| T off: bitwise to v6.2iLu ER; cfr15 (T off): bitwise to v6.4 "all"; 14 sessions × gates as shipped / off | Leonardo `v5A*`, `v5Ao*`, `v5B*`, `v5Bo*` | **56/56 bitwise** |
| T reference (gates as shipped): every routed-tail TSEM step against pview's formula, the stored prior against route4's expression recomputed, windows before routing against T off; recording, within 0 / 1, full 0 / 1, runs 0 / 1 × 14 | Leonardo `v5T*` | **98/98**: TSEM difference 0.0, prior ≤ 3.8e-12, before routing equal everywhere; prior applied on 17,605 windows from window 10; 8 labels changed against T off, posterior change ≤ 0.099 |
| On the candidate: T reference (against cfr15), r15 reference, filter reference (recording × 14 each) | Leonardo `v5Kt*`, `v5Kr*`, `v5Kf*` | **42/42** |
| Forced exception on the candidate: TSEM site (window 30, as shipped), fine-tuning site (window 30, gates off), P01–P03 | Leonardo `v5Xt*`, `v5Xf*` | **6/6** |
| record.py stress / extreme / faults / long / switch / lapse / perm / missing / unknown on the candidate | Leonardo `v5S*` | **9/9 complete**, outputs finite, labels valid; extreme: one caught error |
| Worker image on a GPU (cand, T, off) | Galvani 3102382 (cn225) | **pass**: cand 0.9945788530 in 145.1 s, T 142.6, off 138.7, files unchanged, no traceback (v5.1b 121.6 / 118.8 s). The public replay sessions are recognised before the first refit, so the image never fine-tunes there; GPU fine-tuning of v6.4 / v6.5 code is timed in 3102619 |
| **GPU fine-tuning function check of the candidate ZIP** (`b015bb27…`, unzipped; `r15_reference.py --device cuda --compare-cpu`; P01 and P10\|1, unrecognised, gates off; native `.venv`, torch 2.14.1+cu126, **RTX 2080 Ti**, not the worker image) | Galvani 3102651 (cn122; duplicate 3102652 on cn112 agreed on P01 and was cancelled) | **pass**: err 0 on GPU and CPU; refits = relabels = schedule (17 and 18); r15 relabel reference difference 0.0 (16 and 17 labels changed against the as-emitted ones); GPU against CPU: label agreement 100 %, output max difference 0.039 (P01) / 0.044 (P10\|1), the non-determinism of GPU fine-tuning; relabelling 0.3 ms per refit |
| A100 timing of the candidate ZIP (Galvani A100-PCIE-40GB; against v5.1b and against its all-off ZIP; 800 windows) | Galvani 3102619 (`runs/diag/gcpu/t65-3102619.out`) | **pass**: cand vs off 190 win ratio 0.991, 378 win 0.977 (no cost of the keys beyond noise), err 0, relabels = refits (17 / 36); 800 windows (cap 400 active, 39 refits): v5.1b 73.8 s, v6.5 131.9-135.3 s, **x1.808**, err 0; sealed estimate 31.3 min at 57 % unrecognised, 38.8 min at 100 %. **Finding**: long unrecognised sessions raise CUDA caching-allocator retry warnings (24 lines in the log): peak_alloc 3.3 / 6.65 / 7.17 GB (190 / 378 / 800), peak_reserved 30.4 / 41.55 / 41.54 GB of 42.4. Fix: v6.5b below |
| **Function check in the worker image on an A100** (`r15_reference.py runs/sealed/zt65/cand --device cuda --compare-cpu --only 0`; P01 recording order) | Galvani 3102646 (`runs/diag/gcpu/f65-3102646.out`; twin 3102647 cancelled / partner) | **pass**: A100-PCIE-40GB, torch 2.14.1+cu130, cuda True; err 0; refits = relabels = schedule = 17; relabel diff 0.0; GPU-vs-CPU label agreement 100 %, output max diff 0.033; `RESULT ok true` (raise_at -1), EXIT 0 |

## v6.5b: empty_cache after each refit (14 Nov, ledger §261)

`lean/src65b` → `submissions/lean_v65b` (`lean/tools/bundle.py --v65b`; `build6.py ... --v65b` with the v6.5 candidate's keys).
The only code difference from v6.5 (`diff` of the bundles): `FineTune.refit` wraps the refit in try/finally and calls
`torch.cuda.empty_cache()` in the `finally` when `torch.cuda.is_initialized()`. CPU runs never reach it. The package
directory `runs/sealed/cand_v65b_r1iLu_cand` differs from `cand_v65_r1iLu_cand` in `submission.py` only (all 38 other files cmp-equal).

| ZIP | SHA-256 | Local replay (Mac CPU) |
|---|---|---|
| `v65b_r1iLu_cand` | `a7b13e57f6163b4a26e2cc326c0fed3f14ceadfc23c02de3f9f5a24dbbe850d6` (66,786,277 B) | 0.9945788530 (= v6.5), files unchanged, 122.0 s (v6.5 104.8 s; machine load differs), 4 `[count-filter]` lines |

| Check (v6.5b, bundle `5323fd8b…3397f2`, the sha printed inside each job's log) | Where | Result |
|---|---|---|
| GPU function check (`r15_reference.py --device cuda --compare-cpu`, P01 and P10\|1, gates off, native `.venv`) | Galvani 3103438 (RTX 2080 Ti, cn102; widened partitions) | **pass**: err 0 on GPU and CPU, refits = relabels = schedule (17 / 18), relabel reference diff 0.0, GPU-vs-CPU label agreement 100 %, output max diff 0.039 / 0.044 (same non-determinism band as v6.5: 0.033-0.044), 0 allocator warnings |
| A100 timing, paired v6.5 vs v6.5b (190 and 378 windows) | Galvani 3103439 (A100-PCIE-40GB, cn221) | **pass**: peak_reserved **4.34 / 8.81 GB** (v6.5 in the same job: 30.38 / 41.55), peak_alloc unchanged (3.26 / 6.64); 6 allocator OOM-retry warnings, **all in the v6.5 arm**, none in v6.5b; run time v6.5b / v6.5 **1.019** (190) and **1.000** (378); err 0, refits and relabels equal |
| A100 timing against v5.1b (190 / 378 / 800 windows) | same job | v6.5b ratio 1.972 / 2.284 / 1.837 (800: reserved 9.52 GB, err 0, capped 39); sealed estimate 33.3-37.1 min at 57 % unrecognised (31.7 on the 800 basis), 42.3-49.0 at 100 %. **Reading**: these ratios are higher than 3102619's (v6.5 x1.808 at 800) because this node's v5.1b baseline ran 2x faster (16.2 s against 31.2 s at 190 windows) while the v6 arm's times are unchanged (31.5-32 s); the paired v6.5-vs-v6.5b section is the comparison that matters and shows no change. The node-to-node spread is as large as the v5.1b/v6.5b difference, so the sealed minutes carry that spread |
| CPU bitwise identity v6.5 vs v6.5b (`studies/14nov/cpu_identity.py`, P01 and P10\|1, unrecognised path, 17 / 18 refits, capped 0) | Galvani 3103433 (CPU node, 47.7 min) | **pass**: max_diff 0.0, bitwise_equal true on both sessions, err 0, refits equal; log `evidence/v65b_14nov/cid65b-3103433.out` |

§261 prediction check: (2) holds, (3) holds (reserved <= 12 GB, no warning in the v6.5b arm, +1.9 % / 0.0 %); (1) holds (CPU bitwise identity on both sessions; Mac replay equal: 0.9945788530). All three predictions hold: v6.5b is the upload candidate in place of v6.5 (needs the user's go-ahead to upload).

## v6.6 (development, 14 Nov, ledger §262): cheaper refits, separate tree

`lean/src66` (copy of `lean/src65b`), bundle `submissions/lean_v66`; the v6.5b candidate ZIP and bundle stay untouched.
**Pre-registered target:** the refit path from 14-20 sealed minutes to 2-4, run time at 378 windows v6.6/v6.5b <= 0.6;
identity against v6.5b bitwise on CPU for the arithmetic-neutral changes (GPU buffer, one-sync dtheta), CPU output tolerance
1e-5 with 100 % labels for the cached-FIR / shared-batch forms, GPU <= 0.05 with labels 100 %. Forms and refutation rule in §262.

**Stage-0 result and decision (14 Nov 22:55 CET; ledger §262; `evidence/v66_14nov/rprof-3103933.out`, 2080 Ti):** one refit 1285 ms at
prefix 190, 2535 ms at 378; backward + Adam 40 %, augmented forward 25 %, clean forward 25 % (half of it the frozen FIR), final pass 5 %,
staging 2 %. The pre-registered prediction (>= 50 % in the cacheable / staging parts, >= 1.8x for the stack) is **refuted**: the
identity-preserving forms reach ~1.2x. **Track B stopped (coordinator, 14 Nov):** no `lean/src66` tree, no new bundle; v6.5b stays the candidate.
Arithmetic-changing forms (TF32 on A100 within the <= 0.05 GPU band, subsampled-prefix refits via §257) are revisited when §265 or §257 reads out.

## v6.6: v6.5b plus capped causal personalisation (15 Nov, ledger §268)

`lean/src66` (copy of `lean/src65b`, plus `cpers.py`), bundle `submissions/lean_v66`, key `v6_cp_on` (0 = v6.5b path). Form, predictions
and refutation rule are in §268 (pre-registered before any check). v6.5b stays the candidate until every check below passes.

| Check (v6.6, bundle `37c666a8…fb69f`, key v6_cp_on 1; dirs `cand_v66_cp0` / `cand_v66_cp1`) | Where | Result |
|---|---|---|
| Package form against the study form: `lean/src66/cpers.py` on the study's inputs (`studies/14nov/cp_replay.py`), draw d1 full0 base 0; d2 within3 base 2; d2 runs1 base 4; 14 sessions x 2655 windows each, 1395 fused | Mac CPU | **pass**: max difference of the personal posterior 2.1e-11 / 9.6e-12 / 8.1e-12, **0 label differences** |
| Predict-level identity (`cp_identity.py`) v6.5b vs v6.6 cp off, P01 and P10\\|1 unrecognised | Mac CPU | **pass**: 0 labels differ (err 0) |
| Same, cp on, recognised sessions (P01, P10\\|1, P04\\|2) | Mac CPU | **pass**: 0 differ; stage stopped at window 10 (first routed), 0 refits, 0 fused |
| Same, cp on, unrecognised P01 and P10\\|1 | Mac CPU | pass: 9 refits each, 100 / 102 windows fused, cp err 0, 0 labels differ (cap 0.10 on the log weight moves few labels; dev +0.22) |
| Official local replay of `v66_cp1.zip` (66,788,283 B, `056fe4a1279b2152573f693ca1ab74d18bd3ab8cf9132edc8d6618687fd78877`) | Mac CPU | **pass**: 0.9945788530 (= v6.5b), files unchanged, 117.8 s, 4 `[cpers]` lines (stopped at 10) |
| CPU identity / GPU function / A100 and worker-CPU timing | Galvani 3105109-3105112 | **pending** (queued; Galvani unreachable from the Mac at 09:10 CET, eduVPN off). The §268 cap rule is applied when the timing is read |

| Galvani results of v6.6 cap 400 (3105109-12; logs `evidence/v66_15nov/`; every job printed the bundle sha `37c666a8…fb69f`) | Where | Result |
|---|---|---|
| Step-level bitwise identity cp0 vs v6.5b (P01); predict-level cp0 unrecognised equal; cp1 recognised equal (P01, P10\\|1, P04\\|2; stage stopped at window 10); cp1 unrecognised late-only | Galvani CPU 3105109 | **pass** (0 labels differ everywhere; cp1 unrecognised: 9 refits, 100 / 102 fused, cp err 0) |
| GPU function check: r15_reference cp1 (P01, P10\\|1) and the predict path on cuda | Galvani 3105110 (2080 Ti) | **pass**: err 0, relabel diff 0.0, labels 100 %, output diff 0.039 / 0.044; predict path err 0, cp err 0; 378 windows: 1 late label differs (window 219 >= 89) |
| A100 predict timing v6.6 / v6.5b (190 / 378 / 800 windows) | Galvani 3105111 (A100-PCIE-40GB) | ratio **1.036 / 1.042 / 1.082** (+6.8 / 8.8 / 14.4 ms per window, growing with the prefix: the personalisation refit is uncapped); err 0; labels differ only from window 203 (2 / 6 windows) |
| Worker-like CPU timing | Galvani 3105112 (loaded cpu node, 3.7-7.7 s per window) | function pass (err 0); not a usable cost estimate |
| §268 cap rule | | worst case 43.8 / **51.1** / 42.6 min (v6.5b's 42.3 / 49.0 / 39.4 x ratio) = slack 1.37 / **1.18** / 1.41: below 1.3 on the 378 basis -> key-only `--cap 250` build (`cand_v66_cp1_cap250`, same bundle `37c666a8…`; identity pairing `cand_v66_cp0_cap250` against `cand_v65b_cap250`) queued as Galvani 3105586 and a Mac replay; verdict after that |

### v6.6 cap 250: verdict (15 Nov, ledger §268)
`v66_cp1_cap250.zip` (66,788,282 B, sha256 `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed`; bundle `37c666a8…fb69f`, keys: v6.5b's plus `v6_cp_*`, `v6_ft_cap` 250). Mac replay 0.9945788530, files unchanged, 116.6 s.
A100 (Galvani 3105586, sha checked in the job): cp off cap 250 vs v6.5b cap 250, 378 windows: 0 labels differ, err 0. cp on cap 250 vs v6.5b cap 400: ratio 1.018 / 0.888 / 0.983 (190 / 378 / 800; per-pair noise about ±5 %), err 0, labels differ only from window 203. Worst case 43.1 / 43.5 / 38.7 min, slack **1.39 / 1.38 / 1.55** (1.24 on the 378 basis if the 800 ratio is applied there).
**Verdict: v6.6 cap 250 passes every pre-registered check** (identity off, recognised sessions, study match, replay, GPU function, timing rule). Dev value +0.22 [+0.07, +0.38] (§260, new people), cap 250 free (§257). Not validated: the real worker, sealed session lengths, the personalisation on returning people (stopped at recognition). Upload only on the user's go-ahead.
