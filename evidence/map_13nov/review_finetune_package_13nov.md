Read-only review finished. Nothing was written to the repository; my check scripts are in the scratchpad (paths at the end). The repository changed under me while I worked (HEAD moved from ab3e6b4 to e813090; another session edited `studies/12nov/ttt5.py`, `studies/13nov/ttt6.py`, `ttt_reduce_score.py` and added `ttt8.py`, `ttt9.py`, `make_labels.py`). Line numbers refer to the files as I read them; the findings still hold on the current versions.

Summary: I found no leak and no inactive arm in the fine-tuning studies. The main problem on side (i) is that the "six bases" are not six replications in the TTT scorers, so the gain is less certain than the per-base counts suggest. On side (ii) the package code is sound; the weak points are the dev evidence for v5.2 (not the package form, and inside noise) and 12 of 30 personal traces built on retrained twins.

## (i) Correctness of the reported fine-tuning gains

**1. The six bases are almost one system in the TTT scorers. CONFIRMED.**
- Where: `studies/13nov/ttt_reduce_score.py` 16, 35–46, 71–72; `studies/12nov/ttt3_score.py` 9, 22–36, 52–55; `replay/alib.py` 16–17, 26–30.
- What it does: every base gets slots 5, 6, 7, 8 replaced by the TTT run's own six-seed nets, in both the reference and the arm. The bases differ only in slots 5–9 (I checked all eleven member slots of `Ps`). After the replacement only slot 9 (weight 1/15) still differs.
- What it was meant to do: show the gain is robust across six bases (the §108 rule "ahead on ≥ 5 of 6 bases").
- Check I ran (`chk6.py`, `runs/ttt/probs6`): the TTT reference chains differ from B2_pool's in the argmax on 0.2–1.0 % of new-person windows (packaged chain: 1.0–2.8 %). Per-person deltas correlate 0.83–0.92 between bases for fixA and cum10.
- Effect: per-base counts and base ranges carry almost no information. The real uncertainty is across people:

| Arm | Six-base mean, new | Per base | 80 % person bootstrap | 95 % | People ≥ 0 | Without best person |
|---|---|---|---|---|---|---|
| cum10 | +0.46 | 0.57, 0.51, 0.72, 0.16, 0.45, 0.32 | −0.23 to +1.16 | −0.56 to +1.52 | 6/10 | +0.12 (best is P10, +3.5) |
| fixA | +0.37 | 0.44, 0.42, 0.42, 0.15, 0.49, 0.28 | −0.20 to +0.92 | −0.49 to +1.21 | 6/10 | +0.17 |

- An independent replication needs fresh fine-tuned seeds (the pre-registered TTT8F does this) and, ideally, other people.

**2. The pseudo-labels are the frozen v3 chain on B2_pool, and scoring is in the v3 chain. CONFIRMED.**
- Where: `studies/12nov/ttt5.py` 34, 58–61 (same pattern in `ttt.py` 22, 34–37 and ttt2–ttt4); scorers call `A.system`.
- Check: `studies/12nov/chain_final.npz` equals `A.system("B2_pool", A.FAMS)["out"]` exactly (max difference 0). Label accuracy on new people is 76.6 %. `idx_seen` matches the replay rows.
- This is not a leak: the labels are the chain's causal outputs, windows before t only, from nets trained without the test person.
- Mismatch with a package: the package would use its own running outputs (v4.1 gate, v5 stage, with the fine-tuned nets in the mix). The only evidence on that feedback is the 11 Nov round 2 (two families, three seeds: +0.39, then +0.51 with fed-back labels). Likely a small effect.

**3. The "head" arm does not train the output bias. CONFIRMED.**
- Where: `studies/13nov/ttt6.py` 32–37 (same heuristic in `ttt7.py` 42–47 and the `scope == "head"` branches of ttt2–ttt5).
- What it does: `[p for p in net.parameters() if p.dim() > 1 and p.shape[0] >= 3][-1]` picks only the final weight matrix. I built the four nets and listed parameters: `final_layer.bias` (or `conv_classifier.bias`) is excluded in all four families.
- It also runs at lr 5e-4, while the personal head recipe that worked uses the last two parameters at 2e-3 (`studies/11nov/ttt/galv/pft.py` 38–41).
- Effect: §108 "head refuted" (my rerun: −0.16 six-base mean) closes only "SessionBN affine + final weight, lr 5e-4". To confirm, rerun head with `list(net.parameters())[-2:]` and lr 2e-3. This matters because head-only is the cheap form.

**4. Returning sessions lose with fine-tuning unless it is gated off, and the gated form was never measured. CONFIRMED (numbers), reading (gap).**

| Arm | Seen, routed v3 chain | P04 | P10 | Seen, routing off |
|---|---|---|---|---|
| cum10 | −0.70 | −1.53 | +0.13 | about 0 |
| fixA | −0.68 | −1.06 | −0.30 | about 0 |

- No dev arm covers a session recognised late. P04 session 2 is first routed at window 63–64 in the v4.1 replay, so fine-tuning would run for windows 30–62 and then stop.
- In the dev scoring the fine-tuned nets also feed the shadow tail (gate input). v5.2 keeps personal nets out of the shadow. A v5.3 has to state which it does.

**5. Slot 9 is left frozen in dev although it is the same nets as slot 6. LIKELY.**
- Dev fine-tunes `fb` at k = 2 only (`SLOT["fb"] = 6`). In the package `sbn_k2_s*` and `sbn_k5_s*` are the same weights at two k values (their differences from the personal traces are identical tensor by tensor).
- A package must either reuse the fine-tuned weights for k5 (unmeasured) or keep k5 frozen (the dev form). Pre-register the choice.

**6. Things the dev study does that the package must reproduce or will do differently. Reading.**
- After each refit dev recomputes the SessionBN state by running the whole prefix through the new weights (`ttt6.py` 62–64, `net2(X[:hi])`). A package that keeps the old StreamBN sums after changing weights computes something else.
- Full batch over all earlier windows; refit from the frozen net each time (fixA) or continued (cum10); "all6" means 24 nets fine-tuned.
- Dev computes the clean forward with gradients and then detaches it (`ttt6.py` 55). This is wasted work only; `ft_bench2.py` already uses `no_grad`.
- The package's `mixed_proba`, `Jit.proba` and `proba_from_bank` are under `@torch.no_grad()` (v5.2b 1108, 265, 326). Fine-tuning code called from there needs `enable_grad`. Combined with a v5.2-style `except Exception` fallback, a broken fine-tune would become a silent no-op visible only on stderr. The worker-image replay should assert that parameters changed and that the refit count is above zero.

**7. The label source is not part of the output name. POSSIBLE.**
- `run()` skips when the output files exist (`ttt6.py` 70). The label file comes from the env variable `CHAIN_NPZ` (`ttt5.py` 34; `TTT_LABELS` in `ttt.py` 22).
- `ttt9.py` is safe: it uses its own directory and asserts the labels loaded.
- A ttt6 or ttt8 job started with a leftover `CHAIN_NPZ` (for example `chain_true.npz`, the oracle) would write oracle results under packageable names, or silently reuse old files.
- To confirm: grep `scripts/cloud/night13*.sh` for an exported `CHAIN_NPZ`. Fix: store the label file hash in the npz and assert it in the scorer.

**8. Minor.**
- "base-vs-harness max diff" is printed, never asserted (`ttt6.py` 86).
- The older `ttt3_score.py` `load` asserts only that some seed exists. `probs3b` holds 7 or 8 seeds per family (fbraw has 8 for new, 7 for seen), and `probs4` has 3 seeds for sinc against 6 for the others. The §60/§81/§91 numbers mix seed counts; the newer scorer names its seeds.

**Checked and fine.**
- No inactive or duplicate arm. Every variant in ttt3 to ttt7 differs from the frozen net and from fixA (mean |Δp| ≥ 0.007). The first changed window is exactly 30 (20 for gC10T20).
- fix20 = ttt5 fixA = ttt6 fixA = ttt7 fixA bit for bit on fb and fbraw, within 5e-5 on fbl, and within 0.02 on sinc (GPU nondeterminism). The "in-run reference" is real.
- Causality by reading: refit at t uses `X[:t]` and labels `[:t]`; prediction runs in eval mode with causal SessionBN over the prefix; alignment is causal; no augmentation at evaluation; running statistics are not updated.
- Seeds reach the nets (s0 against s1 differ by 0.3–0.8). Folds exclude the test person. The sinc trunk is stated as external corpora only; I did not verify that beyond the docstring.
- `SessionNet._train` asserts on the options it ignores (`experiments/session_bn.py` 146–150).

## (ii) Safety of uploading a v5.2-type package

**1. The dev evidence for v5.2 (§78, +0.17) is not the package form and is inside noise. CONFIRMED by replay.**
- Where: `studies/12nov/pnet.py` 9–14, 17–35.
- How dev differs from `submission.py`:
  - Personal nets replace the slots on every window of the seen sessions from window 0, in the shadow (gate input) as well as the routed tail. The package swaps only on routed windows of the routed tail.
  - Dev adds a SessionBN shallow member "sh" at weight 1 in both arms. The package has no such member.
  - Slot 9 is not swapped in dev; the package swaps it.
  - `all_loso` gets frozen twins, so "new +0.00" is by construction.
- Check I ran (`chk4.py`, same stored outputs, six bases). It reproduces +0.17 exactly.

| Form | Seen | P04 | P10 | Per base |
|---|---|---|---|---|
| Dev form, five families (as §78) | +0.17 | +0.53 | −0.18 | −0.11 to +0.41 |
| Package form, five families | +0.02 | +0.27 | −0.23 | −0.51 to +0.55 |
| Dev form, without "sh" | +0.25 | +0.04 | +0.46 | −0.10 to +0.42 |
| Package form, without "sh" | +0.39 | +0.33 | +0.45 | +0.28 to +0.55 |

- Each person's delta moves in steps of about 0.27, so these are one or two windows. "All of it P04's, costs P10" flips when the shallow member the package lacks is removed.
- The dev says "about 0 to +0.4, unresolved". It is neither support for nor evidence against the §105 expectation of +0.20.
- Checked and fine: §96's proxy (+2.4) does not depend on the personal shallow nets it puts into slots 3 and 4. Without them it is +2.25 to +2.42 (`chk7.py`).

**2. 12 of 30 personal traces are fine-tuned from retrained twins, and the guard cannot see it. CONFIRMED.**
- Where: `scripts/export_sbn.py` 51–72; used by `scripts/build_pnets.py` 21 and `scripts/export_pnets.py` 25–28.
- What it does: `trained()` silently trains and saves a new net when `runs/sbn/<cfg>-s<seed>.pt` is missing. The assertion at `export_pnets.py` 28 compares the personal net's base hash with the local checkpoint, which is the retrained one, so it always passes.
- Check I ran (`chk5.py`, `runs/sealed/cand_v52b_pnets`):
  - Raw FBCNet: spatial weights differ from the shipped trace by up to 0.146.
  - SincShallow: sinc parameters and depthwise weights differ by 0.002, BN variance by 0.025, BN mean by 0.138.
  - FBCNet k2/k5 and FBLight: weights within 2.6e-6; conv bias and running mean are shifted by the same amount (0.0698 / 0.0398), which cancels in StreamBN. Functionally the same nets.
- This is deviation 1 in `docs/V52_BUILD.md`, still open. On routed windows 12 members (0.133 of the mix) change base as well as head, a combination dev never scored. V52_BUILD puts the effect at about 0.01 in the mix at most.
- Fix: make `trained()` fail unless told to train; have `build_v52.py` compare each personal trace's frozen tensors with the shipped base trace; rebuild the 12 from the original checkpoints.
- Related, POSSIBLE: `build_pnets.py` 14 imports `personalise` from `runs/diag/ttt/galv/pft.py`, an untracked cluster copy that is not on this machine. The code that built the shipped personal nets cannot be verified from the repository.
- Checked and fine: the head-only scope held. Between people only the SessionBN affine and the final layer differ; conv weights and running statistics are identical across P01, P04, P10.

**3. A false accept swaps one third of the mix at full strength, and this was never measured. Reading.**
- Where: `submissions/ensemble_v52b/submission.py` 869–879.
- What it does: all 30 SessionBN members (0.3333 of the mix weight, read from `weights.npz`) are replaced for any r > 0. The head and the own-space stage scale with r.
- §65 measured the cost of a false accept for v4.1 and the stage only (1.6 and 2.2 points per session). In the B2_pool LOSO replay no new-person session was routed.
- At a 1.9 % false-accept rate the expected sealed cost stays small unless the swap costs far more than 5 points a session.
- Confirm cheaply: report accuracy in `check_v52.py`'s switch pass (it already forces wrong people's traces), or run `studies/12nov/impostor.py` with another person's pfh60 outputs.

**4. Package code: no defect found. Reading plus small checks.**
- Session isolation: `PersonalNets.__deepcopy__` (939–944) shares only the pristine traces. They are never run: `_compute` (976–986) deep-copies and resets them. `reset_state` rebinds new containers. The `route.pmix` closure is cleared at the end of `predict` (1161). The only module global is the worker thread pool, which holds no data.
- No per-session recompilation: on CPU a fresh deep copy of a trace runs its first call in 2.5 ms, against 32 ms for the first call ever; the deep copy itself takes 1 ms (`chk8.py`).
- Memory: two float32 inputs of 41 × 480 per window are buffered (157 KB per window, 30 MB for 190 windows) and freed with the session copy.
- Exceptions: every personal path falls back to the v5.1b mix and is counted in the `[pnets]` stderr line. I found no path to a silently wrong output. A total failure on the worker would look exactly like v5.1b (score about 0.8198), so the stderr lines "swapped = routed, errors 0" must be read after any run.
- No new assumption about window order. Labels are an argmax over three classes.
- v5.2b prefetch (956–964): after a prefetch, if the window is not routed or the person changes, those traces consume windows one at a time instead of in one catch-up batch. That is a float32-level difference from v5.2, not a logic difference. Prefetch also runs for the last routed person on every later window, routed or not.

**5. The checkers are not vacuous, with these limits.**
- `scripts/check_v52.py`:
  - The reference loads the files named by the package's own `pnet_files` and uses the package's own `pnets.idx`, `pnets.w` and `members` (55, 84, 104, 116). It cannot detect a wrong member-to-file mapping or weight. I verified the mapping by reading `build_v52.py` 30–34 and the constructor: it is right.
  - It streams with `reset_state()` on one object. The deep-copy path is covered by `check_v52b.py` and the official replay.
  - Every public session is in-sample and recognised from window 10. Late recognition, gaps and unrecognised sessions are exercised only by the forced passes.
  - On GPU it passes only with `--ref-schedule`.
- `scripts/check_own_stage.py`: sound (members against the study class to 1e-8; asserts the stage ran on more than half the windows). Routed windows where the stage failed count as "unstaged", so a partial failure would pass. The logs show degenerate 0, errors 0.
- `scripts/check_perm_stage.py`: tests only an exact permutation with all names present.
- `scripts/check_v52b.py`: sound (official streaming, tolerance 0, hit-rate assertion).
- `scripts/export_pnets.py` 28: vacuous with respect to the shipped base (finding 2).

**6. Runtime: what to measure.**
- `predict` time against window index. `SessionEM._adapt` (538), `TSEM._adapt` (594) and `Route3.scores` (830) rebuild arrays over all earlier windows on every call, in both tails. That is work growing with the square of session length. Measure on a 190-window and a synthetic 600-window session.
- First-routed-window latency: 30 trace copies plus catch-up chunks of 32.
- Prefetch hit rate on sessions with intermittent routing. Public data gives 99 % only because 95 % of windows are routed.
- First-ever call of each of the 300 traces on the A100.
- Trace loading in the worker image. §112 says the worker-image replay was not done.
- For fine-tuning: wall time of the chosen schedule inside `predict`, including the prefix replay after each refit, with the tangent thread running.

## Scratch scripts
All read-only, in `/private/tmp/claude-501/-Users-sebastianboehler-Documents-GitHub-neural-interfaces-2026/ca8951cd-9c0d-4449-8fa2-f0de43144db9/scratchpad/`:
- `/private/tmp/claude-501/-Users-sebastianboehler-Documents-GitHub-neural-interfaces-2026/ca8951cd-9c0d-4449-8fa2-f0de43144db9/scratchpad/chk1.py` (slots that differ between bases; origin of the labels)
- `/private/tmp/claude-501/-Users-sebastianboehler-Documents-GitHub-neural-interfaces-2026/ca8951cd-9c0d-4449-8fa2-f0de43144db9/scratchpad/chk2.py` (identical or inactive arms)
- `/private/tmp/claude-501/-Users-sebastianboehler-Documents-GitHub-neural-interfaces-2026/ca8951cd-9c0d-4449-8fa2-f0de43144db9/scratchpad/chk3.py` (head parameter selection)
- `/private/tmp/claude-501/-Users-sebastianboehler-Documents-GitHub-neural-interfaces-2026/ca8951cd-9c0d-4449-8fa2-f0de43144db9/scratchpad/chk4.py` (§78 in dev form and package form; output in `chk4.out` beside it)
- `/private/tmp/claude-501/-Users-sebastianboehler-Documents-GitHub-neural-interfaces-2026/ca8951cd-9c0d-4449-8fa2-f0de43144db9/scratchpad/chk5.py` (shipped against personal traces, tensor by tensor)
- `/private/tmp/claude-501/-Users-sebastianboehler-Documents-GitHub-neural-interfaces-2026/ca8951cd-9c0d-4449-8fa2-f0de43144db9/scratchpad/chk6.py` (base independence, person bootstrap, seen effect)
- `/private/tmp/claude-501/-Users-sebastianboehler-Documents-GitHub-neural-interfaces-2026/ca8951cd-9c0d-4449-8fa2-f0de43144db9/scratchpad/chk7.py` (§96 without personal shallow nets)
- `/private/tmp/claude-501/-Users-sebastianboehler-Documents-GitHub-neural-interfaces-2026/ca8951cd-9c0d-4449-8fa2-f0de43144db9/scratchpad/chk8.py` (first-call timing of deep-copied traces)
