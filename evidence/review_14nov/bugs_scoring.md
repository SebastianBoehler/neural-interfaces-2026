# Scoring-code bug hunt, 14 Nov (independent review)

Scope: `studies/13nov/`, `studies/14nov/`, `replay/`, `scripts/cloud/leo_collect.sh`, `scripts/cloud/galv_*.sh`,
`scripts/check_v6.py`, `scripts/check_v62.py`. Read-only on tracked code. Check scripts: `evidence/review_14nov/checks/`.
Status: IN PROGRESS (confirmed findings first; this file is rewritten as the review goes on).

## Confirmed

### 1. The eye net for returning people is standardised in another delivery order than the stages use
- **Where:** `studies/13nov/ocushuf.py:10` (`ORDER = ...[f"{SCHEME}|{PSEED}|all_loso"]`), used for both protocols in `main()`
  (lines 33–34). Read by `retv6.py:55` and by `shufcombo.py` with `PROTO=seen_session`.
- **Does:** the eye net's causal trace standardisation follows the all_loso permutation for every session, also when the
  outputs are written for seen_session. `shuffle.perm_for` seeds per protocol (`PR.index(p)`), so P04/P10 sessions 2–3 have
  a different permutation under seen_session. Every other re-streamed input (ttt17/18, restshuf, coremember, coreshuf,
  teachshuf, geshuf) uses the protocol's own order.
- **Should:** use `PERMS[f"{SCHEME}|{PSEED}|{proto}"]` per protocol (traces rebuilt per protocol), and the scorers should
  assert it (they only check `index`).
- **Reproduced** (`checks/c02_ocushuf_seen_order.py`): for all four returning sessions the two orders differ. Share of
  windows whose eye-net prefix equals the delivered prefix: 0.00–0.01 (0.18 on one runs session). Rank correlation of the
  two orders: within +0.97, full −0.11 to +0.02, runs −0.55 to +0.43.
- **Affects:** every returning-people number with the eye view under shuffle: retv6 arms E, ER, ERF (§190, §198, §199, §201,
  the "returning" rows of the morning table, v6.2 and v6.2-safe) and §183. Under within the two orders are close, so the
  effect should be small. Under full and runs the eye view at delivered window t was standardised on windows that had not
  been delivered (the same kind of mismatch as §180). Direction not measured; under runs it is likely optimistic early in
  a session (statistics from several runs where the stream has seen one).

### 2. `leo_collect.sh` treats a failed squeue as "all jobs finished", and swallows rsync and scorer failures
- **Where:** `scripts/cloud/leo_collect.sh:10–11, 16–17`.
- **Does:** (a) for job ids the query is `squeue ... | grep ... ; true`, followed by `&& echo __OK`. The `; true` hides
  squeue's own exit code, so a Slurm error on the cluster (controller timeout) gives an empty answer plus `__OK`: the
  loop ends as if the jobs had left the queue. Only an ssh failure is handled. The `name:` form is correct.
  (b) rsync's exit code is lost in `| grep -v`; a broken copy goes on to scoring. There is no `--delete`, so stale files
  stay. (c) the scorer's exit code is ignored (`> /dev/null 2>&1; echo "scored -> $LOG"`), and the script exits 0 unless
  sacct showed a failed job. A scorer that died on an assert looks like a scored run.
- **Should:** keep squeue's status (`set -o pipefail` or test it separately), fail on rsync errors, propagate the scorer's
  exit code.
- **Affects:** any wave collected through it. Together with finding 4 (seed files found by glob), a partial copy is scored
  with fewer seeds and no error. **Read only**; I did not find a concrete past run where it happened.

### 3. Output names still omit settings that change the result (beyond §208's OCU_DIR)
- **Where:** `studies/13nov/shufcombo.py:269`.
- **Still missing after the §208 fix:** `OCU_ORACLE`, `OCU_SYN` (both change every V6 arm), `BASES` (a one-base smoke
  overwrites the six-base file; morning_table and wsp_leverage assert six bases, stackcmp, sweepsel and wsp_stage2 do not),
  `T6_DIR`/`REC_PFX` (the recording-order twin behind P0r), `T17_DIR`, `CS_NPZ`, the number of eye and rest seed files
  found by glob.
- **Reproduced for OCU_ORACLE / OCU_SYN:** `runs/formA_within{0,1}_oracle0.log` and `runs/syn_within{0,1}_k0_d*.log`
  (13 Nov 16:43–18:01 UTC) ran with the regular settings otherwise (same header, `OCU_TS=0`), so each wrote
  `shufcombo_within{0,1}_ocu_gC5_all_ge_hb_ts_gp0_ttt18within{0,1}L.npz`, the regular v6f file. Those files were
  rewritten later (form B, then §208), so the present files are clean (`checks/c07_local_vs_leo.py`: all local files
  equal the Leonardo copies). Anything read from the local within 0/1 files between 16:43 and the next regular rewrite
  was an oracle or synthetic result.
- **Other writers with the same problem:**
  - `geshuf.py:35` always writes `runs/geshuf.npz`: a run with another `ORDERS` replaces the file and drops the earlier
    orders (loud only under STRICT).
  - `shufpkg.py:63` (`shufpkg.npz` unless `SHUFPKG_OUT`), `toracle.py:46`, `emceil.py:71`, `shuffle.py:97`: fixed names,
    whatever SCHEMES / PERMS / BASES / HS were.
  - `wsp_final.py:84`: `REG` is not in the name (`final_<comp>_<fam><LAM>.npz`); for `acc` the name carries a LAM that
    is not used.
  - `retv6.py:151`: every base subset other than six is called `smoke`; `RT_PROTO`, gate, form, suffix and orders are in.
  - `ttt7.run` / `ttt6.run` skip when the output exists (`exists`), keyed by tag only: the label file's content is not
    part of the key (ttt6 stores `labels=LABEL_TAG` in the file, ttt7/ttt18 do not). Rebuilt labels under the same file
    name are silently not used unless the old outputs are deleted (as §180 did by hand).

### 4. Seed sets found by glob differ between orders and arms; STRICT does not check them
- **Where:** `shufcombo.py:46, 89` (eye and rest seeds by glob), `retv6.py:55–56`.
- **Does:** takes the mean over whatever seed files exist. Confirmed from `runs/retv6*.log`, `runs/s199_*.log` and the
  folders: rest net 9 seeds for within 0/1, 3 seeds for every other order; eye net 9 seeds for shuffled orders, 3 seeds
  for `rec`. STRICT only requires that at least one file exists.
- **Should:** name the seeds explicitly (as `pkg.py` does with `OCU_SEEDS` / `REST_SEEDS`) and match the package's count.
- **Affects:** locally scored within 0/1 against other within orders (retv6 returning rows; local shufcombo runs): a
  9-seed rest member is a stronger member than the 3-seed one. "rec against shuffled" comparisons of any eye arm mix
  3 and 9 eye seeds. Direction: favours within 0/1 and the shuffled eye arms; size not measured.

### 5. The "recording order" combination line is not in recording order
- **Where:** `shufcombo.py:211–214` with `OCU_DIR` set: arm `V6Frec` takes `REST` (recording) but the eye view from
  `OCU`, which is the delivery-order re-stream.
- **Reproduced:** "combination over v5.1b, recording order" in `runs/allorder_*_gp0_L_ge_ts_hb.log` runs from +0.66 to
  +0.98 over the 20 orders; a recording-order number cannot depend on the delivery order.
- **Affects:** wherever that line was quoted as the recording-order gain of the combination next to the shuffled one
  (secondary column since §164); ±0.15.

### 6. §207 leverage verdicts after §208: two more components meet the stage-2 rule
- **Where:** ledger §208 says "the rule verdicts are unchanged except s6 (now undecided)".
- **Reproduced** (`checks/c08_s207_rebase.py`, clean files, rule as pre-registered: ≥ 0.05 on at least two of the three
  orders → stage 2; below 0.03 everywhere → no lever): s3 (0.057 / 0.036 / 0.065) and s4 (0.064 / 0.061 / 0.028) are
  now "stage 2"; s6 and s8 (0.034 on full 0) are "undecided". The other corrected values equal §208's.

### 7. morning_table.py: three weaknesses in how files and intervals are chosen
- **Where:** `studies/13nov/morning_table.py:35–38, 28–29, 66, 71–78`.
- (a) The v6f glob `...gp0_ttt18*L.npz` also matches the draw files `ttt18<o>d1L`, `d2L` (they exist locally for within
  0–7), and the last match wins. **Latent today** (`checks/c01_morning_pairs.py`): no local QF file exists for a within
  order, so the pair is never formed; the day one exists, v6f for that order silently becomes draw 2.
- (b) The "v6.1" column mixes three label forms and two machines: KF (within, full 0/1, runs 0/1), RF (full 4–7), QF
  (full 2/3, runs 2/3, local). Stated in the docstring, not in the printed table.
- (c) The interval is a bootstrap over orders of the ten-person mean. The orders share the same ten people and the same
  nets, so it measures order spread only; the person interval that docs/STATE.md §4 asks for is wider (numbers to follow).
  Scheme-averaged J combines new people over 12 / 8 / 4 orders with returning people over 8 / 4 / 4 orders.

## Suspected, not confirmed

- **`shufcombo.py` FT_TAG is not checked against SCHEME/PSEED.** `ld()` checks only `index`; with `FT_TAG=ttt18within0L`
  and `PSEED=1` the FT streams of another order would be scored. The tag is in the output name, so a mismatch is visible
  there; no such file name exists in `runs/` today.
- **`chain_blend_<order>.npz` is not the mean of the stored `chain_v51b` and `chain_v6r` files** (`checks/c05_labels.py`
  and inline check: max difference 0.05–0.10, argmax agreement 97.4–98.6 %, differing from window 0 of a session, so the
  member mix differed, not the stages). The blend run rebuilt both chains itself; some input differed from the runs that
  wrote the two label files (FT tag of the base re-stream, seed sets, or the gate setting). §200 was closed as
  "undecided, not added", so nothing shipped depends on it.
- **core1 as member in delivery order is "stored + (delivery − own rebuild)", unclipped** (`shufcombo.py:181`). The
  rebuild differs from the stored member by up to 0.34 (mean 0.014), so 14–29 % of rows have a negative entry (down to
  −0.20) and values up to 1.20 (`checks/c04_member_deltas.py`). It enters an arithmetic mix at 1/15, so the effect is
  small; using the delivery stream itself would be cleaner. Hilbert and teacher rebuilds are exact (no issue).
- **`check_v6.py` / `check_v62.py`: `worst = max(worst, float(x))` drops NaN** (`max(0.0, nan)` is 0.0), so a NaN in a
  compared quantity passes the tolerance. Inputs go through `nan_to_num`, so it needs a NaN produced inside the package.
- **Transductive constants** (pre-existing, unlabelled): the teacher's EA prior `G` is the mean covariance over all
  public windows, including the held-out person (`teachshuf.py:24`, reproducing the stored slot 10); `geshuf.py:14`
  `global_mad` likewise. No labels involved.

## Checked and found correct

- **Permutation bookkeeping** (`shuffle.apply`, `shufcombo.chain`): the three stores (inputs, both copies of bases) are
  permuted together with `A.C1`; outputs and labels stay aligned (`SC.person` reads the permuted `y`/`cell`); deltas are
  applied as `[P]` on recording-layout arrays once; `np.argsort(P)` undoes it for saved outputs. `bases_fresh*` are not
  permuted but are not read after import.
- **Run blocks** (`checks/c09_run_blocks.py`): `task|run` is a recording block (Acquisition/Online × interface), not the
  class; 6 blocks per session, all contiguous; delivery orders carry no label streaks (0.29–0.33 same-label neighbours).
- **Order file:** pinned by SHA-256 in shufcombo and retv6; stage order asserted equal to the dumped order.
- **pc128 rebuild:** equals the stored view up to sign (|corr| 1.000 on all 128 columns, both protocols), so the
  sign-aligned delta is sound (`checks/c03_pc128_alignment.py`).
- **Label files** (`checks/c05_labels.py`): no true labels (accuracy 0.72–0.83; only `chain_true`/`chain_truelow` are
  oracle files and ttt18 refuses them by name); within 0/1 and runs 0 are the rebuilt files of §180 (0.7552 / 0.7589 /
  0.7533); `rts` = v6r (new) + rt (returning), `rt` new = v51b, as the ledger says.
- **Fine-tuning loop** (`ttt7.stream`): trains on delivered windows 0..t−1 with their pseudo-labels, predicts t..t+9.
- **Mix weights:** replaced slots keep weight 1 (15 → 16 with the rest member; asserted in retv6).
- **Nested selection** (`sweepsel.py`, `select_retune.py`, `selector.py`): the scored half never enters the choice.
  Notes: "126 splits" are 252 ordered halves; the bracket is the 10–90 % spread over overlapping splits, not a
  confidence interval; selector's threshold grid uses all people's label-free criterion values.
- **Per-held-out-person edits** (`wsp_final.py`, `coregain.py`): fitted on the other nine people's rows only.
- **Oracle hooks** (`OCU_ORACLE`, `OCU_SYN`, `SYN_T`, `ocuorc`, `chain_true`): off unless the env var or arm is set;
  each prints a line. (Their output names are finding 3.)
- **Local result files today:** every local v6f / v6.1 file that has a Leonardo copy equals it (`checks/c07`).
