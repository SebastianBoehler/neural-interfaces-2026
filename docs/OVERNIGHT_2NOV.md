# Overnight campaign, 1–2 Nov (Galvani)

**Aim.** Search widely across method classes before narrowing (Learn2Design lesson,
memory `headroom-estimates-mislead`). Every arm must ship after the code freeze
(1 Nov 23:59 CET) as data only: a `Jit` member (any stateless TorchScript net on the
package's aligned input), plain member weights, or `member_weights`.

**Adoption rule, fixed before any result.**
- Measure: sel = mean(ds, dla) of R19e with the arm, against the same pipeline without it.
- Bar: at least +0.5 pooled over six seeds, with both seed halves (0–2, 3–5) positive.
- Teacher variants must also beat the packaged teacher (ref, w 2) by +0.5.
- Report dla on its own: a ds-only gain is P10 only.
- Anything picked from many variants needs a replication on unseen seeds (6–11) first.
- Never tune on official, al, ss, P04/P09 or the warm-up board.

## Arms

| Track | What | Why (mechanism) | Jobs |
|---|---|---|---|
| T1 replication | ref and m7r4 teachers on pretraining seeds 6–11 | Is the teacher gain real or seed-selected? | 3034986–9, screen 3035013 |
| T2 embedding probe | per-net standardised LR on frozen trunk embeddings (C 0.1, 0.01) | Heads compress 19 dims; the trunk may carry more transferable structure | 3035012 |
| T3 architecture pool | 18 more braindecode architectures pretrained on m7r (seeds 0–5); each alone, all together, each + ref | Architecture diversity helped (m7r4 > ref); scale the pool | 3035099–116, screen 3035319 |
| T4 pretraining recipe | 60 epochs (m7r60), classic augmentation (m7ra) for the four base families | Better pretrained heads at no new data | 3035117–22, 3035243–4, screen 3035320 |
| T5 Riemannian teacher | tangent-space LR heads per external corpus (8–13, 13–30, 4–40 Hz), alone and with ref / m7r4 | A non-CNN view of the same external knowledge | 3035263–4 |
| T6 map regularisation | teacher LR C in {0.01, 0.03, 0.3, 1} | Cheap sensitivity check of the packaged map | 3035321–4 |
| T7 co-training | AlignedNet `cotrain` (experiments/cotrain.py): every PROTEUS step adds a balanced batch of the 7 corpora on their own heads (λ 1, 0.3), for sinc (pretrained), fbcnet (± pretrained), fblight, shallow | Fine-tuning can overwrite external knowledge; multi-task keeps it in the trunk | 3035271–5, 3035301 |
| T8 ROCKET member | MiniRocket-style fixed random dilated kernels + PPV/max + linear head (experiments/rocket.py), EA, EA+rot, raw | A non-learned-filter inductive bias for ensemble diversity | 3035325 |
| T9 more member seeds | R15 member configs on seeds 6–11 | Variance reduction of the ensemble (data-only: more seeds in weights.npz) | 3035302–3 |

Member arms (T7–T9) are scored with `scripts/diag/eval_members.py` (add or swap a member
in R15, then rebalancing and the R19e stage). Its R15 reference gives dla 74.93 against
the cached 74.82 (ds identical); deltas use its own reference.

## Results

(filled in as jobs finish)

**T6 map C (1 Nov ~22:10).** Pooled sel Δ at w 2 / best w:

| C | ref w2 | ref best | m7r4 w2 | m7r4 best |
|---|---|---|---|---|
| 0.01 | +0.64 | +1.61 (w4) | +0.97 | **+2.05 (w4; dla 77.09, ds 88.79)** |
| 0.03 | +0.64 | +1.06 (w3) | +1.14 | +1.53 (w4) |
| 0.1 (packaged) | +0.84 | +0.84 (w2) | +1.11 | +1.58 (w4) |
| 0.3 | +0.61 | +0.61 (w2) | +1.27 | +1.56 (w4) |
| 1 | +0.77 | +0.77 (w2) | +1.14 | +1.48 (w4) |

m7r4 beats ref at w 2 for every C (by +0.3 to +0.7). A smaller C makes the teacher's
probabilities flatter, so it wants more weight. Pre-registered addition to the seed 6–11
replication (3035357; 3035356 at C 0.1): m7r4 at C 0.01, w 4.

**T5 Riemannian teacher (1 Nov ~22:30, 3035361; `riemann_heads.py`, map C 0.01).** The
heads' training accuracy is 0.68–0.93 per corpus. Pooled sel Δ at w 0.5 / 1 / 2 / 3 / 4:
riem alone +0.28 / +0.09 / +0.45 / −0.05 / −0.33. ref+riem peaks at +0.76, below ref
alone at C 0.01 (+1.61 at w 4). m7r4+riem peaks at +1.38 (w 3), below m7r4 (+2.05). No
added value over the CNN heads. The run with LR C 0.1 gave infinite log-probs (it would
need clipping) and was not redone. Closed.

**T1 seed 6–11 replication (1 Nov ~23:10, 3035356 C 0.1, 3035357 C 0.01). Passes.**
Pre-registered criterion: m7r4 at w 2 and w 4 (and C 0.01 w 4) beats the packaged ref
(C 0.1, w 2; s611 sel +0.65, dla 76.06) by ≥ +0.5 sel on unseen seeds 6–11, dla not lower.

| s611 | sel Δ | dla | ds | vs ref w2 |
|---|---|---|---|---|
| ref C 0.1 w2 (packaged) | +0.65 | 76.06 | 87.03 | — |
| m7r4 C 0.1 w2 | +1.52 | 76.74 | 88.10 | +0.87 ✓ |
| m7r4 C 0.1 w4 | +1.92 | 77.26 | 88.38 | +1.27 ✓ |
| m7r4 C 0.01 w4 | +2.30 | 77.26 | 89.12 | +1.65 ✓ |

The dla concern from seeds 0–5 (m7r4 w4 dla 76.41 vs 76.43) does not recur: on 6–11
dla gains +1.2. Pooled over twelve seeds (s011): m7r4 C 0.01 w4 +2.33 (dla 77.41,
ds 89.05) against packaged ref +0.79 (dla 76.33). Candidate arm: **m7r4, C 0.01, w 4,
12 pretraining seeds** (48 nets). w 4 is the grid edge; see W2b. sh alone (shallow only)
at C 0.01 w4: s611 +1.29, below m7r4.

**T2 embedding probe (1 Nov ~23:30, 3035012).** A standardised LR per net on the trunk
embedding (32–1152 dims), probabilities averaged. Best pooled sel Δ, seeds 0–5:
m7r4 +1.49 (C 0.1, w 4) and +1.22 (C 0.01, w 4); ref +1.39 / +1.20. The 19 head
log-probs score higher (m7r4 +1.58 / +2.05). fbcnet's 1152-dim embedding is negative
at every w ≥ 1. The pretrained heads compress better than a 10-participant probe can
learn. Closed.

Lost tasks: node cn122 fails every job with "CUDA device busy or unavailable" (45 tasks
since 18:00, 38 of them T3/T4 pretraining). All resubmitted with cn122 excluded
(3035743–6, 3035750–73); the screens' dependencies now include them.

**W2b weight grid (1 Nov ~23:40, 3035803–4).** m7r4, C 0.01, sel Δ at w 3 / 4 / 5 / 6 / 8:
s05 +1.35 / +2.05 / +2.10 / +1.75 / +1.60; s611 +1.89 / +2.27 / +2.20 / +2.27 / +1.51;
s011 +1.52 / **+2.33** / +1.92 / +1.55 / +1.30. A plateau at w 4–5 that falls off by
w 8. On s05, w 5 leads w 4 by only 0.05, and w 4 holds better on 6–11 and pooled, so
**w 4 stays**. C 0.03 is lower throughout (m7r4 s011 at most +1.98). For ref at C 0.01
the peak is w 5–6 (s011 +2.08), still below m7r4.

**T7 co-training and T8 ROCKET (1 Nov 23:46, 3035335; eval_members, R15 ref sel 80.95).**
Pooled sel Δ (halves 0–2 / 3–5):

| Arm | add | swap |
|---|---|---|
| sinc ptm7r ct1 | **+0.61 (+0.50 / +0.88), dla +0.62** | −0.23 |
| sinc ptm7r ct03 | **+0.63 (+0.55 / +0.80), dla +0.67** | −0.45 |
| fbcnet ct1 | +0.25 (+0.41 / +0.02) | +0.11 |
| fbcnet ptm7r ct1 | −0.06 | −0.02 |
| fblight ct1 | +0.21 (−0.14 / +0.24) | +0.09 |
| shallow ct1 | −0.49 | −0.42 |
| all ct1 (swap 5) | — | −0.07 |
| ROCKET ea / ea rot / raw | −0.46 / −0.49 / −0.41 (×2: −0.87 / −0.68 / −1.26) | — |

ROCKET is closed: every variant and both halves are negative (seeds 0 and 2 were lost
to cn122, but the 3–5 half alone is −0.29 to −0.83). Co-training passes the bar only
as an added sinc ptm7r member, never as a swap. Two follow-ups are needed: it was picked
from 12 arms, and adding any extra sinc member also raises that family's weight.
Pre-registered, before any result:
- replication: ct1 or ct03 trained on seeds 6–11 and added must reach ≥ +0.5 sel;
- control: it must also beat adding the plain sinc ptm7r member (seeds 6–11) by
  ≥ +0.3.

Control 3036246; replication 3036244 (training) → 3036247 (eval), with λ 3 (3036245)
on seeds 0–5 in the same eval.

**Co-training control (3036246): fails.** Adding the plain sinc ptm7r member (seeds
6–11) gives +0.65 (halves +0.64 / +0.72): the same as co-training (+0.61 / +0.63;
ct1+ct03 together +0.68). Co-training adds nothing beyond extra sinc weight. Closed.
The replication (3036247) also fails: on seeds 6–11, adding ct1 gives +0.37 (halves +0.64 / +0.37)
and ct03 +0.35, both below +0.5 and below the plain-sinc control (+0.65). λ 3 on seeds 0–5
gives +0.46.

**Sinc member weight (3036267, post hoc).** Raising sinc ptm7r from 1 to 2 gives sel
Δ +0.61 (seeds 0–5), +0.60 (6–11) and +0.65 (0–11); w 1.5 gives +0.22 and w 3 gives
+0.46. More seeds at w 1 do not help (0–11: −0.30). **With the m7r4 teacher (C 0.01;
3036269–70) it adds nothing:** teacher w 4 alone gives +2.05 / +2.30 / +2.33 (s05 /
s611 / s011); with sinc w 2 it gives +2.20 / +2.15 / +2.11. The teacher already
carries what the extra sinc weight supplies. Closed: member weights stay as packaged.

**T9 member seeds 6–11 (3036217, complete; 3035336 lacked seed 11 for 7 configs).**
Swapping all eight R15 members to seeds 6–11 gives −0.83 sel (dla 75.03 vs 74.93; ds
85.21 vs 86.97). Averaging seeds 0–11 gives −0.33. This points to some selection on seeds 0–5 in R15's
weights and members: the R15 dev score is optimistic by about 0.5. That is not a
reason to change the sealed members (no dev gain). It also means a +0.5 bar sits
near seed noise, which is why replications are mandatory.

Resubmit bug: `sacct`'s SubmitLine drops the quotes around sweep.sbatch's config list,
so 3035744 and 3035746 failed (ValueError protocol). Re-run with quotes: 3036215
(seed 11) and 3036216 (ROCKET seeds 0, 2).

## Second wave (1 Nov 23:20)

| Arm | What | Jobs |
|---|---|---|
| W2a teacher seeds | m7r4 families pretrained on seeds 12–23; heads for 0–23; halves s011 / s1223 / s023 at C 0.01 and 0.1 | 3035805–8, screen 3035809 |
| W2b weight grid | ref and m7r4 at w 3–8 on the s12 screen (C 0.01, 0.03); w is chosen on s05 and checked on s611 | 3035803–4 |
| W3 rotation strength | m7r4 families pretrained with rot 0.10 (m7r10) and 0.20 (m7r20), seeds 0–5; per family and ×4, C 0.1 / 0.01, w 2–5 (`runs/diag/wave3.sh`) | 3036464–71, screen 3036472 |

**W2a teacher seeds 12–23 (corrected, 3036357; 96 checkpoints).** This is a second
independent replication. m7r4 at C 0.01, w 4 gives sel Δ +2.11 on s1223 (dla 77.20,
ds 88.80), against +2.33 on s011 and +2.12 pooled over 24 seeds (s023). The packaged
ref (C 0.1, w 2) gives +0.80 on s1223. m7r4 now holds across three disjoint sets of
pretraining seeds (+2.05 / +2.30 / +2.11 for 0–5 / 6–11 / 12–23). More seeds do not
help: 24 seeds are no better than 12, so the package stays at 12 seeds (48 nets), or
6 if worker time or size demands it.

**W3 rotation strength (3036472).** Best pooled sel Δ on seeds 0–5 (C 0.1 / C 0.01) for
the four families together: m7r10x4 +0.56 / +0.94, m7r20x4 −0.52 / −0.26, against m7r4
(rot 0.05) +1.58 / +2.05. Rot 0.10 helps no family (best single family: sinc +1.51 at
C 0.01), and rot 0.20 is negative for all of them. Pretraining augmentation has a
narrow optimum near 0.05. Closed.

**T4 recipe, corrected (screenB2 3036356; best pooled sel Δ on seeds 0–5, C 0.1 / 0.01).**
| Family | C 0.1 | C 0.01 |
|---|---|---|
| m7r4 (incumbent) | +1.58 (w4) | **+2.05 (w4)** |
| fbcnet m7ra | **+2.29 (w4)** | +1.53 (w4) |
| all m7r60 (4 families) | +2.04 (w3) | +1.92 (w4) |
| all m7ra | +1.67 | +1.65 |
| m7r60 / m7ra + ref | ≤ +1.53 | ≤ +2.00 |

Nothing beats the incumbent by +0.5. fbcnet m7ra is a single family (12 nets instead
of 48), but it is the best of about 400 cells. **Pre-registered check, W4 (3036638–42 →
screen 3036643):** fbcnet m7ra at C 0.1, w 4 and all m7r60 at C 0.1, w 3, pretrained on
seeds 6–11. Either one is adopted only if it beats m7r4 (C 0.01, w 4; s611 +2.30) by
≥ +0.3 on s611. Otherwise m7r4 stays. m7r4 + fbcnet m7ra is reported as well.

**W4 recipe replication (3036643): both fail; m7r4 stays.** s611 against m7r4 (C 0.01, w 4)
at +2.30, pre-registered settings: fbcnet m7ra (C 0.1, w 4) +1.64 (−0.66); all m7r60
(C 0.1, w 3) +1.89 (−0.41). Their seed 0–5 leads (+2.29, +2.04) were selection
noise, as the ~400-cell grid suggested. m7r4 + fbcnet m7ra (12 seeds) is not better
either: s011 at most +2.28 (C 0.1, w 3) against m7r4's +2.33.

**Finer C (3036866–7, m7r4, 12-seed screen).** Best sel Δ (s05 / s611 / s011):
C 0.003: +2.05 / +2.19 / +2.36 (w 5–6); C 0.001: +2.17 / +2.20 / +2.16 (w 5–8); against
C 0.01, w 4: +2.05 / +2.30 / +2.33. C has plateaued and the optimal weight drifts
upward as C shrinks. **C 0.01, w 4 stays (as packaged in cand_m7r4).**

**Incident (2 Nov ~00:15): pretraining without rotation.** pretrain.sbatch takes
`ARCH ALIGN TAG CORPORA "CFGS" "PROTOS" "EXTRA"`. Two submissions dropped the two empty
strings: the cn122 resubmits (rebuilt from sacct's SubmitLine, which drops quotes) and
the W2a seeds 12–23. `--rot 0.05` therefore landed in the fine-tune slot. The
pretraining ran without rotation and saved files under the m7r tag; the 60-epoch runs
crashed (`--epochs` without a value), and the classic-augmentation runs got `--aug`
without rotation. 77 such checkpoints were moved to `runs/pretrain_norot/`, not
deleted: 48 seeds 12–23, 21 architecture-pool, 8 m7ra. The one still running
(3035753_5) was cancelled. **Invalid results: screen 3035809 (s24) and screenB
3035320.** Neither was read for a decision. Corrected resubmission
(`runs/diag/resub2.sh` with explicit `"" ""`): architecture pool 3036271–82,
recipes 3036283–90, seeds 12–23 3036291–4. Screens re-run on them at C 0.1 and 0.01:
screenA2 3036355, screenB2 3036356, s24b 3036357.

## Packaging the candidate (2 Nov ~02:00, no upload)

`scripts/export_external_teacher.py` now takes `--families`, `--seeds` and `--C`; its
defaults reproduce the packaged arm. Galvani chain (`runs/diag/build_m7r4.sh`):
- 3036669: regression; defaults → `runs/sealed/regress_ext_g`, which must match
  final_r19e_ext_g (teacher.pt and weights.npz);
- 3036670: candidate `runs/sealed/cand_m7r4` (sincshallow, fbcnet, fblight, shallow;
  m7r seeds 0–11; C 0.01; weight 4 × v2);
- 3036671: official A100 replay of the candidate;
- 3036672: worker-image replica replay of the candidate.

**Results (2 Nov ~03:00).**
- **Regression passes.** weights.npz is identical to final_r19e_ext_g (every array), and
  so are all check numbers: in-sample accuracy 0.5653, 18 labels changed, traced vs eager
  0.0, CUDA 2.0e-6. teacher.pt has the same state_dict and identical logits on random
  input; only `serialization_id` and `*.debug_pkl` (source line numbers) differ. Earlier
  exports of the same arm already had different shas (65777454…, 9dbc31b8…), so the
  hash was never deterministic.
- **cand_m7r4 passes every staging gate:**
  - harness agreement 1.0;
  - traced vs eager 0.0, batch 1 7e-7, CUDA 1.5e-6;
  - 47 of 2655 labels change against final_r19e_g, teacher weight 0.5;
  - official A100 replay 3036671: files unchanged, ingestion 148 s;
  - worker-image replica 3036672: files unchanged, ingestion 120 s, same score;
  - zip 13.7 MB, sha 060850ee…; teacher.pt c21a03dd…, weights.npz 95c74d1a…,
    submission.py unchanged (25d51a6e…).
- **Not done:** no upload (needs explicit authorisation; sealed phase 30483 opens
  5 Nov). The real Codabench worker has not run this package; the replica and the
  earlier m7r teacher (1026185) are the evidence for compatibility.

## Architecture pool, corrected (screenA2 3036355, 2 Nov 07:45)

22 architectures pretrained as m7r (seeds 0–5), each as a teacher alone and with the
packaged ref added (`+x`). Only the C 0.1 pass ran: the C 0.01 pass would have hit the
8 h limit, so the job was cancelled at 6 h 43 and replaced by a focused screen (wave 5).
Sel Δ vs R19e at w 4 (s02 / s35 / s05); m7r4 at C 0.1 for reference:

| teacher | s02 | s35 | s05 |
|---|---|---|---|
| m7r4 (C 0.1; packaged at C 0.01: s05 +2.05) | +1.51 | +2.41 | +1.58 |
| **EEGNeX** | +2.61 | +2.67 | **+2.54** (ds 89.07) |
| **IFNet** | +2.41 | +2.14 | **+2.13** (ds 89.33) |
| SCCNet | +2.06 | +1.33 | +1.88 |
| EEGTCNet | +1.85 | +1.17 | +1.80 |
| CTNet | +1.66 | +1.94 | +1.56 |
| TCFormer | +1.28 | +1.48 | +1.61 |
| 16 others | | | ≤ +1.54 |
| all 22 together | | | −0.62 |

Adding ref to a single architecture (`+x`) never helps (best +1.72). EEGNeX and IFNet
lead with both halves positive. Picked from 22 architectures on s05, so the lead
needs unseen seeds before it counts.

## Wave 5 (2 Nov ~07:50, `runs/diag/wave5.sh`)

- focus5 3037208: s05 screen from screenA2.npz at C 0.01 and 0.1, w 2–6; m7r4,
  EEGNeX, IFNet, both together, each with m7r4, all three; SCCNet and CTNet alone.
- probe5 3037209: export test `runs/sealed/probe_nxif` (EEGNeX + IFNet, seeds 0–5,
  C 0.01, w 4) to show that both trace to TorchScript and match the harness. Not
  a candidate.
- 3037210 / 3037211: EEGNeX and IFNet m7r pretraining, seeds 6–11.
- screenW5 3037212 (after the pretraining): s05 / s611 / s011 for the focus5 families.

**Pre-registered (before any s611 data).** The arm and its (C, w) are fixed from
focus5 (s05 only) and written here before screenW5 reports. Adopt only if, in
screenW5, that arm beats m7r4 at C 0.01, w 4 by ≥ +0.5 sel on s611 and on s011,
with dla not below m7r4's, and probe5 passes (traced = eager, harness 1.0).
Otherwise cand_m7r4 stays.
