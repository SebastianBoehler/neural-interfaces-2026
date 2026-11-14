# Panel answer: BCI signal processing and Riemannian transfer (14 Nov 2026)

**Bottom line:** nothing in the geometric toolbox is worth more than about +1 point on unseen people at this level.
Recentring is already doing its job, label-free rotation loses, and the one new lever I found is removing run offsets
inside the template stages (oracle +3.0 in a simplified setting, not yet in the chain).

**Status of my numbers.** I ran six quick CPU checks on the public data. They are not pre-registered, not scored in
the chain, and tuned on nothing but also confirmed on nothing: ten people, one pass. Two harnesses:

- **Toy core:** 7 bands, CAR, 0.1 trace shrinkage, log-Euclidean tangent, ridge head, leave one person out (57 % alone;
  your core1 is 69 %).
- **core1 PCs:** your cached 256 components (`runs/replay_data/feats_exact`), session-standardised, shrinkage LDA
  (64.6 %, the same number as your §103c).

Scripts: `/private/tmp/claude-501/-Users-sebastianboehler-Documents-GitHub-neural-interfaces-2026/ca8951cd-9c0d-4449-8fa2-f0de43144db9/scratchpad/bci/a1.py` to `a6.py` in the same folder.

## 1. Diagnosis

- **The tangent view is close to its own within-session ceiling.** With the session's true class means (leave-one-out)
  and the population covariance, core1's 256 PCs reach 78.6 %. Your chain is at 80 %. More geometry on the same view
  cannot give much.
- **The dominant structure inside a session is the run, not the class.** Share of within-session variance of the
  log-covariances, by band:

  | Band (Hz) | 1–4 | 4–8 | 8–13 | 13–20 | 20–30 | 30–45 | 45–58 | chance |
  |---|---|---|---|---|---|---|---|---|
  | Run | 11 % | 11 % | 13 % | 21 % | 25 % | 31 % | 30 % | 2.6 % |
  | Interface | 4 % | 3 % | 2 % | 4 % | 6 % | 7 % | 6 % | 0.5 % |
  | Class | 17 % | 13 % | 5 % | 2 % | 2 % | 2 % | 2 % | 1.1 % |

- **Class information sits below 8 Hz.** Above 13 Hz the class share is barely above chance. This is a slow-potential
  and ocular task far more than a mu/beta task. Variance share is not discriminability, so confirm with per-band
  accuracy before acting on it.
- **What the system cannot capture:** run offsets enter the template stages as if they were isotropic noise (TSEM
  shrinks its covariance 0.7 towards identity). Free-running mixture fitting finds runs, not classes, which is why
  your clustering got 47 %.

## 2. Five proposals, ranked

### 2.1 Remove run offsets in the template views (shared templates, no split)

- **Mechanism.** Each view vector gets its run's mean subtracted before SessionEM and TSEM see it: session reference
  plus a shrunk run offset, j/(j+κ) for j earlier windows of that run. Templates and covariance stay pooled over the
  session. This is not §168 (which split the session into runs and lost 3 points for lack of data) and not §170
  (which down-weighted distant windows).
- **Why it should work.** Every public run is class-balanced (Graz 12/12/12 and 10/10/10; BrainHero online 10/10/10;
  BrainHero acquisition 9 to 12 per class), so a run mean carries almost no class signal.
- **Evidence.** With true run ids and true leave-one-out class templates: 78.6 → 81.6, all ten people up (+1 to +6).
  On the population LDA alone it is only +0.7, and on the toy core +0.9. So it is a template-stage lever, not a
  member lever.
- **Expected gain.** Unseen +0.3 to +1.0 if runs are contiguous in the sealed stream, 0 to +0.4 under a full shuffle.
  Returning people about the same; their 8-point loss in Graz feedback runs is a run-type offset. Confidence about
  35 % that the chain gains at least +0.3.
- **Cheapest test (CPU, an hour).** True run ids in the real stages under your three shuffle schemes, causal offsets.
- **Kill.** Below +0.4 with true run ids. Then stop; a label-free version can only be worse.
- **Notes.** Causal, a few accumulators per window. Under contiguous runs, boundaries come from a change-point
  statistic on the 20–58 Hz log-variance profile. Under a full shuffle it needs online clustering into six groups,
  which I did not test.

### 2.2 Change the template metric: population within-class covariance as the shrinkage target, 256 components

- **Mechanism.** TSEM's covariance becomes a mix of seed residuals, the population within-class covariance in the
  same space, and identity.
- **Evidence.** With true templates and the population metric, accuracy keeps rising with dimension: 70.4, 73.8, 77.1,
  78.6 at 32, 64, 128, 256 components. Your optimum at 128 may come from the identity target, not from the data.
  Adding 20 % of the session's own total covariance gave +3.0 on true templates (81.6; 83.7 with run centring) but
  −9 on the population LDA alone. That sign conflict means undecided, not positive.
- **Expected gain.** Unseen +0.2 to +0.6, returning 0 to +0.3. Confidence 30 %.
- **Test.** CPU replay on six bases. **Kill:** below +0.3 with the person interval above 0.

### 2.3 A well-conditioned sub-8 Hz view for the template stage

- **Mechanism.** A 2 Hz band over 4 s has about 16 degrees of freedom (8 in a half window) for a 41 × 41 matrix, so the
  log map mostly sees the shrinkage floor. Reduce to 12–16 spatial components per band before the log. Add an
  ERP-prototype augmented covariance (population class-average slow waveforms stacked on the window), which keeps the
  phase-locked structure that plain covariances discard. Your map lists it as never run.
- **Use it as a view for the template stage, not as a member.** Your own rule applies: +1 on a member gives +0.1 to
  +0.2.
- **Expected gain.** Unseen 0 to +0.5, returning 0 to +0.3. Confidence 25 %. Its errors will correlate with the eye
  view, which is the risk.
- **Test.** CPU, half a day. **Kill:** view alone below 55 %, or streak-correlated with the eye net.

### 2.4 Detect the shuffle scheme inside the session

- **Mechanism.** If neighbouring delivered windows share a run, their nuisance features are more alike than the
  session average. Thirty to forty windows decide it. Then use the stage constants tuned for that scheme instead of
  a compromise, and switch 2.1 on only when runs are contiguous.
- **Expected gain.** +0.1 to +0.4, bounded by the gap between your scheme-optimal and compromise constants, which you
  already have. Confidence 40 %.
- **Test.** CPU, minutes: detector accuracy on your fixed permutations. **Kill:** under 90 % correct at window 40.

### 2.5 Outlier-weighted session statistics

- **Mechanism.** A Riemannian potato score per window down-weights artefact windows in the reference, the
  standardisation and the templates (not in their own classification).
- **Expected gain.** 0 to +0.3 on average, concentrated on artefact-heavy sessions. Confidence 20 %. I have no
  measurement for it.
- **Test.** CPU replay. **Kill:** no gain on the three weakest people.

### What I would not do

- **Label-free rotation.** Rotation fitted by soft assignment lost 0.5 to 14 points from 64.6 at every setting, also
  when regularised towards identity and cross-fitted. One person fell from 66 to 37.
- **More EM.** Whole-session soft template fitting also collapsed (51 to 59 in 256 components, cross-fitted or not).
  Your confident-seed, single-pass, causal design is the right one.
- **Changing the causal reference.** Under a full shuffle, k = 2 costs 1.9 points in the first ten windows and nothing
  after. A larger k is worse (k = 10: −3.6, −1.8, −0.3 over the first three position bins). A per-feature
  empirical-Bayes prior count recovers 0.7 of the 1.9. That is under 0.1 point per session. The first-fifth deficit is
  the stages warming up, and your §92 shows a population template prior does not fix it.
- **Per-interface anything.** Interface explains 2 to 7 % of variance and an interface reference gave −0.3 on the toy
  core. The run is the unit.
- **Hierarchical or pairwise heads.** A multinomial head in tangent space already contains the contrasts; I expect at
  most +0.2, in line with your §6.
- **Run-level quotas, connectivity or cross-frequency members.** At most +0.3 each in my experience, and you have
  closed the first and last.

## 3. Contrarian claim

**The "64.6 → 89.1" class-geometry ceiling in STATE §5 is mostly label leakage.** I reproduced it (88.9). With class
means that exclude the window being classified it is 76.4, and 78.1 with a rotation regularised towards identity.
So 12.5 of the 24.4 points come from each window sitting in its own class mean, in 256 dimensions with 63 windows per
class. The honest ceiling of that lever is at or below the chain.

A smaller flaw in the same scripts: `orthogonal_procrustes` on three class means in 256 dimensions rotates the other
254 dimensions arbitrarily, which costs 1 to 1.7 points. §104's refutation stands anyway.

**Follow-up worth checking:** whether your non-causal "whole session at once" test (the one that recovered almost
nothing) used in-sample templates. If it did, the same self-confirmation would hide a real gain. I did not check it.

## 4. One check on the data

**Can the run of a window be recovered without labels under a full shuffle?** Cluster each public session's windows
into six groups on a nuisance view and report purity. Above about 70 %, proposal 2.1 survives every shuffle scheme;
below, it depends on runs being contiguous in the sealed stream, which only 2.4 can tell you at run time.

## 5. The two uploads

- **Upload 1 (now):** your measured candidate as it stands (fine-tuning, eye view and rest member on unrecognised
  sessions; v5.1b on recognised ones). It has +1.0 to +1.2 under every shuffle scheme. Nothing from this answer goes
  in.
- **Upload 2 (20–21 Nov):** upload 1 plus whichever of 2.1, 2.2 and 2.4 passed its bar in the chain under all three
  schemes, with 2.1 inactive unless contiguous runs are detected. If none passes, spend it on the returning-person
  route instead.

At your transfer ratio of 0.41 to 0.46, all five proposals together are worth roughly +0.2 to +0.7 sealed. That is
not enough for third place without upload 1's gain.
