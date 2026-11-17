# §342 frozen spatial readout — completed 17Nov11:16CET

Scope: INDIVIDUAL-MODEL SOURCE6/P09 only. No FULL-SYSTEM DEV/SEALED/rank result.
Pilot3118608 COMPLETE57s/0:0; all32fits/source/numericalguards beforelabels.
Pretrained pooled44.630/temporal40.086/spatial51.347/full44.199%; random36.369/36.782/41.002/42.511%.
Spatial−pooled+6.717pp but below strongerhistoricalraw64.061%; no promotion/outerexpansion.
Rootcollected exactguard/resultJSON; compact final reportspatial342.md owns outcome. No ownedjobsremain.
Prior §329 retained1600 averages all41channels, retaining4time. Upstream token order is channel-major/time-minor
(`rearrange B N A T -> B (N A) T`), not an implementation-error proof. Spatial and full layouts are new.
Card `studies/17nov/spatial_readout342/CARD.md`; prospective registrations appended LIVE LEDGER by this owner.
Source initial checkpoint31e97ce pushed; root owns canonical LEDGER reconciliation, otherchanges preserved.

## Completed engineering, no held metrics

- v1CPU3118519 PASS16s; v1A1003118520 PASS32s. Independent C=.001/.01/.1 source-only CPU-oracle/GPU checks pass.
- Full source48×65600 pinned309GPU fit .06982s/26eval/maxgrad1.0066e−6, original stopping limits unchanged.
- v1CPU/A100 independently reconstructed random initial state SHA differs, output within old feature bound.
  Cause not identified. Pretrained state SHA exact. Preservev1; do not assert exact random initialization identity.
- v2CPU3118526 PASS14s, independent CPU oracle numeric loss/grad recorded. Stronger v2GPU pending below.
- v2GPU computes CPUprobe then moves SAME encoder toA100, requires EXACT frozen parameter SHA.
  Saves seed329random INITIAL state(noNNfit); pilot reuses EXACT savedstate for all four layouts.
- Three compact guardJSONs stored locally as spatial342_v1_cpu_guard/v1_A100_guard/v2_cpu_guard.

## Actual queued DAG

- Strengthened A100 smoke3118527, workspace `/mnt/lustre/work/gehler/gfh527/ni26_spatial342_v2_20261117`.
  CPU2/16GiB/A10010min; scheduler estimate11:11CET, not actual start.
- Releasecontroller3118541 afterok3118527, CPU1/4GiB1min, workspace `ni26_spatial342_release_v1_20261117`.
  Validates actualguard/artifacts/cost + account<244active/<32running. No automatic retry on capacity failure.
  Conservative full32largest-head projection must fit1200s, else STOP without score.
- If actual smoke passes, controller submits pilot CPU2/16GiB/A10020min in
  `/mnt/lustre/work/gehler/gfh527/ni26_spatial342_pilot_v2_20261117`.
  Pilot source manifest93d996cdd21b69ba81632e534bd2264a7d5286d106ae5c45dac3a59d2fd9df90.
- Controller `release.json` records actual pilotjob/projection/capacity. No pilotjob exists yet.
- Worst total requested .352A100h/~.75CPUh includingbothsmokes/entries/controller, under .5A100h+2CPUh.

## Pilot design and guards

Frozen exactdefaultMITAtom-L a28e97…a52b,425strictkeys/MATH/TF32off/seed329matchedrandom;
41×480/120Hz→polyphase800/200Hz/100/exactchannels. No newNNfit/rawconcat/assets/provider.
4layouts×2priors; pooled400,temporal1600,spatial16400(time mean),full65600.
SOURCE6P04–P08/P10; P09 held, previously exposed/nonblind; noP01/P02/P03labels/inference.
Source-only scaler /sqrt(D/400); fixedP10 minimumcellNLL selectsC in.001/.01/.1, tie lowerC.
All24inner+8finalfits and actualfittedCPU/A100probe≤1e−3prob/zeronontiedchanges complete before P09labels.
Pinned309GPUlib027df043…b394e; independently C-aware meanCE+||W||²/(2*C*n), bias unpenalized.
Allfit/scaler/posterior/pre-metricSHA outputs persisted before labels; officialmetric actualpath/hash checked.
Oldpooled/temporal frozenbanks verified SHA524610…2541a/exactrows/oldfeaturebounds only, not head reuse.
No new raw reference fit; historical338A6 64.061 is contextual originalPyTorchrawreference, differentdevice/seed.
Predictions spatial−pooled+3pp[−4,+10],full−spatial0[−3,+4],pre−random+1[−3,+5].
No automatic winner expansion/outer/current-system/promotion/upload. Report all8arms and limits.

## 17Nov10:15UTC engineering continuation (supersedes live DAG above)

v2A1003118527PASS22s; controller3118541PASS3s releasedpilot3118549 FAILED25s beforeanyheads/metrics.
Source-onlycacheddiag3118593PASS6s: allSOURCE6oldpretrainedpool/temporal PASSunchangedbound.
CachedP09arrayaudit3118599PASS6s: temporal8coordinatesfailonlyP09lastrows2652/2654,maxabs.0004697.
Same-weightsourceA100diag3118600PASS27s isolatesbatchgeometry: oldGPU_B3→newGPU_B3 BITWISE0;
B3→B8 alone temporal.00045848/fullpatch.00429249, originalCPU/GPUcombinedboundPASS.
SameB8oldtemporal max5.72e−6PASS. Oldsource7N2087 ends7rows; old329sourceendsharesB8withP02query,
newSOURCE6+P09 endsB7. NoP02/P03inferencewasdone. CachedP09arrayauditreadnolabels/newinference.
Prospectiverepair documentedLEDGER: oldfeatureboundenforcedSOURCE6pretrainedonly; preservefailedqueryfeatureclaim.
ALL8currentCPU/batch fittedprobguards remain≤1e−3/zeronontiedchanges; ADDprepool/temporaloldcachedP09fittedprobguard.
RandomhistoricalidentityNOTproved; new4randomlayouts useEXACTsamev2initialstate/currentpatches.
Pilot3118608QUEUED in `ni26_spatial342_pilot_v3_20261117`, source2ace9a0075c34e8e50f26b04caae1cd4e36b75201d7c4c2c012f614ccd8c2628.
Same20minA100/CPU2/16GiB; actualpriorA100106s, hardaggregate1306s=.363h, CPU<2h. No342performancecomputedyet.
