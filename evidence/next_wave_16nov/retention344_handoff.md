# §344 concurrent covariance retention — queued17Nov2026

- Source6P04–P08/P10; P09 already-exposed individual-model instrument; noP02/P03.
- Exact constrained-head native1152 rawFBCNet/native40; no architecture/head/batch/RNG change.
- A native40; B original337C64pretext→40task; C samepretext→40task+covarianceaux;
  D sameC with only auxiliary target associations deranged within sourceperson×class each epoch.
- D whole44239-vectors preserve EXACT epoch exposure multisets perperson/class, zero rowidentity.
- C/D samefixedalpha .5150670026694046, from TRUE first-sourcebatch bodygradient ratio.1.
- COMMON3 P04–P06 teacher unchanged. C++ teacherfeatures reused; NNtraining STILL PyTorch.
- Real-source native-loop zeroaux twin exact8updates/bodySHA/RNG/batches PASS; sourcegradients/controlPASS.
- Firstsmoke3118674 failed18s callbackdefault; v23118676 guardsPASS/costSTOP59s duecoldinitialization.
- Finalv33118695 PASS58s after2step sourcewarmup/resetnative seed; forecast22111.58coresec=6.142CPUh.
- ActualALLsmokes810CPUcoresec=.225CPUh; noquality/heldlabels opened bysmokes.
- Immutable source `/mnt/lustre/work/gehler/gfh527/ni26_retention344_pilot_v1_20261117`.
- ManifestSHA `cecb10cc2d7cc7fb1da4b37bf49ec9f8498db80784004af66b9c1566927d455d`.
- Teacher `ni26_pretrain337_targets_v1_20261117/teacher.npz` SHA
  `b679c40c5234c2f1de14052c73a36c3e754c1f9f98b42a7d1bdca393ae1b8dcc`.
- Owning337source `ni26_pretrain337_pilot_v3_20261117`, manifestb9970a7e…137a1.
- Queued3118755 array0–1 A/B CPU6/24GiB/max10min;3118756 array2–3 C/D CPU6/24GiB/max20min.
- Collector3118759 afterokBOTH, CPU6/24GiB/max5min. No automaticretry/extension/outermatrix.
- Hardnewfullallocations6.5CPUh+actualsmokes.225=6.725CPUh<8CPUh;0A100.
- Eacharm saves actual body+aux and fullsource/held populationBNfeatures initial/fine0/fine10/fine40.
- All4eight-ordercausalbanks/16independentSOURCE6ridge(.1)maps guarded BEFORE P09 labels.
- A/B must reproduce historical338A/337C finalbody and ALL8posteriors EXACTLY orSTOP beforelabels.
- Finalreport `result.json`; compactsmoke `retention344_smoke.json`; no completedqualityresult yet.
- Prior INDIVIDUAL-MODEL C−B+.7pp[−2,+3],C−A+.3[−2,+3],C−D+.5[−1,+2].
- Retention without C>A remains negative strongercomparison; no forgettingclaim withoutprefineevidence.
- NoFULL-SYSTEM/peopleinterval/rank/package/upload claim. Root owns canonicaldocs.
