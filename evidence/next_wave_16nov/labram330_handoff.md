# Atom-L continuation handoff

Prepared17November2026, before any§329sourceaccuracy. Rootowns STATE/STATUS.
This continuation doesnotreplace scheduled06:51CETresearchblock.

## Active artifacts/jobs

- §329 A100engineering3117332: immutable `ni26_labram329_v3_20261117`,
  source936aecfa1040d3c75c767feef448da3f17662b5f908587b59d8a2405ec21118f;
  independentcompatibilityonly. Forecast06:58CET, noactualresultatpreparation.
- §329 chosen2080guard→source3117344: `ni26_labram329_2080_v1_20261117`,
  source057627db1f85aab13e0cab308f0a21d4e7c6645202b86fe2202160f4e4e44833;
  forecast06:23:56CET. ONE20minallocation, explicitdevice/SDPAMATH/TF32off.
  OriginalA100source3117338/3117341canceledpendingbeforeresults; noduplicateaccuracy.
- §330 releasecontroller3117398 afterany3117344: `ni26_finetune330_v1_20261117`,
  source3c58729c7a33f78c383064b4cbc5ffbeb21fa4c438759432e43a8c9697dedc5b.
  ReadsONLYengineering/fitted-readoutguards+cost/memory/sourcehashes, never329queryaccuracy.
  Ifpassed/capacityavailable, releasesONE≤30min2080trainingallocation; actualIDinrelease.json.
  Failedreleaseisablocker, notanaccuracyresult; noautomaticretry/widening.

## Shared upstream assets and source protocol

Officialcode59e4e7ea2d855cda5573c312b380b109f1856764;
HF72189644cef9a91b7b436c5f37f083c1366e2f21, defaultpretrain-onlyAtom-L,
212001576B, SHAa28e97f7bd73bd7ff55f3b5d680f48807c335cf3134d1281228a061056aaf52b.
MITcode+weightsverifiedfromactualtext/metadata; ungated/public. Noaccount/termsacceptance.
Inputcurrent41×480@120Hz→polyphase800@200Hz→µV/100, exactchannelmap, retainedpostnormpatches.
No physicalbandwidthrestoredabove60Hz; unknownparticipantoverlapwithupstreampretraining.
PretraininginventorycontainsDreyer2023/EEGMAT; noliteralPROTEUSnameisnotproofnofamilyexposure.

P04–P10fit; P02/P03previouslyexposedsourcequeries; P01excludedallfit/scalers/query.
Statelessqueryinference, officialcellBA, notouter/current-system/returning/sealedproof.
§329: ninefixedStandardScaler+LR C=.01heads: pretrained/random × pooled400/retained1600
×with/withoutidenticalraw287, plusrawalone. Unchangedsource740epochcausalrawSBNcontext.
Allsourcefits/actualCPU2080fitted-headparitybeforequerylabels; savebanks/source-fitquality.

## Separate §330 full-FT form

Fourarms pretrainedFT/pretrainedfrozen/randomFT/randomfrozen; all12FIXEDepochs/batch16.
All46.3Mencoderparametersadapt inFT. Sameheadseed330; classbalancedCE; bodyAdamWlr1e−4,
head1e−3,wd.01, cosine1%start; identicalsource-onlyinitialconditioning andraw287.
Frozencontrols haveequalclassifierupdatesoncachedsameinitialfeatures, notmatchedFLOPs.
Finalepochonly; noqueryHPO/earlystop. Fixed12epochsareaconvergencelimit, notaconvergenceclaim.

Beforefulltraining: realthree-classsourcebatchsmoke; activefinitebodygradient/nonzeroencoderupdate,
lastsamplegradient, CPU/GPUactualfitted-headprobability≤1e−3/zero nontiedargmaxchanges,
repeat/interleaving/deepcopy, restoresmokeinitialstatebeforefixedschedule.
Warmactualstepcost×completefourarms+180soverheadmustbe≤24min; peakGPU≤8GiB.
EveryfinalfittedheadCPU/GPUcheckandallfourtrainedoutputsfinishbeforequerylabels.
Epochlosses/finalgradients/sourceclassrecalls/param/checkpoint/scaler/bankhashesreported.

Selecteduncertainpriors: pretrainedFT65%[50,78]; FT−matchedfrozen+5pp[−3,+12];
pretrainedFT−randomFT+3pp[−5,+10]. WeakfrozenprobeorunderconvergedFTdoesnotclosefamily.
Noautomaticoutermatrix,upload,packageoradditionaltrainingform.
