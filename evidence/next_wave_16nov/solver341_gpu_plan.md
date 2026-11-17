# §341 resident GPU reuse cost plan — UNRUN

This is a plan, not an implemented GPUhandle, speed claim or allocated GPUjob.
The existing §309 CUDAfit ALREADY copies the standardized matrix/labels once perFULLFIT and keeps
them plus optimizer state resident until completion. It does not copy the matrix each LBFGSclosure.
The opportunity is reuse ACROSSfits and avoiding repeated host validation/feature roundtrips, not “addresidency”.

## Exact usable solver for new research

GPU-proven library `/mnt/lustre/work/gehler/gfh527/ni26_s309_device_20261116/runs/s309_build/libhead_device.so`
SHA027df0434ed7096b6110d0c500bbd83c796a473d21426f3953dbc245b11b394e, reverified17Nov.
`nc_head_fit_device(x,y,n,d,C,device,theta,loss,max_gradient,evaluations)`:
xCPUfloat64Ccontiguous[n,d], yCPUint64[n]class0/1/2; device1CUDA; thetaCPUfloat64[3,d+1], lastcolumnbias.
Loss=meanCE+sum(W²)/(2*C*n), biasunpenalized. Zero-init, LBFGS2000/2500/history10/grad1e−8/change1e−12/Wolfe.
Source-only centering/scaling is EXTERNAL. Class balancing/sampleweights are NOT supported bythis API.
The demo Head.fit AND independentobjective inhead_device_bench.py HARDCODEC=.3; parameterizeBOTH
for a registered alternativeC. Never silently reuse that wrapper forC=.01 or weightedCE.
New341singlepassvalidationlibrary9675…f5d8c has ONLYCPUproof; do not call it GPUproven.

## Minimal prospective comparison, not released

Freeze exactunscoredP01source341matrix/scaler/source-rowhash/C=.3/options; no targetaccuracy.
A: existing309 one-copy-per-fit API. B: fresh opaqueCUDAhandle validatesHOSTX/yONCE, transfersONCE,
owns deviceX/y, performs three independent ZEROinitialized fits with freshoptimizerstate, exports finaltheta.
No warmstartedsolver, mergedobjective/sharedline-search or alteredstops. Three identical fits here are
benchmark repetitions, not claimed useful production work. Laterdifferentfolds/options require separate guards.
C: sameB with sourceproducer tensor already onGPU only if actualproducer exports a device tensor/DLPack;
current ctypes outputbuffer feature APIs return CPUdata, so this zero-copyproducer route DOES NOT EXISTyet.
Do not claim existingGPUfeatureproduction implies complete downstream GPUresidency.

Firstreal mini/full objective/gradient guards and CPUpath identity. On actualA100, assertdevice/dtype/
sourcehash, no CPUfallback, finitecoefficients/gradient≤1e−5/commonobjectivegap≤1e−5. Reject incorrectcache keys.
Record matrix/scaler/reference/row/objective/options/C/weights/version/device inhandle/cache identity.
Separate coldcontext/build/validation/transfer/allocation, synchronizedend-to-endwall and CUDAevent/kernel
costs; eventtime alone is not wholecost. ProfileMemcpy events if separating DMA from CPUqueue gaps.
Use repeatedcounterbalancedA/Bgroups, full3fit cost INCLUDINGcreate/destroy and finalcopies; warmrepeat
only as a separately labeled measure. Report objective evaluations/trajectory and memory, not just ratios.
No sessionfuturebatching: batchSOURCEwindows/independentfits, never unavailablefuturetarget EEG.
No mutable handle/state may cross targetsession boundaries in any later package. Training source tensors
are fixed source artifacts; that does not authorize target-state caching across sessions.

Expectedgain is limited to measuredvalidation/transfer/setup fraction; it is NOT the historical224×device ratio.
The CPUprofile cannot predict GPUend-to-end benefit quantitatively. A future5minA100/CPU4/8GiBpilot could
size a separately authorized experiment; no suchallocation was requested orqueued under341.

## Memory and shape limits before attempting batches

P01matrix2465×44239float64=~.872GB(decimal), finalhead3×44240float64=~1.06MB; §309actualGPUpeak~.924GB.
For proposed frozen spatial readouts, 2277source×16400F64≈.299GB;×65600≈1.195GB beforetemporary/probe state.
These are estimates, not an import/optimization/latency proof for new inputs. Checkactual n/d/source/C/scaler.
Sourcecovariances, priors and per-foldstandardization may differ; do not reuse a resident standardizedmatrix
when any identity changes. Batchmultipleheads only with independentobjectives/line searches/stopping; a
single summedLBFGSobjective changes the solver and cannot silently replace independentfits.

Future patch admission: actualGPUcompatibility/source/optimizeractivity/zero/reset/probability/objective checks
first, boundedcost thenfixedscientific comparison. No automaticaccuracygrid, active-snapshotreplacement,
externalassets/provider, upload or packageclaim follows from this plan.
