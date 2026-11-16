# §314 time-local waveform representation

The registered whole-decoder study compares a 2,952-feature supervised time-local
power/mean baseline with the same baseline plus 6,144 fixed multiscale asymmetric
waveform-response features. It retains all 41 input channels in the matched
baseline. The new bank uses 16 bipolar channel pairs and does not use tangent
matrices, complex connectivity or pretrained FBCNet/Shallow weights. It is
ROCKET-inspired, not a reproduction. Both models are stateless.

The deterministic-feature and time-reversal unit test passed locally and on
Galvani; isolated-mode imports passed. Real-source pilot 3115983 passed 0:0 in
`/mnt/lustre/work/gehler/gfh527/ni26_s314_time_patterns_20261117/smoke`; it
read no held accuracy. Its source-only fit and all eight stateless streams took
155.9 seconds after setup. Its source manifest SHA-256 is
`baa34aa59fc34c0722dcf0c65a02b6d1fc1b70e93e32190b5ac7712dde49af0a`.

Production callback 3115987 passed its independent fold/source/order proofs and
queue-capacity check. It submitted 20 bundled fold/seed jobs as array 3116021
at %2. Original collector 3116022 was canceled while pending because its report
used non-serializable tuple keys and indexed those keys incorrectly. Corrected
collector 3116190 depends on all 20 fits and keeps the original producer source
manifest unchanged. Its independent report-only source manifest SHA-256 is
`338fa57a4adf88dc553bb8180c8b7e01662caf621d5e0cb9ace2ae1426f9ad1b`;
focused JSON/report and actual-source compilation tests passed. The
assembler writes 32 canonical 2,655-row files named
`ttt18<order>_tf314_base-all_loso-s<seed>.npz` and the analogous `tfd314`
files, with seeds 0/1 and eight orders. These are not six-seed streams.
No score, returning-person result, worker compatibility or package gain is
claimed before the relevant complete checks finish.
