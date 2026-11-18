# D-EA historical delivery preflight

The historical B2 replay leaves two order-sensitive slots in recording order. Slot 1 is **trainable-FIR FBCNet with EA**, the exact mean of seeds 0–4. Slot 3 is **shallow EA**, the exact mean of seeds 0–5. Their raw-input counterparts occupy slots 2 and 4. All four means matched bit for bit on 2,655 rows across all six B2 archive bases. The old `tf` suffix means trainable filter bank.

On P01, both source snapshots used the original k=5 prior from 2,465 fit rows across nine other people. The first full0 delivery is recording row 94. Its B1 alignment and the first 16 inputs matched the independent first-window calculation and short prefix exactly. Looking up the recording-order aligned row instead changes the input by a maximum 3.072798. The two actual source implementations produced the same first-window tensor.

| P01, seed 0 reconstruction | Recording-order decision changes | Max recording posterior error | Shuffled member bank |
| --- | ---: | ---: | --- |
| Trainable-FIR FBCNet with EA | 0 of 190 | 0.0000000 | Passed prefix and reset |
| Shallow EA | 3 of 190 | 0.0494459 | Blocked by recording parity |

Trainable-FIR FBCNet with EA kept its state fingerprint and reproduced the independent full0 prefix within 4.404e−7. It changed 18 argmax decisions versus the recording-order lookups on the same 190 rows. The maximum posterior movement was 0.3996684. These are label-blind delivery differences. The preflight saved one causal member bank and the fit fingerprint. It did not score a chain or estimate an accuracy change.

The historical CV files saved probabilities and row indices. They contain no original fitted weights. The recorded Galvani `1459dcd` checkout lacked `experiments/`. The round-two evidence documents the rsynced, uncommitted source, first committed with probability export as `82120152`. The trainable-FIR reconstruction used `c625ecc7`, the source introducing that recipe just before A100 job 3002616. The shallow archive came from A100 job 2975282. The FBCNet result establishes numerical reproduction on this fold. The original training bytes and shallow discrepancy remain unresolved.

Galvani input preflight 3129221 passed at 18:56:54 CET. A100 3129222_0 failed the intended shallow recording gate and 3129222_1 passed the trainable-FIR FBCNet gate at 18:58:52 CET. Release `ni26_friday_ea_restream_20261118`, code `8e4394b`.

| Artifact | SHA-256 |
| --- | --- |
| Source snapshot and script manifest | `38056e8b786f71ed4db963c46c4ed2db3d3c06976d531c1b168e7816c3304749` |
| Historical bank manifest | `f4169ebbc0fe7ad51c55b18c6f2ded5b9223b467086bfed190daf9b641385385` |
| FBCNet recording and causal proof | `2d716f3ac9b996a15023c119d354e3407d6bfc269e9bf8b2ba1d0bc1aba66458` |
| Shallow recording failure proof | `2dc6e068e16f284accf495d589739827970a7eb67ec2a1bacac4cd369aefc7bc` |
| FBCNet causal full0 member bank | `57d643becb9233863317901a4854f53b628cd172c3ac704b0e3c4ea218cd850b` |

## D-EAa additional FBCNet anchors

Both preregistered anchors reproduced their entire archived recording-order posterior arrays bit for bit. The causal first-window alignment agreed exactly with the independent calculation in all four sessions. Within each session, the first 16 model outputs agreed within 6.849e−7 and interleaved reset reproduced the prefix exactly.

| Held fold | Source and held rows | Full0 first rows | Full0 decisions changed versus recording lookup |
| --- | ---: | --- | ---: |
| P10, seed 0 | 2,078 and 577 | 1077, 1300, 1390 | 40 |
| P01, seed 1 | 2,465 and 190 | 94 | 22 |

All input and A100 checks finished by 19:04:44 CET. Galvani 3129231 and 3129232_0/1 exited 0:0. The release `ni26_friday_ea_anchors_20261118` independently verified both CPU-readable checkpoints against their fitted-state fingerprints. New code `a00fe54` and source manifest `32612c9c02caaa8adf74dbedeed566bcfec7c9ad9e2a1b6e7d21731cefe1313c`. The original weights' provenance is still unavailable. The result covers these member replays only.

| Fold | Recording proof SHA-256 | Reconstructed checkpoint SHA-256 | Full0 bank SHA-256 |
| --- | --- | --- | --- |
| P10, seed 0 | `61efb8998668bcb49486f46fccd3ef22e26b3eb2ebd4da1eb935cb04e5e4d9b5` | `9ad6f449e0ee115be10749caff2561e29f1f33f4242550d6431d3c57cdf38cf8` | `8e7943dcca5078dab6c042667eb5129a302b8de44b8e8da0e72baae067410f25` |
| P01, seed 1 | `8fe26e92634521335e9f4ce49896d5e196bac475f33367ea13611e7c95256443` | `ee79a87315c620bb1173f6b75ca4c9eea362ae1b8a1dff105a7bdd58c417486e` | `f22fe30300874903c2adac70e7cc0a42e244270d12fe2b80cb0f18aec07bc0c8` |
