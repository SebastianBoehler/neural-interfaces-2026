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
