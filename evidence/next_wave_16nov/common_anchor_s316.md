# §316 shared-anchor causal adaptation

§307's independent P09 refit failed its registered 1e-5 anchor check. §316 is
a separate study: it starts every adaptation arm from the exact same §305
source-only anchor and scaler. It tests the adaptation conditional on that
anchor; it does not validate independent reproduction of the optimizer.

The adapter verifies the §305 fold shard, exact source-person rows, scaler,
cache, input and permutation hashes before loading the reference coefficients.
The original causal stream and fixed/learned update rules remain unchanged.
The assembler requires explicit common-anchor provenance for each of ten newly
generated targets. Eight local tests passed across the adapter, scaler,
adaptation and person-ID tests.

P09 no-score pilot **3116532** precedes the capacity-checked release **3116533**.
The latter schedules all ten folds afresh, the complete guarded assembler and
the independent full-chain dispatcher only after the pilot passes. Separate
immutable pilot, production and chain workspaces are under
`/mnt/lustre/work/gehler/gfh527/ni26_s316_common_20261117/`. Target source
manifest SHA-256 is
`da00d4603be02274254324b47762cd927dc9451dec4f4012107dfa2c4d2ccf82`;
chain manifest SHA-256 is
`0d6660f5d33d1527fb76f6aee907161ab920c2b81e43580c7846368677c82ab6`.
Original §307 outputs remain unchanged. P09 pilot **3116532 completed 0:0**
in 1:36. Across the eight static streams, the largest difference from §305
was 2.33e-14, with no changed non-tied labels. Zero/update-disabled identity,
session reset, prefix and bounded updates also passed. No score was read.

Release **3116533 completed 0:0** and submitted ten fresh target cells as
**3116543** (%2), followed by complete assembly **3116544** and the guarded
chain dispatcher. The obsolete pending §307 assembler 3116402 was cancelled
only after this new dispatch; its failed target evidence remains intact.
