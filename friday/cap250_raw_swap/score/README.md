# F-C13 exact Scherer scorer

Place this directory at `$release/score`. From the immutable worker release, run the scorer after all nine person jobs complete and the registered smoke has qualified each supplied actor.

```bash
sbatch --kill-on-invalid-dep=yes --dependency="afterok:$people_job" score/score.sbatch "$worker_source_sha" "$seed0_sha" "$seed1_sha"
```

For a primary-only run omit the final seed-1 SHA. Every receipt must then cover exactly stock and seed 0, and the result marks replication pending. The supplied SHA values bind the worker manifest and actor bytes independently of the worker outputs.

The scorer verifies the pinned §411b data, metric, stock labels, stock counters and row permutations. It requires one native replacement call per candidate window. Counters may change in candidate arms when the real downstream chain responds. Errors and refits beyond the cap invalidate the bank. `before_labels.json` seals receipt and decision hashes read-only before the scorer opens held gold. `result.json` reports absolute official accuracy, class recalls within each session and order, paired 80% contrasts, decision changes and complete warm-session runtime ratios.

Only the preregistered seed 0 can qualify for fresh held-person validation. The interval and both order point estimates must support a gain, and A100 timing must pass. Scherer participants are reused development evidence.
