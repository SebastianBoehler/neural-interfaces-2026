# §362 Label-assisted ceiling for selecting between two complete branches

Compare untouched cap250 and the fixed §318 JS full-chain branch on all 10 new
people, eight pinned delivery orders and six dependent bases. Both branches
evolve their own original state. At a disagreement, an oracle selects JS when
JS is correct, otherwise it selects the original branch. This uses the current
true label and is a bound for these two frozen choices. Scores come from the
unchanged official cell metric, with the paired people/order 80% intervals.
Check that every evaluation cell contains all three classes so the per-window
choice indeed maximizes the metric. Report the unique affected rows separately
from the multiplied order/base count.

Predict an oracle lift of +1.0 points over always-JS, plausible range
[+0.2,+2.5]. Use 1.0 point as a resource-sizing bar, together with per-person
coverage and paired cost. A smaller ceiling leaves limited room for a
practical selector. It does not settle adaptive routing in general. This
comparison is distinct from §320's one-person, weaker forward branches and
§350's mixed-state switching policy.

Score the same label-assisted comparison separately on the conditional §321
returning controls, keeping the original two-person and retained-prior caveats.
A real selector must never use labels or future windows. It would need both
branches' past states and both current outputs, so a timed single branch alone
cannot prove feasibility. Report the older §321 independent A100 branch-pair
timings separately. Their sum is a serial work proxy, with unmeasured
shared-state optimization and native execution costs.

One CPU2/6GiB/5min archive scoring allocation after frozen source, row, zero,
class, order and replay checks. No new worker inference, policy fits, candidate
promotion or upload from this registration.

## Result

Job 3122392 completed in 2m36s on two CPUs. The scheme-equal six-base oracle
adds **1.2914 pp** [80% official interval 0.9215, 1.6215] over always-JS.
On B2_pool, control/JS/oracle are **76.2793/76.5517/77.8829%**. The B2_pool
oracle adds +1.6035 pp [1.2235, 1.9314] over control and +1.3311 pp
[0.8523, 1.7602] over JS. Over all six bases the branches disagree 4,021
times in 127,440 comparisons, covering 987 distinct EEG rows. The B2_pool
figures are 682 disagreements in 21,240 comparisons and 533 distinct rows.
This supplies enough headroom to test one selector.

The two-person conditional returning oracle adds **0.1046 pp** over original at
the .2/.8 P04/P10 weighting. The models disagree only 47 and 8 times on those
two people across eight orders. This small ceiling supports keeping the
returning decision separate. The source priors and auxiliary weights retain
the exposure documented in §321. No returning selector was evaluated.

The original A100 study's median separate base-plus-JS serial work was
82.57/154.63/319.97 seconds on 190/378/800-window sessions. It did not run
fused dual-state inference or a selector. [Full results](branch_ceiling362_result.json)
