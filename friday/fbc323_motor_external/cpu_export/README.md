# Qualify the original native renderer on CPU

RTX v1 seed 0 joint missed the strict CPU-oracle posterior bound by 3.331e−7. Float64 FIR v2 also missed at seed 0, with 2.10812e−5 maximum drift on hard. Both cases kept every decision on the failing probe. Restore the original v1 renderer. Qualify CPU one-window inference in the same worker image against the independent CPU batch oracle. Preserve thresholds, fixtures and states. The evaluation device is part of this receipt. This measures the source-only CPU export and leaves the failed RTX limit explicit.
