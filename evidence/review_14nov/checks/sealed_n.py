"""Infer the sealed window count N from the 10-digit plain accuracies logged in docs/SEALED_UPLOADS.md.
accuracy = k / N exactly, so N must make both logged values land on an integer k within the print rounding."""
acc = {"upload4 v41": 0.8159139571, "upload5 v51b": 0.8195488722}
hits = []
for N in range(1000, 80001):
    ok = True
    for a in acc.values():
        k = round(a * N)
        if abs(k / N - a) > 6e-11:
            ok = False; break
    if ok:
        hits.append(N)
print("N consistent with both accuracies:", hits)
for N in hits[:5]:
    print(N, {n: round(a * N) for n, a in acc.items()}, "sessions at 189.6 windows:", round(N / 189.64, 1))
