# Sealed Track 2 composition (audit, 20 Nov)

## 1. Withdrawn texts (upstream 2026-competition, quoted exactly)
Introduced in d5d2cbf (2026-10-12, "DOC Codabench: add complete scientific FAQs"); present in the pinned submodule a66bc23; gone at upstream 8447e181 (2026-11-19 13:10Z).
- `codabench/pages/competition_bci.md` L61: "Codabench uses private recordings from the same 20-participant, six-session acquisition programme described in the main website's dataset directory. Evaluation combines later sessions from represented participants with participants absent from the public release. Recordings and labels remain hidden."
- `codabench/pages/scientific_faq_bci.md` L25-30: "It combines later sessions from the ten participants represented in the public release with ten participants absent from it. No sealed-participant calibration session, label, or other data is supplied through `fit`: ... Exact identities, session allocation, and evaluation order remain private."
- Public release (same FAQ L39-40): "14 sessions of 10 participants. Two participants contribute their first three sessions and eight contribute their first session."
- 8447e181 replaces both with "later sessions and new participants from the PROTEUS acquisition programme" and FAQ "Exact cohort size, identities, session allocation, and evaluation order remain private." No numeric cohort statement is left upstream. The 20x6 text was live for the first 5 uploads' period (removal date between 10 Nov and 19 Nov; commits 8734b2f6/bddaff3d of 10-11 Nov no longer contain "20-participant").

## 2. Window total (docs/SEALED_UPLOADS.md)
Upload 4 accuracy 0.8159139571, upload 5 accuracy 0.8195488722. Searching N in 19,000-21,000: N = 20,083 is the only integer for which both are k/N to 10 digits (k = 16,386 and 16,459). The third value (16,662, accuracy 0.82966) is not in docs/SEALED_UPLOADS.md or the other docs I grepped, so it is UNVERIFIED here; the two verified uploads already pin N.
Implied visits: 20,083 / 189.64 (public mean) = 105.9; /190 = 105.7; /189.46 = 106. Visit length 179-201 spans 100-112 visits, so N does not identify the count, but 106 (= 120 - 14) fits to <0.1 %.

## 3. Cell count
No Codabench result JSON with n_cells is stored (the n_cells hits in runs/ are local replay diagnostics, 8 cells, and docs/LEDGER.md planning text). Codabench's own diagnostics were not saved; nothing to verify. If cells = 2 interfaces per visit as in public (14 visits -> 28 cells), the sealed set would be ~212 cells.

## 4. Conclusion
Arithmetic of the withdrawn design: 20 x 6 = 120 visits; minus 14 public = 106. New people 10 x 6 = 60 (56.6 %); returning 46 (43.4 %), split P04 3, P10 3, eight one-visit public people 5 each = 40. Confidence: total ~106 visits high-medium (N fits, design stated for 4 weeks of the competition, upstream only made it private, not contradicted); exact per-person allocation medium-low ("session allocation private"; the design could be unbalanced, and the cohort could differ since the text was withdrawn).
Cell shares (if 2 cells per visit): new ~57 %, returning ~43 %. Returning by person: P04 6.5 %, P10 6.5 %, each of the 8 one-visit people 10.9 % (87 % jointly). Of all sealed cells: P04 2.8 %, P10 2.8 %, 8 one-visit people 4.7 % each (37.7 % jointly).
Dev weighting: keep 57/43 as a prior, but the returning part should be 0.87 on the pool of the eight one-visit public people (enroll on public visit 1, score on a later visit; not available as labelled dev data, only partly via NI-V B-ret and Scherer proxies) and 0.13 on P04+P10 (about 0.065 each), replacing 0.2 P04 + 0.8 P10. A returning P04/P10 dev gain therefore represents only ~6 % of sealed cells; do not treat it as the returning measure. Report P04, P10 and a "8 unobserved" sensitivity (a +/-1 pp returning effect for those 8 moves the total ~0.38 pp).
