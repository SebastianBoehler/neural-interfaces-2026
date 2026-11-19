# F34 analytic precision repair

Initial source smoke 3135174 recovered exact native hard, then failed at the original phase tolerance before any held prediction. The float32 analytic token change reached 2.86102e−6 against 2e−6. Mean energy error was 2.38107e−7 and the real-power token moved 1.70956 under the same rotation. The two mathematical power definitions and all model architecture remain fixed.

V2 computes the identical projected finite-window FFT, real/Hilbert powers and normalized averaging in float64 for both arms. Round the resulting token traces to float32 before the same original temporal layers. Preserve the actual source phase tolerance, independent CPU held replay, fit, sham and exact hard. The matched timing and memory limits still apply. No held labels were opened during v1.
