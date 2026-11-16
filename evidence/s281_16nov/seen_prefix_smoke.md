# §281 seen-session route and prefix diagnostic, 16 November 2026

This is a diagnostic, not a treatment score. `within0` is the only order checked here.
The pinned v6.6 cap-250 ZIP SHA-256 is `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed`.
The saved `within0` route trace SHA-256 is `d5810af6deccceaae1c53c50505815b1a6f23434e63b557452f793cc66041e1c`.
The reference `runs/chainout/ret_within0.npz` SHA-256 is `e26685b271d5e016b038af57baab3e80f39c24ee4430c61a4dc3c0635e358950`.
The permutations file SHA-256 is `6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f`.

The route diagnostic compared six base streams over four held-out sessions and 762 rows.
Its archived routed-output maximum absolute difference was 2.98e-8, with no changed labels.
The recognition mask was latched from the first recognized window. The diagnostic uses dev gate emulation;
it does not verify the packaged gate directly or validate downstream routed-tail changes.

The explicitly indexed `within0_B2_pool.npz` and `within0_B2_s0.npz` prefix artifacts have respective
SHA-256 values `c09232684ef70f3835abaf60075859866ea9a930807b77a379986305b6f1a4f0` and
`9b131eac09b8bffc2d9ca9aebcee42232fe0e163a1a5d7b62cc013b46022321b`. Each checked eight
pre-recognition refits. The package's SessionEM and TSEM outputs and their prefix reapplications
matched the study implementation exactly on these data. Future-window and other-session perturbations
left the protected prefixes unchanged while changing other outputs by at least 0.125.
Across 280 paired prefix rows, the two bases differed on eight argmax labels; their largest posterior
absolute difference was 0.24572. A prefix consumer must validate its order and base metadata.

The producer was subsequently strengthened with guards for package settings and matching artifact
metadata; these guards were checked separately. The older all-LOSO archived study reuses pooled labels
for non-pool bases, so those streams are an approximation and cannot establish per-base parity.
The old unindexed seen prefix files also lack base metadata and must not be used for training.
No fine-tuning model was trained and no package-level end-to-end fine-tuning parity was established.

Source SHA-256 after those guard additions: `s281_seen_relabel.py` is
`4a76719822444b87d05a5251c456251e17095f17ce4fc5b8b48b121b798a4596`;
`s281_seen_route.py` is `0a3eb86a03e629b943c4d627e26f2716387d12a37dc484a9204aa71ec321e91a`;
`relabel.py` is `4c8ad10bdf24d6f7229ab68da39cf2e3354a7547048a60607ff94503bb72eb90`;
and `shufcombo.py` is `800ac13c8a1e1d080681da3d580076d46575b114565a0821bebb36f9c7fbd5c0`.
