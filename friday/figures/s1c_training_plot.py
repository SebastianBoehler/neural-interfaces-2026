"""Plot authenticated S1C training and inference contrasts."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot import BLUE, GOLD, INK, MUTED, save


LAYOUTS = {
    "82fe550621a8eb72a0762c507fb81ee1a9d4d3cc65983a3e8bbdf2d923aed420": dict(
        title="Original training gave the strongest waveform scores",
        stem="held-s1c-training", xlim=(-5, 2.2),
        caption="Original training with bounded inference scored 59.25% and 60.26%.",
        contrasts=(
            ("original_bounded_eval", "Bounded inference\nOriginal training"),
            ("bounded_training", "Add bounding during training\nSame bounded inference"),
            ("causal_training_ordinary_eval", "Causal training\nSame ordinary inference"),
            ("whole_decoder", "Bounded training + inference\nVersus original S1C"),
        )),
    "eb06cd1fa8c52540706045850a4a76fa74113f7febe668e5229dc1a12984842e": dict(
        title="Masking fixed features helped when retained at inference",
        stem="held-s1c-feature-dropout", xlim=(-1.6, 1.72),
        caption="Masking during training with all features at inference scored 59.94% and 60.60%.",
        contrasts=(
            ("dropout_vs_scaled_on", "Masking vs 75% scaling in training\nFull features at inference"),
            ("dropout_vs_original_on", "Masking vs original training\nFull features at inference"),
            ("dropout_vs_zero_zero", "Masking vs zero features in training\nZero fixed features at inference"),
        )),
    "9c2ef83eb850d671950be72024393c43c2b2e9952303d5f5bf0718b618e62916": dict(
        title="Matched jitter and phase gave similar S1C gains",
        stem="held-s1c-matched-phase", xlim=(-.85, 1.75),
        caption="Bounded inference. Both augmentations helped seed 1 over hard CE.",
        contrasts=(
            ("bounded_phase_minus_hard", "Dispersive phase\nvs hard CE"),
            ("bounded_jitter_minus_hard", "Rigid jitter\nvs hard CE"),
            ("bounded_phase_minus_jitter", "Dispersive phase\nvs rigid jitter"),
        )),
    "3b759900d0add003dd10b2eb832348deab3a6e61ec1343851f81f452d53fa9ba": dict(
        title="Spatial sensor withholding produced mixed S1C effects",
        stem="held-s1c-spatial-withholding", xlim=(-1.40, 1.55),
        caption="Bounded inference. The masked arms matched electrode count and removed energy.",
        contrasts=(
            ("bounded_local_minus_hard", "Local electrodes\nvs hard CE"),
            ("bounded_dispersed_minus_hard", "Dispersed electrodes\nvs hard CE"),
            ("bounded_local_minus_dispersed", "Local electrodes\nvs dispersed electrodes"),
        )),
    "f504fad70c3e8df062e434163fc838d06a999d69df310c694f9f2a7f7fc2ff2b": dict(
        title="The hierarchical head reduced jittered S1C accuracy",
        stem="held-s1c-hierarchy-jitter", xlim=(-1.95, 1.78),
        caption="Bounded inference. Hierarchy plus jitter lost to the exact jitter control at both seeds.",
        contrasts=(
            ("bounded_jitter_minus_hard", "Jitter\nvs hard CE"),
            ("bounded_hierarchy_minus_hard", "Hierarchy\nvs hard CE"),
            ("bounded_hierarchy_jitter_minus_jitter", "Hierarchy plus jitter\nvs jitter"),
            ("bounded_interaction", "Factorial\ninteraction"),
        )),
    "64678ecd1cb61b6869e4d54940b49581531e8dfe04162c661e4c687431094790": dict(
        title="Cross-person interpolation changed sign across seeds",
        stem="held-s1c-donor-interpolation", xlim=(-1.42, 1.30),
        caption="Bounded inference. Both interpolation arms used the same 489,600 donor appearances.",
        contrasts=(
            ("bounded_cross_minus_hard", "Cross-person\nvs hard CE"),
            ("bounded_within_minus_hard", "Same-session\nvs hard CE"),
            ("bounded_cross_minus_within", "Cross-person\nvs same-session"),
        )),
    "fe563b10da46916df384e1e823a2dc418eb60b187bcc780ccf6617f42efd7d76": dict(
        title="Reconstruction training effects varied across seeds",
        stem="held-s1c-reconstruction-consistency", xlim=(-1.65, 1.33),
        caption="Bounded inference. The held recordings retained the original electrodes.",
        contrasts=(
            ("bounded_alternating_minus_hard", "Alternating reconstructions\nvs clean CE"),
            ("bounded_consistent_minus_hard", "Reconstruction consistency\nvs clean CE"),
            ("bounded_consistent_minus_alternating", "Reconstruction consistency\nvs alternating CE"),
        )),
    "ef8d307bb460f65886bcd03de70ffdf31ec8a6dffc525d0b5152c918c8596931": dict(
        title="The 50 ms lag improved one raw S1C seed",
        stem="held-s1c-within-block-lag", xlim=(-.85, 1.17),
        caption="Bounded inference. Scrambled pairs used the same shape and number of parameters.",
        contrasts=(
            ("bounded_lag_minus_hard", "50 ms lag\nvs hard CE"),
            ("bounded_scrambled_minus_hard", "Scrambled pairs\nvs hard CE"),
            ("bounded_lag_minus_scrambled", "50 ms lag\nvs scrambled pairs"),
        )),
    "363de60158d284da9979286fba1b08f04d8a247d4a9f18c1b19f3163e2ffff49": dict(
        title="Lag plus jitter tracked the original jitter decoder",
        stem="held-s1c-lag-jitter", xlim=(-1.39, 1.70),
        caption="Bounded inference. All three trained parent states and their decisions matched exactly.",
        contrasts=(
            ("bounded_joint_minus_hard", "Joint model\nvs hard CE"),
            ("bounded_joint_minus_jitter", "Joint model\nvs jitter"),
            ("bounded_joint_minus_lag", "Joint model\nvs 50 ms lag"),
            ("bounded_interaction", "Factorial\ninteraction"),
        )),
    "2f3748d9c72f94b72c95906eeca2fa582f12222c3e6767b8367607807acbbceb": dict(
        title="Cross-person calibration lowered held accuracy",
        stem="held-s1c-calibration-context", xlim=(-2.22, .61),
        caption="Bounded inference. Source disagreement fell 91% and 85% relative to jitter.",
        contrasts=(
            ("bounded_cross_minus_jitter", "Cross-person context\nvs jitter"),
            ("bounded_cross_minus_within", "Cross-person context\nvs within-session"),
            ("bounded_within_minus_jitter", "Within-session context\nvs jitter"),
        )),
    "64d2182825146f30486781e7a847407556a6acb32cdec55e08ecfc35182d52af": dict(
        title="Multiresolution EEG gating improved one seed",
        stem="held-s1c-multiresolution", xlim=(-1.08, 1.98),
        caption="Bounded inference. Both temporal gates add 4,032 trainable parameters.",
        contrasts=(
            ("bounded_multi_minus_hard", "Multiresolution gate\nvs hard CE"),
            ("bounded_multi_minus_single", "Multiresolution gate\nvs conventional gate"),
            ("bounded_single_minus_hard", "Conventional gate\nvs hard CE"),
        )),
    "bed4bf0abc95f1c75613c40b6caefab545e4d1f1e14c96ca598d5caf5a3fc3eb": dict(
        title="Common-average referencing changed sign across seeds",
        stem="held-s1c-reference-geometry", xlim=(-2.28, 2.72),
        caption="Bounded inference. Both reference variants matched output energy per window.",
        contrasts=(
            ("bounded_car_minus_hard", "Common-average reference\nvs hard CE"),
            ("bounded_car_minus_sham", "Common-average reference\nvs artificial direction"),
            ("bounded_sham_minus_hard", "Artificial direction\nvs hard CE"),
        )),
    "5eb8c7e938a1b14a90562253b5aaa1b867c0e79ddcbb5f61439b49d210698e54": dict(
        title="Physical sensor neighbors changed sign across seeds",
        stem="held-s1c-electrode-neighbors", xlim=(-1.07, .97),
        caption="Bounded inference. Both learned graphs preserved each electrode’s directed degree.",
        contrasts=(
            ("bounded_physical_minus_hard", "Physical neighbors\nvs hard CE"),
            ("bounded_physical_minus_scrambled", "Physical neighbors\nvs scrambled"),
            ("bounded_scrambled_minus_hard", "Scrambled neighbors\nvs hard CE"),
        )),
    "62f73d696cbbdbf59594a676d322e27d66a570533e35e910fb223589ecc277cd": dict(
        title="Teacher relation gains varied across seeds",
        stem="held-s1c-cross-fitted-relations", xlim=(-.87, 1.33),
        caption="Bounded inference. Each teacher excluded both people in the source pair.",
        contrasts=(
            ("bounded_literal_minus_hard", "Literal relations\nvs hard CE"),
            ("bounded_literal_minus_deranged", "Literal relations\nvs deranged targets"),
            ("bounded_deranged_minus_hard", "Deranged targets\nvs hard CE"),
        )),
    "285a0b6426557c39cdf16f1211a6b151c14587ed0c8382bfbef8afeaf4938979": dict(
        title="Both recurrent readouts fell below the plain decoder",
        stem="held-s1c-recurrent-readout", xlim=(-12.4, 5.2),
        caption="Bounded inference. Preserving token order improved the recurrent readout.",
        contrasts=(
            ("bounded_chronological_minus_hard", "Chronological readout\nvs hard CE"),
            ("bounded_chronological_minus_scrambled", "Chronological readout\nvs scrambled readout"),
            ("bounded_scrambled_minus_hard", "Scrambled readout\nvs hard CE"),
        )),
    "892022a7c6cd5dae4e8888ea596dbbce4dbf6dbf636c7cea5629fcaa93fe847f": dict(
        title="Cross-person support did not consistently improve S1C",
        stem="held-s1c-episodic-support", xlim=(-.99, 1.22),
        caption="Bounded inference. Both temporary-head controls saw exactly the same source rows.",
        contrasts=(
            ("bounded_cross_minus_hard", "Cross-person support\nvs hard CE"),
            ("bounded_cross_minus_within", "Cross-person support\nvs within-person"),
            ("bounded_within_minus_hard", "Within-person support\nvs hard CE"),
        )),
    "d4f491be1e192e5c5a01fcf1d0a20b74aa90d5f0707e1bc21e6d21a525adf3e0": dict(
        title="Learning waveform order gave mixed held gains",
        stem="held-s1c-order-pretext", xlim=(-1.49, 1.24),
        caption="Bounded inference. The order head was used during training only.",
        contrasts=(
            ("bounded_ordered_minus_hard", "True order\nvs hard CE"),
            ("bounded_ordered_minus_sham", "True order\nvs shuffled labels"),
            ("bounded_sham_minus_hard", "Shuffled labels\nvs hard CE"),
        )),
}


def draw(receipt, out):
    raw = receipt.read_bytes()
    key = hashlib.sha256(raw).hexdigest()
    if key not in LAYOUTS:
        raise ValueError("receipt differs from the verified sources")
    data = json.loads(raw)
    layout = LAYOUTS[key]
    contrasts = layout["contrasts"]

    fig, ax = plt.subplots(figsize=(9.6, 4.75), layout="constrained")
    fig.get_layout_engine().set(rect=(.02, .085, .96, .91))
    fig.suptitle(layout["title"], x=.025, ha="left", fontsize=19, weight="bold", color=INK)
    ax.set_title("Ten held-out PROTEUS participants · eight delivery orders",
                 loc="left", fontsize=10, color=MUTED, pad=10)
    for seed, color, marker, offset in (("0", BLUE, "o", -.12), ("1", GOLD, "D", .12)):
        results = data["seeds"][seed]["paired"]
        for i, (key, _) in enumerate(contrasts):
            mean = results[key]["mean"]
            lo, hi = results[key]["interval"]
            ax.errorbar(mean, i + offset, xerr=[[mean - lo], [hi - mean]],
                        color=color, fmt=marker, ms=6, capsize=3, lw=1.65,
                        label=f"Seed {seed}" if i == 0 else None)
    ax.axvline(0, lw=1, ls="--", color=MUTED)
    ax.set_yticks(range(len(contrasts)), [label for _, label in contrasts])
    ax.set_ylim(len(contrasts) - .45, -.55)
    ax.set(xlim=layout["xlim"], xlabel="Paired change in standalone balanced accuracy (percentage points)")
    ax.grid(axis="x", color="#e1e7ee")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    fig.text(.025, .029, f'Bars show paired 80% intervals. {layout["caption"]}', fontsize=9.2, color=MUTED)
    out.mkdir(parents=True, exist_ok=True)
    save(fig, out, layout["stem"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    draw(args.receipt, args.output)
