"""Plot signed deferred-covariance timings on the exact RTX proxy worker."""

import argparse
from pathlib import Path
from statistics import mean

from adaptation import layout
from native_adaptation import locked
from plot import BLUE, GOLD, INK, MUTED, save


RECEIPTS = (
    '04e8a1162bfa8fd91cafbe597d9402ff3b7d34bb1f147e81c24f2451f0e571ef',
    '6ad9d58a7ca7b1daf2280599a0f40c091a69169e38891402b67cc7ca5eba6c42',
)


def draw(receipt_dir, out):
    fig, (ax,) = layout('Deferred covariance reduced stage time on the RTX proxy')
    fig.supxlabel('Deferred time ÷ eager time', fontsize=10, y=.076)
    ax.axvline(1, color=MUTED, lw=1, ls='--')
    ax.set_title('Unrecognized Scherer G visit 1 · lower ratios are faster', loc='left', fontsize=10, color=MUTED)
    for order, (color, offset, digest) in enumerate(zip((BLUE, GOLD), (-.13, .13), RECEIPTS, strict=True)):
        data = locked(receipt_dir / f'rtx-covariance-profile-{order}.json', digest)
        overall = data['profile']['unrecognized']
        assert overall['independent_verified'] and data['order'] == order
        eager_stage, lazy_stage = data['shared']['unrecognized']['stage_seconds']
        p95_ratio = mean(x['p95_s'] for x in overall['deferred']) / mean(x['p95_s'] for x in overall['stock'])
        for row, value in enumerate((lazy_stage / eager_stage, overall['ratio'], p95_ratio)):
            ax.scatter(value, row + offset, color=color, s=42, label=f'Order {order}' if row == 0 else None)
            offset_x, align = (-9, 'right') if .95 < value < 1 else (9, 'left')
            ax.annotate(f'{value:.3f}×', (value, row + offset), xytext=(offset_x, 0),
                        textcoords='offset points', ha=align, va='center', fontsize=9, color=INK)
    ax.set_yticks(range(3), ['Personalizer stage', 'Whole warm worker', 'Per-window p95'])
    ax.set(xlim=(.62, 1.27), ylim=(2.55, -.5))
    ax.legend(loc='upper right', frameon=False, ncol=2)
    fig.text(.028, .024, 'Two delivery orders for one reused participant · real cap250 worker on RTX 2080 Ti', fontsize=9, color=MUTED)
    out.mkdir(parents=True, exist_ok=True)
    save(fig, out, 'rtx-deferred-covariance')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipts', type=Path)
    parser.add_argument('output', type=Path)
    options = parser.parse_args()
    draw(options.receipts, options.output)
