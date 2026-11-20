# Lane FT-B6s: Sinc family on the B6 extra-people trunk (X) vs re-pretrained control (C), stock fold packages (pp)

X = m7r + ds007221g/lr + netbci Sinc trunk; C = shipped m7r recipe re-run. New people: ten held-new folds, official cell mean. J uses the UNVERIFIED 57/43 assumption with the eight other returning people counted 0.

FT chunks complete for all arms: ['o71000', 'o71004', 'o71008', 'o71012', 'o71016', 'o71020']; incomplete: none.

## X-stock: X minus stock

- 24 orders (71000..71023). New people cell mean **+0.058**; people-only paired 80 % [-0.02, +0.14]; people-nested 80 % [-0.05, +0.18]; positive in 14/24 orders; per-order: +0.04, +0.12, -0.03, +0.50, -0.31, -0.11, +0.36, +0.12, -0.22, +0.16, +0.09, -0.01, -0.26, +0.30, +0.19, -0.26, +0.20, -0.15, +0.03, -0.27, -0.11, +0.37, +0.31, +0.32
- per person (mean over orders; cells per person 2/2/2/6/2/2/2/2/2/6): P01 -0.06, P02 -0.20, P03 +0.25, P04 +0.21, P05 +0.15, P06 +0.18, P07 -0.02, P08 +0.18, P09 +0.10, P10 -0.13
- changed decisions per new-person fold, all orders pooled: P01 2.11 %, P02 0.62 %, P03 1.50 %, P04 4.46 %, P05 4.58 %, P06 1.72 %, P07 0.90 %, P08 0.90 %, P09 1.68 %, P10 2.10 %
- returning (24 orders) P04 -0.18, P10 -0.02 (neither decides); J (UNVERIFIED 0.57 new + 0.43 x [0.2 P04 + 0.8 P10]) +0.011
- registered rule: upper (people-nested) +0.18 -> undecided

## C-stock: C minus stock

- 24 orders (71000..71023). New people cell mean **-0.018**; people-only paired 80 % [-0.10, +0.06]; people-nested 80 % [-0.13, +0.10]; positive in 12/24 orders; per-order: +0.05, +0.19, +0.34, +0.00, -0.04, -0.42, -0.03, -0.34, -0.08, +0.60, -0.04, -0.04, -0.07, +0.13, +0.16, -0.45, +0.04, +0.00, +0.03, +0.04, -0.03, +0.03, -0.40, -0.10
- per person (mean over orders; cells per person 2/2/2/6/2/2/2/2/2/6): P01 +0.02, P02 -0.05, P03 +0.29, P04 +0.14, P05 +0.09, P06 -0.02, P07 -0.02, P08 -0.09, P09 -0.25, P10 -0.21
- changed decisions per new-person fold, all orders pooled: P01 2.21 %, P02 0.75 %, P03 1.23 %, P04 3.85 %, P05 3.93 %, P06 1.72 %, P07 0.75 %, P08 0.68 %, P09 1.70 %, P10 1.80 %
- returning (24 orders) P04 -0.13, P10 +0.09 (neither decides); J (UNVERIFIED 0.57 new + 0.43 x [0.2 P04 + 0.8 P10]) +0.009
- registered rule: upper (people-nested) +0.10 -> undecided

## X-C: X minus ft:C

- 24 orders (71000..71023). New people cell mean **+0.076**; people-only paired 80 % [+0.03, +0.12]; people-nested 80 % [-0.04, +0.19]; positive in 14/24 orders; per-order: -0.00, -0.08, -0.37, +0.50, -0.27, +0.31, +0.39, +0.46, -0.14, -0.44, +0.13, +0.04, -0.18, +0.17, +0.03, +0.19, +0.16, -0.15, +0.00, -0.31, -0.07, +0.34, +0.71, +0.43
- per person (mean over orders; cells per person 2/2/2/6/2/2/2/2/2/6): P01 -0.09, P02 -0.15, P03 -0.04, P04 +0.08, P05 +0.07, P06 +0.20, P07 -0.00, P08 +0.26, P09 +0.34, P10 +0.08
- changed decisions per new-person fold, all orders pooled: P01 2.06 %, P02 0.84 %, P03 1.28 %, P04 4.48 %, P05 4.58 %, P06 1.74 %, P07 0.90 %, P08 0.88 %, P09 1.87 %, P10 1.93 %
- returning (24 orders) P04 -0.06, P10 -0.11 (neither decides); J (UNVERIFIED 0.57 new + 0.43 x [0.2 P04 + 0.8 P10]) +0.002
- registered rule: upper (people-nested) +0.19 -> undecided
