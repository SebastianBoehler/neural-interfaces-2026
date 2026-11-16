"""Summarize saved-stage effects using the existing scorer's metrics and intervals."""
import numpy as np


def scheme_mean(array, orders):
    groups = [[i for i, o in enumerate(orders) if o.rstrip('0123456789') == kind]
              for kind in ('within', 'full', 'runs')]
    return np.mean([array[g].mean(axis=0) for g in groups], axis=0)


def report(scores, y, cells, people, s, metrics, arms, stages):
    counts = np.array([len(np.unique(cells[people == p])) for p in s.PEOPLE])
    accuracy_by_stage = {}
    for arm in arms:
        for stage in stages:
            accuracy, ups, downs, confusions, controls, briers, margins = [], [], [], [], [], [], []
            for order in s.ORDERS:
                arows, urows, drows, crows, controlrows, brows, mrows = [], [], [], [], [], [], []
                for base in s.BASES:
                    before = scores['control', stage, order, base]
                    after = scores[arm, stage, order, base]
                    delta = (s.person_accuracy(after, y, cells, people, s.PEOPLE)
                             - s.person_accuracy(before, y, cells, people, s.PEOPLE))
                    up, down, confusion = metrics.transitions(before.argmax(1), after.argmax(1),
                                                               y, cells, people, s.PEOPLE)
                    if not np.allclose(up.sum(1) - down.sum(1), delta, rtol=0, atol=1e-10):
                        raise ValueError('weighted correction/destruction does not match existing metric')
                    arows.append(delta); urows.append(up); drows.append(down); crows.append(confusion)
                    if arm == arms[0]:
                        _, _, ctl = metrics.transitions(before.argmax(1), before.argmax(1), y, cells, people, s.PEOPLE)
                        controlrows.append(ctl)
                    if stage != 'CP':
                        bb, mb = metrics.calibration(before, y, cells, people, s.PEOPLE)
                        ba, ma = metrics.calibration(after, y, cells, people, s.PEOPLE)
                        brows.append(ba - bb); mrows.append(ma - mb)
                accuracy.append(np.mean(arows, axis=0)); ups.append(np.mean(urows, axis=0))
                downs.append(np.mean(drows, axis=0)); confusions.append(np.mean(crows, axis=0))
                if controlrows: controls.append(np.mean(controlrows, axis=0))
                if stage != 'CP': briers.append(np.mean(brows, axis=0)); margins.append(np.mean(mrows, axis=0))
            acc = np.asarray(accuracy); accuracy_by_stage[arm, stage] = acc
            if controls:
                ctl = scheme_mean(np.asarray(controls), s.ORDERS)
                print('control', stage, 'official_confusion_true_by_predicted', np.average(ctl, weights=counts, axis=0).tolist(), flush=True)
            up = scheme_mean(np.asarray(ups), s.ORDERS)
            down = scheme_mean(np.asarray(downs), s.ORDERS); conf = scheme_mean(np.asarray(confusions), s.ORDERS)
            print(arm, stage, 'accuracy', s.summarize(acc, s.ORDERS, s.PEOPLE),
                  'official_equal_cell', s.cell_weighted_summary(acc, s.ORDERS, counts), flush=True)
            print(arm, stage, 'official_class_wrong_to_correct', np.average(up, weights=counts, axis=0).tolist(),
                  'official_class_correct_to_wrong', np.average(down, weights=counts, axis=0).tolist(),
                  'official_confusion_true_by_predicted', np.average(conf, weights=counts, axis=0).tolist(), flush=True)
            if stage != 'CP':
                for name, values in (('brier_delta', briers), ('true_class_margin_delta', margins)):
                    arr = np.asarray(values)
                    print(arm, stage, name, s.summarize(arr, s.ORDERS, s.PEOPLE),
                          'official_equal_cell', s.cell_weighted_summary(arr, s.ORDERS, counts), flush=True)
    for arm in ('E05', 'E1CS'):
        difference = accuracy_by_stage[arm, 'CP'] - accuracy_by_stage[arm, 'V6Fr']
        print(arm, 'CP_delta_minus_V6Fr_delta', s.summarize(difference, s.ORDERS, s.PEOPLE),
              'official_equal_cell', s.cell_weighted_summary(difference, s.ORDERS, counts), flush=True)
