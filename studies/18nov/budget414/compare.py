"""Report matched timing deltas and conditional 60-minute break-even counts."""
import json
from math import floor
from pathlib import Path
from statistics import mean


def measurements(arm, rep):
    run = json.loads(Path(f'{arm}_{rep}.json').read_text())
    wall = float(Path(f'{arm}-{rep}.wall').read_text())
    durations = [record['total'] for record in run['streams']]
    first_return, first_active, warm_active, warm_return = durations
    return dict(wall=wall, residual=wall-sum(durations), load=run['load'],
                copy=mean(record['copy'] for record in run['streams']),
                first_return=first_return, first_active=first_active,
                warm_return=warm_return, warm_active=warm_active)


def scenario(delta):
    headroom = 3600 - 1287
    warm = delta['warm_return'] + delta['warm_active']
    startup = delta['residual'] + delta['first_return'] + delta['first_active'] - warm
    assert min(warm, warm+startup) > 0
    return dict(warm_pair=warm, process_premium=startup,
                one_process=floor((headroom-startup)/warm),
                two_processes=floor((headroom-2*startup)/warm),
                fresh_process_each_pair=floor(headroom/(warm+startup)))


def native_budget(old):
    # Even if all 1,287 s were available for this one standardized workload.
    warm = old['warm_return'] + old['warm_active']
    premium = old['residual'] + old['first_return'] + old['first_active'] - warm
    return dict(one_process=floor((1287-premium)/warm),
                two_processes=floor((1287-2*premium)/warm),
                fresh_process_each_pair=floor(1287/(warm+premium)))


def main():
    pairs = []
    for rep in range(2):
        old, cap = [measurements(arm, rep) for arm in ('old', 'cap')]
        delta = {k: cap[k]-v for k, v in old.items()}
        pairs.append(dict(old=old, cap=cap, delta=delta, scenarios=scenario(delta),
                          incumbent_budget_pairs=native_budget(old)))
    avg = {k: mean(pair['delta'][k] for pair in pairs) for k in pairs[0]['delta']}
    result = dict(pairs=pairs, mean_delta=avg, mean_scenarios=scenario(avg),
                  first_active_excess=avg['first_active']-avg['warm_active'])
    Path('result414.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
