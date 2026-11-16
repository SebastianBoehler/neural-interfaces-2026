"""Fail closed if a traced CPU-only process leaves its assigned CPU set."""
import argparse
import ast
import json
import re
from pathlib import Path

DEVICE = re.compile(r'/dev/(?:nvidia[^" ]*|dri/(?:renderD|card)[^" ]*|char/\d+:\d+)')
INDIRECT = re.compile(r'(?:open_by_handle_at|io_uring_setup|io_uring_enter|pidfd_getfd|recvmsg)\(')
AFFINITY = re.compile(r'^\s*(\d+)\s+sched_setaffinity\((\d+),\s*\d+,\s*\[([^]]*)\]')
ASSIGNED = re.compile(r'affinity_logical_cpus= (\[[^]]*\])')


def check(trace, allowed):
    allowed = set(allowed)
    if not allowed:
        raise ValueError('empty CPU assignment')
    calls = 0
    with Path(trace).open(errors='strict') as f:
        for number, line in enumerate(f, 1):
            if DEVICE.search(line) or INDIRECT.search(line):
                raise ValueError(f'forbidden device or indirect access at trace line {number}')
            if 'sched_setaffinity(' not in line:
                continue
            match = AFFINITY.search(line)
            if not match:
                raise ValueError(f'unparsed affinity at trace line {number}')
            tid, target, text = match.groups()
            if int(target) not in (0, int(tid)):
                raise ValueError(f'cross-process affinity at trace line {number}')
            mask = set()
            for item in text.replace(',', ' ').split():
                if '-' in item:
                    first, last = map(int, item.split('-', 1))
                    if last < first:
                        raise ValueError('reversed CPU range')
                    mask.update(range(first, last + 1))
                else:
                    mask.add(int(item))
            if not mask or not mask <= allowed:
                raise ValueError(f'affinity escapes assigned CPUs at trace line {number}')
            calls += 1
    return calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace'); ap.add_argument('assignment')
    ap.add_argument('--cpus', type=int, required=True)
    ap.add_argument('--json-assignment', action='store_true')
    args = ap.parse_args()
    text = Path(args.assignment).read_text()
    if args.json_assignment:
        allowed = json.loads(text)
    else:
        matches = ASSIGNED.findall(text)
        if len(matches) != 1:
            raise ValueError('expected one recorded Slurm affinity')
        allowed = ast.literal_eval(matches[0])
    if len(set(allowed)) != args.cpus:
        raise ValueError('assignment count differs from Slurm CPU count')
    calls = check(args.trace, allowed)
    print(f'TRACE OK: CPUs={sorted(allowed)}, affinity_calls={calls}, no forbidden device opens', flush=True)


if __name__ == '__main__':
    main()
