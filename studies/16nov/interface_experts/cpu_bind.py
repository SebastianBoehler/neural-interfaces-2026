"""Bind a fresh Python process to six logical CPUs inside Slurm's step mask."""
import json
import os
import sys


def select(allowed):
    allowed = sorted(set(allowed))
    if len(allowed) < 6: raise ValueError('Slurm assigned fewer than six logical CPUs')
    return allowed[:6]


def main():
    if os.environ.get('SLURM_CPUS_PER_TASK') != '6' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('expected six-CPU, no-GPU Slurm allocation')
    assigned = sorted(os.sched_getaffinity(0)); selected = select(assigned)
    os.sched_setaffinity(0, set(selected))
    if sorted(os.sched_getaffinity(0)) != selected or not set(selected) <= set(assigned):
        raise ValueError('CPU binding did not remain inside Slurm mask')
    os.environ['EXPERT_ASSIGNED_CPUS'] = json.dumps(assigned)
    os.environ['EXPERT_SELECTED_CPUS'] = json.dumps(selected)
    print('Slurm-assigned logical CPUs:', assigned, 'selected:', selected, flush=True)
    os.execv(sys.executable, [sys.executable, '-B', '-I', 'run.py', *sys.argv[1:]])


if __name__ == '__main__': main()
