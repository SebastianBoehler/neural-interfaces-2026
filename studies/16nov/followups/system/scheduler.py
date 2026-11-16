"""Do not ask Slurm to depend on completed jobs purged from its registry."""


def live_dependencies(parents):
    dependencies = []
    for job, state, live, artifacts_valid in parents:
        if state == 'COMPLETED':
            if not artifacts_valid:
                raise ValueError(f'{job}: completed parent artifacts not verified')
        elif state in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING') and live:
            dependencies.append(str(job))
        else:
            raise ValueError(f'{job}: unsafe parent {state}, live={live}')
    return dependencies
