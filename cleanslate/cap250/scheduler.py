"""Submit cap250 DAG nodes as Slurm jobs and address individual fit array tasks."""

import json
import os
import shlex
import subprocess


class Slurm:
    def __init__(self, args):
        self.run = args.run
        self.root = args.release
        self.ids = {}
        self.state = self.run / "jobs.jsonl"
        self.runner = str(self.root / "cleanslate/cap250/slurm.sh")
        self.worker = "cleanslate/cap250/array_worker.py"
        self.regular = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes",
                        f"--account={args.account}", f"--chdir={self.run}",
                        f"--output={self.run}/logs/%x-%A_%a.out"]
        self.cpu, self.gpu, self.gres = args.cpu_partition, args.gpu_partition, args.gres
        if args.exclude:
            self.regular.append(f"--exclude={args.exclude}")
        rev = args.revision or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=self.root).strip()
        self.env = {**os.environ, "CAP250_RELEASE": str(self.root), "CAP250_RUN": str(self.run),
                    "CAP250_ASSETS": str(args.assets), "CAP250_DATA": str(args.data), "CAP250_REV": rev,
                    "CAP250_PYTHON": str(args.python or ""), "CAP250_IMAGE": str(args.image or ""),
                    "CAP250_SMOKE": str(int(args.smoke))}
        (self.run / "logs").mkdir()
        (self.run / "arrays").mkdir()

    def sbatch(self, name, resources, deps, tail):
        flags = [f"--job-name={name}", f"--cpus-per-task={resources.cpus}",
                 f"--mem={resources.memory}", f"--time={resources.minutes}",
                 f"--partition={self.gpu if resources.gpu else self.cpu}"]
        if resources.gpu:
            flags.append(f"--gres={self.gres}")
        if deps:
            flags.append("--dependency=afterok:" + ":".join(deps))
        command = [*self.regular, *flags, *tail]
        result = subprocess.run(command, env=self.env, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().partition(";")[0]
        if not job_id.isdecimal():
            raise RuntimeError(f"invalid sbatch job ID: {result.stdout!r}\n{result.stderr}")
        return job_id

    def submit_group(self, name, group):
        first = group[0]
        deps = tuple(self.ids[key] for key in first.deps)
        if len(group) == 1:
            wrapped = ["env", *(f"{key}={value}" for key, value in first.env),
                       "bash", self.runner, first.program, *first.args]
            job_id = self.sbatch(name, first.resources, deps, ["--wrap=" + shlex.join(wrapped)])
            jobs = [(first.key, job_id)]
        else:
            manifest = self.run / "arrays" / f"{name}.json"
            manifest.write_text(json.dumps([dict(program=node.program, args=node.args, env=dict(node.env))
                                            for node in group], indent=2) + "\n")
            wrapped = ["bash", self.runner, self.worker, str(manifest)]
            job_id = self.sbatch(name, first.resources, deps,
                                 [f"--array=0-{len(group) - 1}", "--wrap=" + shlex.join(wrapped)])
            jobs = [(node.key, f"{job_id}_{i}") for i, node in enumerate(group)]
        with self.state.open("a") as handle:
            for key, slurm_id in jobs:
                handle.write(json.dumps(dict(node=key, job=slurm_id)) + "\n")
        self.ids.update(jobs)
        print(f"{job_id} {name} ({len(jobs)} job{'s' if len(jobs) != 1 else ''})", flush=True)

    def submit(self, nodes):
        pending = {node.key: node for node in nodes}
        index = 0
        while pending:
            groups = {}
            for node in pending.values():
                if set(node.deps) <= self.ids.keys():
                    family_seed = tuple(node.args[1:4]) if node.program.endswith("/worker.py") else ()
                    signature = (node.program, node.resources, node.deps, family_seed)
                    groups.setdefault(signature, []).append(node)
            if not groups:
                raise ValueError("cap250 DAG has an unsatisfied dependency")
            for group in groups.values():
                index += 1
                name = f"cap250-{index:03d}-{group[0].program.rsplit('/', 1)[-1].removesuffix('.py')}"
                try:
                    self.submit_group(name, group)
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(f"sbatch failed for {group[0].key}: {error.stderr.strip()}. "
                                       f"Submitted jobs are recorded in {self.state}") from error
                for node in group:
                    del pending[node.key]
