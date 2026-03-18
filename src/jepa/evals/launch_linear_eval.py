#!/usr/bin/env python3
import argparse
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import List

import submitit

MODULE = "jepa.evals.train_probe"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def pre_steps() -> List[str]:
    return [
        "module load gcc/system",
        "module load Mambaforge/23.3.1-1-hpc1-bdist",
        "mamba activate neodev",
    ]


class EvalJob:
    def __init__(
        self,
        *,
        workdir: Path,
        base_config: Path,
        checkpoint_path: Path,
        nodes: int,
        gpus_per_node: int,
        retries: int,
    ):
        self.workdir = workdir
        self.base_config = base_config
        self.checkpoint_path = checkpoint_path
        self.nodes = int(nodes)
        self.gpus_per_node = int(gpus_per_node)
        self.retries = int(retries)

    def _pre_shell(self) -> str:
        return "; ".join(pre_steps())

    def _torch_cmd(self, port: int) -> str:
        args = [
            "torchrun",
            "--nnodes",
            str(self.nodes),
            "--nproc-per-node",
            str(self.gpus_per_node),
            "--rdzv-backend",
            "c10d",
            "--rdzv-endpoint",
            f"localhost:{port}",
            "-m",
            MODULE,
            "--config",
            str(self.base_config),
            "--checkpoint",
            str(self.checkpoint_path),
        ]
        return " ".join(shlex.quote(a) for a in args)

    def __call__(self) -> None:
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "16")

        tries = self.retries + 1
        for _ in range(tries):
            port = find_free_port()
            cmd = f"{self._pre_shell()} ; {self._torch_cmd(port)}"
            proc = subprocess.run(["bash", "-lc", cmd], cwd=self.workdir, env=env)
            if proc.returncode == 0:
                return
        raise RuntimeError(f"Eval failed after {tries} attempt(s): {self.checkpoint}")

    def checkpoint(self) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(
            EvalJob(
                workdir=self.workdir,
                base_config=self.base_config,
                checkpoint_path=self.checkpoint_path,
                nodes=self.nodes,
                gpus_per_node=self.gpus_per_node,
                retries=self.retries,
            )
        )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt-root",
        type=Path,
        required=True,
        help="Directory containing subdirs with checkpoint.pth files",
    )
    p.add_argument("--base-config", type=Path, required=True)
    p.add_argument(
        "--exp-name", type=str, help="Name for logs; default ckpt-root basename"
    )
    p.add_argument("--workdir", type=Path, default=Path("."))
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument("--timeout-min", type=int, default=300)
    p.add_argument(
        "-C", dest="constraint", choices=["thin", "fat", "none"], default="thin"
    )
    p.add_argument("--array-parallelism", type=int, default=64)
    p.add_argument("--retries", type=int, default=0)
    args = p.parse_args(argv)

    workdir = args.workdir.resolve()
    base_config = args.base_config.resolve()
    ckpt_root = args.ckpt_root.resolve()
    exp_name = args.exp_name or ckpt_root.name

    ckpts = sorted(ckpt_root.rglob("checkpoint.pth"))
    if not ckpts:
        print(f"No checkpoint.pth files under {ckpt_root}", file=sys.stderr)
        return 2

    logs_root = workdir / "experiments" / f"linear_eval_{exp_name}" / "slurm_logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(logs_root))

    slurm_params = dict(
        name=f"lin_eval_{exp_name}",
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        tasks_per_node=1,
        timeout_min=args.timeout_min,
        slurm_array_parallelism=args.array_parallelism,
    )
    if args.constraint == "thin":
        slurm_params["slurm_constraint"] = "thin"
    elif args.constraint == "fat":
        slurm_params["slurm_constraint"] = "fat"
    executor.update_parameters(**slurm_params)

    jobs = []
    with executor.batch():
        for ckpt in ckpts:
            job = EvalJob(
                workdir=workdir,
                base_config=base_config,
                checkpoint_path=ckpt,
                nodes=args.nodes,
                gpus_per_node=args.gpus_per_node,
                retries=args.retries,
            )
            jobs.append(executor.submit(job))

    for ckpt, j in zip(ckpts, jobs):
        print(f"{j.job_id}\t{ckpt}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
