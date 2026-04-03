import argparse
import itertools
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import submitit
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


# ---------- sweep expansion ----------


def expand_sweep_param(spec):
    """Expand a single sweep parameter specification into a list of values.

    Supports:
      - plain list:  [1e-3, 3e-4, 1e-4]
      - log2 from min: {log2: {min: 0.1, n: 3}}  -> [0.1, 0.2, 0.4]
      - log2 from max: {log2: {max: 0.4, n: 3}}  -> [0.1, 0.2, 0.4]
      - linspace:      {linspace: {min: 0.01, max: 0.1, n: 5}}
      - scalar:        0.01  (becomes [0.01])
    """
    if isinstance(spec, list):
        return list(spec)
    if isinstance(spec, dict):
        if "log2" in spec:
            args = spec["log2"]
            n = int(args["n"])
            if "min" in args:
                return [float(args["min"]) * (2 ** i) for i in range(n)]
            if "max" in args:
                return [float(args["max"]) / (2 ** i) for i in range(n - 1, -1, -1)]
            raise ValueError("log2 requires 'min' or 'max'")
        if "linspace" in spec:
            args = spec["linspace"]
            return np.linspace(float(args["min"]), float(args["max"]), int(args["n"])).tolist()
    return [spec]


RANGE_KEYS = {"log2", "linspace"}


def flatten_sweep_params(d, prefix=""):
    """Flatten nested sweep params to dotted keys.

    {training: {lr: [1e-3, 1e-4]}} -> {"training.lr": [1e-3, 1e-4]}
    """
    result = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, (dict, DictConfig)) and not (RANGE_KEYS & set(v.keys())):
            result.update(flatten_sweep_params(v, full_key))
        else:
            result[full_key] = v
    return result


def expand_sweep(sweep_cfg) -> dict[str, list[Any]]:
    """Parse sweep config into {dotted_key: [values]} grid."""
    if sweep_cfg is None:
        return {}
    params = OmegaConf.to_container(sweep_cfg.get("params", {}), resolve=True)
    if not params:
        return {}
    flat = flatten_sweep_params(params)
    return {k: expand_sweep_param(v) for k, v in flat.items()}


def cartesian(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    if not vals:
        return [{}]
    return [dict(zip(keys, prod)) for prod in itertools.product(*vals)]


# ---------- utils ----------


def short_hash(d: dict[str, Any]) -> str:
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str).encode()
    import hashlib

    return hashlib.sha1(blob).hexdigest()[:8]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ---------- job ----------


class TrainJob:
    def __init__(
        self,
        *,
        workdir: Path,
        cfg_path: Path,
        sweep_name: str,
        run_id: str,
        gpus_per_node: int,
        nodes: int,
        setup_commands: list[str],
        retries: int,
    ):
        self.workdir = workdir
        self.cfg_path = cfg_path
        self.sweep_name = sweep_name
        self.run_id = run_id
        self.gpus_per_node = int(gpus_per_node)
        self.nodes = int(nodes)
        self.setup_commands = list(setup_commands)
        self.retries = int(retries)

    def _pre_shell(self) -> str:
        if not self.setup_commands:
            return "true"
        return " ; ".join(self.setup_commands)

    def _torch_cmd(self, master_port: int) -> str:
        return " ".join(
            [
                "torchrun",
                "--nnodes",
                str(self.nodes),
                "--nproc-per-node",
                str(self.gpus_per_node),
                "--rdzv-backend",
                "c10d",
                "--rdzv-endpoint",
                f"localhost:{master_port}",
                "-m",
                "jepa.train",
                "--config",
                str(self.cfg_path),
            ]
        )

    def __call__(self) -> None:
        env = os.environ.copy()
        env.setdefault("WANDB_RUN_GROUP", self.sweep_name)
        env.setdefault("OMP_NUM_THREADS", "16")

        tries = self.retries + 1
        for _ in range(tries):
            port = find_free_port()
            cmd = f"{self._pre_shell()} ; {self._torch_cmd(port)}"
            proc = subprocess.run(["bash", "-lc", cmd], cwd=self.workdir, env=env)
            if proc.returncode == 0:
                return
        raise RuntimeError("Training failed after retries")

    def checkpoint(self) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(
            TrainJob(
                workdir=self.workdir,
                cfg_path=self.cfg_path,
                sweep_name=self.sweep_name,
                run_id=self.run_id,
                gpus_per_node=self.gpus_per_node,
                nodes=self.nodes,
                setup_commands=self.setup_commands,
                retries=self.retries,
            )
        )


# ---------- config helpers ----------


def build_run_config(base_cfg: DictConfig, overrides: dict[str, Any], seed: int) -> DictConfig:
    """Clone base config, apply sweep overrides and seed, strip launcher-only keys."""
    run_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    OmegaConf.update(run_cfg, "training.seed", seed)
    for k, v in overrides.items():
        OmegaConf.update(run_cfg, k, v)
    # Remove keys that train.py doesn't need
    for key in ("cluster", "sweep"):
        if key in run_cfg:
            del run_cfg[key]
    return run_cfg


def save_run_config(run_cfg: DictConfig, cfg_dir: Path, run_id: str) -> Path:
    cfg_path = cfg_dir / f"{run_id}.yaml"
    OmegaConf.save(run_cfg, cfg_path)
    return cfg_path


# ---------- launcher ----------


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Launch training jobs (locally or via SLURM).",
        epilog="Positional args are Hydra overrides, e.g.: +experiment=toy_craftax cluster=hopper training.lr=1e-3",
    )
    p.add_argument("--sweep-name", required=True)
    p.add_argument("--seeds", type=int, default=None, help="Override sweep.seeds")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--retries", type=int, default=0)
    p.add_argument("--workdir", type=Path, default=Path("."))
    p.add_argument("overrides", nargs="*", help="Hydra config overrides")

    args = p.parse_args(argv)
    workdir = args.workdir.resolve()
    config_dir = str((workdir / "configs").resolve())

    # Compose config via Hydra
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose("config", overrides=args.overrides)

    cluster = cfg.cluster
    use_slurm = bool(cluster.get("slurm", False))

    # Expand sweep
    sweep_cfg = OmegaConf.select(cfg, "sweep", default=None)
    grid = expand_sweep(sweep_cfg)
    combos = cartesian(grid)

    # Seeds: CLI > sweep config > default 1
    seeds_count = args.seeds
    if seeds_count is None:
        seeds_count = int(sweep_cfg.get("seeds", 1)) if sweep_cfg is not None else 1
    seeds = [args.seed_offset + i for i in range(seeds_count)]

    run_specs = list(itertools.product(combos, seeds))
    print(f"Sweep: {len(combos)} configs x {len(seeds)} seeds = {len(run_specs)} runs")

    # Prepare output dirs
    cfg_dir = workdir / "experiments" / args.sweep_name / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Build and save all run configs
    jobs_info = []
    for overrides, seed in run_specs:
        run_key = dict(overrides, seed=seed)
        run_id = short_hash(run_key)
        run_cfg = build_run_config(cfg, overrides, seed)
        cfg_path = save_run_config(run_cfg, cfg_dir, run_id)
        jobs_info.append((run_id, cfg_path))

    if use_slurm:
        _submit_slurm(args, workdir, cluster, jobs_info)
    else:
        _run_local(workdir, cluster, jobs_info, args.sweep_name)


def _submit_slurm(args, workdir, cluster, jobs_info):
    logs_root = workdir / "experiments" / args.sweep_name / "slurm_logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(logs_root))

    slurm_params = dict(
        name=args.sweep_name,
        nodes=int(cluster.nodes),
        gpus_per_node=int(cluster.gpus_per_node),
        tasks_per_node=1,
        timeout_min=int(cluster.timeout_min),
        slurm_array_parallelism=int(cluster.get("array_parallelism", 64)),
    )
    constraint = cluster.get("constraint")
    if constraint:
        slurm_params["slurm_constraint"] = constraint

    executor.update_parameters(**slurm_params)

    setup_commands = list(cluster.get("setup_commands", []))

    jobs = []
    with executor.batch():
        for run_id, cfg_path in jobs_info:
            job = TrainJob(
                workdir=workdir,
                cfg_path=cfg_path,
                sweep_name=args.sweep_name,
                run_id=run_id,
                gpus_per_node=int(cluster.gpus_per_node),
                nodes=int(cluster.nodes),
                setup_commands=setup_commands,
                retries=args.retries,
            )
            jobs.append(executor.submit(job))

    for j in jobs:
        print(j.job_id)


def _run_local(workdir, cluster, jobs_info, sweep_name):
    gpus = int(cluster.get("gpus_per_node", 1))
    for run_id, cfg_path in jobs_info:
        print(f"Running {run_id} ({cfg_path})")
        env = os.environ.copy()
        env.setdefault("WANDB_RUN_GROUP", sweep_name)

        port = find_free_port()
        cmd = [
            "torchrun",
            "--nproc-per-node", str(gpus),
            "--rdzv-backend", "c10d",
            "--rdzv-endpoint", f"localhost:{port}",
            "-m", "jepa.train",
            "--config", str(cfg_path),
        ]

        proc = subprocess.run(cmd, cwd=workdir, env=env)
        if proc.returncode != 0:
            print(f"Run {run_id} failed with exit code {proc.returncode}")
            sys.exit(proc.returncode)


if __name__ == "__main__":
    sys.exit(main())
