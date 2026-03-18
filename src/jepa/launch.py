import argparse
import itertools
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import submitit
import yaml

# ---------- small utils ----------


def yaml_load(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def yaml_dump(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    ks = dotted_key.split(".")
    d = cfg
    for k in ks[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[ks[-1]] = value


def parse_scalar(s: str):
    return yaml.safe_load(s)


def _split_top_level_commas(s: str):
    tokens, buf, depth, stack = [], [], 0, []
    matching = {"(": ")", "[": "]", "{": "}"}
    inv = {v: k for k, v in matching.items()}
    for ch in s:
        if ch in matching:
            depth += 1
            stack.append(matching[ch])
        elif ch in inv:
            if stack and stack[-1] == ch:
                stack.pop()
                depth -= 1
        if ch == "," and depth == 0:
            tok = "".join(buf).strip()
            if tok:
                tokens.append(tok)
            buf = []
        else:
            buf.append(ch)
    tok = "".join(buf).strip()
    if tok:
        tokens.append(tok)
    return tokens


def _expand_value_token(tok: str):
    tok = tok.strip()
    if tok.startswith("log2[") and tok.endswith("]"):
        inner = tok[5:-1]
        start_s, steps_s = _split_top_level_commas(inner)
        a = float(parse_scalar(start_s))
        steps = int(parse_scalar(steps_s))
        if a <= 0:
            raise ValueError("log2 range requires positive start.")
        if steps <= 0:
            raise ValueError("steps must be >= 1.")
        # divide by 2 each step
        return [a / (2.0**i) for i in range(steps)]
    return [parse_scalar(tok)]


def parse_grid_arg(grid_items: List[str]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {}
    for item in grid_items:
        k, v = item.split("=", 1)
        tokens = _split_top_level_commas(v)
        vals: List[Any] = []
        for t in tokens:
            vals.extend(_expand_value_token(t))
        out[k] = vals
    return out


def cartesian(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    if not vals:
        return [{}]
    return [dict(zip(keys, prod)) for prod in itertools.product(*vals)]


def short_hash(d: Dict[str, Any]) -> str:
    blob = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
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
        base_config: Path,
        sweep_name: str,
        overrides: Dict[str, Any],
        seed: int,
        gpus_per_node: int,
        nodes: int,
        retries: int,
    ):
        self.workdir = workdir
        self.base_config = base_config
        self.sweep_name = sweep_name
        self.overrides = dict(overrides)
        self.seed = int(seed)
        self.gpus_per_node = int(gpus_per_node)
        self.nodes = int(nodes)
        self.retries = int(retries)

        run_key = dict(overrides)
        run_key["seed"] = self.seed
        self.run_id = short_hash(run_key)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cfg_dir = self.workdir / "temp_configs" / self.sweep_name
        self.cfg_path = (
            self.cfg_dir / f"{self.base_config.stem}_{self.run_id}_{ts}.yaml"
        )

    def _make_config(self) -> Path:
        cfg = yaml_load(self.base_config)
        set_nested(cfg, "training.seed", self.seed)
        for k, v in self.overrides.items():
            set_nested(cfg, k, v)
        yaml_dump(cfg, self.cfg_path)
        return self.cfg_path

    def _pre_shell(self) -> str:
        return "; ".join(
            [
                "ls data/kinetics-dataset/k700-2020-processed-tars/*.tar | xargs -n1 -P8 -I{} tar -xf {} -C /scratch/local/",
                "module load gcc/system",
                "module load Mambaforge/23.3.1-1-hpc1-bdist",
                "mamba activate neodev",
            ]
        )

    def _torch_cmd(self, cfg_path: Path, master_port: int) -> str:
        # One task per GPU
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
                str(cfg_path),
            ]
        )

    def __call__(self) -> None:
        cfg_path = self._make_config()
        env = os.environ.copy()
        # Group runs in W&B by sweep name; no resume logic here.
        env.setdefault("WANDB_RUN_GROUP", self.sweep_name)
        env.setdefault("OMP_NUM_THREADS", "16")

        tries = self.retries + 1
        for _ in range(tries):
            port = find_free_port()
            cmd = f"{self._pre_shell()} ; {self._torch_cmd(cfg_path, port)}"
            proc = subprocess.run(["bash", "-lc", cmd], cwd=self.workdir, env=env)
            if proc.returncode == 0:
                return
        raise RuntimeError("Training failed after retries")

    def checkpoint(self) -> submitit.helpers.DelayedSubmission:
        # Requeue fresh on preemption/timeout. New port chosen in __call__.
        return submitit.helpers.DelayedSubmission(
            TrainJob(
                workdir=self.workdir,
                base_config=self.base_config,
                sweep_name=self.sweep_name,
                overrides=self.overrides,
                seed=self.seed,
                gpus_per_node=self.gpus_per_node,
                nodes=self.nodes,
                retries=self.retries,
            )
        )


# ---------- launcher ----------


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-name", required=True)
    p.add_argument("--base-config", required=True, type=Path)
    p.add_argument(
        "--grid",
        action="append",
        default=[],
        help="key=v1,v2 or key=lin(start,end,count) or key=log2(start,steps) (repeatable)",
    )
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed-offset", type=int, default=0)

    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument("--timeout-min", type=int, default=15 * 60)
    p.add_argument(
        "-C", dest="constraint", choices=["thin", "fat", "none"], default="thin"
    )

    p.add_argument("--array-parallelism", type=int, default=64)
    p.add_argument("--retries", type=int, default=0)
    p.add_argument("--workdir", type=Path, default=Path("."))

    args = p.parse_args(argv)
    workdir = args.workdir.resolve()

    grid = parse_grid_arg(args.grid)
    combos = cartesian(grid)

    seeds = [args.seed_offset + i for i in range(args.seeds)]
    run_specs = [(ov, sd) for ov, sd in itertools.product(combos, seeds)]

    logs_root = workdir / "experiments" / args.sweep_name / "slurm_logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(logs_root))

    slurm_params = dict(
        name=args.sweep_name,
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
        for overrides, seed in run_specs:
            job = TrainJob(
                workdir=workdir,
                base_config=args.base_config.resolve(),
                sweep_name=args.sweep_name,
                overrides=overrides,
                seed=seed,
                gpus_per_node=args.gpus_per_node,
                nodes=args.nodes,
                retries=args.retries,
            )
            jobs.append(executor.submit(job))

    for j in jobs:
        print(j.job_id)


if __name__ == "__main__":
    sys.exit(main())
