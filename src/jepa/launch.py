import argparse
import datetime
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


def prepend_pythonpath(env: dict[str, str], worktree: Path | None) -> None:
    """Force `import jepa` to resolve to the worktree's snapshot.

    Prepending to PYTHONPATH wins over an editable install at the repo root.
    Leaves cwd at the repo root so data/ paths and experiments/ writes stay sane.
    """
    if worktree is None:
        return
    src = str(Path(worktree) / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}:{existing}" if existing else src


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
        worktree: Path | None = None,
        wandb_run_group: str | None = None,
    ):
        self.workdir = workdir
        self.cfg_path = cfg_path
        self.sweep_name = sweep_name
        self.run_id = run_id
        self.gpus_per_node = int(gpus_per_node)
        self.nodes = int(nodes)
        self.setup_commands = list(setup_commands)
        self.retries = int(retries)
        self.worktree = Path(worktree) if worktree is not None else None
        self.wandb_run_group = wandb_run_group or sweep_name

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
        env.setdefault("WANDB_RUN_GROUP", self.wandb_run_group)
        env.setdefault("OMP_NUM_THREADS", "16")
        prepend_pythonpath(env, self.worktree)

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
                worktree=self.worktree,
                wandb_run_group=self.wandb_run_group,
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


# ---------- launch log ----------


def git_info(workdir: Path) -> dict | None:
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=workdir, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=workdir, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"hash": h, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ---------- snapshot (tag + worktree) ----------


def _git(workdir: Path, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git"] + args, cwd=workdir, capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result


def is_dirty(workdir: Path) -> bool:
    return bool(_git(workdir, ["status", "--porcelain"]).stdout.strip())


def tag_exists(workdir: Path, tag: str) -> bool:
    return _git(
        workdir, ["rev-parse", "--verify", "--quiet", f"refs/tags/{tag}"], check=False
    ).returncode == 0


def create_tag(workdir: Path, tag: str, message: str) -> None:
    _git(workdir, ["tag", "-a", tag, "-m", message])


def delete_tag(workdir: Path, tag: str) -> None:
    _git(workdir, ["tag", "-d", tag], check=False)


def create_worktree(workdir: Path, path: Path, ref: str) -> None:
    _git(workdir, ["worktree", "add", str(path), ref])


def remove_worktree(workdir: Path, path: Path) -> None:
    _git(workdir, ["worktree", "remove", "--force", str(path)], check=False)


def sweep_tag(study: str, sweep: str) -> str:
    return f"sweep/{study}/{sweep}"


def tag_message(study: str, sweep: str, launch_record: dict) -> str:
    return (
        f"study: {study}\n"
        f"sweep: {sweep}\n"
        f"date: {launch_record['timestamp']}\n"
        f"runs: {launch_record['n_runs']} ({launch_record['n_combos']} configs × {launch_record['n_seeds']} seeds)\n"
        f"overrides: {' '.join(launch_record['cli_overrides'])}\n"
    )


def sweep_dir_for(workdir: Path, study: str | None, sweep: str) -> Path:
    """Resolve the sweep's artifact directory.

    Non-smoke launches require `study` and live under `experiments/<study>/<sweep>/`.
    Smoke launches pass `study=None` and use the flat `experiments/<sweep>/` layout.
    """
    if study is None:
        return workdir / "experiments" / sweep
    return workdir / "experiments" / study / sweep


def write_launch_record(
    record: dict, workdir: Path, study: str | None, sweep_name: str
) -> None:
    central = workdir / "experiments" / "launches.jsonl"
    per_sweep = sweep_dir_for(workdir, study, sweep_name) / "launches.jsonl"
    central.parent.mkdir(parents=True, exist_ok=True)
    per_sweep.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str) + "\n"
    for path in (central, per_sweep):
        with open(path, "a") as f:
            f.write(line)


# ---------- experiment notes (Obsidian-style) ----------


_TOP_JOURNAL_HEADER = (
    "# Experiment Journal — Studies\n\n"
    "Reverse-chronological log of studies. Click a `[[study]]` to open its README.\n\n"
)


def _study_journal_header(study: str) -> str:
    return (
        f"# Study `{study}` — Sweep Journal\n\n"
        f"Reverse-chronological log of sweeps within this study. "
        f"Click a `[[sweep]]` to open its README.\n\n"
    )


def _sweep_readme_template(study: str, sweep_name: str, launch_record: dict) -> str:
    cli = " ".join(launch_record["cli_overrides"])
    date = launch_record["timestamp"][:10]
    cluster_kind = "slurm" if launch_record["slurm"] else "local"
    return f"""---
study: {study}
sweep: {sweep_name}
date: {date}
status: running
tags: []
---

# {sweep_name}

Part of study [[{study}]].

## Hypothesis
_What are we testing? What do we expect to learn?_

## Setup
- Launched: {launch_record["timestamp"]}
- Cluster: {cluster_kind}
- Runs: {launch_record["n_runs"]} ({launch_record["n_combos"]} configs × {launch_record["n_seeds"]} seeds)
- Tag: `{launch_record["git"]["tag"]}`
- Worktree: `{launch_record["worktree"]}`
- Overrides: `{cli}`

## Runs
Configs in `configs/`, status in `status/`, metrics in `metrics/`, results in `results/`.
See `launches.jsonl` for the full launch record (run_id ↔ slurm_job_id mapping included).

## Results
_Fill in after analyzing._

## Conclusion
_Fill in after analyzing — what was learned, what to try next._
"""


def _study_readme_template(study: str, launch_record: dict) -> str:
    date = launch_record["timestamp"][:10]
    return f"""---
study: {study}
opened: {date}
status: in-progress
tags: []
---

# Study: {study}

## Goal
_What research question is this study answering? What does success look like
(pass criterion, anti-overfit criterion, baseline to beat)?_

## Decisions log
_Append chronologically as evidence accumulates. Each entry: date — observation
or decision. Keep terse._

- {date} — Study opened with first sweep.

## Sweep summary
_See `journal.md` in this directory for the chronological list. Update this
section with a curated summary table once enough sweeps have run._

## Conclusion
_Fill in once the study reaches a settled answer: final config + rationale +
caveats. Link to the sweeps that produced the evidence._
"""


def scaffold_study_and_sweep(
    workdir: Path, study: str, sweep_name: str, launch_record: dict
) -> None:
    """Create / update the four notebook artifacts for a non-smoke launch.

    1. Per-sweep README under experiments/<study>/<sweep>/README.md (idempotent).
    2. Per-study journal entry prepended to experiments/<study>/journal.md.
    3. Per-study README at experiments/<study>/README.md (only on first sweep).
    4. Top-level journal entry prepended to experiments/journal.md (only on first sweep).
    """
    study_dir = workdir / "experiments" / study
    study_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir = study_dir / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    sweep_readme = sweep_dir / "README.md"
    if not sweep_readme.exists():
        sweep_readme.write_text(_sweep_readme_template(study, sweep_name, launch_record))

    study_readme = study_dir / "README.md"
    is_first_sweep = not study_readme.exists()
    if is_first_sweep:
        study_readme.write_text(_study_readme_template(study, launch_record))

    cli = " ".join(launch_record["cli_overrides"])
    date = launch_record["timestamp"][:10]
    sweep_entry = f"- {date} · [[{sweep_name}]] · {launch_record['n_runs']} runs · `{cli}`\n"
    study_journal = study_dir / "journal.md"
    header = _study_journal_header(study)
    if not study_journal.exists():
        study_journal.write_text(header)
    text = study_journal.read_text()
    if text.startswith(header):
        study_journal.write_text(header + sweep_entry + text[len(header):])
    else:
        study_journal.write_text(text + sweep_entry)

    if is_first_sweep:
        top_journal = workdir / "experiments" / "journal.md"
        if not top_journal.exists():
            top_journal.write_text(_TOP_JOURNAL_HEADER)
        study_entry = f"- {date} · [[{study}]] · opened\n"
        text = top_journal.read_text()
        if text.startswith(_TOP_JOURNAL_HEADER):
            top_journal.write_text(
                _TOP_JOURNAL_HEADER + study_entry + text[len(_TOP_JOURNAL_HEADER):]
            )
        else:
            top_journal.write_text(text + study_entry)


# ---------- launcher ----------


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Launch training jobs (locally or via SLURM).",
        epilog="Positional args are Hydra overrides, e.g.: +experiment=toy_craftax cluster=hopper training.lr=1e-3",
    )
    p.add_argument(
        "--study",
        default=None,
        help=(
            "Study (research goal) this sweep belongs to. Required for non-smoke "
            "launches. Use `tmp` for one-off iteration that doesn't belong to a "
            "real study. Smoke launches ignore this flag."
        ),
    )
    p.add_argument("--sweep-name", required=True)
    p.add_argument("--seeds", type=int, default=None, help="Override sweep.seeds")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--retries", type=int, default=0)
    p.add_argument("--workdir", type=Path, default=Path("."))
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Quick smoke test: small total_steps, single seed, wandb disabled, "
            "capped val. Flat layout under experiments/<sweep>/, no tag, no worktree, "
            "no scaffolding. Wrap in `interactive --gpus 1` rather than running "
            "on the login node."
        ),
    )
    p.add_argument("overrides", nargs="*", help="Hydra config overrides")

    args = p.parse_args(argv)
    workdir = args.workdir.resolve()
    config_dir = str((workdir / "configs").resolve())

    if not args.smoke and args.study is None:
        sys.exit("--study is required for non-smoke launches (use `--study tmp` for one-offs).")

    if args.smoke:
        smoke_overrides = [
            "training.total_steps=300",
            "training.val_fraction=0.5",
            "training.ckpt_fraction=1.0",
            "++training.wandb=disabled",
            "++training.val_max_steps=5",
            "++training.final_val_max_steps=5",
        ]
        # Smoke overrides go last so they win over user CLI overrides
        # (later overrides win in Hydra). Cluster is intentionally NOT forced —
        # pick a non-slurm cluster appropriate for the host (e.g. `cluster=local`
        # on a workstation, `cluster=berzelius_interactive` inside an
        # `interactive --gpus 1` session).
        args.overrides = list(args.overrides) + smoke_overrides
        args.seeds = 1
        args.retries = 0

    # Compose config via Hydra
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose("config", overrides=args.overrides)

    cluster = cfg.cluster
    use_slurm = bool(cluster.get("slurm", False))

    if args.smoke and use_slurm:
        sys.exit(
            "--smoke must run synchronously; got cluster.slurm=true. "
            "Pass `cluster=local` or `cluster=berzelius_interactive`."
        )

    study = None if args.smoke else args.study
    sweep_dir = sweep_dir_for(workdir, study, args.sweep_name)

    # Pre-flight for non-smoke: clean tree + no collisions with prior launches.
    tag = None
    worktree_path: Path | None = None
    if not args.smoke:
        tag = sweep_tag(study, args.sweep_name)
        worktree_path = sweep_dir / "code"
        if is_dirty(workdir):
            sys.exit(
                "Refusing to launch: working tree is dirty. Commit (or stash) first, "
                "or use --smoke for dirty iteration."
            )
        if tag_exists(workdir, tag):
            sys.exit(
                f"Refusing to launch: git tag `{tag}` already exists. "
                f"Pick a different sweep name, or to redo this sweep run:\n"
                f"  git tag -d {tag} && git worktree remove {worktree_path} && rm -rf {sweep_dir}"
            )
        if worktree_path.exists():
            sys.exit(f"Refusing to launch: worktree path `{worktree_path}` already exists.")
        if (sweep_dir / "launches.jsonl").exists():
            sys.exit(
                f"Refusing to launch: `{sweep_dir}/launches.jsonl` already exists. "
                f"This sweep name has artifacts from a prior launch — pick a different name."
            )

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
    cfg_dir = sweep_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Build and save all run configs
    cli_overrides = tuple(sorted(args.overrides))
    jobs_info = []
    for overrides, seed in run_specs:
        run_key = dict(overrides, seed=seed, _cli=cli_overrides)
        run_id = short_hash(run_key)
        run_cfg = build_run_config(cfg, overrides, seed)
        cfg_path = save_run_config(run_cfg, cfg_dir, run_id)
        jobs_info.append((run_id, cfg_path))

    launch_record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "study": study,
        "sweep_name": args.sweep_name,
        "argv": sys.argv,
        "cli_overrides": list(args.overrides),
        "cwd": str(workdir),
        "hostname": socket.gethostname(),
        "slurm": use_slurm,
        "cluster": OmegaConf.to_container(cluster, resolve=True),
        "n_combos": len(combos),
        "n_seeds": len(seeds),
        "n_runs": len(jobs_info),
        "run_ids": [rid for rid, _ in jobs_info],
        "git": git_info(workdir),
        "worktree": str(worktree_path) if worktree_path else None,
    }
    if tag is not None:
        launch_record["git"] = {**(launch_record["git"] or {}), "tag": tag}

    # Snapshot creation: tag first, then worktree. Tear both down on any
    # failure that happens before submit returns — atomic launch.
    submitted = False
    try:
        if not args.smoke:
            create_tag(workdir, tag, tag_message(study, args.sweep_name, launch_record))
            create_worktree(workdir, worktree_path, tag)

        wandb_group = (
            f"{study}/{args.sweep_name}" if study is not None else args.sweep_name
        )

        if use_slurm:
            slurm_job_ids = _submit_slurm(
                args, workdir, cluster, jobs_info, sweep_dir,
                worktree=worktree_path, wandb_run_group=wandb_group,
            )
            submitted = True
            launch_record["slurm_job_ids"] = {
                rid: jid for (rid, _), jid in zip(jobs_info, slurm_job_ids)
            }
            write_launch_record(launch_record, workdir, study, args.sweep_name)
            if not args.smoke:
                scaffold_study_and_sweep(workdir, study, args.sweep_name, launch_record)
        else:
            write_launch_record(launch_record, workdir, study, args.sweep_name)
            if not args.smoke:
                scaffold_study_and_sweep(workdir, study, args.sweep_name, launch_record)
            submitted = True  # local run starts now; worktree must persist
            _run_local(
                workdir, cluster, jobs_info, args.sweep_name,
                worktree=worktree_path, wandb_run_group=wandb_group,
            )
    except BaseException:
        if not submitted and not args.smoke:
            if worktree_path is not None and worktree_path.exists():
                remove_worktree(workdir, worktree_path)
            if tag is not None and tag_exists(workdir, tag):
                delete_tag(workdir, tag)
        raise


def _submit_slurm(
    args, workdir, cluster, jobs_info, sweep_dir, worktree=None, wandb_run_group=None
):
    logs_root = sweep_dir / "slurm_logs"
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
    group = wandb_run_group or args.sweep_name

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
                worktree=worktree,
                wandb_run_group=group,
            )
            jobs.append(executor.submit(job))

    for j in jobs:
        print(j.job_id)

    return [j.job_id for j in jobs]


def _run_local(
    workdir, cluster, jobs_info, sweep_name, worktree=None, wandb_run_group=None
):
    gpus = int(cluster.get("gpus_per_node", 1))
    setup_commands = list(cluster.get("setup_commands", []) or [])
    pre_shell = " ; ".join(setup_commands) if setup_commands else ""
    group = wandb_run_group or sweep_name

    for run_id, cfg_path in jobs_info:
        print(f"Running {run_id} ({cfg_path})")
        env = os.environ.copy()
        env.setdefault("WANDB_RUN_GROUP", group)
        prepend_pythonpath(env, worktree)

        port = find_free_port()
        torch_cmd = (
            f"torchrun --nproc-per-node {gpus} "
            f"--rdzv-backend c10d --rdzv-endpoint localhost:{port} "
            f"-m jepa.train --config {cfg_path}"
        )
        cmd = f"{pre_shell} ; {torch_cmd}" if pre_shell else torch_cmd

        proc = subprocess.run(["bash", "-lc", cmd], cwd=workdir, env=env)
        if proc.returncode != 0:
            print(f"Run {run_id} failed with exit code {proc.returncode}")
            sys.exit(proc.returncode)


if __name__ == "__main__":
    sys.exit(main())
