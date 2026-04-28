import argparse
import datetime
import itertools
import json
import os
import re
import shlex
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import submitit
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


# ---------- conventions ----------

RESEARCH_RESULTS_DIRNAME = "research_results"
RESEARCH_DIRNAME = "research"
TAG_PREFIX = "experiment"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def research_results_dir(workdir: Path) -> Path:
    return workdir / RESEARCH_RESULTS_DIRNAME


def find_main_repo(cwd: Path) -> Path:
    """Resolve the main worktree's root from any cwd inside the repo.

    Used so the launcher's artifact paths (research_results/, data/) and the
    SLURM workers' cwd are always the main repo, even when the user invoked
    `jepa.launch` from inside a dev worktree under research/<study>/<exp>/.
    git -C <cwd> rev-parse --git-common-dir resolves to the shared .git/
    directory; its parent is the main worktree's root.
    """
    try:
        common_dir = subprocess.check_output(
            ["git", "-C", str(cwd), "rev-parse", "--git-common-dir"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        sys.exit(f"Not inside a git repository: {cwd}")
    return (Path(cwd) / common_dir).resolve().parent


def find_caller_repo(cwd: Path) -> Path:
    """Resolve the caller's *own* worktree root (not the main repo's).

    Used for Hydra config discovery. Configs for an experiment live on the
    dev worktree's branch; the launcher must read them from there, not from
    main. `git rev-parse --show-toplevel` returns the current worktree's
    root regardless of subdirectory level.
    """
    try:
        toplevel = subprocess.check_output(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        sys.exit(f"Not inside a git repository: {cwd}")
    return Path(toplevel).resolve()


def validate_name(kind: str, value: str) -> None:
    if not NAME_RE.match(value):
        sys.exit(
            f"--{kind} {value!r} must match {NAME_RE.pattern} "
            f"(lowercase letters, digits, hyphens; must start with a letter or digit)."
        )


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
    Leaves cwd at the repo root so data/ paths and research_results/ writes stay sane.
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
        experiment: str,
        run_id: str,
        gpus_per_node: int,
        nodes: int,
        setup_commands: list[str],
        retries: int,
        worktree: Path | None = None,
        wandb_run_group: str | None = None,
        code_hash: str | None = None,
    ):
        self.workdir = workdir
        self.cfg_path = cfg_path
        self.experiment = experiment
        self.run_id = run_id
        self.gpus_per_node = int(gpus_per_node)
        self.nodes = int(nodes)
        self.setup_commands = list(setup_commands)
        self.retries = int(retries)
        self.worktree = Path(worktree) if worktree is not None else None
        self.wandb_run_group = wandb_run_group or experiment
        self.code_hash = code_hash

    def _pre_shell(self) -> str:
        if not self.setup_commands:
            return "true"
        return " ; ".join(self.setup_commands)

    def torch_argv(self, master_port: int) -> list[str]:
        return [
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

    def __call__(self) -> None:
        env = os.environ.copy()
        env.setdefault("WANDB_RUN_GROUP", self.wandb_run_group)
        env.setdefault("SOURCE_WORKDIR", str(self.workdir))
        env.setdefault("OMP_NUM_THREADS", "16")
        if self.code_hash is not None:
            env["SOURCE_GIT_HASH"] = self.code_hash
        prepend_pythonpath(env, self.worktree)

        tries = self.retries + 1
        for _ in range(tries):
            port = find_free_port()
            cmd = f"{self._pre_shell()} ; {shlex.join(self.torch_argv(port))}"
            proc = subprocess.run(["bash", "-lc", cmd], cwd=self.workdir, env=env)
            if proc.returncode == 0:
                return
        raise RuntimeError("Training failed after retries")

    def checkpoint(self) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(
            TrainJob(
                workdir=self.workdir,
                cfg_path=self.cfg_path,
                experiment=self.experiment,
                run_id=self.run_id,
                gpus_per_node=self.gpus_per_node,
                nodes=self.nodes,
                setup_commands=self.setup_commands,
                retries=self.retries,
                worktree=self.worktree,
                wandb_run_group=self.wandb_run_group,
                code_hash=self.code_hash,
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


def run_id_material(run_cfg: DictConfig) -> dict:
    """Plain-dict view of the run config for hashing into run_id.

    Strips fields that are observability/wiring rather than part of the run's
    training identity. Equivalent CLI invocations that resolve to the same
    config collapse to the same run_id; YAML edits that genuinely change the
    config produce a new run_id.
    """
    cfg = OmegaConf.to_container(run_cfg, resolve=True)
    training = cfg.get("training") if isinstance(cfg, dict) else None
    if isinstance(training, dict):
        training.pop("project", None)
    return cfg


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


def tag_points_to_head(workdir: Path, tag: str) -> bool:
    """True if `tag` resolves to the same commit as the current HEAD."""
    # ^{} dereferences annotated tags to the underlying commit.
    tag_sha = _git(workdir, ["rev-parse", f"refs/tags/{tag}^{{}}"], check=False).stdout.strip()
    head_sha = _git(workdir, ["rev-parse", "HEAD"]).stdout.strip()
    return bool(tag_sha) and tag_sha == head_sha


def tag_is_ancestor_of_head(workdir: Path, tag: str) -> bool:
    """True if the tag's commit is an ancestor of HEAD (or equal)."""
    return _git(
        workdir,
        ["merge-base", "--is-ancestor", f"refs/tags/{tag}^{{}}", "HEAD"],
        check=False,
    ).returncode == 0


def force_update_tag(workdir: Path, tag: str, message: str) -> None:
    """Move an annotated tag to the current HEAD."""
    _git(workdir, ["tag", "-f", "-a", tag, "-m", message])


def existing_run_ids(experiment_dir: Path) -> set[str]:
    """Collect all run_ids from prior launches in this experiment."""
    launches = experiment_dir / "launches.jsonl"
    if not launches.exists():
        return set()
    out: set[str] = set()
    with open(launches) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.update(rec.get("run_ids", []))
    return out


def create_tag(workdir: Path, tag: str, message: str) -> None:
    _git(workdir, ["tag", "-a", tag, "-m", message])


def delete_tag(workdir: Path, tag: str) -> None:
    _git(workdir, ["tag", "-d", tag], check=False)


def create_worktree(workdir: Path, path: Path, ref: str) -> None:
    _git(workdir, ["worktree", "add", str(path), ref])


def remove_worktree(workdir: Path, path: Path) -> None:
    _git(workdir, ["worktree", "remove", "--force", str(path)], check=False)


def experiment_tag(study: str, experiment: str) -> str:
    return f"{TAG_PREFIX}/{study}/{experiment}"


def tag_message(study: str, experiment: str, launch_record: dict) -> str:
    return (
        f"study: {study}\n"
        f"experiment: {experiment}\n"
        f"date: {launch_record['timestamp']}\n"
        f"runs: {launch_record['n_runs']} ({launch_record['n_combos']} configs × {launch_record['n_seeds']} seeds)\n"
        f"overrides: {' '.join(launch_record['cli_overrides'])}\n"
    )


def experiment_dir_for(workdir: Path, study: str | None, experiment: str) -> Path:
    """Resolve the experiment's artifact directory.

    Non-smoke launches require `study` and live under
    `research_results/<study>/<experiment>/`. Smoke launches pass `study=None`
    and use the flat `research_results/<experiment>/` layout.
    """
    base = research_results_dir(workdir)
    if study is None:
        return base / experiment
    return base / study / experiment


def write_launch_record(
    record: dict, workdir: Path, study: str | None, experiment: str
) -> None:
    central = research_results_dir(workdir) / "launches.jsonl"
    per_experiment = experiment_dir_for(workdir, study, experiment) / "launches.jsonl"
    central.parent.mkdir(parents=True, exist_ok=True)
    per_experiment.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str) + "\n"
    for path in (central, per_experiment):
        with open(path, "a") as f:
            f.write(line)


# ---------- experiment notes (Obsidian-style) ----------


_TOP_JOURNAL_HEADER = (
    "# Research Journal — Studies\n\n"
    "Reverse-chronological log of studies. Click a `[[study]]` to open its README.\n\n"
)


def _study_journal_header(study: str) -> str:
    return (
        f"# Study `{study}` — Experiment Journal\n\n"
        f"Reverse-chronological log of experiments within this study. "
        f"Click an `[[experiment]]` to open its README.\n\n"
    )


def _experiment_readme_template(study: str, experiment: str, launch_record: dict) -> str:
    cli = " ".join(launch_record["cli_overrides"])
    date = launch_record["timestamp"][:10]
    cluster_kind = "slurm" if launch_record["slurm"] else "local"
    return f"""---
study: {study}
experiment: {experiment}
date: {date}
status: running
tags: []
---

# {experiment}

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


def scaffold_experiment(
    workdir: Path, study: str, experiment: str, launch_record: dict
) -> None:
    """Create / update experiment-level notebook artifacts for a non-smoke launch.

    Study-level docs (README.md, journal.md, top-level journal entry) are the
    /study skill's responsibility — the launcher refuses earlier if the study
    README is missing, so they're guaranteed to exist by the time we get here.
    The launcher adds:

    1. Per-experiment README at research_results/<study>/<experiment>/README.md (idempotent).
    2. Per-experiment entry prepended to research_results/<study>/journal.md.
    """
    study_dir = research_results_dir(workdir) / study
    experiment_dir = study_dir / experiment
    experiment_dir.mkdir(parents=True, exist_ok=True)

    experiment_readme = experiment_dir / "README.md"
    if not experiment_readme.exists():
        experiment_readme.write_text(_experiment_readme_template(study, experiment, launch_record))

    cli = " ".join(launch_record["cli_overrides"])
    date = launch_record["timestamp"][:10]
    experiment_entry = f"- {date} · [[{experiment}]] · {launch_record['n_runs']} runs · `{cli}`\n"
    study_journal = study_dir / "journal.md"
    header = _study_journal_header(study)
    if not study_journal.exists():
        study_journal.write_text(header)
    text = study_journal.read_text()
    if text.startswith(header):
        study_journal.write_text(header + experiment_entry + text[len(header):])
    else:
        study_journal.write_text(text + experiment_entry)


def require_study_readme(workdir: Path, study: str) -> None:
    """Refuse if the study README is missing — `/study` skill must run first."""
    readme = research_results_dir(workdir) / study / "README.md"
    if not readme.exists():
        sys.exit(
            f"Refusing to launch: study {study!r} has no README at {readme}. "
            f"Open the study first: invoke the /study skill (it asks the framing "
            f"questions, suggests a first experiment, and writes the README)."
        )


# ---------- launcher ----------


@dataclass(frozen=True)
class LaunchPaths:
    cwd: Path
    workdir: Path
    config_dir: str
    caller_repo: Path


@dataclass(frozen=True)
class SnapshotMode:
    tag: str | None = None
    worktree_path: Path | None = None
    extending: bool = False
    ffwd_tag: bool = False


@dataclass(frozen=True)
class RunPlan:
    combos: list[dict[str, Any]]
    seeds: list[int]
    jobs_info: list[tuple[str, Path]]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch training jobs (locally or via SLURM).",
        epilog="Positional args are Hydra overrides, e.g.: +train=toy_craftax cluster=hopper training.lr=1e-3",
    )
    parser.add_argument(
        "--study",
        default=None,
        help=(
            "Study (research goal) this experiment belongs to. Required for "
            "non-smoke launches. Use `--study tmp` for one-off iteration that "
            "doesn't belong to a real study. Smoke launches ignore this flag."
        ),
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment name within the study. One launch == one experiment.",
    )
    parser.add_argument("--seeds", type=int, default=None, help="Override sweep.seeds")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Quick smoke test: small total_steps, single seed, wandb disabled, "
            "capped val. Flat layout under research_results/<experiment>/, no tag, "
            "no worktree, no scaffolding. Wrap in `interactive --gpus 1` rather "
            "than running on the login node."
        ),
    )
    parser.add_argument("overrides", nargs="*", help="Hydra config overrides")
    return parser.parse_args(argv)


def resolve_launch_paths(args: argparse.Namespace) -> LaunchPaths:
    cwd = Path.cwd().resolve()
    # workdir is the main repo's root — used for artifact paths and SLURM
    # workers' cwd. Auto-resolved from cwd when --workdir is the default,
    # so the launcher can be invoked from inside a dev worktree under
    # research/<study>/<experiment>/ and still write artifacts to the main
    # repo's research_results/ tree.
    if args.workdir == Path("."):
        workdir = find_main_repo(cwd)
    else:
        workdir = args.workdir.resolve()
    # Configs come from the *caller's* worktree, not main — experiment-local
    # configs (e.g. new train presets) live on the dev branch and shouldn't
    # need to land on main to be picked up. Caller worktree is also the
    # PYTHONPATH source for `--smoke` runs, so smokes test the dev branch's
    # code rather than main's editable install.
    caller_repo = find_caller_repo(cwd).resolve()
    config_dir = str((caller_repo / "configs").resolve())
    return LaunchPaths(
        cwd=cwd, workdir=workdir, config_dir=config_dir, caller_repo=caller_repo
    )


def validate_launch_request(args: argparse.Namespace) -> None:
    if not args.smoke and args.study is None:
        sys.exit("--study is required for non-smoke launches (use `--study tmp` for one-offs).")

    validate_name("experiment", args.experiment)
    if args.study is not None:
        validate_name("study", args.study)


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    smoke_overrides = [
        "training.total_steps=30",
        "training.val_fraction=0.5",
        "training.ckpt_fraction=1.0",
        "++training.wandb=disabled",
        "++training.val_max_steps=5",
        "++training.final_val_max_steps=5",
    ]
    # Smoke overrides go last so they win over user CLI overrides (later
    # overrides win in Hydra). Cluster is intentionally NOT forced — pick a
    # non-slurm cluster appropriate for the host.
    args.overrides = list(args.overrides) + smoke_overrides
    args.seeds = 1
    args.retries = 0


def compose_launch_config(config_dir: str, overrides: list[str]) -> DictConfig:
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose("config", overrides=overrides)


def resolve_study(args: argparse.Namespace) -> str | None:
    return None if args.smoke else args.study


def require_sync_smoke(args: argparse.Namespace, use_slurm: bool) -> None:
    if args.smoke and use_slurm:
        sys.exit(
            "--smoke must run synchronously; got cluster.slurm=true. "
            "Pass `cluster=local` or `cluster=berzelius_interactive`."
        )


def resolve_snapshot_mode(
    args: argparse.Namespace,
    cwd: Path,
    workdir: Path,
    study: str | None,
    experiment_dir: Path,
) -> SnapshotMode:
    # Pre-flight for non-smoke: study must already exist (framed via /study),
    # working tree clean, no run_id collisions with prior launches.
    #
    # Re-launching into an existing experiment is supported as an *extension*:
    # additional runs (e.g. a wider sweep grid) are submitted into the same
    # experiment dir + same wandb group. Three cases for the tag:
    #   - Tag at HEAD: reuse tag + snapshot as-is.
    #   - Tag is an ancestor of HEAD: fast-forward the tag and recreate the
    #     snapshot at HEAD. The original commit is still in git history (the
    #     dev branch points past it), and per-launch git.hash in launches.jsonl
    #     records what each run actually ran.
    #   - Tag is on a divergent branch: refuse (probably wants a new experiment).
    if args.smoke:
        return SnapshotMode()
    if study is None:
        raise ValueError("non-smoke launches require a study")

    require_study_readme(workdir, study)
    tag = experiment_tag(study, args.experiment)
    worktree_path = experiment_dir / "code"
    if is_dirty(cwd):
        sys.exit(
            "Refusing to launch: working tree is dirty. Commit (or stash) first, "
            "or use --smoke for dirty iteration."
        )

    if tag_exists(cwd, tag):
        if tag_points_to_head(cwd, tag):
            return SnapshotMode(tag=tag, worktree_path=worktree_path, extending=True)
        if tag_is_ancestor_of_head(cwd, tag):
            return SnapshotMode(
                tag=tag, worktree_path=worktree_path, extending=True, ffwd_tag=True
            )
        sys.exit(
            f"Refusing to launch: git tag `{tag}` exists but is on a "
            f"divergent branch (not an ancestor of HEAD). Either rebase "
            f"or use a new experiment name.\n"
            f"  To redo this experiment from scratch:\n"
            f"    git tag -d {tag} && "
            f"git worktree remove {worktree_path} && rm -rf {experiment_dir}"
        )
    if worktree_path.exists():
        sys.exit(
            f"Refusing to launch: worktree path `{worktree_path}` exists but "
            f"the experiment tag does not. Likely orphan state — clean it up:\n"
            f"  git worktree remove {worktree_path} && rm -rf {experiment_dir}"
        )
    return SnapshotMode(tag=tag, worktree_path=worktree_path)


def build_run_plan(args: argparse.Namespace, cfg: DictConfig, experiment_dir: Path) -> RunPlan:
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
    print(f"Experiment: {len(combos)} configs x {len(seeds)} seeds = {len(run_specs)} runs")

    # Prepare output dirs
    cfg_dir = experiment_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Build and save all run configs
    jobs_info = []
    for overrides, seed in run_specs:
        run_cfg = build_run_config(cfg, overrides, seed)
        run_id = short_hash(run_id_material(run_cfg))
        cfg_path = save_run_config(run_cfg, cfg_dir, run_id)
        jobs_info.append((run_id, cfg_path))
    return RunPlan(combos=combos, seeds=seeds, jobs_info=jobs_info)


def require_new_run_ids(
    args: argparse.Namespace, experiment_dir: Path, jobs_info: list[tuple[str, Path]]
) -> None:
    # When extending an existing experiment, refuse if any new run_id collides
    # with a prior one (deterministic hash → same config; we'd silently
    # overwrite status / metrics / results).
    if args.smoke:
        return
    prior_run_ids = existing_run_ids(experiment_dir)
    new_run_ids = [rid for rid, _ in jobs_info]
    collisions = sorted(set(new_run_ids) & prior_run_ids)
    if collisions:
        sys.exit(
            f"Refusing to launch: run_id collision with prior launches in "
            f"this experiment: {collisions}. Same (config, seed) → same hash. "
            f"Change the sweep grid (or the seed offset) so the new runs differ."
        )


def build_launch_record(
    args: argparse.Namespace,
    paths: LaunchPaths,
    cluster: DictConfig,
    use_slurm: bool,
    study: str | None,
    snapshot: SnapshotMode,
    run_plan: RunPlan,
) -> dict[str, Any]:
    launch_record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "study": study,
        "experiment": args.experiment,
        "argv": sys.argv,
        "cli_overrides": list(args.overrides),
        "cwd": str(paths.cwd),
        "workdir": str(paths.workdir),
        "hostname": socket.gethostname(),
        "slurm": use_slurm,
        "cluster": OmegaConf.to_container(cluster, resolve=True),
        "n_combos": len(run_plan.combos),
        "n_seeds": len(run_plan.seeds),
        "n_runs": len(run_plan.jobs_info),
        "run_ids": [rid for rid, _ in run_plan.jobs_info],
        "git": git_info(paths.cwd),
        "worktree": str(snapshot.worktree_path) if snapshot.worktree_path else None,
    }
    if snapshot.tag is not None:
        launch_record["git"] = {**(launch_record["git"] or {}), "tag": snapshot.tag}
    return launch_record


def prepare_snapshot(
    cwd: Path,
    snapshot: SnapshotMode,
    study: str | None,
    experiment: str,
    launch_record: dict[str, Any],
) -> None:
    # Snapshot creation: tag first, then worktree. Tear both down on any
    # failure that happens before submit returns — atomic launch. Three
    # branches:
    #   - Fresh launch: create tag and snapshot.
    #   - Extending, tag already at HEAD: reuse both, no-op.
    #   - Extending with HEAD ahead of tag: fast-forward the tag and
    #     recreate the snapshot at HEAD.
    if snapshot.tag is None:
        return
    if study is None or snapshot.worktree_path is None:
        raise ValueError("snapshot launches require a study and worktree path")

    if not snapshot.extending:
        create_tag(cwd, snapshot.tag, tag_message(study, experiment, launch_record))
        create_worktree(cwd, snapshot.worktree_path, snapshot.tag)
    elif snapshot.ffwd_tag:
        print(
            f"Extending {snapshot.tag}: fast-forwarding tag and recreating snapshot "
            f"(per-run git.hash in launches.jsonl preserves the original "
            f"runs' code identity)."
        )
        if snapshot.worktree_path.exists():
            remove_worktree(cwd, snapshot.worktree_path)
        force_update_tag(cwd, snapshot.tag, tag_message(study, experiment, launch_record))
        create_worktree(cwd, snapshot.worktree_path, snapshot.tag)


def rollback_fresh_snapshot(cwd: Path, snapshot: SnapshotMode) -> None:
    if snapshot.worktree_path is not None and snapshot.worktree_path.exists():
        remove_worktree(cwd, snapshot.worktree_path)
    if snapshot.tag is not None and tag_exists(cwd, snapshot.tag):
        delete_tag(cwd, snapshot.tag)


def wandb_group_for(study: str | None, experiment: str) -> str:
    return f"{study}/{experiment}" if study is not None else experiment


def persist_launch_record(
    launch_record: dict[str, Any],
    workdir: Path,
    study: str | None,
    experiment: str,
    smoke: bool,
) -> None:
    write_launch_record(launch_record, workdir, study, experiment)
    if not smoke:
        if study is None:
            raise ValueError("non-smoke launches require a study")
        scaffold_experiment(workdir, study, experiment, launch_record)


def main(argv=None):
    args = parse_args(argv)
    paths = resolve_launch_paths(args)
    validate_launch_request(args)
    apply_smoke_overrides(args)

    cfg = compose_launch_config(paths.config_dir, args.overrides)
    cluster = cfg.cluster
    use_slurm = bool(cluster.get("slurm", False))
    require_sync_smoke(args, use_slurm)

    study = resolve_study(args)
    experiment_dir = experiment_dir_for(paths.workdir, study, args.experiment)
    snapshot = resolve_snapshot_mode(
        args, paths.cwd, paths.workdir, study, experiment_dir
    )
    run_plan = build_run_plan(args, cfg, experiment_dir)
    require_new_run_ids(args, experiment_dir, run_plan.jobs_info)

    launch_record = build_launch_record(
        args, paths, cluster, use_slurm, study, snapshot, run_plan
    )

    submitted = False
    try:
        prepare_snapshot(paths.cwd, snapshot, study, args.experiment, launch_record)
        wandb_group = wandb_group_for(study, args.experiment)
        code_hash = (launch_record["git"] or {}).get("hash")

        # Smokes import code from the caller's dev worktree so they validate the
        # branch under iteration. Non-smokes import from the tag-pinned snapshot
        # to preserve reproducibility.
        runtime_worktree = paths.caller_repo if args.smoke else snapshot.worktree_path

        if use_slurm:
            slurm_job_ids = _submit_slurm(
                args, paths.workdir, cluster, run_plan.jobs_info, experiment_dir,
                worktree=runtime_worktree, wandb_run_group=wandb_group,
                code_hash=code_hash,
            )
            submitted = True
            launch_record["slurm_job_ids"] = {
                rid: jid for (rid, _), jid in zip(run_plan.jobs_info, slurm_job_ids)
            }
            persist_launch_record(
                launch_record, paths.workdir, study, args.experiment, args.smoke
            )
        else:
            persist_launch_record(
                launch_record, paths.workdir, study, args.experiment, args.smoke
            )
            submitted = True  # local run starts now; worktree must persist
            _run_local(
                paths.workdir, cluster, run_plan.jobs_info, args.experiment,
                worktree=runtime_worktree, wandb_run_group=wandb_group,
                code_hash=code_hash,
            )
    except BaseException:
        # Roll back tag + worktree only if we created them this invocation.
        # Extension launches reuse a prior tag/worktree; tearing those down
        # would destroy the earlier launch's snapshot.
        if not submitted and not args.smoke and not snapshot.extending:
            rollback_fresh_snapshot(paths.cwd, snapshot)
        raise


def _submit_slurm(
    args, workdir, cluster, jobs_info, experiment_dir,
    worktree=None, wandb_run_group=None, code_hash=None,
):
    logs_root = experiment_dir / "slurm_logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(logs_root))

    slurm_params = dict(
        name=args.experiment,
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
    group = wandb_run_group or args.experiment

    jobs = []
    with executor.batch():
        for run_id, cfg_path in jobs_info:
            job = TrainJob(
                workdir=workdir,
                cfg_path=cfg_path,
                experiment=args.experiment,
                run_id=run_id,
                gpus_per_node=int(cluster.gpus_per_node),
                nodes=int(cluster.nodes),
                setup_commands=setup_commands,
                retries=args.retries,
                worktree=worktree,
                wandb_run_group=group,
                code_hash=code_hash,
            )
            jobs.append(executor.submit(job))

    for j in jobs:
        print(j.job_id)

    return [j.job_id for j in jobs]


def _run_local(
    workdir, cluster, jobs_info, experiment, worktree=None, wandb_run_group=None,
    code_hash=None,
):
    gpus = int(cluster.get("gpus_per_node", 1))
    setup_commands = list(cluster.get("setup_commands", []) or [])
    pre_shell = " ; ".join(setup_commands) if setup_commands else ""
    group = wandb_run_group or experiment

    for run_id, cfg_path in jobs_info:
        print(f"Running {run_id} ({cfg_path})")
        env = os.environ.copy()
        env.setdefault("WANDB_RUN_GROUP", group)
        env.setdefault("SOURCE_WORKDIR", str(workdir))
        if code_hash is not None:
            env["SOURCE_GIT_HASH"] = code_hash
        prepend_pythonpath(env, worktree)

        port = find_free_port()
        torch_argv = [
            "torchrun",
            "--nproc-per-node",
            str(gpus),
            "--rdzv-backend",
            "c10d",
            "--rdzv-endpoint",
            f"localhost:{port}",
            "-m",
            "jepa.train",
            "--config",
            str(cfg_path),
        ]
        torch_cmd = shlex.join(torch_argv)
        cmd = f"{pre_shell} ; {torch_cmd}" if pre_shell else torch_cmd

        proc = subprocess.run(["bash", "-lc", cmd], cwd=workdir, env=env)
        if proc.returncode != 0:
            print(f"Run {run_id} failed with exit code {proc.returncode}")
            sys.exit(proc.returncode)


if __name__ == "__main__":
    sys.exit(main())
