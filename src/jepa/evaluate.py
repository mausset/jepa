import argparse
import datetime
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import submitit
from omegaconf import OmegaConf

from jepa.launch import (
    find_main_repo,
    git_info,
    prepend_pythonpath,
    research_results_dir,
    validate_name,
)


# ---------- eval registry ----------
#
# Each entry maps an eval name to a spec describing how to run it on a single
# checkpoint. Adding a new eval later: write the per-checkpoint script (or
# extend an existing one), then add a registry entry naming it.
#
# A runner builds the bash argv for one (checkpoint, output, env, kwargs)
# invocation. The launcher executes that argv via `bash -lc` so cluster
# `setup_commands` can prepend the env (module loads, conda activate, etc.).


def _planning_runner_factory(planner: str):
    """Build a runner for one of the planner-style evals.

    `planner_kwargs` keys map to `eval/planning.py` CLI flags (snake_case
    becomes kebab-case). Only the keys present are passed; unrecognized flags
    surface as the per-checkpoint script's own error.
    """
    def runner(checkpoint, output, env_name, planner_kwargs):
        argv = [
            "python", "-m", "jepa.eval.planning",
            "--checkpoint", str(checkpoint),
            "--output", str(output),
            "--env-name", env_name,
            "--planner", planner,
        ]
        for k, v in planner_kwargs.items():
            argv.extend([f"--{k.replace('_', '-')}", str(v)])
        return argv
    return runner


EVALS: dict[str, dict] = {
    "planning_mppi": {
        "runner": _planning_runner_factory("mppi"),
        "output_basename": "planning_mppi.json",
    },
}


# ---------- job ----------


class EvalJob:
    def __init__(
        self,
        *,
        workdir: Path,
        run_id: str,
        wandb_run_id: str,
        checkpoint: Path,
        experiment_eval_dir: Path,
        eval_names: list[str],
        env_name: str,
        planner_kwargs: dict,
        worktree: Path | None,
        setup_commands: list[str],
        retries: int,
    ):
        self.workdir = Path(workdir)
        self.run_id = run_id
        self.wandb_run_id = wandb_run_id
        self.ckpt_path = Path(checkpoint)
        self.experiment_eval_dir = Path(experiment_eval_dir)
        self.eval_names = list(eval_names)
        self.env_name = env_name
        self.planner_kwargs = dict(planner_kwargs)
        self.worktree = Path(worktree) if worktree is not None else None
        self.setup_commands = list(setup_commands)
        self.retries = int(retries)

    def _pre_shell(self) -> str:
        if not self.setup_commands:
            return "true"
        return " ; ".join(self.setup_commands)

    def __call__(self) -> None:
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "16")
        env.setdefault("JEPA_WORKDIR", str(self.workdir))
        prepend_pythonpath(env, self.worktree)

        out_dir = self.experiment_eval_dir / self.wandb_run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for eval_name in self.eval_names:
            spec = EVALS[eval_name]
            output_path = out_dir / spec["output_basename"]
            argv = spec["runner"](
                self.ckpt_path, output_path, self.env_name, self.planner_kwargs
            )
            inner = " ".join(argv)
            cmd = f"{self._pre_shell()} ; {inner}"

            tries = self.retries + 1
            ok = False
            for _ in range(tries):
                proc = subprocess.run(["bash", "-lc", cmd], cwd=self.workdir, env=env)
                if proc.returncode == 0:
                    ok = True
                    break
            if not ok:
                raise RuntimeError(
                    f"eval `{eval_name}` failed for run {self.run_id} after {tries} tries"
                )

    def checkpoint(self) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(
            EvalJob(
                workdir=self.workdir,
                run_id=self.run_id,
                wandb_run_id=self.wandb_run_id,
                checkpoint=self.ckpt_path,
                experiment_eval_dir=self.experiment_eval_dir,
                eval_names=self.eval_names,
                env_name=self.env_name,
                planner_kwargs=self.planner_kwargs,
                worktree=self.worktree,
                setup_commands=self.setup_commands,
                retries=self.retries,
            )
        )


# ---------- discovery ----------


def discover_runs(experiment_dir: Path, runs_filter: str | None, include_crashed: bool) -> list[dict]:
    """Return run records (run_id, wandb_run_id, status, checkpoint) for the
    runs we'll evaluate.

    Filters: by status (`done` always; `crashed` only with --include-crashed),
    optionally by an explicit run_id allowlist (--runs).
    Skips runs with missing checkpoints.
    """
    statuses = []
    for status_file in sorted((experiment_dir / "status").glob("*.json")):
        with open(status_file) as f:
            statuses.append(json.load(f))

    valid_states = {"done"} | ({"crashed"} if include_crashed else set())
    selected = [s for s in statuses if s.get("status") in valid_states]

    if runs_filter:
        wanted = {r.strip() for r in runs_filter.split(",") if r.strip()}
        selected = [s for s in selected if s["run_id"] in wanted]

    out = []
    for s in selected:
        wandb_id = s.get("wandb_run_id")
        if wandb_id is None:
            print(f"Warning: status for {s['run_id']} has no wandb_run_id; skipping")
            continue
        ckpt = experiment_dir / "checkpoints" / wandb_id / "checkpoint.pth"
        if not ckpt.exists():
            print(f"Warning: no checkpoint at {ckpt}; skipping run {s['run_id']}")
            continue
        out.append({
            "run_id": s["run_id"],
            "wandb_run_id": wandb_id,
            "status": s["status"],
            "checkpoint": ckpt,
        })
    return out


def all_outputs_exist(experiment_eval_dir: Path, wandb_run_id: str, eval_names: list[str]) -> bool:
    out_dir = experiment_eval_dir / wandb_run_id
    return all((out_dir / EVALS[name]["output_basename"]).exists() for name in eval_names)


# ---------- launch ----------


def submit_slurm_eval(
    args, workdir: Path, cluster: dict, jobs: list[EvalJob], experiment_eval_dir: Path
) -> list[str]:
    logs_root = experiment_eval_dir / "slurm_logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(logs_root))

    slurm_params = dict(
        name=f"eval-{args.study}-{args.experiment}",
        nodes=int(cluster.get("nodes", 1)),
        gpus_per_node=int(cluster.get("gpus_per_node", 1)),
        tasks_per_node=1,
        timeout_min=int(args.timeout_min),
        slurm_array_parallelism=int(cluster.get("array_parallelism", 64)),
    )
    constraint = cluster.get("constraint")
    if constraint:
        slurm_params["slurm_constraint"] = constraint
    executor.update_parameters(**slurm_params)

    submitted = []
    with executor.batch():
        for job in jobs:
            submitted.append(executor.submit(job))

    for j in submitted:
        print(j.job_id)
    return [j.job_id for j in submitted]


def run_local_eval(jobs: list[EvalJob]) -> None:
    for job in jobs:
        print(f"Evaluating run {job.run_id} ({job.wandb_run_id})")
        job()


def load_cluster(workdir: Path, name: str) -> dict:
    cluster_yaml = workdir / "configs" / "cluster" / f"{name}.yaml"
    if not cluster_yaml.exists():
        sys.exit(f"Cluster config not found: {cluster_yaml}")
    return OmegaConf.to_container(OmegaConf.load(cluster_yaml), resolve=True)


def read_experiment_launch_record(experiment_dir: Path) -> dict:
    launches = experiment_dir / "launches.jsonl"
    if not launches.exists():
        sys.exit(f"No launches.jsonl at {experiment_dir}; is this a real experiment dir?")
    with open(launches) as f:
        first_line = f.readline().strip()
    if not first_line:
        sys.exit(f"Empty launches.jsonl at {experiment_dir}")
    return json.loads(first_line)


def build_planner_kwargs(args) -> dict:
    """Pass-through kwargs forwarded to `eval/planning.py`.

    Keep keys aligned with that script's CLI flags (snake_case here is mapped
    to kebab-case at runner-build time).
    """
    return {
        "num_episodes": args.num_episodes,
        "horizon": args.horizon,
        "batch_size": args.batch_size,
        "population": args.population,
        "iterations": args.iterations,
        "temperature": args.temperature,
        "alpha": args.alpha,
    }


def write_launch_record(experiment_eval_dir: Path, record: dict) -> None:
    experiment_eval_dir.mkdir(parents=True, exist_ok=True)
    with open(experiment_eval_dir / "launches.jsonl", "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate trained checkpoints in an experiment against one or more evals."
    )
    p.add_argument("--study", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--env-name", required=True,
                   help="Env name used by `eval/planning.py` (e.g. pointmaze, keydoor).")
    p.add_argument("--evals", default="planning_mppi",
                   help="Comma-separated eval names from the registry.")
    p.add_argument("--cluster", default="thin",
                   help="Cluster name (yaml under configs/cluster/). Defaults to thin.")
    p.add_argument("--force", action="store_true",
                   help="Re-run evals even if their JSON already exists.")
    p.add_argument("--runs", default=None,
                   help="Comma-separated run_ids to evaluate (default: all `done` runs).")
    p.add_argument("--include-crashed", action="store_true",
                   help="Also evaluate runs with status=crashed (rarely useful).")
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--timeout-min", type=int, default=30,
                   help="SLURM per-task timeout. Eval is short; 30 min default.")
    p.add_argument("--workdir", type=Path, default=Path("."))

    # Forwarded to eval/planning.py:
    p.add_argument("--num-episodes", type=int, default=200)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--population", type=int, default=256)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.1)

    args = p.parse_args(argv)
    cwd = Path.cwd().resolve()
    if args.workdir == Path("."):
        workdir = find_main_repo(cwd)
    else:
        workdir = args.workdir.resolve()

    validate_name("study", args.study)
    validate_name("experiment", args.experiment)

    experiment_dir = research_results_dir(workdir) / args.study / args.experiment
    experiment_launch = read_experiment_launch_record(experiment_dir)

    worktree = Path(experiment_launch["worktree"]) if experiment_launch.get("worktree") else None
    if worktree is None or not worktree.exists():
        sys.exit(
            f"Experiment worktree {worktree} missing — refuse to eval against drifted code. "
            f"Recreate via: git worktree add {worktree} {experiment_launch['git']['tag']}"
        )

    eval_names = [name.strip() for name in args.evals.split(",") if name.strip()]
    for name in eval_names:
        if name not in EVALS:
            sys.exit(f"Unknown eval `{name}`; known: {sorted(EVALS)}")

    cluster = load_cluster(workdir, args.cluster)
    use_slurm = bool(cluster.get("slurm", False))

    runs = discover_runs(experiment_dir, args.runs, args.include_crashed)

    experiment_eval_dir = experiment_dir / "eval"
    if not args.force:
        runs = [
            r for r in runs
            if not all_outputs_exist(experiment_eval_dir, r["wandb_run_id"], eval_names)
        ]

    if not runs:
        print("Nothing to evaluate (all selected runs already have outputs; pass --force to redo).")
        return

    print(f"Evaluating {len(runs)} runs × {len(eval_names)} evals on cluster=`{args.cluster}` "
          f"(slurm={use_slurm}, timeout={args.timeout_min}m)")

    planner_kwargs = build_planner_kwargs(args)
    setup_commands = list(cluster.get("setup_commands", []) or [])

    jobs = []
    for r in runs:
        jobs.append(EvalJob(
            workdir=workdir,
            run_id=r["run_id"],
            wandb_run_id=r["wandb_run_id"],
            checkpoint=r["checkpoint"],
            experiment_eval_dir=experiment_eval_dir,
            eval_names=eval_names,
            env_name=args.env_name,
            planner_kwargs=planner_kwargs,
            worktree=worktree,
            setup_commands=setup_commands,
            retries=args.retries,
        ))

    slurm_job_ids = None
    if use_slurm:
        slurm_job_ids = submit_slurm_eval(args, workdir, cluster, jobs, experiment_eval_dir)
    else:
        run_local_eval(jobs)

    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "study": args.study,
        "experiment": args.experiment,
        "argv": sys.argv,
        "cwd": str(cwd),
        "workdir": str(workdir),
        "hostname": socket.gethostname(),
        "cluster_name": args.cluster,
        "cluster": cluster,
        "evals": eval_names,
        "planner_kwargs": planner_kwargs,
        "env_name": args.env_name,
        "n_runs": len(jobs),
        "run_ids": [j.run_id for j in jobs],
        "wandb_run_ids": [j.wandb_run_id for j in jobs],
        "git": git_info(cwd),
        "worktree": str(worktree),
        "slurm_job_ids": (
            {j.run_id: jid for j, jid in zip(jobs, slurm_job_ids)}
            if slurm_job_ids is not None else None
        ),
    }
    write_launch_record(experiment_eval_dir, record)


if __name__ == "__main__":
    sys.exit(main())
