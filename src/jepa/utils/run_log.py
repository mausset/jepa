import json
import os
import socket
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slurm_job_id_from_env() -> str | None:
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_job and array_task:
        return f"{array_job}_{array_task}"
    return os.environ.get("SLURM_JOB_ID")


def to_scalar_dict(d, prefix: str = "") -> dict[str, float]:
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                out[prefix + k] = float(v.detach().cpu().item())
        elif isinstance(v, (int, float)):
            out[prefix + k] = float(v)
    return out


class RunStatus:
    """Atomically writes a per-run status JSON. Rank-0 only.

    Fields capture enough state to answer "is run X alive, where, and how is it doing"
    without attaching to the process or querying W&B.
    """

    def __init__(
        self,
        sweep_dir: Path,
        run_id: str,
        *,
        total_steps: int,
        slurm_job_id: str | None,
        wandb_run_id: str | None = None,
    ):
        self.path = Path(sweep_dir) / "status" / f"{run_id}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.total_steps = total_steps
        self.slurm_job_id = slurm_job_id
        self.wandb_run_id = wandb_run_id
        self.hostname = socket.gethostname()
        self.start_time = _now()
        self._latest_metrics: dict[str, float] = {}
        self.write(status="running", step=0)

    def _write(self, payload: dict) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.replace(self.path)

    def write(
        self,
        *,
        status: str,
        step: int,
        exception: str | None = None,
    ) -> None:
        payload = {
            "run_id": self.run_id,
            "status": status,
            "step": step,
            "total_steps": self.total_steps,
            "fraction": (step / self.total_steps) if self.total_steps else None,
            "start_time": self.start_time,
            "last_heartbeat": _now(),
            "hostname": self.hostname,
            "slurm_job_id": self.slurm_job_id,
            "wandb_run_id": self.wandb_run_id,
            "latest_metrics": self._latest_metrics,
            "exception": exception,
        }
        self._write(payload)

    def set_wandb_run_id(self, wandb_run_id: str | None) -> None:
        self.wandb_run_id = wandb_run_id

    def heartbeat(self, step: int, metrics: dict[str, float] | None = None) -> None:
        if metrics:
            self._latest_metrics = {**self._latest_metrics, **metrics}
        self.write(status="running", step=step)

    def done(self, step: int, metrics: dict[str, float] | None = None) -> None:
        if metrics:
            self._latest_metrics = {**self._latest_metrics, **metrics}
        self.write(status="done", step=step)

    def crashed(self, step: int, exc: BaseException) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.write(status="crashed", step=step, exception=tb)


class MetricsLogger:
    """Append-only jsonl of train/val scalar metrics. Rank-0 only.

    Each line: {"step": int, "stage": "train"|"val", **scalar_metrics}.
    """

    def __init__(self, sweep_dir: Path, run_id: str):
        self.path = Path(sweep_dir) / "metrics" / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, step: int, stage: str, metrics: dict[str, float]) -> None:
        if not metrics:
            return
        record = {"step": int(step), "stage": stage, **metrics}
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
