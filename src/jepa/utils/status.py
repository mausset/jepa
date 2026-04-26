"""Print compact status tables for experiments in a study.

Reads `research_results/<study>/<experiment>/launches.jsonl` to find runs,
infers the swept hyperparameters from the resolved per-run configs, and
displays one row per run with its current `status/<run_id>.json` state +
latest val metrics. No hand-maintained run_id ↔ hyperparameter mappings.

Usage:
    python -m jepa.utils.status <study>
    python -m jepa.utils.status <study> <experiment> [<experiment>...]
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from jepa.launch import find_main_repo, research_results_dir


METRIC_KEYS = (
    "val/mse",
    "val/kl",
    "val/state_sigreg",
    "val/action_acc",
    "val/rollout_action_acc",
)
METRIC_HEADERS = ("mse", "kl", "sigreg", "acc", "rollout")


def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, path))
        else:
            out[path] = v
    return out


def find_swept_keys(configs: list[dict]) -> list[str]:
    """Dotted-paths whose leaf values differ across the given configs."""
    flats = [flatten(c) for c in configs]
    keys: set[str] = set().union(*[set(f) for f in flats]) if flats else set()
    swept = []
    for k in sorted(keys):
        values = {json.dumps(f.get(k), sort_keys=True, default=str) for f in flats}
        if len(values) > 1:
            swept.append(k)
    return swept


def collect_run_ids(experiment_dir: Path) -> list[str]:
    """Run_ids in submission order, across all launches in the experiment."""
    launches = experiment_dir / "launches.jsonl"
    if not launches.exists():
        return []
    seen: set[str] = set()
    out: list[str] = []
    with launches.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for rid in rec.get("run_ids", []):
                if rid not in seen:
                    seen.add(rid)
                    out.append(rid)
    return out


def gather_runs(experiment_dir: Path) -> list[dict]:
    runs: list[dict] = []
    for rid in collect_run_ids(experiment_dir):
        cfg_path = experiment_dir / "configs" / f"{rid}.yaml"
        st_path = experiment_dir / "status" / f"{rid}.json"
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        st = json.loads(st_path.read_text()) if st_path.exists() else {}
        runs.append({"run_id": rid, "config": cfg, "status": st})
    return runs


def fmt_metric(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if v != v:
            return "nan"
        absv = abs(v)
        if absv == 0 or 1e-3 <= absv < 1e5:
            return f"{v:.4f}"
        return f"{v:.2e}"
    return str(v)


def fmt_axis(v: Any) -> str:
    if isinstance(v, float):
        # Keep float axes (lambda, kl_beta) compact and sortable in printout.
        if v == 0:
            return "0"
        absv = abs(v)
        if 1e-3 <= absv < 1e3:
            return f"{v:g}"
        return f"{v:.2e}"
    if isinstance(v, list):
        return "[" + ",".join(fmt_axis(x) for x in v) + "]"
    return str(v)


def render_experiment(experiment_dir: Path) -> None:
    runs = gather_runs(experiment_dir)
    if not runs:
        print("  (no runs)")
        return
    swept = find_swept_keys([r["config"] for r in runs])
    # The seed dimension is uninteresting if it's the only swept key (single seed
    # is the project default); otherwise keep it as a column.
    if swept == ["training.seed"]:
        swept = []

    rows = []
    for r in runs:
        flat = flatten(r["config"])
        st = r["status"]
        m = st.get("latest_metrics") or {}
        row: dict[str, Any] = {k: flat.get(k) for k in swept}
        row["run_id"] = r["run_id"]
        row["status"] = st.get("status", "-")
        row["step"] = st.get("step", 0) or 0
        row["total"] = st.get("total_steps", 0) or 0
        for mk in METRIC_KEYS:
            row[mk] = m.get(mk)
        rows.append(row)

    def sort_key(row):
        return tuple(
            (1, row.get(k)) if row.get(k) is None
            else (0, row.get(k))
            for k in swept
        ) + (row["run_id"],)

    rows.sort(key=sort_key)

    headers = [k.split(".")[-1] for k in swept] + [
        "run_id", "status", "step", "%",
        *METRIC_HEADERS,
    ]

    def stringify(row):
        out = [fmt_axis(row[k]) for k in swept]
        out.append(row["run_id"])
        out.append(row["status"])
        out.append(str(row["step"]))
        out.append(f"{100*row['step']/row['total']:.1f}%" if row["total"] else "-")
        for mk in METRIC_KEYS:
            out.append(fmt_metric(row.get(mk)))
        return out

    str_rows = [stringify(r) for r in rows]
    widths = [
        max(len(h), max((len(r[i]) for r in str_rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "  "
    print(sep.join(h.ljust(w) for h, w in zip(headers, widths)))
    print(sep.join("-" * w for w in widths))
    for r in str_rows:
        print(sep.join(c.ljust(w) for c, w in zip(r, widths)))


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m jepa.utils.status",
        description="Print compact status tables for experiments in a study.",
    )
    p.add_argument("study")
    p.add_argument("experiment", nargs="*",
                   help="Experiment names; omit to show all in the study.")
    args = p.parse_args(argv)

    cwd = Path.cwd().resolve()
    workdir = find_main_repo(cwd)
    study_dir = research_results_dir(workdir) / args.study
    if not study_dir.exists():
        sys.exit(f"Study {args.study!r} not found at {study_dir}")

    if args.experiment:
        names = args.experiment
    else:
        names = sorted(
            d.name for d in study_dir.iterdir()
            if d.is_dir() and (d / "launches.jsonl").exists()
        )
    if not names:
        sys.exit(f"No experiments found under {study_dir}")

    for i, name in enumerate(names):
        exp_dir = study_dir / name
        if i:
            print()
        print(f"=== {name} ===")
        if not exp_dir.exists():
            print("  (no experiment dir)")
            continue
        render_experiment(exp_dir)


if __name__ == "__main__":
    main()
