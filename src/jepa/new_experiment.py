"""Spawn a per-experiment dev worktree on a fresh branch.

Creates `research/<study>/<experiment>/` as a git worktree on a new branch
`research/<study>/<experiment>`, with absolute symlinks back to the main
repo's `data/` and `research_results/` so paths resolve identically whether
the caller's cwd is main or the dev worktree.

Must be run from the main worktree's root, on the `main` branch. The main
worktree is identified by `.git` being a directory (secondary worktrees
have `.git` as a file pointing back to the main repo).
"""
import argparse
import subprocess
import sys
from pathlib import Path

from jepa.launch import (
    RESEARCH_DIRNAME,
    RESEARCH_RESULTS_DIRNAME,
    research_results_dir,
    validate_name,
)


def find_main_repo_root() -> Path:
    """Return the main repo root from cwd; exit if cwd isn't the main worktree's root."""
    cwd = Path.cwd().resolve()

    try:
        toplevel = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.PIPE
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit(f"Refusing: cwd ({cwd}) is not inside a git repository.")

    toplevel = Path(toplevel).resolve()

    if cwd != toplevel:
        sys.exit(
            f"Refusing: cwd ({cwd}) is not the worktree root ({toplevel}). "
            f"cd to the repo root and try again."
        )

    if not (cwd / ".git").is_dir():
        sys.exit(
            f"Refusing: cwd ({cwd}) is a secondary worktree. "
            f"Run jepa.new_experiment from the main worktree."
        )

    return cwd


def current_branch(workdir: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(workdir), "rev-parse", "--abbrev-ref", "HEAD"],
        text=True,
    ).strip()


def branch_exists(workdir: Path, branch: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(workdir), "rev-parse", "--verify", "--quiet",
         f"refs/heads/{branch}"],
        capture_output=True,
    )
    return result.returncode == 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Spawn a per-experiment dev worktree on a fresh branch. "
            "Run from the main worktree's root, on `main`."
        ),
    )
    p.add_argument("study", help="Study (research goal). Lowercase letters/digits/hyphens.")
    p.add_argument("experiment", help="Experiment name within the study.")
    args = p.parse_args(argv)

    validate_name("study", args.study)
    validate_name("experiment", args.experiment)

    workdir = find_main_repo_root()

    head = current_branch(workdir)
    if head != "main":
        sys.exit(
            f"Refusing: HEAD is {head!r}, expected `main`. "
            f"Switch to main first (new experiments branch off main)."
        )

    study_readme = research_results_dir(workdir) / args.study / "README.md"
    if not study_readme.exists():
        sys.exit(
            f"Refusing: study {args.study!r} has no README at {study_readme}. "
            f"Open the study first: invoke the /study skill (it asks the framing "
            f"questions, suggests a first experiment, and writes the README)."
        )

    branch = f"research/{args.study}/{args.experiment}"
    dev_worktree = workdir / RESEARCH_DIRNAME / args.study / args.experiment

    if dev_worktree.exists():
        sys.exit(f"Refusing: dev worktree path {dev_worktree} already exists.")
    if branch_exists(workdir, branch):
        sys.exit(
            f"Refusing: branch `{branch}` already exists. "
            f"Pick a different experiment name, or delete the branch first "
            f"(`git branch -D {branch}`)."
        )

    # Ensure the dev worktree's parent dir exists. The artifact root
    # (research_results/) is guaranteed to exist because the study README is
    # in research_results/<study>/, which we just verified.
    dev_worktree.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["git", "-C", str(workdir), "worktree", "add", str(dev_worktree),
         "-b", branch],
        check=True,
    )

    # Absolute symlinks: data/ and research_results/ resolve identically from
    # any cwd, so SLURM workers find datasets and write artifacts to the same
    # shared tree whether they were launched from main or the dev worktree.
    (dev_worktree / "data").symlink_to(workdir / "data")
    (dev_worktree / RESEARCH_RESULTS_DIRNAME).symlink_to(workdir / RESEARCH_RESULTS_DIRNAME)

    print()
    print(f"Created {dev_worktree}")
    print(f"On branch {branch}")
    print()
    print("Next steps:")
    print(f"  cd {dev_worktree}")
    print(f"  # edit code/configs, commit on this branch")
    print(
        f"  python -m jepa.launch --study {args.study} "
        f"--experiment {args.experiment} \\"
    )
    print(f"    +train=<config> cluster=thin [overrides ...]")


if __name__ == "__main__":
    sys.exit(main())
