"""
Result recording, CSV export, and aggregate summary.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class ResultRecord:
    """One row of output: one algorithm run on one instance."""

    instance_name: str
    solver: str         # "gurobi" | "scip"
    success: bool
    timed_out: bool
    n_iters: int
    n_restarts: int
    restart_ratio: float   # n_restarts / n_iters
    wall_time: float       # total wall-clock seconds
    lp_time: float         # LP-solve seconds within wall_time

    # Hyperparameters (for traceability)
    eta: float = 1.0
    gamma: float = 1.0
    beta: float = 1.0
    lam: float = 0.0
    p: float = 1.0
    q: int = 2
    use_argmin_feas: bool = False
    eps_soft: float = 0.15
    eps_feas: float = 0.0
    seed: int = 0


_COLUMNS = [f.name for f in fields(ResultRecord)]


def write_csv(records: list[ResultRecord], path: str | Path) -> None:
    """Write per-instance records to a CSV file (one row per record)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def load_csv(path: str | Path) -> list[ResultRecord]:
    """Read per-instance records from a CSV file."""
    path = Path(path)
    records = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row["success"]          = row["success"] == "True"
            row["timed_out"]        = row["timed_out"] == "True"
            row["use_argmin_feas"]  = row["use_argmin_feas"] == "True"
            row["n_iters"]          = int(row["n_iters"])
            row["n_restarts"]       = int(row["n_restarts"])
            row["restart_ratio"]    = float(row["restart_ratio"])
            row["wall_time"]        = float(row["wall_time"])
            row["lp_time"]          = float(row["lp_time"])
            row["eta"]              = float(row["eta"])
            row["gamma"]            = float(row["gamma"])
            row["beta"]             = float(row["beta"])
            row["lam"]              = float(row["lam"])
            row["p"]                = float(row["p"])
            row["q"]                = int(row["q"])
            row["eps_soft"]         = float(row["eps_soft"])
            row["eps_feas"]         = float(row["eps_feas"])
            row["seed"]             = int(row["seed"])
            records.append(ResultRecord(**row))
    return records


def aggregate(records: list[ResultRecord]) -> dict[str, Any]:
    """Compute aggregate statistics over all records."""
    if not records:
        return {}
    n          = len(records)
    n_fail     = sum(1 for r in records if not r.success)
    total_iter = sum(r.n_iters for r in records)
    total_rst  = sum(r.n_restarts for r in records)
    total_wall = sum(r.wall_time for r in records)
    total_lp   = sum(r.lp_time for r in records)
    return {
        "n_instances":   n,
        "fail_rate_pct": 100.0 * n_fail / n,
        "total_iters":   total_iter,
        "restart_ratio": 100.0 * total_rst / max(total_iter, 1),
        "wall_time_s":   total_wall,
        "lp_time_s":     total_lp,
        "mean_wall_s":   total_wall / n,
    }


def print_summary(agg: dict[str, Any], file=None) -> None:
    """Print an aggregate summary to stdout (or `file`)."""
    if file is None:
        file = sys.stdout
    if not agg:
        return
    metrics = [
        ("fail_rate_pct", "Fail rate %",   "6.2f"),
        ("total_iters",   "Total iters",   "8d"),
        ("restart_ratio", "Restart %",     "6.2f"),
        ("mean_wall_s",   "Mean time (s)", "8.3f"),
    ]
    print(file=file)
    for key, label, fmt in metrics:
        v = agg.get(key, float("nan"))
        print(f"{label:<18s}  {v:{fmt}}", file=file)
    print(file=file)
