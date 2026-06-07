"""
CLI entry point for the Differentiable Feasibility Pump.

Usage
-----
    diffpump path/to/*.mps
    diffpump instances.txt --json-dir output/

All hyperparameters default to the original feasibility pump (η=γ=β=1, λ=0,
p=1).  Override any of them to explore the differentiable variants.

See  diffpump --help  for full option list.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import sys
from pathlib import Path

from diffpump.problem import load_instance
from diffpump.pump import run_instance
from diffpump.results import ResultRecord
from diffpump.variants import VariantConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="diffpump",
        description="Differentiable Feasibility Pump — run on one or more instances",
    )

    p.add_argument(
        "instances", nargs="+",
        help="Path(s) or glob(s) to .mps/.lp instance files, or a file "
             "containing one path per line (if a single .txt/.list file is given).",
    )

    p.add_argument("--eta",    type=float, default=1.0,  help="Step size η (default 1.0)")
    p.add_argument("--gamma",  type=float, default=1.0,  help="Regularisation weight γ (default 1.0)")
    p.add_argument("--beta",   type=float, default=1.0,  help="Integrality loss weight β (default 1.0)")
    p.add_argument("--lam",    type=float, default=0.0,  help="Feasibility loss weight λ (default 0.0)")
    p.add_argument("--p",      type=float, default=1.0,  help="Integrality loss order p (default 1.0)")
    p.add_argument("--q",      type=int,   default=2,    help="Argmin feasibility exponent q (default 2)")
    p.add_argument("--eps",    type=float, default=0.15, help="Soft-rounding ε (default 0.15)")
    p.add_argument("--eps-feas", type=float, default=0.0,
                   help="Feasibility-loss ε offset (default 0)")
    p.add_argument("--use-argmin-feas", action="store_true", default=False,
                   help="Use argmin feasibility loss (eq. 21) instead of the linear form")

    p.add_argument("--max-iters", type=int, default=1000, help="Iteration limit (default 1000)")
    p.add_argument("--time-limit", type=float, default=None,
                   help="Wall-clock time limit in seconds (default: no limit)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default 0)")
    p.add_argument(
        "--solver", default="scip",
        choices=["gurobi", "scip"],
        help="LP solver backend for the pump loop (default: scip).",
    )
    p.add_argument(
        "--json-dir", default="jsons",
        help="Directory for per-instance JSON files and combined results.json (default: jsons/)",
    )

    return p


def _resolve_instances(specs: list[str]) -> list[Path]:
    """Expand globs and list files into a sorted list of instance paths."""
    paths: list[Path] = []
    for spec in specs:
        p = Path(spec)
        if p.is_file() and p.suffix in (".txt", ".list"):
            lines = p.read_text().splitlines()
            paths.extend(Path(line.strip()) for line in lines if line.strip())
        else:
            matched = sorted(glob.glob(spec, recursive=True))
            if not matched:
                if p.exists():
                    paths.append(p)
                else:
                    print(f"Warning: no files matched '{spec}'", file=sys.stderr)
            else:
                paths.extend(Path(m) for m in matched)
    return sorted(set(paths))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)

    instances = _resolve_instances(args.instances)
    if not instances:
        print("Error: no instance files found.", file=sys.stderr)
        return 1

    json_dir = Path(args.json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(instances)
    width   = len(str(n_total))
    sep     = "─" * 60

    tl_str = f"{args.time_limit}s" if args.time_limit else "none"
    print(sep)
    print(f"  solver     : {args.solver}   seed: {args.seed}   max-iters: {args.max_iters}   time-limit: {tl_str}")
    print(f"  instances  : {n_total}   json-dir: {json_dir}/")
    print(sep)

    cfg = VariantConfig(
        eta=args.eta,
        gamma=args.gamma,
        beta=args.beta,
        lam=args.lam,
        p=args.p,
        q=args.q,
        use_argmin_feas=args.use_argmin_feas,
        eps_soft=args.eps,
        eps_feas=args.eps_feas,
        max_iters=args.max_iters,
        time_limit=args.time_limit,
    )

    records: list[ResultRecord] = []

    for idx, inst_path in enumerate(instances, 1):
        prefix = f"  [{idx:{width}d}/{n_total}] {inst_path.name}"
        print(f"{prefix}  loading ...", end=" ", flush=True)
        try:
            inst = load_instance(inst_path, seed=args.seed)
        except Exception as exc:
            print(f"LOAD ERROR: {exc}", file=sys.stderr)
            continue

        print("solving ...", end=" ", flush=True)
        try:
            rec = run_instance(inst, cfg, seed=args.seed, solver=args.solver)
        except Exception as exc:
            print(f"RUN ERROR: {exc}", file=sys.stderr)
            continue

        records.append(rec)

        if rec.success:
            status = "OK     "
        elif rec.timed_out:
            status = "TIMEOUT"
        else:
            status = "FAIL   "

        lp_pct = 100.0 * rec.lp_time / max(rec.wall_time, 1e-9)
        print(
            f"{status}  "
            f"iters={rec.n_iters:{width+3}d}  "
            f"restarts={rec.n_restarts:{width+3}d}  "
            f"wall={rec.wall_time:7.2f}s  "
            f"lp={rec.lp_time:7.2f}s ({lp_pct:.0f}%)"
        )

        inst_json = json_dir / f"{inst_path.stem}.json"
        with open(inst_json, "w") as fh:
            json.dump(dataclasses.asdict(rec), fh, indent=2)

    if not records:
        print("No results produced.", file=sys.stderr)
        return 2

    print(sep)
    print(f"  → {json_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
