# DiffPump

Differentiable Feasibility Pump for general integer and mixed-integer programs.

## Install

```bash
pip install -e ".[test]"
```

Requires a Gurobi license for instance loading (academic licenses are free).

## Usage

```bash
diffpump path/to/instance.mps
diffpump "data/*.mps" --solver scip --max-iters 500 --time-limit 60
diffpump instances.txt --json-dir output/
```

Results are written as JSON files to `jsons/` (one per instance). Use `--json-dir` to change the output directory.

### Key options

| Flag | Default | Description |
|---|---|---|
| `--solver` | `scip` | `gurobi` or `scip` |
| `--max-iters` | `1000` | Iteration limit |
| `--time-limit` | none | Wall-clock limit in seconds |
| `--json-dir` | `jsons` | Output directory |
| `--seed` | `0` | Random seed |
| `--eta`, `--gamma`, `--beta`, `--lam`, `--p` | `1,1,1,0,1` | Algorithm hyperparameters |
