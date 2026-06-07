# DiffPump

**Differentiable Feasibility Pump** — general integer and mixed-integer extension.

Implementation of Algorithm 2 from:

> Cacciola, Emine, Forel, Frangioni, Lodi.
> *The Differentiable Feasibility Pump.*
> Mathematical Programming, 2025.

Extends the binary-only IPCO version to **general integer and mixed-integer**
problems. All algorithm hyperparameters are directly configurable; two LP
solver backends are available (Gurobi and SCIP).

---

## Requirements

- Python ≥ 3.10
- [Gurobi](https://www.gurobi.com/) with a valid license — used for **instance loading** (academic licenses available free)
- PyTorch ≥ 2.0
- NumPy, SciPy, PyYAML, pyscipopt

---

## Install

```bash
pip install diffpump
pip install 'diffpump[test]'   # + test dependencies
```

Or from source in editable mode:

```bash
git clone https://github.com/eminyous/diffpump.git
cd diffpump
pip install -e ".[test]"
```

---

## CLI usage

```bash
# Default parameters (original feasibility pump behaviour)
diffpump --instances path/to/instance.mps --out results.csv

# Tune hyperparameters freely
diffpump --instances path/to/instance.mps \
         --eta 0.6 --gamma 0.7 --beta 0.3 --lam 0.1 --p 2 \
         --eps 0.15 --max-iters 1000 --time-limit 300 --seed 42 \
         --out results.csv

# Use SCIP for LP solves (open-source, no license required for solving)
diffpump --instances path/to/instance.mps --solver scip --out results.csv

# Enable argmin feasibility loss
diffpump --instances path/to/instance.mps --lam 0.9 --use-argmin-feas --out results.csv
```

**Output**: a CSV with one row per instance (success, timed\_out, solver, iters,
restarts, wall\_time, lp\_time, all hyperparameters) plus an aggregate summary
to stdout and a `.summary.json`.

### Full option reference

| Flag | Default | Description |
|---|---|---|
| `--instances` | *(required)* | Path(s), glob(s), or a `.txt`/`.list` file of paths |
| `--solver` | `gurobi` | LP backend for pump iterations: `gurobi \| scip` |
| `--eta` | `1.0` | Step size η |
| `--gamma` | `1.0` | Regularisation weight γ |
| `--beta` | `1.0` | Integrality loss weight β |
| `--lam` | `0.0` | Feasibility loss weight λ |
| `--p` | `1.0` | Integrality loss order p |
| `--q` | `2` | Argmin feasibility exponent q |
| `--eps` | `0.15` | Soft-rounding temperature ε |
| `--eps-feas` | `0.0` | Feasibility-loss ReLU offset |
| `--use-argmin-feas` | off | Use argmin feasibility loss (eq. 21) instead of linear form |
| `--max-iters` | `1000` | Iteration limit |
| `--time-limit` | none | Wall-clock time limit in seconds |
| `--seed` | `0` | Random seed |
| `--out` | `results/out.csv` | Output CSV path |

The defaults reproduce the original feasibility pump (η=γ=β=1, λ=0, p=1).

---

## Python API

```python
from diffpump import load_instance, run_instance, VariantConfig

inst   = load_instance("path/to/instance.mps", seed=0)

# Default config (original feasibility pump)
config = VariantConfig()

# Or tune freely
config = VariantConfig(eta=0.6, gamma=0.7, beta=0.3, lam=0.1, p=2.0)

# Gurobi backend (default)
result = run_instance(inst, config, seed=0)

# SCIP backend
result = run_instance(inst, config, seed=0, solver="scip")

print(result.success, result.timed_out, result.n_iters, result.wall_time)
```

### `VariantConfig` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `eta` | `float` | `1.0` | Gradient step size |
| `gamma` | `float` | `1.0` | Regularisation weight |
| `beta` | `float` | `1.0` | Integrality loss weight |
| `lam` | `float` | `0.0` | Feasibility loss weight |
| `p` | `float` | `1.0` | Integrality loss order |
| `use_argmin_feas` | `bool` | `False` | Use argmin feasibility loss (eq. 21) |
| `q` | `int` | `2` | Argmin feasibility exponent |
| `eps_feas` | `float` | `0.0` | Feasibility-loss ReLU offset |
| `eps_soft` | `float` | `0.15` | Soft-rounding temperature |
| `max_iters` | `int` | `1000` | Iteration limit |
| `time_limit` | `float \| None` | `None` | Wall-clock limit in seconds |

### `ResultRecord` fields (one row per run)

| Field | Type | Description |
|---|---|---|
| `instance_name` | `str` | Stem of the instance file |
| `solver` | `str` | LP backend used (`"gurobi"` or `"scip"`) |
| `success` | `bool` | Feasible solution found |
| `timed_out` | `bool` | Time limit reached before convergence |
| `n_iters` | `int` | Total pump iterations |
| `n_restarts` | `int` | Number of cycle-breaking restarts |
| `restart_ratio` | `float` | `n_restarts / n_iters` |
| `wall_time` | `float` | Total wall-clock seconds |
| `lp_time` | `float` | LP-solve seconds within `wall_time` |
| `eta … seed` | | All hyperparameters, for traceability |

---

## Solver backends

| Backend | License needed | Notes |
|---|---|---|
| `gurobi` (default) | Yes (academic: free) | Always used for **loading**; fastest LP solver |
| `scip` | No | Open-source; good for large instances without a Gurobi solve license |

Gurobi is always used to **load** instances (parse the MPS file, extract the
constraint matrix). The `--solver` flag controls only the LP solves inside the
pump loop, where the bulk of compute time is spent.

---

## Run tests

```bash
pytest tests/ -v
```

---

## Package structure

```
diffpump/
  __init__.py       Public API
  problem.py        MIPInstance dataclass, load_instance() (Gurobi)
  gurobi_model.py   LP relaxation + pumping model (Gurobi backend)
  scip_model.py     LP relaxation + pumping model (SCIP backend)
  surrogate.py      −I surrogate Jacobian (eq. 7) via torch.autograd.Function
  rounding.py       Hard rounding + soft rounding (eqs. 24, 25)
  losses.py         Integrality loss (eq. 8), feasibility losses (eqs. 19–21),
                    regularisation, total loss (eq. 18)
  restarts.py       Cycle detection, flip, perturbation (§2.1)
  pump.py           Algorithm 2 main loop
  variants.py       VariantConfig dataclass
  results.py        ResultRecord, CSV I/O, aggregate summary
  cli.py            diffpump console entry point

tests/
  test_losses.py    Unit tests for all loss components
  test_rounding.py  Unit tests + finite-difference checks for soft rounding
  test_gradient.py  Surrogate Jacobian and gradient update verification
  test_pump.py      End-to-end tests on tiny synthetic instances
  test_cli.py       Parser, _resolve_instances, and main() integration tests
```

---

## Key equations

| Symbol | Description | Equation |
|---|---|---|
| φ(x; θ, y) = Σ\|θᵢ\|\|xᵢ−yᵢ\| | Pumping function | eq. 13 |
| f(x) = Σ\|xᵢ−⌊xᵢ⌉\|^p | Integrality loss | eq. 8 |
| L = βf + λg + γΩ | Total loss | eq. 18 |
| g(x) = (1/m)Σ ReLU(sⱼ−ε) | Pure-integer feasibility loss | eq. 19 |
| g(x,u) = (1/m)Σ gⱼ(x,u) | Mixed-integer feasibility loss | eq. 20 |
| g = argminᵤ G(x,u) | Argmin feasibility loss | eq. 21 |
| J\_x ⌊x⌉\_ε = (1/ε)φ\_N((0.5−x+⌊x⌋)/ε) | Soft rounding Jacobian | eq. 25 |
| J\_θ x\*(θ) = −I | Surrogate Jacobian | eq. 7 |
