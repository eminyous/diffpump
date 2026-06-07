"""
Instance loading and representation.

Reads a MIPLIB .mps/.lp file via Gurobi and extracts:
  - constraint matrix A, RHS b (in Ax >= b form)
  - variable bounds, integrality flags, cost vector
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB


@dataclass
class Instance:
    """All static data needed to run the pump on one instance."""

    name: str
    path: str

    # Variable arrays (length = n_int + n_cont)
    n_vars: int = 0        # total variables
    n_int: int = 0         # integer (binary + general integer) variables
    n_cont: int = 0        # continuous variables

    # Integer-variable indices into the original variable ordering
    int_idx: list[int] = field(default_factory=list)
    cont_idx: list[int] = field(default_factory=list)

    lb: np.ndarray = field(default_factory=lambda: np.empty(0))  # (n_vars,)
    ub: np.ndarray = field(default_factory=lambda: np.empty(0))  # (n_vars,)
    is_binary: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))  # (n_int,)
    c: np.ndarray = field(default_factory=lambda: np.empty(0))   # (n_vars,) cost

    # Constraint data (Ax >= b, m constraints)
    A: np.ndarray | None = None    # (m, n_vars)  dense; use sparse for large instances
    b: np.ndarray = field(default_factory=lambda: np.empty(0))   # (m,)
    row_norms: np.ndarray = field(default_factory=lambda: np.empty(0))  # (m,) ‖[Aⱼ bⱼ]‖₂



def load_instance(path: str | Path, seed: int = 0) -> Instance:
    """
    Load a MIP instance from a .mps or .lp file.

    Returns a MIPInstance with the constraint matrix extracted in Ax >= b form.
    The Gurobi model is *not* stored in the dataclass; call `build_gurobi_models`
    separately to get the solver objects.
    """
    path = Path(path)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    model = gp.read(str(path), env=env)
    model.Params.OutputFlag = 0

    vars_ = model.getVars()
    n_vars = len(vars_)

    lb = np.array([v.LB for v in vars_])
    ub = np.array([v.UB for v in vars_])
    c  = np.array([v.Obj for v in vars_])

    int_idx  = [i for i, v in enumerate(vars_) if v.VType in (GRB.INTEGER, GRB.BINARY)]
    cont_idx = [i for i in range(n_vars) if i not in set(int_idx)]

    n_int  = len(int_idx)
    n_cont = len(cont_idx)

    is_binary = np.array([ub[i] <= 1 + 1e-9 and lb[i] >= -1e-9 for i in int_idx], dtype=bool)

    # --- Extract constraint matrix in Ax >= b form ---
    # Gurobi stores constraints as: lhs sense rhs  (sense ∈ {<, >, =})
    constrs = model.getConstrs()
    A_rows, b_rows = [], []
    for c_obj in constrs:
        row = model.getRow(c_obj)
        sense = c_obj.Sense
        rhs   = c_obj.RHS

        coeffs = np.zeros(n_vars)
        for k in range(row.size()):
            vi = row.getVar(k).index
            coeffs[vi] = row.getCoeff(k)

        if sense == GRB.LESS_EQUAL:       # Ax <= b  →  -Ax >= -b
            A_rows.append(-coeffs)
            b_rows.append(-rhs)
        elif sense == GRB.GREATER_EQUAL:  # Ax >= b  →  keep
            A_rows.append(coeffs)
            b_rows.append(rhs)
        else:                             # equality: add both directions
            A_rows.append(coeffs)
            b_rows.append(rhs)
            A_rows.append(-coeffs)
            b_rows.append(-rhs)

    A = np.array(A_rows, dtype=float)  # (m, n_vars)
    b = np.array(b_rows,  dtype=float)  # (m,)

    # Row norms  ‖[Aⱼ bⱼ]‖₂  for normalisation in feasibility loss (eq. 19)
    Ab = np.concatenate([A, b[:, None]], axis=1)
    row_norms = np.linalg.norm(Ab, axis=1)
    # Avoid division by zero for degenerate rows
    row_norms = np.where(row_norms > 1e-12, row_norms, 1.0)

    env.dispose()

    return Instance(
        name=path.stem,
        path=str(path),
        n_vars=n_vars,
        n_int=n_int,
        n_cont=n_cont,
        int_idx=int_idx,
        cont_idx=cont_idx,
        lb=lb,
        ub=ub,
        is_binary=is_binary,
        c=c,
        A=A,
        b=b,
        row_norms=row_norms,
    )


def check_feasibility(y_int: np.ndarray, inst: Instance, tol: float = 1e-6) -> bool:
    """
    Check whether the integer vector y_int (values for integer variables only,
    in int_idx order) is feasible for the original MIP.

    Checks A[:, int_idx] @ y_int >= b (integer-variable slice).  For
    mixed-integer instances this approximates feasibility by ignoring the
    continuous-variable contribution; a precise check would require solving a
    fixing LP.
    """
    x_full = np.zeros(inst.n_vars)
    for local_i, global_i in enumerate(inst.int_idx):
        x_full[global_i] = y_int[local_i]

    slacks = inst.b - inst.A @ x_full
    return bool(np.all(slacks <= tol))
