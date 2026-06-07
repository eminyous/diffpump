"""
Gurobi model management.

Provides:
  - GurobiPumpModel: wraps the LP relaxation + pumping problem structure.
  - solve_relaxation(): initial solve with cost c to get x^(0).
  - solve_pumping(): solve min φ(x;θ,y) s.t. (x,u)∈X̂  (eq. 13 / eq. 17).

Gurobi settings (paper §5):
  - Single thread
  - OptimalityTol = FeasibilityTol = 1e-9 (tightest available)
  - Silent output
"""

from __future__ import annotations

import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from diffpump.problem import Instance

# Global tolerance settings matching paper §5
_OPT_TOL  = 1e-9
_FEAS_TOL = 1e-9


class GurobiPumpModel:
    """
    Holds the LP relaxation and the pumping-problem Gurobi model for one instance.

    The pumping model is a copy of the relaxation augmented with auxiliary variables
    d_i >= 0 to linearise φ(x; θ, y) = Σᵢ |θᵢ| |xᵢ - yᵢ|.

    Layout
    ------
    The pumping LP has variables [x_relax, d_0, ..., d_{n_int-1}].
    Objective coefficients on d_i are |θ_i|; all others are 0.
    Two inequality constraints per integer variable i:
        xᵢ - dᵢ ≤  yᵢ     ↔    dᵢ ≥  xᵢ - yᵢ
       -xᵢ - dᵢ ≤ -yᵢ     ↔    dᵢ ≥  yᵢ - xᵢ
    """

    def __init__(self, inst: Instance, seed: int = 0) -> None:
        self.inst = inst
        self._seed = seed
        self._relax_model: gp.Model | None = None
        self._pump_model:  gp.Model | None  = None
        self._d_vars: list = []          # auxiliary d variables in pump model
        self._x_pump_vars: list = []     # x variables in pump model (same order as relax)
        self._ub_constrs_pos: list = []  # dᵢ ≥ xᵢ - yᵢ  constraints
        self._ub_constrs_neg: list = []  # dᵢ ≥ yᵢ - xᵢ  constraints

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build the LP relaxation and pumping models. Call once per instance."""
        self._build_relaxation()
        self._build_pump_model()

    def _build_relaxation(self) -> None:
        inst = self.inst
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()

        m = gp.read(inst.path, env)
        m.Params.OutputFlag    = 0
        m.Params.Threads       = 1
        m.Params.OptimalityTol = _OPT_TOL
        m.Params.FeasibilityTol= _FEAS_TOL
        m.Params.Seed          = self._seed

        # Relax all integrality constraints
        self._relax_model = m.relax()
        m.dispose()   # original MIP model no longer needed
        self._relax_model.Params.OutputFlag    = 0
        self._relax_model.Params.Threads       = 1
        self._relax_model.Params.OptimalityTol = _OPT_TOL
        self._relax_model.Params.FeasibilityTol= _FEAS_TOL
        self._relax_model.Params.Seed          = self._seed

    def _build_pump_model(self) -> None:
        """
        Clone the relaxation and add auxiliary variables + linearisation constraints
        for φ(x; θ, y).
        """
        inst  = self.inst
        relax = self._relax_model
        pump  = relax.copy()
        pump.Params.OutputFlag    = 0
        pump.Params.Threads       = 1
        pump.Params.OptimalityTol = _OPT_TOL
        pump.Params.FeasibilityTol= _FEAS_TOL
        pump.Params.Seed          = self._seed

        # Zero out the original objective — pumping problem uses φ only
        for v in pump.getVars():
            v.Obj = 0.0
        pump.ModelSense = GRB.MINIMIZE

        x_vars = pump.getVars()  # same ordering as in the relaxation
        self._x_pump_vars = x_vars

        d_vars, ub_pos, ub_neg = [], [], []

        for local_i, global_i in enumerate(inst.int_idx):
            xi = x_vars[global_i]
            di = pump.addVar(lb=0.0, ub=GRB.INFINITY, obj=1.0, name=f"d_{local_i}")
            d_vars.append(di)
            # dᵢ ≥  xᵢ - yᵢ   →   -dᵢ + xᵢ ≤ yᵢ
            c_pos = pump.addConstr(xi - di <= 0.0,  name=f"dp_{local_i}")
            # dᵢ ≥  yᵢ - xᵢ   →   -dᵢ - xᵢ ≤ -yᵢ
            c_neg = pump.addConstr(-xi - di <= 0.0, name=f"dn_{local_i}")
            ub_pos.append(c_pos)
            ub_neg.append(c_neg)

        pump.update()
        self._pump_model     = pump
        self._d_vars         = d_vars
        self._ub_constrs_pos = ub_pos
        self._ub_constrs_neg = ub_neg

    # ------------------------------------------------------------------
    # Solves
    # ------------------------------------------------------------------

    def solve_relaxation(self, theta: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """
        Solve the LP relaxation with objective θ (or the original c if θ is None).

        Returns (x_star, lp_time_seconds).
        x_star is a numpy array of length n_vars.
        """
        inst  = self.inst
        relax = self._relax_model

        if theta is None:
            # Restore original cost
            for i, v in enumerate(relax.getVars()):
                v.Obj = inst.c[i]
        else:
            for i, v in enumerate(relax.getVars()):
                v.Obj = float(theta[i])

        relax.ModelSense = GRB.MINIMIZE
        t0 = time.perf_counter()
        relax.optimize()
        lp_time = time.perf_counter() - t0

        if relax.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            # LP infeasible/unbounded — return zeros (pump will restart)
            return np.zeros(inst.n_vars), lp_time

        x_star = np.array([v.X for v in relax.getVars()])
        return x_star, lp_time

    def solve_pumping(
        self,
        theta: np.ndarray,
        y_int: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Solve the pumping problem (eq. 13 / eq. 17):
            min φ(x; θ, y) = Σᵢ |θᵢ| |xᵢ - yᵢ|
            s.t. (x, u) ∈ X̂

        Returns (x_star_int, u_star, lp_time).
          x_star_int : (n_int,)  — integer-variable slice of optimal solution
          u_star      : (n_cont,) — continuous-variable slice
          lp_time     : wall seconds for the Gurobi solve
        """
        inst = self.inst
        pump = self._pump_model

        # Update d-variable objective coefficients to |θᵢ|
        for local_i, di in enumerate(self._d_vars):
            di.Obj = float(abs(theta[local_i]))

        # Update RHS of linearisation constraints to current yᵢ
        for local_i, (c_pos, c_neg) in enumerate(
            zip(self._ub_constrs_pos, self._ub_constrs_neg, strict=True)
        ):
            yi = float(y_int[local_i])
            c_pos.RHS =  yi   # xᵢ - dᵢ ≤ yᵢ
            c_neg.RHS = -yi   # -xᵢ - dᵢ ≤ -yᵢ

        pump.update()
        t0 = time.perf_counter()
        pump.optimize()
        lp_time = time.perf_counter() - t0

        if pump.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            # Fallback: return current y padded with zeros for continuous vars
            x_star_int = y_int.copy().astype(float)
            u_star = np.zeros(inst.n_cont)
            return x_star_int, u_star, lp_time

        x_all = np.array([v.X for v in self._x_pump_vars])
        x_star_int = x_all[inst.int_idx]
        u_star     = x_all[inst.cont_idx] if inst.cont_idx else np.zeros(0)

        return x_star_int, u_star, lp_time

    def dispose(self) -> None:
        """Free Gurobi resources."""
        if self._pump_model is not None:
            self._pump_model.dispose()
        if self._relax_model is not None:
            self._relax_model.dispose()
