"""
SCIP-based LP solver for the pump (pyscipopt backend).

Instance loading always uses Gurobi (load_instance).  This module uses
pyscipopt only for the LP solves inside the pump loop, building both
models from the already-extracted MIPInstance data so that variable
ordering is guaranteed to match.
"""

from __future__ import annotations

import time

import numpy as np
import pyscipopt

from diffpump.problem import Instance


class ScipPumpModel:
    """
    LP relaxation and pumping-problem model backed by SCIP.

    Mirrors the GurobiPumpModel interface exactly so it can be swapped in
    transparently in the pump loop.

    Layout
    ------
    Relaxation model  : min c·x  s.t. Ax ≥ b,  lb ≤ x ≤ ub  (all continuous)
    Pumping model     : min Σᵢ |θᵢ| dᵢ  s.t. Ax ≥ b  +  linearisation of φ(x;θ,y)
      Two constraints per integer variable i (RHS updated each iteration):
          xᵢ − dᵢ ≤  yᵢ    (dᵢ ≥  xᵢ − yᵢ)
         −xᵢ − dᵢ ≤ −yᵢ   (dᵢ ≥  yᵢ − xᵢ)
    """

    def __init__(self, inst: Instance, seed: int = 0) -> None:
        self.inst   = inst
        self._seed  = seed
        self._relax_model  = None
        self._relax_vars:  list = []
        self._pump_model   = None
        self._pump_x_vars: list = []
        self._pump_d_vars: list = []
        self._pump_c_pos:  list = []   # dᵢ ≥  xᵢ − yᵢ  constraints
        self._pump_c_neg:  list = []   # dᵢ ≥  yᵢ − xᵢ  constraints

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build both LP models. Call once per instance."""
        self._build_relaxation()
        self._build_pump_model()

    def _new_model(self):
        m = pyscipopt.Model()
        m.hideOutput()
        m.setIntParam("randomization/randomseedshift", self._seed)
        return m

    def _add_constraints(self, m, x_vars) -> None:
        """Add A @ x >= b to model m, skipping all-zero rows."""
        inst = self.inst
        qs   = pyscipopt.quicksum
        for j in range(len(inst.b)):
            row = inst.A[j]
            nz  = np.nonzero(row)[0]
            if len(nz) == 0:
                continue
            m.addCons(
                qs(float(row[k]) * x_vars[k] for k in nz) >= float(inst.b[j]),
                name=f"c{j}",
            )

    def _build_relaxation(self) -> None:
        inst = self.inst
        m    = self._new_model()

        x_vars = [
            m.addVar(name=f"x_{i}", vtype="C",
                     lb=float(inst.lb[i]), ub=float(inst.ub[i]),
                     obj=0.0)
            for i in range(inst.n_vars)
        ]
        self._add_constraints(m, x_vars)

        self._relax_model = m
        self._relax_vars  = x_vars

    def _build_pump_model(self) -> None:
        inst = self.inst
        m    = self._new_model()

        x_vars = [
            m.addVar(name=f"x_{i}", vtype="C",
                     lb=float(inst.lb[i]), ub=float(inst.ub[i]),
                     obj=0.0)
            for i in range(inst.n_vars)
        ]
        self._add_constraints(m, x_vars)

        d_vars, c_pos, c_neg = [], [], []
        for local_i, global_i in enumerate(inst.int_idx):
            xi = x_vars[global_i]
            d  = m.addVar(name=f"d_{local_i}", vtype="C", lb=0.0, obj=0.0)
            d_vars.append(d)
            # Initialised with yᵢ = 0; RHS updated each iteration.
            c_pos.append(m.addCons(xi - d <= 0.0,        name=f"dp_{local_i}"))
            c_neg.append(m.addCons(-1.0 * xi - d <= 0.0, name=f"dn_{local_i}"))

        self._pump_model   = m
        self._pump_x_vars  = x_vars
        self._pump_d_vars  = d_vars
        self._pump_c_pos   = c_pos
        self._pump_c_neg   = c_neg

    # ------------------------------------------------------------------
    # Solves
    # ------------------------------------------------------------------

    @staticmethod
    def _free_if_solved(m) -> None:
        if m.getStatus() != "unknown":
            m.freeTransform()

    def solve_relaxation(self, theta: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """
        Solve the LP relaxation with objective θ (or original c if θ is None).

        Returns (x_star, lp_time_seconds).
        """
        inst = self.inst
        m    = self._relax_model
        qs   = pyscipopt.quicksum

        self._free_if_solved(m)

        coeffs = theta if theta is not None else inst.c
        m.setObjective(
            qs(float(coeffs[i]) * v for i, v in enumerate(self._relax_vars)),
            sense="minimize",
        )

        t0 = time.perf_counter()
        m.optimize()
        lp_time = time.perf_counter() - t0

        if m.getStatus() != "optimal":
            return np.zeros(inst.n_vars), lp_time

        return np.array([m.getVal(v) for v in self._relax_vars]), lp_time

    def solve_pumping(
        self,
        theta: np.ndarray,
        y_int: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Solve the pumping LP (eq. 13 / eq. 17).

        Returns (x_star_int, u_star, lp_time).
        """
        inst = self.inst
        m    = self._pump_model
        qs   = pyscipopt.quicksum

        self._free_if_solved(m)

        m.setObjective(
            qs(float(abs(theta[i])) * d for i, d in enumerate(self._pump_d_vars)),
            sense="minimize",
        )

        for local_i, (cp, cn) in enumerate(zip(self._pump_c_pos, self._pump_c_neg, strict=True)):
            yi = float(y_int[local_i])
            m.chgRhs(cp,  yi)
            m.chgRhs(cn, -yi)

        t0 = time.perf_counter()
        m.optimize()
        lp_time = time.perf_counter() - t0

        if m.getStatus() != "optimal":
            return y_int.copy().astype(float), np.zeros(inst.n_cont), lp_time

        x_all  = np.array([m.getVal(v) for v in self._pump_x_vars])
        u_star = x_all[inst.cont_idx] if inst.cont_idx else np.zeros(0)
        return x_all[inst.int_idx], u_star, lp_time

    def dispose(self) -> None:
        """Release pyscipopt resources (garbage-collected if not called)."""
        self._pump_model  = None
        self._relax_model = None
