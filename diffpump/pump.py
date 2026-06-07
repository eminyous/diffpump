"""
Algorithm 2 — Differentiable Feasibility Pump main loop.

Implements the general integer / mixed-integer version of the DFP
(Cacciola, Emine, Forel, Frangioni, Lodi, MPB 2025).

Entry point: run_instance(inst, config, gurobi_model)

The loop:
  1. Initialise x^(0) by solving the LP relaxation with cost c.
  2. Round x^(0) → y^(0).
  3. At each iteration k:
       a. Check if y^(k) is feasible  → SUCCESS.
       b. Detect cycles → flip or perturb y^(k).
       c. Solve the pumping problem to get x^(k+1).
       d. Compute loss L(θ) using the surrogate Jacobian.
       e. Update θ ← θ − η ∇_θ L.
       f. Round x^(k+1) → y^(k+1).
  4. If k ≥ N_max → FAILURE.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from diffpump.losses import (
    feasibility_loss_mi_argmin,
    feasibility_loss_mi_first,
    feasibility_loss_pi,
    integrality_loss,
    regularisation,
)
from diffpump.problem import Instance, check_feasibility
from diffpump.restarts import CycleDetector, Restarter
from diffpump.results import ResultRecord
from diffpump.rounding import hard_round, soft_round
from diffpump.surrogate import surrogate_pump

if TYPE_CHECKING:
    from diffpump.variants import VariantConfig


def _make_pump_model(inst: Instance, solver: str, seed: int):
    if solver == "gurobi":
        from diffpump.gurobi_model import GurobiPumpModel
        return GurobiPumpModel(inst, seed=seed)
    if solver == "scip":
        from diffpump.scip_model import ScipPumpModel
        return ScipPumpModel(inst, seed=seed)
    raise ValueError(f"Unknown solver '{solver}'. Choose 'gurobi' or 'scip'.")


def run_instance(
    inst: Instance,
    config: VariantConfig,
    seed: int = 0,
    solver: str = "gurobi",
) -> ResultRecord:
    """
    Run the differentiable feasibility pump on one instance.

    Parameters
    ----------
    inst   : MIPInstance  — loaded and pre-processed instance
    config : VariantConfig — algorithm variant and hyperparameters
    seed   : int — random seed for restarts and Gurobi

    Returns
    -------
    ResultRecord with per-instance metrics (success, iters, restarts, time).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # --- Pre-compute constraint matrix slices (used in feasibility loss) ---
    A     = inst.A        # (m, n_vars)
    b_np  = inst.b        # (m,)
    norms = inst.row_norms  # (m,)

    A_int_np  = A[:, inst.int_idx]                          # (m, n_int)
    A_cont_np = A[:, inst.cont_idx] if inst.cont_idx else np.zeros((len(b_np), 0))

    A_int_t  = torch.tensor(A_int_np,  dtype=torch.double)
    A_cont_t = torch.tensor(A_cont_np, dtype=torch.double)
    b_t      = torch.tensor(b_np,      dtype=torch.double)
    norms_t  = torch.tensor(norms,     dtype=torch.double)

    lb_int = inst.lb[inst.int_idx]
    ub_int = inst.ub[inst.int_idx]

    gm = _make_pump_model(inst, solver, seed)
    gm.build()

    # --- Initialise x^(0) by solving LP with original cost c ---
    t_wall_start = time.perf_counter()

    x0_full, lp0_time = gm.solve_relaxation(theta=None)
    x0_int = x0_full[inst.int_idx]

    # --- Initialise θ^(0) = c restricted to integer variables ---
    c_int = inst.c[inst.int_idx]
    if np.all(c_int == 0):
        theta_np = np.ones(inst.n_int, dtype=float)
    else:
        theta_np = c_int.astype(float).copy()

    # --- Round x^(0) → y^(0) ---
    y_int = hard_round(x0_int, lb_int, ub_int)

    # --- Set up cycle detection and restarts ---
    detector = CycleDetector()
    restarter = Restarter(lb_int, ub_int, inst.is_binary, rng)

    detector.record(y_int)

    n_iters     = 0
    n_restarts  = 0
    lp_time_tot = lp0_time
    success     = False
    timed_out   = False
    for _ in range(config.max_iters):
        n_iters += 1

        # (a) Feasibility check
        if check_feasibility(y_int, inst):
            success = True
            break

        if (config.time_limit is not None
                and time.perf_counter() - t_wall_start >= config.time_limit):
            timed_out = True
            break

        # (b) Cycle detection + restart
        restart_type = detector.check(y_int)
        if restart_type != "none":
            y_int, _ = restarter.apply(restart_type, y_int)
            n_restarts += 1

        detector.record(y_int)

        # (c) Solve pumping problem with current θ (surrogate Jacobian −I).
        # theta_t requires_grad=True so the backward of _SurrogatePump routes
        # ∂L/∂θ = −∂L/∂x_int through it automatically.
        theta_t = torch.tensor(theta_np, dtype=torch.double, requires_grad=True)

        x_int_t, u_t, lp_time_t = surrogate_pump(theta_t, y_int, gm)
        lp_time_tot += lp_time_t.item()

        x_int_np = x_int_t.detach().numpy()
        u_np     = u_t.detach().numpy()

        # (d) Compute loss L(θ)  — all terms must flow through theta_t's graph
        L = torch.zeros(1, dtype=torch.double, requires_grad=False).squeeze()

        if config.beta != 0.0:
            L = L + config.beta * integrality_loss(x_int_t, config.p)

        if config.lam != 0.0:
            # Apply soft rounding to get ⌊x*⌉ with differentiable Jacobian
            y_soft_t = soft_round(x_int_t, config.eps_soft)

            if config.use_argmin_feas:
                # DP5: eq. 21 with q
                feas_t = feasibility_loss_mi_argmin(
                    y_soft_t, A_int_t, A_cont_t, b_t, norms_t,
                    eps_feas=config.eps_feas, q=config.q,
                )
            elif inst.n_cont > 0:
                # Mixed-integer: eq. 20 first approach
                u_t_det = torch.tensor(u_np, dtype=torch.double)
                feas_t = feasibility_loss_mi_first(
                    y_soft_t, u_t_det, A_int_t, A_cont_t, b_t, norms_t,
                    eps_feas=config.eps_feas,
                )
            else:
                # Pure-integer: eq. 19
                feas_t = feasibility_loss_pi(
                    y_soft_t, A_int_t, b_t, norms_t,
                    eps_feas=config.eps_feas,
                )
            L = L + config.lam * feas_t

        L = L + config.gamma * regularisation(theta_t)

        # (e) Gradient update  θ ← θ − η ∇_θ L
        L.backward()
        with torch.no_grad():
            grad = theta_t.grad
            theta_np = theta_np - config.eta * grad.numpy()

        # (f) Round new x* → y^(k+1)
        y_int = hard_round(x_int_np, lb_int, ub_int)

    wall_time = time.perf_counter() - t_wall_start

    gm.dispose()

    return ResultRecord(
        instance_name=inst.name,
        solver=solver,
        success=success,
        timed_out=timed_out,
        n_iters=n_iters,
        n_restarts=n_restarts,
        restart_ratio=n_restarts / max(n_iters, 1),
        wall_time=wall_time,
        lp_time=lp_time_tot,
        eta=config.eta,
        gamma=config.gamma,
        beta=config.beta,
        lam=config.lam,
        p=config.p,
        q=config.q,
        use_argmin_feas=config.use_argmin_feas,
        eps_soft=config.eps_soft,
        eps_feas=config.eps_feas,
        seed=seed,
    )
