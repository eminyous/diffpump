"""
Loss components for the Differentiable Feasibility Pump.

General loss (eq. 18):
    L(θ) = α C(θ) + β f(x*(θ)) + λ g(⌊x*(θ)⌉) + γ Ω(θ)

In all paper experiments α = 0.

Implemented components
----------------------
  integrality_loss(x, p)     — f(x) = Σᵢ |xᵢ − ⌊xᵢ⌉|^p           (eq. 8)
  feasibility_loss_pi(x, A_int, b, norms, eps_feas)
                             — g(x) pure-integer  (eq. 19)
  feasibility_loss_mi_first(x_int, u, A, b, norms, eps_feas)
                             — g(x,u) mixed-integer first approach (eq. 20),
                               gradient propagated to x_int only
  feasibility_loss_mi_argmin(x_int, A, b, norms, eps_feas, q)
                             — g(x) = argminᵤ G(x,u)               (eq. 21)
  regularisation(theta, gamma) — Ω(θ) = ‖θ‖²/2
  total_loss(...)            — combines all four components

The surrogate Jacobian J_θ x*(θ) = −I (eq. 7) is implemented via a
custom autograd Function in surrogate.py and is threaded through the
loss computation automatically.

All functions accept and return torch.Tensor so that L.backward() gives
∇_θ L through PyTorch autograd.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize as sco
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Integrality loss  f(x) = Σᵢ |xᵢ − ⌊xᵢ⌉|^p   (eq. 8)
# ---------------------------------------------------------------------------

def integrality_loss(x: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    """
    Integrality loss (eq. 8).

    Parameters
    ----------
    x : (n_int,) continuous values for integer variables (from LP solve)
    p : loss order (p=1 recovers FP; p=2 is convex on each unit interval)

    Returns
    -------
    Scalar tensor  f(x) = Σᵢ |xᵢ − ⌊xᵢ⌉|^p.

    The gradient is computed by PyTorch autograd; at half-integer points
    torch.floor is not differentiable but the gradient is zero there by
    convention (the loss is at a local max/min).
    """
    # floor(x) has zero gradient everywhere (step function), so we detach
    # it to avoid zero gradients blocking the computation.  The fractional
    # part frac = x - floor(x) is differentiable in x; gradient is 1.
    floor_x = torch.floor(x.detach())
    frac    = x - floor_x                    # fractional part in [0,1)
    # Distance to nearest integer = min(frac, 1-frac)
    dist    = torch.where(frac <= 0.5, frac, 1.0 - frac)
    if p == 1.0:
        return dist.sum()
    return (dist ** p).sum()


# ---------------------------------------------------------------------------
# Feasibility loss helpers
# ---------------------------------------------------------------------------

def _slack(x_full: torch.Tensor, A: torch.Tensor, b: torch.Tensor,
           norms: torch.Tensor) -> torch.Tensor:
    """
    Normalised slack of all constraints:
        sⱼ(x) = (bⱼ − Aⱼ x) / ‖[Aⱼ bⱼ]‖₂     (paper §4.2)

    Returns (m,) tensor.  Positive sⱼ means constraint j is violated.
    """
    lhs = (A @ x_full)           # (m,)
    slack = (b - lhs) / norms    # (m,)
    return slack


def _relu_avg(slack: torch.Tensor, eps_feas: float) -> torch.Tensor:
    """Average ReLU infeasibility with ε offset: (1/m) Σⱼ ReLU(sⱼ − ε)."""
    return F.relu(slack - eps_feas).mean()


# ---------------------------------------------------------------------------
# Pure-integer feasibility loss  g(x)  (eq. 19)
# ---------------------------------------------------------------------------

def feasibility_loss_pi(
    x_int: torch.Tensor,
    A_int: torch.Tensor,
    b: torch.Tensor,
    norms: torch.Tensor,
    eps_feas: float = 0.0,
) -> torch.Tensor:
    """
    Feasibility loss for pure-integer instances (eq. 19).

    The constraint matrix slice A_int contains only the columns corresponding
    to integer variables (pre-selected by the caller).

    Parameters
    ----------
    x_int    : (n_int,) LP solution for integer variables (differentiable)
    A_int    : (m, n_int) constraint matrix integer-variable columns
    b        : (m,) RHS vector
    norms    : (m,) row norms ‖[Aⱼ bⱼ]‖₂
    eps_feas : ε offset to avoid numerical instability (paper §4.2; default 0)

    Returns
    -------
    Scalar g(x).
    """
    slack = _slack(x_int, A_int, b, norms)
    return _relu_avg(slack, eps_feas)


# ---------------------------------------------------------------------------
# Mixed-integer feasibility loss — first approach  g(x, u)  (eq. 20)
# Gradient propagated to x_int only; u is detached.
# ---------------------------------------------------------------------------

def feasibility_loss_mi_first(
    x_int: torch.Tensor,
    u: torch.Tensor,
    A_int: torch.Tensor,
    A_cont: torch.Tensor,
    b: torch.Tensor,
    norms: torch.Tensor,
    eps_feas: float = 0.0,
) -> torch.Tensor:
    """
    Mixed-integer feasibility loss, first approach (eq. 20).

    Uses current u (from pumping solve) in the forward pass.
    Gradient is propagated only to x_int; u is detached.

    Parameters
    ----------
    x_int  : (n_int,)  integer-variable slice of x* (differentiable)
    u      : (n_cont,) continuous-variable slice of x* (detached)
    A_int  : (m, n_int)  integer-variable columns of A
    A_cont : (m, n_cont) continuous-variable columns of A
    b, norms, eps_feas : same as feasibility_loss_pi
    """
    u_det = u.detach()
    # Forward: Aⱼ x = Aⱼˣ xᵢₙₜ + Aⱼᵘ u
    lhs   = A_int @ x_int + A_cont @ u_det   # (m,)
    slack = (b - lhs) / norms
    return _relu_avg(slack, eps_feas)


# ---------------------------------------------------------------------------
# Mixed-integer feasibility loss — argmin approach  g(x) = argminᵤ G  (eq. 21)
# ---------------------------------------------------------------------------

class _ArgminFeasLoss(torch.autograd.Function):
    """
    Custom Function for the argmin feasibility loss (eq. 21).

    Forward:  solve  min_u G(x,u) = (1/m) Σⱼ gⱼ(x,u)^q
    Backward: ∂g/∂x = ∂G/∂x|_{u=u*(x)}   (closed-form, eq. after 21)
              The ∂G/∂u · ∂u/∂x term vanishes because ∇_u G = 0 at optimum.
    """

    @staticmethod
    def forward(ctx, x_int: torch.Tensor,
                A_int_t, A_cont_t, b_t, norms_t,
                eps_feas: float, q: int):
        x_np = x_int.detach().numpy()
        A_int_np  = A_int_t.numpy()
        A_cont_np = A_cont_t.numpy()
        b_np      = b_t.numpy()
        norms_np  = norms_t.numpy()

        n_cont = A_cont_np.shape[1]

        def G(u):
            lhs   = A_int_np @ x_np + A_cont_np @ u
            slack = (b_np - lhs) / norms_np
            g_j   = np.maximum(slack - eps_feas, 0.0)
            return np.sum(g_j ** q) / len(b_np)

        def dG_du(u):
            lhs   = A_int_np @ x_np + A_cont_np @ u
            slack = (b_np - lhs) / norms_np
            g_j   = np.maximum(slack - eps_feas, 0.0)
            # ∂G/∂u = (q/m) Σⱼ gⱼ^(q-1) · ∂gⱼ/∂u
            # ∂gⱼ/∂u = -(Aⱼᵘ / ‖[Aⱼ bⱼ]‖₂)   where gⱼ > 0, else 0
            active = (g_j > 0).astype(float)
            coeff  = q * g_j ** max(q - 1, 0) * active / len(b_np)
            dg_du  = -(A_cont_np.T / norms_np) @ coeff   # (n_cont,)
            return dg_du

        u0     = np.zeros(n_cont)
        result = sco.minimize(G, u0, jac=dG_du, method="L-BFGS-B",
                              options={"maxiter": 200, "ftol": 1e-12})
        u_opt  = result.x

        g_val  = float(result.fun)

        ctx.save_for_backward(x_int,
                              torch.tensor(u_opt, dtype=torch.double),
                              A_int_t, A_cont_t, b_t, norms_t)
        ctx.eps_feas = eps_feas
        ctx.q = q
        return torch.tensor(g_val, dtype=torch.double)

    @staticmethod
    def backward(ctx, grad_output):
        x_int, u_opt, A_int_t, A_cont_t, b_t, norms_t = ctx.saved_tensors
        eps_feas = ctx.eps_feas
        q        = ctx.q

        x_np = x_int.numpy()
        u_np = u_opt.numpy()
        A_int_np  = A_int_t.numpy()
        A_cont_np = A_cont_t.numpy()
        b_np      = b_t.numpy()
        norms_np  = norms_t.numpy()

        lhs   = A_int_np @ x_np + A_cont_np @ u_np
        slack = (b_np - lhs) / norms_np
        g_j   = np.maximum(slack - eps_feas, 0.0)

        # ∂G/∂x|_{u=u*(x)}:
        # ∂gⱼ/∂x = -(Aⱼˣ / ‖[Aⱼ bⱼ]‖₂)   where gⱼ > 0, else 0
        active = (g_j > 0).astype(float)
        coeff  = q * g_j ** max(q - 1, 0) * active / len(b_np)
        dG_dx  = -(A_int_np.T / norms_np) @ coeff   # (n_int,)

        grad_x = torch.tensor(dG_dx, dtype=torch.double) * grad_output
        # gradients for non-Tensor args: None
        return grad_x, None, None, None, None, None, None


def feasibility_loss_mi_argmin(
    x_int: torch.Tensor,
    A_int: torch.Tensor,
    A_cont: torch.Tensor,
    b: torch.Tensor,
    norms: torch.Tensor,
    eps_feas: float = 0.0,
    q: int = 2,
) -> torch.Tensor:
    """
    Mixed-integer feasibility loss, argmin approach (eq. 21).

    Solves min_u G(x,u) = (1/m) Σⱼ gⱼ(x,u)^q for fixed x (L-BFGS-B),
    then returns g = G(x, u*(x)).

    Gradient wrt x is the closed-form ∂G/∂x at u=u*(x) (the ∂u/∂x term
    drops out because ∇_u G = 0 at the optimum).

    Parameters
    ----------
    x_int  : (n_int,) integer-variable slice (differentiable)
    A_int  : (m, n_int)
    A_cont : (m, n_cont)
    b      : (m,)
    norms  : (m,)
    eps_feas : ε offset
    q      : 1 or 2 (paper uses q=2 for DP5)
    """
    return _ArgminFeasLoss.apply(
        x_int.double(),
        A_int.double(), A_cont.double(),
        b.double(), norms.double(),
        eps_feas, q,
    )


# ---------------------------------------------------------------------------
# Regularisation  Ω(θ) = ‖θ‖²/2
# ---------------------------------------------------------------------------

def regularisation(theta: torch.Tensor) -> torch.Tensor:
    """L2 regularisation on θ:  Ω(θ) = ‖θ‖²/2."""
    return 0.5 * (theta * theta).sum()


# ---------------------------------------------------------------------------
# Total loss  L(θ)  (eq. 18, with α = 0)
# ---------------------------------------------------------------------------

def total_loss(
    theta: torch.Tensor,
    x_int: torch.Tensor,
    y_int_rounded: torch.Tensor | None,
    *,
    beta: float,
    lam: float,
    gamma: float,
    p: float,
    feas_loss_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    L(θ) = β f(x*) + λ g(⌊x*⌉) + γ Ω(θ)     (eq. 18, α = 0)

    Parameters
    ----------
    theta           : (n_int,) parameter vector (differentiable)
    x_int           : (n_int,) LP solution for integer vars (differentiable via surrogate)
    y_int_rounded   : (n_int,) soft-rounded x_int for feasibility loss (or None if λ=0)
    beta            : weight on integrality loss
    lam             : weight on feasibility loss
    gamma           : regularisation weight
    p               : integrality loss order
    feas_loss_tensor: pre-computed feasibility loss tensor (or None if λ=0)

    Returns
    -------
    Scalar loss tensor.
    """
    L = torch.tensor(0.0, dtype=theta.dtype, requires_grad=False)

    if beta != 0.0:
        L = L + beta * integrality_loss(x_int, p)

    if lam != 0.0 and feas_loss_tensor is not None:
        L = L + lam * feas_loss_tensor

    L = L + gamma * regularisation(theta)
    return L
