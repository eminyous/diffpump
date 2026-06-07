"""
Configuration for the Differentiable Feasibility Pump.

All hyperparameters can be set freely; the defaults reproduce the behaviour of
the original feasibility pump (η=γ=β=1, λ=0, p=1, no argmin feasibility).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VariantConfig:
    """Full configuration for one run of the pump algorithm."""

    # Gradient descent step size η
    eta: float = 1.0
    # Regularisation weight γ  (Ω term)
    gamma: float = 1.0
    # Integrality loss weight β
    beta: float = 1.0
    # Feasibility loss weight λ
    lam: float = 0.0
    # Integrality loss order p
    p: float = 1.0

    # Feasibility loss options
    use_argmin_feas: bool = False   # True → argmin feasibility (eq. 21); False → eq. 19/20
    q: int = 2                      # exponent in argmin feasibility (eq. 21)
    eps_feas: float = 0.0           # ε in ReLU(sⱼ − ε)

    # Soft rounding temperature ε
    eps_soft: float = 0.15

    # Stopping criteria
    max_iters: int = 1000
    time_limit: float | None = None
