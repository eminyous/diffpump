"""
Hard rounding and soft (differentiable) rounding.

Hard rounding
-------------
  y_i = clamp(round(x_i), lb_i, ub_i)

Soft rounding (§4.3)
--------------------
Binary case (eq. 24):
  J_x ⌊x⌉_ε = (1/ε) φ_N( (0.5 − x) / ε )

General-integer case (eq. 25):
  J_x ⌊x⌉_ε = (1/ε) φ_N( (0.5 − x + ⌊x⌋) / ε )

where φ_N is the standard-Gaussian PDF.

Both are implemented as a torch.autograd.Function so they participate in the
autograd graph and the Gaussian-PDF Jacobian is used in the backward pass.
"""

from __future__ import annotations

from math import pi, sqrt

import numpy as np
import torch

_SQRT2PI = sqrt(2.0 * pi)
_EPS_GUARD = 1e-7  # avoid division by zero inside the PDF


def hard_round(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """
    Component-wise nearest-integer rounding, clamped to [lb, ub].

    Parameters
    ----------
    x   : (n_int,) continuous values for integer variables only
    lb  : (n_int,) lower bounds
    ub  : (n_int,) upper bounds

    Returns
    -------
    y : (n_int,) integer array (dtype float for compatibility)
    """
    y = np.round(x)
    y = np.clip(y, lb, ub)
    return y


# ---------------------------------------------------------------------------
# Soft rounding — differentiable surrogate (eqs. 24, 25)
# ---------------------------------------------------------------------------

class SoftRoundFunction(torch.autograd.Function):
    """
    Forward: return hard rounding (or an approximation thereof).
    Backward: use the closed-form Gaussian-PDF Jacobian (eq. 24 / eq. 25).

    The forward pass returns the *hard* rounded value so that the actual
    integer iterate y^(k) used in the pumping problem is unchanged.
    Only the backward gradient is "soft" (uses the PDF surrogate).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float) -> torch.Tensor:
        """
        x   : (...) float tensor of continuous values (integer-variable slice)
        eps : soft-rounding temperature parameter ε > 0

        Returns ⌊x⌉  (component-wise nearest integer, as float).
        """
        with torch.no_grad():
            floor_x = torch.floor(x)
            frac    = x - floor_x
            y_hard  = floor_x + (frac >= 0.5).to(x.dtype)
        ctx.save_for_backward(x)
        ctx.eps = eps
        return y_hard

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Jacobian surrogate: J = diag( (1/ε) φ_N( (0.5 − x + ⌊x⌋) / ε ) )   (eq. 25)
        For binary x ∈ [0,1]: ⌊x⌋ = 0 so this reduces to eq. 24.
        """
        (x,) = ctx.saved_tensors
        eps   = ctx.eps
        floor_x = torch.floor(x)
        z       = (0.5 - x + floor_x) / (eps + _EPS_GUARD)
        # Standard Gaussian PDF: φ(z) = exp(-z²/2) / √(2π)
        phi     = torch.exp(-0.5 * z * z) / _SQRT2PI
        jacobian_diag = phi / (eps + _EPS_GUARD)  # (1/ε) φ(z)
        return grad_output * jacobian_diag, None


def soft_round(x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Apply soft rounding to x with temperature ε.

    x   : (...) float tensor (integer-variable slice of x*)
    eps : ε > 0, default 0.15 in paper §5.1

    Returns ⌊x⌉ (hard, numerically) with gradient given by eq. 25.
    """
    return SoftRoundFunction.apply(x, eps)


def soft_round_jacobian(x: np.ndarray, eps: float) -> np.ndarray:
    """
    Compute the diagonal of the soft-rounding Jacobian J_x⌊x⌉_ε (eq. 25)
    without building a torch graph.  Used for finite-difference checks.

    Returns (n_int,) array of Jacobian diagonal entries.
    """
    floor_x = np.floor(x)
    z       = (0.5 - x + floor_x) / (eps + _EPS_GUARD)
    phi     = np.exp(-0.5 * z * z) / _SQRT2PI
    return phi / (eps + _EPS_GUARD)
