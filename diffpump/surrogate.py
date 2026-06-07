"""
Surrogate LP layer: wraps the Gurobi pumping solve as a torch.autograd.Function
with the -I surrogate Jacobian (eq. 7).

Only the integer-variable slice of x* is connected to θ through this Jacobian;
continuous variables are returned detached (§3.3, Proposition 1).
"""

from __future__ import annotations

import numpy as np
import torch

from diffpump.gurobi_model import GurobiPumpModel


class _SurrogatePump(torch.autograd.Function):
    """
    Forward : call the Gurobi pumping solver and return x*_int as a Tensor.
    Backward: J_θ x*_int = -I  →  ∂L/∂θ = -∂L/∂x*_int   (eq. 7)
    """

    @staticmethod
    def forward(ctx, theta: torch.Tensor, y_int_np, gurobi_model: GurobiPumpModel):
        """
        theta      : (n_int,) float Tensor — the parameter vector for integer vars
        y_int_np   : (n_int,) numpy array — current integer iterate y^(k)
        gurobi_model: GurobiPumpModel instance

        Returns x_int_tensor : (n_int,) Tensor (connected to theta via -I backward)
                u_tensor      : (n_cont,) Tensor (detached — not used in gradient)
                lp_time       : scalar Tensor (for timing; no gradient)
        """
        theta_np = theta.detach().numpy()

        x_int_np, u_np, lp_time = gurobi_model.solve_pumping(theta_np, y_int_np)

        x_int_tensor = torch.tensor(x_int_np, dtype=theta.dtype)
        u_tensor     = torch.tensor(u_np,     dtype=theta.dtype)
        return x_int_tensor, u_tensor, torch.tensor(lp_time)

    @staticmethod
    def backward(ctx, grad_x_int, grad_u, grad_lp_time):
        """
        Surrogate Jacobian  J_θ x* = -I  →  grad_theta = -grad_x_int.
        Gradients for y_int_np and gurobi_model are None (not Tensors).
        """
        return -grad_x_int, None, None


def surrogate_pump(
    theta: torch.Tensor,
    y_int_np: np.ndarray,
    gurobi_model: GurobiPumpModel,
):
    """
    Run the pumping LP and return (x_int, u, lp_time) with the -I surrogate Jacobian.

    x_int is connected to theta in the autograd graph.
    u is detached (continuous variables not updated by gradient).
    lp_time is a scalar Tensor (for recording; no gradient).
    """
    return _SurrogatePump.apply(theta, y_int_np, gurobi_model)
