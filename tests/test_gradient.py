"""
Tests for the DFP gradient mechanism (Algorithm 2, eq. 7).

Verifies that:
  1. The -I surrogate Jacobian routes grad_theta = -grad_x_int exactly.
  2. The gradient of L w.r.t. theta is nonzero after each pump step.
  3. theta is updated by the gradient step (the loop is not a no-op).
"""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from diffpump.gurobi_model import GurobiPumpModel
from diffpump.losses import integrality_loss, regularisation
from diffpump.problem import load_instance
from diffpump.rounding import hard_round
from diffpump.surrogate import surrogate_pump

_MPS_BINARY = textwrap.dedent("""\
NAME          tiny_binary
ROWS
 N  OBJ
 G  CON1
COLUMNS
    MARKER   'MARKER'  'INTORG'
    x1       OBJ   1.0   CON1  1.0
    x2       OBJ   1.0   CON1  1.0
    MARKER   'MARKER'  'INTEND'
RHS
    RHS      CON1  1.0
BOUNDS
 UI BND      x1    1
 UI BND      x2    1
ENDATA
""")


def _write_tmp_mps(content: str) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mps", delete=False) as f:
        f.write(content)
        return Path(f.name)


@pytest.fixture(scope="module")
def binary_setup():
    """Load the tiny binary instance and build the Gurobi pump model once."""
    inst = load_instance(_write_tmp_mps(_MPS_BINARY), seed=0)
    gm = GurobiPumpModel(inst, seed=0)
    gm.build()
    x0, _ = gm.solve_relaxation()
    lb = inst.lb[inst.int_idx]
    ub = inst.ub[inst.int_idx]
    y_int = hard_round(x0[inst.int_idx], lb, ub)
    yield inst, gm, y_int
    gm.dispose()


class TestSurrogateJacobian:
    """Directly verify the -I surrogate Jacobian (eq. 7)."""

    def test_jacobian_is_minus_identity(self, binary_setup):
        """
        With loss = sum(x_int), grad_x_int = ones(n_int).
        The -I surrogate must give grad_theta = -ones(n_int), regardless of x*.
        This holds even when x* is integer — the backward path is unconditional.
        """
        inst, gm, y_int = binary_setup
        theta_t = torch.ones(inst.n_int, dtype=torch.double, requires_grad=True)

        x_int_t, _, _ = surrogate_pump(theta_t, y_int, gm)
        x_int_t.sum().backward()

        np.testing.assert_allclose(
            theta_t.grad.numpy(), -np.ones(inst.n_int), atol=1e-10
        )

    def test_jacobian_scales_with_upstream_gradient(self, binary_setup):
        """
        With loss = 3 * sum(x_int), grad_x_int = 3*ones.
        The -I surrogate must give grad_theta = -3*ones.
        """
        inst, gm, y_int = binary_setup
        theta_t = torch.ones(inst.n_int, dtype=torch.double, requires_grad=True)

        x_int_t, _, _ = surrogate_pump(theta_t, y_int, gm)
        (3.0 * x_int_t.sum()).backward()

        np.testing.assert_allclose(
            theta_t.grad.numpy(), -3.0 * np.ones(inst.n_int), atol=1e-10
        )


class TestGradientUpdate:
    """Verify the gradient step in Algorithm 2 is live."""

    def test_combined_gradient_formula(self, binary_setup):
        """
        With y_int feasible the pumping LP returns x* = y_int (integer).
        At integer x*, integrality_loss = 0 but its subgradient w.r.t. x is ones
        (torch.where selects the frac branch whose slope is 1 at frac=0).
        Via the -I surrogate: grad_theta_int = -ones.
        Regularisation adds theta itself: grad_theta_reg = theta.
        Total: grad_theta = theta - ones (component-wise).

        We verify this formula with theta = [2.0, 3.0], expecting grad = [1.0, 2.0].
        """
        inst, gm, y_int = binary_setup
        theta_vals = np.array([2.0, 3.0])
        theta_t = torch.tensor(theta_vals, dtype=torch.double, requires_grad=True)

        x_int_t, _, _ = surrogate_pump(theta_t, y_int, gm)
        L = integrality_loss(x_int_t, p=1) + regularisation(theta_t)
        L.backward()

        expected = theta_vals - 1.0  # [1.0, 2.0]
        np.testing.assert_allclose(theta_t.grad.numpy(), expected, atol=1e-10)

    def test_theta_changes_after_one_step(self, binary_setup):
        """
        After one gradient step, theta must differ from its initial value.
        Uses theta=[0.5, 0.5] where grad = theta - 1 = [-0.5, -0.5] != 0.
        """
        inst, gm, y_int = binary_setup
        theta_np = np.array([0.5, 0.5])
        theta_t = torch.tensor(theta_np, dtype=torch.double, requires_grad=True)

        x_int_t, _, _ = surrogate_pump(theta_t, y_int, gm)
        L = integrality_loss(x_int_t, p=1) + regularisation(theta_t)
        L.backward()

        new_theta = theta_np - 0.5 * theta_t.grad.numpy()
        assert not np.allclose(new_theta, theta_np, atol=1e-12), (
            "theta did not change after gradient step"
        )

    def test_theta_changes_over_multiple_steps(self, binary_setup):
        """
        Over three gradient steps starting from theta=[0.5, 0.5], theta must
        keep changing each iteration — the update does not collapse to zero.
        (The fixed point is theta=ones; starting away from it, updates are nonzero.)
        """
        inst, gm, y_int = binary_setup
        theta_np = np.array([0.5, 0.5])
        eta = 0.3
        seen = [theta_np.copy()]

        for _ in range(3):
            theta_t = torch.tensor(theta_np, dtype=torch.double, requires_grad=True)
            x_int_t, _, _ = surrogate_pump(theta_t, y_int, gm)
            L = integrality_loss(x_int_t, p=1) + regularisation(theta_t)
            L.backward()
            theta_np = theta_np - eta * theta_t.grad.numpy()
            seen.append(theta_np.copy())

        for i in range(len(seen) - 1):
            assert not np.allclose(seen[i], seen[i + 1], atol=1e-12), (
                f"theta did not change between step {i} and {i + 1}"
            )
