"""
Unit tests for integrality loss, feasibility loss, and regularisation.
All tests use numpy / torch only — no Gurobi required.
"""

import numpy as np
import pytest
import torch

from diffpump.losses import (
    feasibility_loss_mi_argmin,
    feasibility_loss_mi_first,
    feasibility_loss_pi,
    integrality_loss,
    regularisation,
)

# ---------------------------------------------------------------------------
# integrality_loss
# ---------------------------------------------------------------------------

class TestIntegralityLoss:

    def test_zero_at_integers(self):
        x = torch.tensor([0.0, 1.0, 2.0, -1.0], dtype=torch.double)
        assert integrality_loss(x, p=1).item() == pytest.approx(0.0)
        assert integrality_loss(x, p=2).item() == pytest.approx(0.0)

    def test_max_at_half_integer(self):
        x = torch.tensor([0.5], dtype=torch.double)
        assert integrality_loss(x, p=1).item() == pytest.approx(0.5)

    def test_p2_is_smaller_than_p1_midrange(self):
        x = torch.tensor([0.3], dtype=torch.double)
        f1 = integrality_loss(x, p=1).item()
        f2 = integrality_loss(x, p=2).item()
        # For 0.3 (closer to 0 than 0.5): dist=0.3; 0.3^2 = 0.09 < 0.3
        assert f2 < f1

    def test_gradient_p1(self):
        x = torch.tensor([0.3], dtype=torch.double, requires_grad=True)
        loss = integrality_loss(x, p=1)
        loss.backward()
        # frac=0.3, dist=min(0.3, 0.7)=0.3, gradient of dist wrt x = 1
        assert x.grad.item() == pytest.approx(1.0, abs=1e-6)

    def test_gradient_p2(self):
        x = torch.tensor([0.3], dtype=torch.double, requires_grad=True)
        loss = integrality_loss(x, p=2)
        loss.backward()
        # dist=0.3, d(dist^2)/dx = 2*0.3*1 = 0.6
        assert x.grad.item() == pytest.approx(0.6, abs=1e-6)

    def test_symmetry_around_half(self):
        x1 = torch.tensor([0.3], dtype=torch.double)
        x2 = torch.tensor([0.7], dtype=torch.double)
        assert integrality_loss(x1, p=2).item() == pytest.approx(
            integrality_loss(x2, p=2).item(), abs=1e-10)


# ---------------------------------------------------------------------------
# feasibility_loss_pi  (eq. 19)
# ---------------------------------------------------------------------------

class TestFeasibilityLossPI:
    """Two-constraint, one-variable example: x >= 1, x <= 3 → -x >= -3."""

    @pytest.fixture
    def simple_problem(self):
        # Constraints: x >= 1  and  -x >= -3
        A_int = torch.tensor([[1.0], [-1.0]], dtype=torch.double)
        b     = torch.tensor([1.0, -3.0],    dtype=torch.double)
        # Row norms: ‖[1,1]‖ = √2, ‖[-1,-3]‖ = √10
        norms = torch.tensor([np.sqrt(2), np.sqrt(10)], dtype=torch.double)
        return A_int, b, norms

    def test_feasible_point(self, simple_problem):
        A_int, b, norms = simple_problem
        x = torch.tensor([2.0], dtype=torch.double)  # x=2 satisfies both
        g = feasibility_loss_pi(x, A_int, b, norms, eps_feas=0.0)
        assert g.item() == pytest.approx(0.0, abs=1e-10)

    def test_infeasible_point(self, simple_problem):
        A_int, b, norms = simple_problem
        x = torch.tensor([0.0], dtype=torch.double)  # violates x >= 1
        g = feasibility_loss_pi(x, A_int, b, norms, eps_feas=0.0)
        assert g.item() > 0.0

    def test_gradient_nonzero_at_infeasible(self, simple_problem):
        A_int, b, norms = simple_problem
        x = torch.tensor([0.0], dtype=torch.double, requires_grad=True)
        g = feasibility_loss_pi(x, A_int, b, norms, eps_feas=0.0)
        g.backward()
        assert x.grad is not None and x.grad.item() != 0.0

    def test_eps_offset(self, simple_problem):
        A_int, b, norms = simple_problem
        # With a large eps, even a feasible point may have g > 0
        x = torch.tensor([2.0], dtype=torch.double)
        g_no_eps = feasibility_loss_pi(x, A_int, b, norms, eps_feas=0.0)
        g_with_eps = feasibility_loss_pi(x, A_int, b, norms, eps_feas=1.0)
        assert g_no_eps.item() == pytest.approx(0.0, abs=1e-10)
        # With eps=1: slack - 1 may be positive → g > 0
        assert g_with_eps.item() >= 0.0


# ---------------------------------------------------------------------------
# feasibility_loss_mi_first  (eq. 20)
# ---------------------------------------------------------------------------

class TestFeasibilityLossMIFirst:

    def test_zero_when_feasible(self):
        # 1 integer var x, 1 continuous var u: x + u >= 2, u >= 0
        A_int  = torch.tensor([[1.0], [0.0]], dtype=torch.double)
        A_cont = torch.tensor([[1.0], [1.0]], dtype=torch.double)
        b      = torch.tensor([2.0, 0.0],    dtype=torch.double)
        norms  = torch.tensor([np.sqrt(6), np.sqrt(1)], dtype=torch.double)

        x_int = torch.tensor([1.0], dtype=torch.double)  # x=1, u=1 → feasible
        u     = torch.tensor([1.0], dtype=torch.double)
        g = feasibility_loss_mi_first(x_int, u, A_int, A_cont, b, norms, eps_feas=0.0)
        assert g.item() == pytest.approx(0.0, abs=1e-10)

    def test_gradient_propagates_to_x_not_u(self):
        A_int  = torch.tensor([[1.0]], dtype=torch.double)
        A_cont = torch.tensor([[1.0]], dtype=torch.double)
        b      = torch.tensor([5.0],  dtype=torch.double)
        norms  = torch.tensor([np.sqrt(26)], dtype=torch.double)

        x_int = torch.tensor([1.0], dtype=torch.double, requires_grad=True)
        u     = torch.tensor([1.0], dtype=torch.double, requires_grad=True)
        g = feasibility_loss_mi_first(x_int, u, A_int, A_cont, b, norms, eps_feas=0.0)
        g.backward()
        # Gradient should flow to x_int (infeasible) but u is detached
        assert x_int.grad is not None and x_int.grad.item() != 0.0
        assert u.grad is None   # u detached inside the function


# ---------------------------------------------------------------------------
# feasibility_loss_mi_argmin  (eq. 21)
# ---------------------------------------------------------------------------

class TestFeasibilityLossMIArgmin:

    def test_zero_when_continuously_feasible(self):
        # 1 integer var x=0, 1 continuous u: u >= 2  → minimum u=2 is feasible
        A_int  = torch.tensor([[0.0]], dtype=torch.double)
        A_cont = torch.tensor([[1.0]], dtype=torch.double)  # u >= 2
        b      = torch.tensor([2.0],  dtype=torch.double)
        norms  = torch.tensor([np.sqrt(5)], dtype=torch.double)

        x_int = torch.tensor([0.0], dtype=torch.double, requires_grad=True)
        g = feasibility_loss_mi_argmin(x_int, A_int, A_cont, b, norms, eps_feas=0.0, q=2)
        # Optimal u solves the argmin; since u can be made >= 2, g should be 0
        assert g.item() == pytest.approx(0.0, abs=1e-6)

    def test_gradient_computed(self):
        # Simple: only integer var, no continuous, Ax >= b
        A_int  = torch.tensor([[1.0]], dtype=torch.double)
        A_cont = torch.zeros((1, 0), dtype=torch.double)
        b      = torch.tensor([3.0], dtype=torch.double)
        norms  = torch.tensor([np.sqrt(10)], dtype=torch.double)

        x_int = torch.tensor([1.0], dtype=torch.double, requires_grad=True)
        g = feasibility_loss_mi_argmin(x_int, A_int, A_cont, b, norms, eps_feas=0.0, q=2)
        g.backward()
        assert x_int.grad is not None


# ---------------------------------------------------------------------------
# regularisation
# ---------------------------------------------------------------------------

class TestRegularisation:

    def test_value(self):
        theta = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
        reg = regularisation(theta)
        assert reg.item() == pytest.approx(0.5 * (1 + 4 + 9), abs=1e-10)

    def test_gradient(self):
        theta = torch.tensor([1.0, -2.0], dtype=torch.double, requires_grad=True)
        reg = regularisation(theta)
        reg.backward()
        # ∂Ω/∂θᵢ = θᵢ
        np.testing.assert_allclose(theta.grad.numpy(), [1.0, -2.0], atol=1e-10)
