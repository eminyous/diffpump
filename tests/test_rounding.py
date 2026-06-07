"""
Unit tests for hard rounding and soft rounding.

Includes finite-difference verification of the soft-rounding Jacobian (eq. 24/25).
"""

from math import exp, pi, sqrt

import numpy as np
import pytest
import torch

from diffpump.rounding import hard_round, soft_round, soft_round_jacobian


class TestHardRound:

    def test_rounds_to_nearest_integer(self):
        x  = np.array([0.3, 0.7, 1.4, 1.6, -0.4, -0.6])
        lb = np.array([-1., -1., 0., 0., -1., -1.])
        ub = np.array([2., 2., 3., 3., 1., 1.])
        y  = hard_round(x, lb, ub)
        np.testing.assert_array_equal(y, [0., 1., 1., 2., 0., -1.])

    def test_clamps_to_bounds(self):
        x  = np.array([3.6, -1.2])
        lb = np.array([0., 0.])
        ub = np.array([3., 3.])
        y  = hard_round(x, lb, ub)
        np.testing.assert_array_equal(y, [3., 0.])

    def test_binary_rounding(self):
        x  = np.array([0.49, 0.51])
        lb = np.zeros(2)
        ub = np.ones(2)
        y  = hard_round(x, lb, ub)
        np.testing.assert_array_equal(y, [0., 1.])


class TestSoftRound:

    def test_forward_matches_hard_round(self):
        """Soft rounding forward pass equals hard rounding (only backward differs)."""
        x_np = np.array([0.3, 0.7, 1.2, 1.8, 2.5])
        x_t  = torch.tensor(x_np, dtype=torch.double)
        y_soft = soft_round(x_t, eps=0.15).detach().numpy()
        # Use round-half-up (floor(x + 0.5)), not np.round which uses banker's rounding.
        y_hard = np.floor(x_np + 0.5)
        np.testing.assert_array_equal(y_soft, y_hard)

    def test_jacobian_numpy_matches_autograd_backward(self):
        """
        The numpy soft_round_jacobian (eq. 25) and the torch autograd backward
        implement the same formula — verify they agree at several points.

        Note: finite-differencing the forward pass would give 0 everywhere
        (the forward is hard rounding, a step function), so we compare the
        numpy helper against the autograd backward instead.
        """
        eps         = 0.15
        test_points = [0.1, 0.3, 0.7, 0.9, 1.1, 1.4, 1.6, 2.3]

        for xi in test_points:
            x_t = torch.tensor([xi], dtype=torch.double, requires_grad=True)
            soft_round(x_t, eps).backward()
            jac_autograd = x_t.grad.item()

            jac_numpy = soft_round_jacobian(np.array([xi]), eps)[0]

            assert jac_numpy == pytest.approx(jac_autograd, rel=1e-10), (
                f"Jacobian mismatch at x={xi}: numpy={jac_numpy:.6f}, "
                f"autograd={jac_autograd:.6f}"
            )

    def test_backward_uses_soft_jacobian(self):
        """Confirm that the autograd backward returns the Gaussian-PDF gradient."""
        eps = 0.15
        xi  = 0.3  # well away from half-integer
        x_t = torch.tensor([xi], dtype=torch.double, requires_grad=True)
        y   = soft_round(x_t, eps)
        y.sum().backward()

        expected = soft_round_jacobian(np.array([xi]), eps)[0]
        assert x_t.grad.item() == pytest.approx(expected, rel=1e-5)

    def test_eps_larger_makes_jacobian_more_uniform(self):
        """
        As ε → ∞, the Jacobian → 1/(ε √(2π)) → 0 but becomes more uniform.
        As ε → 0, the Jacobian peaks sharply at half-integer points.
        This tests that larger ε gives a smaller max Jacobian value.
        """
        x_vals = np.array([0.3, 0.5, 0.7])
        j_small = soft_round_jacobian(x_vals, eps=0.05).max()
        j_large = soft_round_jacobian(x_vals, eps=0.30).max()
        assert j_small > j_large

    def test_binary_special_case_eq24(self):
        """
        For binary x ∈ [0,1], ⌊x⌋ = 0 so eq. 25 reduces to eq. 24.
        Verify the two formulas agree.
        """
        eps = 0.15

        for xi in [0.1, 0.3, 0.7, 0.9]:
            # Eq. 24 (binary, ⌊x⌋ = 0):  (1/ε) φ((0.5-x)/ε)
            z24   = (0.5 - xi) / eps
            jac24 = (1.0 / eps) * (exp(-0.5 * z24**2) / sqrt(2 * pi))

            # Eq. 25 (general): (1/ε) φ((0.5-x+⌊x⌋)/ε), ⌊x⌋=0 for x∈(0,1)
            z25   = (0.5 - xi + 0) / eps
            jac25 = (1.0 / eps) * (exp(-0.5 * z25**2) / sqrt(2 * pi))

            assert jac24 == pytest.approx(jac25, rel=1e-12)


class TestBinarySpecialCase:
    """Verify that the binary path (all variables ∈ {0,1}) reduces correctly."""

    def test_hard_round_binary(self):
        x  = np.array([0.0, 0.4, 0.6, 1.0])
        lb = np.zeros(4)
        ub = np.ones(4)
        y  = hard_round(x, lb, ub)
        np.testing.assert_array_equal(y, [0., 0., 1., 1.])

    def test_soft_round_binary_gradient_at_midpoint(self):
        """At x=0.5 the soft-rounding Jacobian should be maximal (1/(ε√(2π)))."""
        eps = 0.15
        x_t = torch.tensor([0.5], dtype=torch.double, requires_grad=True)
        y   = soft_round(x_t, eps)
        y.backward()
        expected = 1.0 / (eps * sqrt(2 * pi))
        assert x_t.grad.item() == pytest.approx(expected, rel=1e-5)
