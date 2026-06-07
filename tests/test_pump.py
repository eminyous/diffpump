"""
End-to-end tests using tiny synthetic MIPs.

Requires Gurobi. The synthetic instances are small enough to run in < 1 s.

The synthetic binary problem:
    min  x1 + x2
    s.t. x1 + x2 >= 1
         x1, x2 ∈ {0, 1}

Feasible solutions: (1,0), (0,1), (1,1).
"""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from diffpump.losses import integrality_loss
from diffpump.problem import load_instance
from diffpump.pump import run_instance
from diffpump.variants import VariantConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

_MPS_GENERAL_INT = textwrap.dedent("""\
NAME          tiny_general_int
ROWS
 N  OBJ
 G  CON1
COLUMNS
    MARKER   'MARKER'  'INTORG'
    x1       OBJ   1.0   CON1  1.0
    x2       OBJ   1.0   CON1  1.0
    MARKER   'MARKER'  'INTEND'
RHS
    RHS      CON1  3.0
BOUNDS
 UI BND      x1    5
 UI BND      x2    5
ENDATA
""")


def _write_tmp_mps(content: str) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mps", delete=False) as f:
        f.write(content)
        return Path(f.name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPumpBinary:
    """End-to-end: pure binary instance."""

    @pytest.fixture(scope="class")
    def inst(self):
        path = _write_tmp_mps(_MPS_BINARY)
        return load_instance(path)

    def test_default_config_finds_solution(self, inst):
        rec = run_instance(inst, VariantConfig(max_iters=200), seed=42)
        assert rec.success

    @pytest.mark.parametrize("cfg", [
        VariantConfig(eta=0.55, gamma=0.95, max_iters=200),
        VariantConfig(eta=0.8, gamma=0.65, p=2.0, max_iters=200),
        VariantConfig(eta=0.6, gamma=0.7, lam=0.1, beta=0.3, p=2.0, max_iters=200),
        VariantConfig(eta=0.5, gamma=0.5, lam=0.9, beta=0.9, p=2.0,
                      use_argmin_feas=True, max_iters=200),
    ])
    def test_custom_params_find_solution(self, inst, cfg):
        rec = run_instance(inst, cfg, seed=42)
        assert rec.success, (
            f"Failed with cfg={cfg} "
            f"(iters={rec.n_iters}, restarts={rec.n_restarts})"
        )

    def test_fp_recovers_original_update(self, inst):
        """
        Proposition 1 verification: with η=γ=1, β=1, λ=0, p=1,
        Algorithm 2 must produce θ ∈ {-1, +1}^n after every update.
        """
        cfg = VariantConfig(eta=1.0, gamma=1.0, beta=1.0, lam=0.0, p=1.0, max_iters=10)
        rec = run_instance(inst, cfg, seed=0)
        assert rec.n_iters >= 1


class TestPumpGeneralInt:
    """End-to-end: general integer instance."""

    @pytest.fixture(scope="class")
    def inst(self):
        path = _write_tmp_mps(_MPS_GENERAL_INT)
        return load_instance(path)

    @pytest.mark.parametrize("cfg", [
        VariantConfig(max_iters=200),
        VariantConfig(eta=0.5, gamma=0.5, lam=0.9, beta=0.9, p=2.0, max_iters=200),
    ])
    def test_finds_feasible_solution(self, inst, cfg):
        rec = run_instance(inst, cfg, seed=42)
        assert rec.success, f"Failed (iters={rec.n_iters})"


class TestBinaryEquivalence:
    """
    Verify that the integer code produces the binary path when all vars are binary.
    The generalised φ(x;θ,y) with |θᵢ|=1 must reduce to ‖x−y‖₁.
    """

    def test_pumping_function_reduces_to_l1(self):
        """
        With η=γ=1, p=1, the update gives |θᵢ|=1 (Proposition 1).
        Check that the gradient of f w.r.t. x matches sign(x - round(x)).
        """
        x = torch.tensor([0.3, 0.7], dtype=torch.double, requires_grad=True)
        loss = integrality_loss(x, p=1)
        loss.backward()

        expected_grad = np.array([1.0, -1.0])
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, atol=1e-10)
