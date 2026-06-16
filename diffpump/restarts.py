"""
Cycle detection and restart operations (§2.1, Remark 1).

The paper says restart operations are kept exactly as [FGL05]:
  - Cycle of size 1  (y^(k) == y^(k-1)):           FLIP
  - Cycle of size >1 (y^(k) seen in history window): PERTURBATION

Generalisations for integer variables:
  - Flip:        binary → flip 0↔1; general integer → add ±1, clamped.
  - Perturbation: sample uniformly from {lb, ..., ub} for each variable
                  independently with probability p_perturb.
"""

from __future__ import annotations

import numpy as np

_HISTORY_WINDOW   = 20     # rolling window of past rounded solutions
_P_PERTURB        = 0.3    # per-variable perturbation probability

# Stand-in magnitudes for (semi-)unbounded general integers, matching the
# reference FeasPumpCollection implementation [32] (src/feaspump.cpp:160-161).
# A variable whose domain is wider than _BIGBIGM is treated as unbounded; the
# perturbation then samples a window of half-width _BIGM around the relevant
# finite bound, or around the current value when both bounds are effectively
# infinite.
_BIGM    = 1e9
_BIGBIGM = 1e15


class CycleDetector:
    """
    Tracks the history of rounded solutions and detects cycles.

    Usage
    -----
    detector = CycleDetector()
    for each iteration k:
        restart_type = detector.check(y)   # "none" | "flip" | "perturb"
        if restart_type != "none":
            y = restarter.apply(restart_type, y, ...)
        detector.record(y)
    """

    def __init__(self, window: int = _HISTORY_WINDOW) -> None:
        self._window = window
        self._history: list[tuple] = []
        self._prev: tuple | None = None

    def check(self, y: np.ndarray) -> str:
        """
        Check whether y triggers a restart.

        Returns
        -------
        "none"    — no cycle detected
        "flip"    — cycle of size 1 (y == previous y)
        "perturb" — y appeared in rolling history (cycle of size > 1)
        """
        key = tuple(y.tolist())

        if self._prev is not None and key == self._prev:
            return "flip"

        if key in self._history:
            return "perturb"

        return "none"

    def record(self, y: np.ndarray) -> None:
        """Record y in the history (call AFTER any restart has been applied)."""
        key = tuple(y.tolist())
        self._prev = key
        self._history.append(key)
        if len(self._history) > self._window:
            self._history.pop(0)

    def reset(self) -> None:
        self._history.clear()
        self._prev = None


class Restarter:
    """
    Applies flip or perturbation operations to a rounded solution y.
    """

    def __init__(
        self,
        lb_int: np.ndarray,
        ub_int: np.ndarray,
        is_binary: np.ndarray,
        rng: np.random.Generator,
        p_perturb: float = _P_PERTURB,
    ) -> None:
        self._lb       = lb_int
        self._ub       = ub_int
        self._is_bin   = is_binary
        self._rng      = rng
        self._p_perturb= p_perturb

    def flip(self, y: np.ndarray) -> np.ndarray:
        """
        Flip a random subset of integer variables.

        Binary variables: 0 ↔ 1.
        General integers: add a nonzero ±1 offset (or random value if domain > 2).
        Number of variables flipped ~ Uniform[1, max(1, n_int // 4)].
        """
        y = y.copy()
        n_int = len(y)
        n_flip = int(self._rng.integers(1, max(2, n_int // 4 + 1)))
        idx = self._rng.choice(n_int, size=min(n_flip, n_int), replace=False)

        for i in idx:
            if self._is_bin[i]:
                y[i] = 1.0 - y[i]
            else:
                lb_i, ub_i = self._lb[i], self._ub[i]
                span = ub_i - lb_i
                if np.isfinite(span) and span <= 1:
                    y[i] = lb_i if y[i] == ub_i else ub_i
                else:
                    # Random ±1 shift, stay in bounds.
                    # np.clip handles an infinite ub (no upper clamp applied).
                    shift = self._rng.choice([-1, 1])
                    y[i]  = np.clip(y[i] + shift, lb_i, ub_i)

        return y

    def perturb(self, y: np.ndarray) -> np.ndarray:
        """
        Random perturbation: for each variable independently with probability
        p_perturb, replace yᵢ with a random integer drawn as in the reference
        FeasPumpCollection restart [32] (src/feaspump.cpp:835-839):

          - bounded domain (ub − lb < _BIGBIGM): uniform integer in [lb, ub];
          - finite lb, value near it: uniform in the window [lb, lb + 2·_BIGM);
          - finite ub, value near it: uniform in the window (ub − 2·_BIGM, ub];
          - fully unbounded: uniform in [yᵢ − _BIGM, yᵢ + _BIGM) around yᵢ.

        The result is floored to keep yᵢ integral and clamped to [lb, ub].
        At least one variable is always perturbed to ensure a change.
        """
        y = y.copy()
        n_int = len(y)
        mask = self._rng.random(n_int) < self._p_perturb

        # Ensure at least one variable is perturbed
        if not mask.any():
            mask[self._rng.integers(n_int)] = True

        for i in np.where(mask)[0]:
            lb_i, ub_i = self._lb[i], self._ub[i]
            r = self._rng.random()
            if (ub_i - lb_i) < _BIGBIGM:          # bounded domain
                new_val = np.floor(lb_i + (1.0 + ub_i - lb_i) * r)
            elif (y[i] - lb_i) < _BIGM:           # finite lb, current value near it
                new_val = np.floor(lb_i + (2.0 * _BIGM - 1.0) * r)
            elif (ub_i - y[i]) < _BIGM:           # finite ub, current value near it
                new_val = np.floor(ub_i - (2.0 * _BIGM - 1.0) * r)
            else:                                 # fully unbounded: window around current value
                new_val = np.floor(y[i] + (2.0 * _BIGM - 1.0) * r - _BIGM)
            y[i] = float(np.clip(new_val, lb_i, ub_i))

        return y

    def apply(self, restart_type: str, y: np.ndarray) -> tuple[np.ndarray, str]:
        """Dispatch to flip or perturb and return (new_y, restart_type)."""
        if restart_type == "flip":
            return self.flip(y), "flip"
        elif restart_type == "perturb":
            return self.perturb(y), "perturb"
        return y, "none"
