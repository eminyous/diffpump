"""
DiffPump — Differentiable Feasibility Pump
==========================================

General integer and mixed-integer extension of the Differentiable Feasibility Pump
(Cacciola, Emine, Forel, Frangioni, Lodi — Mathematical Programming 2025).

Entrypoints
-----------
    from diffpump import run_instance, VariantConfig
"""

from diffpump.problem import Instance
from diffpump.pump import run_instance
from diffpump.results import ResultRecord
from diffpump.variants import VariantConfig

__all__ = ["VariantConfig", "run_instance", "Instance", "ResultRecord"]
