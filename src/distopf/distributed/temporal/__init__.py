"""Temporal decomposition algorithms for multi-period OPF.

This module implements three temporal decomposition algorithms:
- TENAPP-1O: First-order temporal decomposition
- TENAPP-APRX: Approximate dual update method
- TENAPP-ADMM: Alternating Direction Method of Multipliers

All algorithms accept a ``Case`` and return a :class:`~distopf.results.PowerFlowResult`.
The underlying solver backend is selected via the ``wrapper`` parameter
(default ``"matrix_bess"``); pass ``wrapper="pyomo"`` together with a
pyomo-compatible objective to use IPOPT instead of CLARABEL.
"""

from .tenapp_1o import solve_tenapp_1o
from .tenapp_aprx import solve_tenapp_aprx
from .tenapp_admm import solve_tenapp_admm
from .objectives import (
    energy_cost_min,
    cp_battery_efficiency,
    tenapp_1o_augmentation,
    tenapp_aprx_augmentation,
    tenapp_admm_augmentation,
)
from .utils import (
    build_timestep_cases,
    update_bat_start_soc,
    combine_temporal_results,
    compile_iteration_summary,
)

__all__ = [
    "solve_tenapp_1o",
    "solve_tenapp_aprx",
    "solve_tenapp_admm",
    "energy_cost_min",
    "cp_battery_efficiency",
    "tenapp_1o_augmentation",
    "tenapp_aprx_augmentation",
    "tenapp_admm_augmentation",
    "build_timestep_cases",
    "update_bat_start_soc",
    "combine_temporal_results",
    "compile_iteration_summary",
]
