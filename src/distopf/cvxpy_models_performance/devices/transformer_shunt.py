"""Transformer no-load / magnetizing shunt (vectorized CVXPY).

``transformer_shunt_g`` and ``transformer_shunt_b`` are arrays ``(n_bus_phase,)``
aligned with ``bus_phase_set``. Parsing and validation are shared with the Pyomo provider.
"""

from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np

from distopf.cvxpy_models_performance.common.model import column
from distopf.pyomo_models.devices.transformer_shunt import (
    TransformerShuntProvider as _Scalar,
)


class TransformerShuntProvider:
    """P and Q draw ``g * v2`` and ``b * v2`` at transformer from-buses."""

    name = "transformer_shunt"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        _Scalar().create_components(model, case, config)
        shunt = model.transformer_shunt
        model.transformer_shunt_g = np.array(
            [shunt.get(key, (0.0, 0.0))[0] for key in model.bus_phase_set], dtype=float
        )
        model.transformer_shunt_b = np.array(
            [shunt.get(key, (0.0, 0.0))[1] for key in model.bus_phase_set], dtype=float
        )

    def active_power_injection(self, model: Any) -> Any:
        if not model.transformer_shunt_g.any():
            return 0
        return -cp.multiply(column(model.transformer_shunt_g), model.v2)

    def reactive_power_injection(self, model: Any) -> Any:
        if not model.transformer_shunt_b.any():
            return 0
        return -cp.multiply(column(model.transformer_shunt_b), model.v2)

    def add_constraints(self, model: Any, config: Any) -> None:
        return None


__all__ = ["TransformerShuntProvider"]
