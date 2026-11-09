"""Line charging as a pi model (vectorized CVXPY).

``line_charging_half_b`` is an array ``(n_bus_phase,)`` aligned with
``bus_phase_set``. Parsing and validation are shared with the Pyomo provider.
"""

from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np

from distopf.cvxpy_models_performance.common.model import column
from distopf.pyomo_models.devices.line_charging import LineChargingProvider as _Scalar


class LineChargingProvider:
    """Reactive injection ``(b/2) * v2`` from both ends of every branch."""

    name = "line_charging"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        _Scalar().create_components(model, case, config)
        half_b = model.line_charging_half_b
        model.line_charging_half_b = np.array(
            [half_b.get(key, 0.0) for key in model.bus_phase_set], dtype=float
        )

    def active_power_injection(self, model: Any) -> Any:
        return 0

    def reactive_power_injection(self, model: Any) -> Any:
        if not model.line_charging_half_b.any():
            return 0
        return cp.multiply(column(model.line_charging_half_b), model.v2)

    def add_constraints(self, model: Any, config: Any) -> None:
        return None


__all__ = ["LineChargingProvider"]
