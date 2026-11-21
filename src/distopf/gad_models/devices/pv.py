"""PV device provider (stub): reads ``case.pv_data`` (GAD schema v1).

Stub only. It creates an empty ``pv_device_set`` and injects nothing. WP-GEN
adds the variables, the available-power computation and the limits (shared
with generators via ``devices/injector.py``).
"""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo  # type: ignore

from distopf.gad_models.common.registry import DeviceProvider


class PVProvider(DeviceProvider):
    """PV systems. Currently no devices and no injections."""

    name = "pv"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        model.pv_device_set = pyo.Set(initialize=[])

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return 0

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return 0

    def add_constraints(self, model: Any, config: Any) -> None:
        return None


__all__ = ["PVProvider"]
