"""Line charging as a pi model: half of each branch's b_{phase} at both ends.

branch_data columns b_a, b_b, b_c are total shunt susceptance per phase in
per unit on the branch z_base (same base as r_*/x_*). Each end contributes a
reactive injection of (b/2) * v2 at its bus. Missing or NaN columns mean 0.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


_PHASES = ("a", "b", "c")


class LineChargingProvider:
    """Reactive injection from line charging, split equally between branch ends."""

    name = "line_charging"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        # (bus, phase) -> total b/2 of all branch ends attached there
        half_b: dict[tuple[int, str], float] = {}
        for _, row in case.branch_data.iterrows():
            for phase in _PHASES:
                b = row.get(f"b_{phase}")
                if b is None or pd.isna(b) or b == 0:
                    continue
                if (
                    int(row["fb"]),
                    int(row["tb"]),
                    phase,
                ) not in model.branch_phase_set:
                    raise ValueError(
                        f"branch ({row['fb']}, {row['tb']}) has b_{phase} but no phase {phase}"
                    )
                for bus in (int(row["fb"]), int(row["tb"])):
                    key = (bus, phase)
                    half_b[key] = half_b.get(key, 0.0) + float(b) / 2
        model.line_charging_half_b = half_b

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return 0

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        half_b = model.line_charging_half_b.get((bus, phase), 0.0)
        if half_b == 0.0:
            return 0
        return half_b * model.v2[bus, phase, time]

    def add_constraints(self, model: Any, config: Any) -> None:
        return None


__all__ = ["LineChargingProvider"]
