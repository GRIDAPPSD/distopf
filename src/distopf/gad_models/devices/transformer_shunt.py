"""Transformer no-load loss and magnetizing shunt, modeled as a constant-impedance load.

branch_data columns g_sh_{a,b,c} and b_sh_{a,b,c} are per unit on s_base. The shunt
draws p = g * v2 and q = b * v2 at the branch from-bus. Missing or NaN columns mean 0.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


_PHASES = ("a", "b", "c")


class TransformerShuntProvider:
    """Load-like P and Q draw proportional to v2 at transformer from-buses."""

    name = "transformer_shunt"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        # (bus, phase) -> (g, b) summed over all branches whose from-bus is bus
        shunt: dict[tuple[int, str], tuple[float, float]] = {}
        for _, row in case.branch_data.iterrows():
            for phase in _PHASES:
                g = row.get(f"g_sh_{phase}")
                b = row.get(f"b_sh_{phase}")
                g = 0.0 if g is None or pd.isna(g) else float(g)
                b = 0.0 if b is None or pd.isna(b) else float(b)
                if g == 0.0 and b == 0.0:
                    continue
                key = (int(row["fb"]), phase)
                # Center-tap shunts sit on the primary phase, not the s1/s2 branch phases.
                if key not in model.bus_phase_set:
                    raise ValueError(
                        f"branch ({row['fb']}, {row['tb']}) has shunt on phase {phase} "
                        f"but from-bus has no phase {phase}"
                    )
                g0, b0 = shunt.get(key, (0.0, 0.0))
                shunt[key] = (g0 + g, b0 + b)
        model.transformer_shunt = shunt

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        g, _ = model.transformer_shunt.get((bus, phase), (0.0, 0.0))
        if g == 0.0:
            return 0
        return -g * model.v2[bus, phase, time]

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        _, b = model.transformer_shunt.get((bus, phase), (0.0, 0.0))
        if b == 0.0:
            return 0
        return -b * model.v2[bus, phase, time]

    def add_constraints(self, model: Any, config: Any) -> None:
        return None


__all__ = ["TransformerShuntProvider"]
