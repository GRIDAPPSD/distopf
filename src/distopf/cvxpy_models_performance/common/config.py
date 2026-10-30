"""Typed build options shared by every provider of a vectorized CVXPY model."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class ModelConfig:
    """Switches that select which constraints and components a model gets.

    Options that are not listed here (for example extension settings such as
    ``zone_edges`` or ``new_pv``) are kept in ``options`` and are readable as
    attributes, so ``config.zone_edges`` works for them.
    """

    linear: bool = True
    control_capacitors: bool = False
    control_regulators: bool = False
    circular_constraints: bool = True
    thermal_constraints: bool = True
    equality_only: bool = False
    free_swing_voltage: bool = False
    free_boundary_loads: bool = False
    voltage_slacks: bool = False
    thermal_slacks: bool = False
    socp_relaxation: bool = False
    sdp_relaxation: bool = False
    sdp_diagonal_relaxation: bool = False
    sdp_test1: bool = False
    reg_tap_change_limit: int | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "ModelConfig":
        """Split ``kwargs`` into known switches and extension options."""
        known = {f.name for f in fields(cls)} - {"options"}
        switches = {k: v for k, v in kwargs.items() if k in known}
        extras = {k: v for k, v in kwargs.items() if k not in known}
        return cls(**switches, options=extras)

    def __getattr__(self, name: str) -> Any:
        options = self.__dict__.get("options", {})
        if name in options:
            return options[name]
        raise AttributeError(name)
