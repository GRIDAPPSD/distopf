"""Generator device provider (GAD schema v1): one total P/Q decision per device.

CSV schema (case.gen_data, one row per OpenDSS Generator)
---------------------------------------------------------
Required columns:
    device_name     Unique name, e.g. "generator.g1".
    bus_name        AC bus; several generators may share a bus.
    phases          e.g. "abc", "a", "s1s2". The totals are split equally over them.
    p               Nominal active power, total over all phases (>= 0).
    s_max           Apparent-power rating, total (> 0).

Optional columns (blank = default):
    p_min           Lowest dispatch in controlled modes (default 0).
    p_max           Cap on the available power (default p).
    q               Reactive-power setpoint, total (default 0).
    q_min, q_max    Reactive-power limits, total (default -s_max, s_max; clamped to them).
    control_variable  "PQ", "P", "Q", or blank (= NONE: P and Q fixed, no rating).
    gen_shape       Schedule column that scales P (and Q, see below). Blank is constant.
                    The name must match the column exactly.
    cost            Cost per unit of active power (default 0).
    type            Constant "generator".

Available P is min(p * gen_shape(t), p_max). Fixed Q (modes NONE and P) is
q * qmult(t), where qmult is the schedule column "<gen_shape>.q" if it exists,
otherwise the P multiplier.

Variables are p_gen[device, t] and q_gen[device, t] (totals); each phase of a
device injects total / number of phases. Per-phase columns (p_a, s_a_max, ...,
balanced) and legacy id/name columns are rejected.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from distopf.gad_models.common.device_data import (
    NetworkContext,
    cell_opt_float,
    cell_required_float,
)
from distopf.gad_models.devices import injector
from distopf.gad_models.devices.injector import PowerTerms

_TABLE = "gen_data"
_PREFIX = "gen"


def _power_terms(row: pd.Series, label: str, errors: list[str]) -> PowerTerms | None:
    p = cell_required_float(row, "p", label, errors, ge=0.0)
    p_min = cell_opt_float(row, "p_min", label, errors)
    p_max = cell_opt_float(row, "p_max", label, errors)
    if p is None:
        return None
    p_min = 0.0 if p_min is None else p_min
    p_max = p if p_max is None else p_max
    if p_min < 0 or p_min > p_max:
        errors.append(f"{label}: needs 0 <= p_min <= p_max, got {p_min:g}, {p_max:g}")
        return None
    return PowerTerms(scale=p, cap=p_max, floor=p_min)


def validate_gen_data(
    data: pd.DataFrame,
    ctx: NetworkContext,
    schedules: pd.DataFrame,
    limits_enforced: bool = True,
) -> None:
    """Raise ValueError / InfeasibleCaseError if the generator table is unusable."""
    injector.read_injectors(
        data, _TABLE, "gen_shape", _power_terms, ctx, schedules, limits_enforced
    )


class GeneratorProvider:
    """Own generator sets, parameters, variables, injections, and constraints."""

    name = "generators"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        ctx = NetworkContext.from_model(model, "GeneratorProvider")
        devices = injector.read_injectors(
            injector.device_table(case, _TABLE),
            _TABLE,
            "gen_shape",
            _power_terms,
            ctx,
            injector.case_schedules(case),
            limits_enforced=not injector.cfg(config, "equality_only", False),
        )
        injector.add_injector_components(model, _PREFIX, devices, ctx.times)

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return injector.injection(model, _PREFIX, "p", bus, phase, time)

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return injector.injection(model, _PREFIX, "q", bus, phase, time)

    def add_constraints(self, model: Any, config: Any) -> None:
        injector.add_injector_constraints(model, _PREFIX, config)


def add_gen_constant_p_constraints(m: Any) -> None:
    """Fix P to available power for generators in control modes NONE and Q."""
    injector.add_constant_p_constraints(m, _PREFIX)


def add_gen_constant_q_constraints(m: Any) -> None:
    """Fix Q to its setpoint for generators in control modes NONE and P."""
    injector.add_constant_q_constraints(m, _PREFIX)


def add_gen_limits(m: Any) -> None:
    """P and Q bounds for every controlled generator (all modes except NONE)."""
    injector.add_limits(m, _PREFIX)


def add_gen_circle_constraints(m: Any) -> None:
    """Exact apparent-power limit p^2 + q^2 <= s_max^2 for PQ-controlled generators."""
    injector.add_circle_constraints(m, _PREFIX)


def add_gen_octagon_constraints(m: Any) -> None:
    """Inscribed-octagon approximation of the rating for PQ-controlled generators."""
    injector.add_octagon_constraints(m, _PREFIX)


__all__ = ["GeneratorProvider", "validate_gen_data"]
