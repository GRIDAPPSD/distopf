"""Battery provider for the legacy-index migration path.

This provider owns battery-specific model components and constraints while
preserving the current one-battery-per-bus component names. It is the first
concrete provider for a built-in device family and is intentionally compatible
with the existing LinDistFlow and BranchFlow model factories.
"""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo  # type: ignore

from distopf.pyomo_models.common.model_types import ControlVariable
from distopf.pyomo_models.common.protocol import LindistModelProtocol
from distopf.pyomo_models.common.data import parse_phases, phase_tuples
from distopf.pyomo_models.common.model_types import CONTROL_VARIABLE_MAP


def create_battery_parameters(model: Any, case: Any) -> None:
    """Create every battery parameter consumed by common LinDistFlow constraints."""
    p_data, q_data, rating, q_min, q_max = {}, {}, {}, {}, {}
    energy, soc_min, soc_max, start_soc = {}, {}, {}, {}
    charge_eff, discharge_eff, cycles, control = {}, {}, {}, {}
    has_phase = {
        (device, phase): False for device in model.bat_set for phase in ("a", "b", "c")
    }
    has_a, has_b, has_c, n_phases = {}, {}, {}, {}
    for _, row in case.bat_data.iterrows():
        device = row.id
        phases = parse_phases(str(row.phases))
        count = len(phases)
        n_phases[device] = count
        has_a[device], has_b[device], has_c[device] = (
            "a" in phases,
            "b" in phases,
            "c" in phases,
        )
        for phase in ("a", "b", "c"):
            has_phase[(device, phase)] = phase in phases
        energy[device] = getattr(row, "energy_capacity", 0)
        soc_min[device] = getattr(row, "min_soc", 0)
        soc_max[device] = getattr(row, "max_soc", 1)
        start_soc[device] = getattr(row, "start_soc", 0.5)
        charge_eff[device] = getattr(row, "charge_efficiency", 1)
        discharge_eff[device] = getattr(row, "discharge_efficiency", 1)
        cycles[device] = getattr(row, "annual_cycle_limit", 365)
        control[device] = CONTROL_VARIABLE_MAP[getattr(row, "control_variable", "P")]
        for phase in phases:
            key = (device, phase)
            if key not in model.bat_phase_set:
                continue
            s_max = getattr(row, "s_max", 1000.0) / count
            rating[key] = s_max
            q_min[key] = getattr(row, "q_min", -getattr(row, "s_max", 1000.0)) / count
            q_max[key] = getattr(row, "q_max", getattr(row, "s_max", 1000.0)) / count
            for time in model.time_set:
                p_data[(*key, time)] = getattr(row, "p", 0.0) / count
                q_data[(*key, time)] = getattr(row, "q", 0.0) / count
    model.p_bat_nom = pyo.Param(
        model.bat_phase_set, model.time_set, initialize=p_data, default=0.0
    )
    model.q_bat_nom = pyo.Param(
        model.bat_phase_set, model.time_set, initialize=q_data, default=0.0
    )
    model.s_bat_rated = pyo.Param(
        model.bat_phase_set, initialize=rating, default=1000.0
    )
    model.q_bat_min = pyo.Param(model.bat_phase_set, initialize=q_min, default=-1000.0)
    model.q_bat_max = pyo.Param(model.bat_phase_set, initialize=q_max, default=1000.0)
    model.bat_control_type = pyo.Param(model.bat_set, initialize=control, default=0)
    model.energy_capacity = pyo.Param(model.bat_set, initialize=energy, default=0)
    model.soc_min = pyo.Param(model.bat_set, initialize=soc_min, default=0)
    model.soc_max = pyo.Param(model.bat_set, initialize=soc_max, default=1)
    model.start_soc = pyo.Param(model.bat_set, initialize=start_soc, default=0.5)
    model.charge_efficiency = pyo.Param(
        model.bat_set, initialize=charge_eff, default=1.0
    )
    model.discharge_efficiency = pyo.Param(
        model.bat_set, initialize=discharge_eff, default=1.0
    )
    model.annual_cycle_limit = pyo.Param(model.bat_set, initialize=cycles, default=365)
    model.battery_has_a_phase = pyo.Param(model.bat_set, initialize=has_a, default=True)
    model.battery_has_b_phase = pyo.Param(model.bat_set, initialize=has_b, default=True)
    model.battery_has_c_phase = pyo.Param(model.bat_set, initialize=has_c, default=True)
    model.battery_n_phases = pyo.Param(model.bat_set, initialize=n_phases, default=3)
    model.battery_has_phase = pyo.Param(
        model.bat_set, ("a", "b", "c"), initialize=has_phase, default=True
    )


class BatteryProvider:
    """Create and constrain battery variables for a Pyomo model."""

    name = "batteries"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        """Attach battery components only when the factory did not create them."""
        if not hasattr(model, "bat_phase_set"):
            model.bat_phase_set = pyo.Set(
                initialize=phase_tuples(case.bat_data, "id"), dimen=2
            )
        if not hasattr(model, "bat_set"):
            model.bat_set = pyo.Set(initialize=case.bat_data.id.tolist())

        if not hasattr(model, "p_charge"):
            model.p_charge = pyo.Var(model.bat_set, model.time_set, initialize=0)
        if not hasattr(model, "p_discharge"):
            model.p_discharge = pyo.Var(model.bat_set, model.time_set, initialize=0)
        if not hasattr(model, "p_bat"):
            model.p_bat = pyo.Var(model.bat_phase_set, model.time_set, initialize=0)
        if not hasattr(model, "q_bat"):
            model.q_bat = pyo.Var(model.bat_phase_set, model.time_set, initialize=0)
        if not hasattr(model, "soc"):
            model.soc = pyo.Var(model.bat_set, model.time_set, initialize=0.5)
        if not hasattr(model, "p_bat_nom"):
            create_battery_parameters(model, case)

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        key = (bus, phase, time)
        return model.p_bat[key] if key in model.p_bat else 0

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        key = (bus, phase, time)
        return model.q_bat[key] if key in model.q_bat else 0

    def add_constraints(self, model: Any, config: Any) -> None:
        """Attach shared battery operating constraints once."""
        if len(model.bat_set) == 0:
            return
        for name, builder in (
            (
                "battery_constant_q_bat",
                add_battery_constant_q_constraints_p_control,
            ),
            ("storage", add_battery_energy_constraints),
            (
                "net_discharge",
                add_battery_net_p_bat_equal_phase_constraints,
            ),
        ):
            if not hasattr(model, name):
                builder(model)

        equality_only = getattr(config, "equality_only", False) if config else False
        if equality_only:
            return
        if not hasattr(model, "battery_discharging_limits"):
            add_battery_power_limits(model)
        if not hasattr(model, "battery_soc_limits"):
            add_battery_soc_limits(model)
        circular = getattr(config, "circular_constraints", True) if config else True
        if circular and not hasattr(model, "bat_circle_constraint"):
            add_circular_battery_constraints_pq_control(model)


# Constraints ----------------------------------------------------------------


def add_battery_power_limits(m: LindistModelProtocol) -> None:
    def _d(m: LindistModelProtocol, _id, ph, t):
        return (0, m.p_discharge[_id, t], m.s_bat_rated[_id, ph])

    def _c(m: LindistModelProtocol, _id, ph, t):
        return (0, m.p_charge[_id, t], m.s_bat_rated[_id, ph])

    m.battery_discharging_limits = pyo.Constraint(m.bat_phase_set, m.time_set, rule=_d)
    m.battery_charging_limits = pyo.Constraint(m.bat_phase_set, m.time_set, rule=_c)


def add_battery_soc_limits(m: LindistModelProtocol) -> None:
    def battery_soc_limits(m: LindistModelProtocol, _id, t):
        return (m.soc_min[_id], m.soc[_id, t], m.soc_max[_id])

    m.battery_soc_limits = pyo.Constraint(
        m.bat_set, m.time_set, rule=battery_soc_limits
    )


def add_battery_net_p_bat_constraints(m: LindistModelProtocol) -> None:
    def net_discharge(m: LindistModelProtocol, _id, t):
        p_bat_a = m.p_bat[_id, "a", t] if m.battery_has_phase[_id, "a"] else 0
        p_bat_b = m.p_bat[_id, "b", t] if m.battery_has_phase[_id, "b"] else 0
        p_bat_c = m.p_bat[_id, "c", t] if m.battery_has_phase[_id, "c"] else 0
        return p_bat_a + p_bat_b + p_bat_c == m.p_discharge[_id, t] - m.p_charge[_id, t]

    m.net_discharge = pyo.Constraint(m.bat_phase_set, m.time_set, rule=net_discharge)


def add_battery_net_p_bat_equal_phase_constraints(m: LindistModelProtocol) -> None:
    def net_discharge_equal_phases(m: LindistModelProtocol, _id, ph, t):
        n_phases = m.battery_n_phases[_id]
        return (
            m.p_bat[_id, ph, t]
            == (m.p_discharge[_id, t] - m.p_charge[_id, t]) / n_phases
        )

    m.net_discharge = pyo.Constraint(
        m.bat_phase_set, m.time_set, rule=net_discharge_equal_phases
    )


def add_battery_energy_constraints(m: LindistModelProtocol) -> None:
    def storage(m: LindistModelProtocol, _id, t):
        eta_d = m.discharge_efficiency[_id]
        eta_c = m.charge_efficiency[_id]
        if t == m.start_step:
            soc0 = m.start_soc[_id]
        else:
            soc0 = m.soc[_id, t - 1]
        return (
            m.soc[_id, t] - soc0
            == eta_c * m.delta_t * m.p_charge[_id, t]
            - (1 / eta_d) * m.delta_t * m.p_discharge[_id, t]
        )

    m.storage = pyo.Constraint(m.bat_set, m.time_set, rule=storage)


def add_battery_constant_q_constraints_p_control(m: LindistModelProtocol) -> None:
    def _rule(m: LindistModelProtocol, _id, ph, t):
        if m.bat_control_type[_id] != ControlVariable.P:
            return pyo.Constraint.Skip
        return m.q_bat[_id, ph, t] == m.q_bat_nom[_id, ph, t]

    m.battery_constant_q_bat = pyo.Constraint(m.bat_phase_set, m.time_set, rule=_rule)


def add_circular_battery_constraints_pq_control(m: LindistModelProtocol) -> None:
    """
    Add circular battery apparent power constraints.

    Enforces the exact quadratic constraint:
        P_bat^2 + Q_bat^2 <= S_rated^2

    This is a nonlinear (quadratic) constraint requiring a nonlinear solver
    (e.g., IPOPT) or a solver supporting second-order cone constraints.
    """

    def bat_circle(m: LindistModelProtocol, _id, ph, t):
        if m.bat_control_type[_id] != ControlVariable.PQ:
            return pyo.Constraint.Skip
        return (
            m.p_bat[_id, ph, t] ** 2 + m.q_bat[_id, ph, t] ** 2
            <= m.s_bat_rated[_id, ph] ** 2
        )

    m.bat_circle_constraint = pyo.Constraint(
        m.bat_phase_set, m.time_set, rule=bat_circle
    )


def add_circular_battery_constraints(m: LindistModelProtocol) -> None:
    """
    Add circular battery apparent power constraints.

    Enforces the exact quadratic constraint:
        P_bat^2 + Q_bat^2 <= S_rated^2

    This is a nonlinear (quadratic) constraint requiring a nonlinear solver
    (e.g., IPOPT) or a solver supporting second-order cone constraints.
    """

    def bat_circle(m: LindistModelProtocol, _id, ph, t):
        return (
            m.p_bat[_id, ph, t] ** 2 + m.q_bat[_id, ph, t] ** 2
            <= m.s_bat_rated[_id, ph] ** 2
        )

    m.bat_circle_constraint = pyo.Constraint(
        m.bat_phase_set, m.time_set, rule=bat_circle
    )


__all__ = ["BatteryProvider", "create_battery_parameters"]
