"""Capacity-expansion setup and constraint helpers (vectorized CVXPY).

Array layout: ``p_max`` is ``(n_zone, n_resource)`` (no time axis), ``p_zr`` is
``(n_zone, n_resource, n_steps)``, ``p_der_inj`` is ``(n_bus_phase, n_steps)``,
``capacity_factor`` is ``(n_steps,)`` and ``relative_capacity`` is a dict by
resource. ``alpha`` is an array ``(n_bus, n_resource)`` aligned with ``bus_set``
and ``resource_set``. ``add_der_capacity_injection_constraints`` uses the number
of phases of the bus (``phase_map``).
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from distopf.api import Case
from distopf.cvxpy_models_performance.common.model import (
    IndexSet,
    add_constraint,
    sparse,
)
from distopf.pyomo_models.extensions.capacity_expansion.capacity_expansion_constraints import (
    create_zones_from_edge_names as create_zones_from_edge_names,
)


def add_capacity_expansion_as_fraction_of_load(
    m,
    case: Case,
    fraction: float = 0.1,
    relative_capacity: dict[str, float] | None = None,
):
    if relative_capacity is None:
        relative_capacity = {"PV": 0.5, "BESS": 0.5}
    total_relative_capacity = sum(relative_capacity.values())
    if total_relative_capacity == 0:
        raise ValueError("Sum of relative capacities cannot be zero.")
    relative_capacity = {
        k: v / total_relative_capacity for k, v in relative_capacity.items()
    }
    m.resource_set = IndexSet(relative_capacity.keys())
    total_load_p = case.bus_data[["pl_a", "pl_b", "pl_c"]].sum().sum()
    m.total_capacity_expansion = fraction * total_load_p
    m.relative_capacity = dict(relative_capacity)


def add_capacity_expansion_from_absolute_capacity(
    m, absolute_capacity={"PV": 0.5, "BESS": 0.5}
):
    """Initialize capacity-expansion budget from absolute per-resource capacities (p.u.)."""
    if not absolute_capacity:
        raise ValueError("absolute_capacity cannot be empty.")
    total_capacity = sum(absolute_capacity.values())
    if any(v < 0 for v in absolute_capacity.values()):
        raise ValueError("Absolute capacities must be non-negative.")

    m.resource_set = IndexSet(absolute_capacity.keys())
    m.total_capacity_expansion = total_capacity
    if total_capacity == 0:
        relative_capacity = {
            k: 1 / len(absolute_capacity) for k in absolute_capacity.keys()
        }
    else:
        relative_capacity = {
            k: v / total_capacity for k, v in absolute_capacity.items()
        }
    m.relative_capacity = relative_capacity


def add_pv_parameters(
    m,
    curtailment_max=1.0,
    capacity_factor=0.8,
    case: Case | None = None,
    pv_shape: str = "PV",
):
    m.curtailment_max = curtailment_max
    if case is not None and pv_shape in case.schedules.columns:
        m.capacity_factor = np.array(
            [float(case.schedules.at[t, pv_shape]) for t in m.time_set]  # type: ignore[arg-type]
        )
    else:
        m.capacity_factor = np.full(len(m.time_set), float(capacity_factor))


def add_bess_parameters(
    m,
    e_max=4.0,
    soc=0.5,
    discharge_derate=1.0,
    charge_derate=1.0,
):
    m.e_max = e_max
    m.soc_new = soc
    m.discharge_derate = discharge_derate
    m.charging_derate = charge_derate


def _compute_load_proportional_allocation_factors(m, case: Case, zones):
    """Load-proportional DER allocation ``alpha[bus, resource]`` (Eq. 4 and 5)."""
    alpha = np.zeros((len(m.bus_set), len(m.resource_set)))
    bus_data_by_id = case.bus_data.set_index("id", drop=False)
    for z, buses in zones.items():
        z_load = bus_data_by_id.loc[buses, ["pl_a", "pl_b", "pl_c"]].sum().sum()
        for _id in buses:
            frac = (
                bus_data_by_id.loc[_id, ["pl_a", "pl_b", "pl_c"]].sum() / z_load
                if z_load > 0
                else 1.0 / len(buses)
            )
            alpha[m.bus_set.pos[_id], :] = frac
    m.alpha = alpha
    for z, buses in zones.items():
        rows = m.bus_set.indices(buses)
        for r in m.resource_set:
            alpha_sum = alpha[rows, m.resource_set.pos[r]].sum()
            assert abs(alpha_sum - 1) < 1e-6, (
                f"Zone {z} resource {r} alpha sum is {alpha_sum}"
            )


def add_capacity_expansion_variables(m, case: Case, zones):
    if not hasattr(m, "resource_set"):
        raise AttributeError(
            "Calling add_capacity_expansion_variables requires m.resource_set to be "
            "defined. Call add_capacity_expansion_as_fraction_of_load first. Then "
            "call add_pv_parameters and add_bess_parameters to set resource-specific "
            "parameters."
        )

    m.zone_set = IndexSet(list(zones.keys()))
    z_of_bus = {_id: None for _id in m.bus_set}
    for z, buses in zones.items():
        for _id in buses:
            z_of_bus[_id] = z
    m.z_of_bus = z_of_bus
    m.add_var("p_max", m.zone_set, m.resource_set, time=False, nonneg=True, value=0.0)
    m.add_var("p_zr", m.zone_set, m.resource_set, value=0.0)
    m.add_var("p_der_inj", m.bus_phase_set, value=0.0)
    _compute_load_proportional_allocation_factors(m, case, zones)


def add_zone_capacity_expansion_constraints(m):
    r"""Eq. 3: ``sum_z p_max[z, r] == T_DER * c_r``."""
    resources = list(m.resource_set)
    target = np.array(
        [m.total_capacity_expansion * m.relative_capacity[r] for r in resources]
    )
    add_constraint(
        m, "der_budget", cp.sum(m.p_max, axis=0) == target, [(r,) for r in resources]
    )


def _zone_resource_view(m, resource: str):
    """``(p_max[:, r] as column, p_zr[:, r, :])`` for resource ``r``."""
    r = m.resource_set.pos[resource]
    n_zone = len(m.zone_set)
    return cp.reshape(m.p_max[:, r], (n_zone, 1), order="C"), m.p_zr[:, r, :]


def add_pv_capacity_constraints(m):
    if "PV" not in m.resource_set:
        return
    m.pv_set = IndexSet(["PV"])
    p_max, p_zr = _zone_resource_view(m, "PV")
    factor = m.capacity_factor.reshape(1, -1)
    keys = [(z, "PV") for z in m.zone_set]
    add_constraint(
        m, "pv_gen_min", p_zr >= (1 - m.curtailment_max) * (p_max @ factor), keys
    )
    add_constraint(m, "pv_gen_max", p_zr <= p_max @ factor, keys)


def add_bess_capacity_constraints(m):
    if "BESS" not in m.resource_set:
        return
    m.bess_set = IndexSet(["BESS"])
    p_max, p_zr = _zone_resource_view(m, "BESS")
    ones = np.ones((1, len(m.time_set)))
    keys = [(z, "BESS") for z in m.zone_set]
    add_constraint(
        m, "bess_charge_max", -p_zr <= m.charging_derate * (p_max @ ones), keys
    )
    add_constraint(
        m, "bess_discharge_max", p_zr <= m.discharge_derate * (p_max @ ones), keys
    )
    add_constraint(m, "bess_energy_1", p_zr <= m.e_max * m.soc_new / m.delta_t, keys)
    add_constraint(
        m, "bess_energy_2", -p_zr <= m.e_max * (1 - m.soc_new) / m.delta_t, keys
    )


def add_der_capacity_injection_constraints(m):
    """Each bus uses its assigned zone's dispatch, split equally across phases."""
    n_res = len(m.resource_set)
    rows, cols, vals = [], [], []
    for i, (_id, _ph) in enumerate(m.bus_phase_set):
        z = m.z_of_bus[_id]
        if z is None:
            continue
        n_phases = len(m.phase_map[_id])
        z_pos = m.zone_set.pos[z]
        for r in range(n_res):
            rows.append(i)
            cols.append(z_pos * n_res + r)
            vals.append(m.alpha[m.bus_set.pos[_id], r] / n_phases)
    mapping = sparse(rows, cols, vals, (len(m.bus_phase_set), len(m.zone_set) * n_res))
    flat = cp.reshape(m.p_zr, (len(m.zone_set) * n_res, len(m.time_set)), order="C")
    add_constraint(
        m, "der_inj_map", m.p_der_inj == mapping @ flat, list(m.bus_phase_set)
    )
