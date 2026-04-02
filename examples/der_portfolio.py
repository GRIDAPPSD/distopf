"""
DER Portfolio Capacity Expansion with Battery Storage
=====================================================
Minimize substation power withdrawal by co-optimizing:
  - New PV capacity allocation by zone (p_max[z])
  - New battery energy capacity by zone (e_max_new[z])
  - Dispatch of existing and new batteries
All DER injections are coupled directly into the LinDistFlow power balance.
"""

import distopf as opf
import pyomo.environ as pyo
import networkx as nx

from distopf.api import create_case
from distopf.pyomo_models.lindist import create_lindist_model
from distopf.pyomo_models.protocol import LindistModelProtocol
from distopf.pyomo_models.objectives import substation_power_objective_rule
from distopf.pyomo_models.constraints import (
    add_battery_constant_q_constraints_p_control,
    add_battery_energy_constraints,
    add_battery_net_p_bat_equal_phase_constraints,
    add_battery_power_limits,
    add_battery_soc_limits,
    add_capacitor_constraints,
    add_circular_generator_constraints_pq_control,
    add_cvr_load_constraints,
    add_generator_constant_p_constraints_q_control,
    add_generator_constant_q_constraints_p_control,
    add_generator_limits,
    add_p_flow_constraints,
    add_q_flow_constraints,
    add_regulator_constraints,
    add_swing_bus_constraints,
    add_voltage_drop_constraints,
    add_voltage_limits,
)

# ---------------------------------------------------------------------------
# Case + system data
# ---------------------------------------------------------------------------
case = create_case(
    data_path=opf.CASES_DIR / "csv" / "ieee123_30der_bat",
    ignore_schedule=True,
    n_steps=1,
    start_step=0,
)
s_base = case.bus_data["s_base"].iloc[0]
bus_data_by_id = case.bus_data.set_index("id", drop=False)
swing_bus_id = int(
    case.bus_data.loc[case.bus_data["bus_type"] == opf.SWING_BUS, "id"].iloc[0]
)
batt_ids = case.bat_data["id"].tolist() if case.bat_data is not None else []

# ---------------------------------------------------------------------------
# Planning parameters
# ---------------------------------------------------------------------------
total_load_p = case.bus_data[["pl_a", "pl_b", "pl_c"]].sum().sum()
T_DER = 0.10 * total_load_p  # PV capacity budget (p.u. MW)
T_BATT = 0.10 * total_load_p  # Battery energy budget (p.u. MWh)

cf = 0.8  # PV capacity factor
c_max = 1.0  # max curtailment fraction (1 = fully curtailable)
eta_c = 0.93  # charge efficiency
eta_d = 0.93  # discharge efficiency
delta_t = 1.0  # time-step length (hours)
c_rate = 1.0  # battery C-rate (power / energy)
soc_init = 0.5
soc_min = 0.2
soc_max = 0.9

# ---------------------------------------------------------------------------
# Zone partitioning (two areas separated by switch sw2)
# ---------------------------------------------------------------------------
sw_row = case.branch_data.loc[case.branch_data["name"] == "sw2"].iloc[0]
topo = nx.Graph()
for _, row in case.branch_data.iterrows():
    topo.add_edge(int(row["fb"]), int(row["tb"]))
topo.remove_edge(int(sw_row["fb"]), int(sw_row["tb"]))
zones = {
    f"Z{i + 1}": [b for b in sorted(c) if b != swing_bus_id]
    for i, c in enumerate(nx.connected_components(topo))
}

# ---------------------------------------------------------------------------
# Allocation factors (load-proportional)
# ---------------------------------------------------------------------------
alpha = {}
for z, buses in zones.items():
    z_load = bus_data_by_id.loc[buses, ["pl_a", "pl_b", "pl_c"]].sum().sum()
    for n in buses:
        frac = (
            bus_data_by_id.loc[n, ["pl_a", "pl_b", "pl_c"]].sum() / z_load
            if z_load > 0
            else 1.0 / len(buses)
        )
        alpha[(z, n, "PV")] = frac
        alpha[(z, n, "BATT")] = frac


# ---------------------------------------------------------------------------
# Baseline OPF (for comparison)
# ---------------------------------------------------------------------------
def _add_standard_constraints(m):
    add_p_flow_constraints(m)
    add_q_flow_constraints(m)
    add_voltage_drop_constraints(m)
    add_swing_bus_constraints(m)
    add_cvr_load_constraints(m)
    add_generator_constant_p_constraints_q_control(m)
    add_generator_constant_q_constraints_p_control(m)
    add_circular_generator_constraints_pq_control(m)
    add_capacitor_constraints(m)
    add_regulator_constraints(m)
    add_voltage_limits(m)
    add_generator_limits(m)
    add_battery_constant_q_constraints_p_control(m)
    add_battery_energy_constraints(m)
    add_battery_net_p_bat_equal_phase_constraints(m)
    add_battery_power_limits(m)
    add_battery_soc_limits(m)


def _substation_power(m):
    return sum(
        pyo.value(m.p_flow[tb, ph, 0])
        for tb, ph in m.branch_phase_set
        if m.from_bus_map[tb] == swing_bus_id
    )


opt = pyo.SolverFactory("ipopt")

m_base = create_lindist_model(case)
_add_standard_constraints(m_base)
m_base.objective = pyo.Objective(
    rule=lambda m: sum(
        (m.p_flow[i, ph, t] ** 2 + m.q_flow[i, ph, t] ** 2) * m.r[i, ph + ph]
        for i, ph in m.branch_phase_set
        for t in m.time_set
    ),
    sense=pyo.minimize,
)
res = opt.solve(m_base, tee=False)
if res.solver.status != pyo.SolverStatus.ok:
    raise RuntimeError("Baseline solve failed")
p_sub_baseline = _substation_power(m_base)

# ---------------------------------------------------------------------------
# Portfolio model
# ---------------------------------------------------------------------------
m = create_lindist_model(case)

# --- portfolio sets and variables (declared before power balance) ----------
m.zone_set = pyo.Set(initialize=list(zones.keys()))
m.resource_set = pyo.Set(initialize=["PV"])

m.p_max = pyo.Var(m.zone_set, m.resource_set, domain=pyo.NonNegativeReals, initialize=0)
m.p_zr = pyo.Var(m.zone_set, m.resource_set, m.time_set, domain=pyo.Reals, initialize=0)

m.e_max_new = pyo.Var(m.zone_set, domain=pyo.NonNegativeReals, initialize=0)
m.p_batt_new_ch = pyo.Var(
    m.zone_set, m.time_set, domain=pyo.NonNegativeReals, initialize=0
)
m.p_batt_new_disch = pyo.Var(
    m.zone_set, m.time_set, domain=pyo.NonNegativeReals, initialize=0
)
m.soc_new = pyo.Var(m.zone_set, m.time_set, domain=pyo.NonNegativeReals, initialize=0)

m.p_der_inj = pyo.Var(m.bus_phase_set, m.time_set, domain=pyo.Reals, initialize=0)


# --- custom power balance (includes p_der_inj) ----------------------------
def _p_balance(m: LindistModelProtocol, _id, ph, t):
    load = m.p_load[_id, ph, t]
    gen = m.p_gen[_id, ph, t] if (_id, ph, t) in m.p_gen else 0
    bat = m.p_bat[_id, ph, t] if (_id, ph, t) in m.p_bat else 0
    out = sum(
        m.p_flow[tb, ph, t]
        for tb in m.to_bus_map[_id]
        if (tb, ph) in m.branch_phase_set
    )
    return m.p_flow[_id, ph, t] == out + load - gen - bat - m.p_der_inj[_id, ph, t]


m.power_balance_p = pyo.Constraint(m.branch_phase_set, m.time_set, rule=_p_balance)

# --- remaining network constraints ----------------------------------------
add_q_flow_constraints(m)
add_voltage_drop_constraints(m)
add_swing_bus_constraints(m)
add_cvr_load_constraints(m)
add_generator_constant_p_constraints_q_control(m)
add_generator_constant_q_constraints_p_control(m)
add_circular_generator_constraints_pq_control(m)
add_capacitor_constraints(m)
add_regulator_constraints(m)
add_voltage_limits(m)
add_generator_limits(m)
add_battery_constant_q_constraints_p_control(m)
add_battery_energy_constraints(m)
add_battery_net_p_bat_equal_phase_constraints(m)
add_battery_power_limits(m)
add_battery_soc_limits(m)

# --- PV portfolio constraints ---------------------------------------------
m.pv_budget = pyo.Constraint(
    rule=lambda m: sum(m.p_max[z, "PV"] for z in m.zone_set) == T_DER
)
m.pv_gen_min = pyo.Constraint(
    m.zone_set,
    m.resource_set,
    m.time_set,
    rule=lambda m, z, r, t: m.p_zr[z, r, t] >= (1 - c_max) * cf * m.p_max[z, r],
)
m.pv_gen_max = pyo.Constraint(
    m.zone_set,
    m.resource_set,
    m.time_set,
    rule=lambda m, z, r, t: m.p_zr[z, r, t] <= cf * m.p_max[z, r],
)

# --- new battery capacity constraints -------------------------------------
m.batt_budget = pyo.Constraint(
    rule=lambda m: sum(m.e_max_new[z] for z in m.zone_set) == T_BATT
)


def _new_soc(m, z, t):
    soc_prev = soc_init * m.e_max_new[z] if t == m.start_step else m.soc_new[z, t - 1]
    return (
        m.soc_new[z, t]
        == soc_prev
        + eta_c * delta_t * m.p_batt_new_ch[z, t]
        - (1 / eta_d) * delta_t * m.p_batt_new_disch[z, t]
    )


m.new_batt_soc_dyn = pyo.Constraint(m.zone_set, m.time_set, rule=_new_soc)
m.new_batt_ch_lim = pyo.Constraint(
    m.zone_set,
    m.time_set,
    rule=lambda m, z, t: m.p_batt_new_ch[z, t] <= c_rate * m.e_max_new[z],
)
m.new_batt_disch_lim = pyo.Constraint(
    m.zone_set,
    m.time_set,
    rule=lambda m, z, t: m.p_batt_new_disch[z, t] <= c_rate * m.e_max_new[z],
)
m.new_batt_soc_lo = pyo.Constraint(
    m.zone_set,
    m.time_set,
    rule=lambda m, z, t: m.soc_new[z, t] >= soc_min * m.e_max_new[z],
)
m.new_batt_soc_hi = pyo.Constraint(
    m.zone_set,
    m.time_set,
    rule=lambda m, z, t: m.soc_new[z, t] <= soc_max * m.e_max_new[z],
)


# --- nodal injection mapping (PV + existing + new batteries) --------------
def _der_inj(m, n, ph, t):
    inj = sum(alpha.get((z, n, "PV"), 0.0) * m.p_zr[z, "PV", t] for z in m.zone_set)
    if n in batt_ids:
        n_ph = pyo.value(m.battery_n_phases[n])
        inj += (m.p_discharge[n, t] - m.p_charge[n, t]) / n_ph
    inj += sum(
        alpha.get((z, n, "BATT"), 0.0)
        * (m.p_batt_new_disch[z, t] - m.p_batt_new_ch[z, t])
        for z in m.zone_set
    )
    return m.p_der_inj[n, ph, t] == inj


m.der_inj_map = pyo.Constraint(m.bus_phase_set, m.time_set, rule=_der_inj)

# --- objective and solve --------------------------------------------------
m.objective = pyo.Objective(rule=substation_power_objective_rule, sense=pyo.minimize)

res = opt.solve(m, tee=False)
if res.solver.status != pyo.SolverStatus.ok:
    raise RuntimeError(f"Portfolio solve failed: {res.solver.status}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
p_sub = pyo.value(m.objective)
t0 = list(m.time_set)[0]

print(f"Substation power  baseline : {p_sub_baseline:.4f} p.u.")
print(
    f"Substation power  portfolio: {p_sub:.4f} p.u.  ({100 * (p_sub_baseline - p_sub) / p_sub_baseline:.1f}% reduction)"
)
print()
for z in zones:
    print(
        f"  PV   {z}: p_max={pyo.value(m.p_max[z, 'PV']):.4f}  dispatch={pyo.value(m.p_zr[z, 'PV', t0]):.4f} p.u."
    )
    print(
        f"  BATT {z}: e_max={pyo.value(m.e_max_new[z]):.4f}  disch={pyo.value(m.p_batt_new_disch[z, t0]):.4f}  ch={pyo.value(m.p_batt_new_ch[z, t0]):.4f} p.u."
    )
for bid in m.bat_set:
    print(
        f"  Existing battery {bid}: disch={pyo.value(m.p_discharge[bid, t0]):.4f}  ch={pyo.value(m.p_charge[bid, t0]):.4f}  SOC={pyo.value(m.soc[bid, t0]):.4f}/{pyo.value(m.energy_capacity[bid]):.4f}"
    )
