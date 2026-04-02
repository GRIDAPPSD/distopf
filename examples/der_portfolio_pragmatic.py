"""
DER Portfolio Capacity Expansion - Pragmatic Baseline Approach
==============================================================

Start with a KNOWN WORKING case (ieee123_30der from pyomo_example.py),
then layer in DER portfolio variables and constraints step-by-step.

This avoids solver configuration issues and validates the portfolio
formulation separately from baseline OPF feasibility.
"""

import distopf as opf
import pyomo.environ as pyo
import numpy as np
import networkx as nx

from distopf.pyomo_models.lindist import create_lindist_model
from distopf.pyomo_models.protocol import LindistModelProtocol
from distopf.pyomo_models.objectives import substation_power_objective_rule
from distopf.api import create_case
from distopf.pyomo_models.constraints import (
    add_capacitor_constraints,
    add_circular_generator_constraints_pq_control,
    add_cvr_load_constraints,
    add_generator_constant_p_constraints_q_control,
    add_generator_constant_q_constraints_p_control,
    add_p_flow_constraints,
    add_q_flow_constraints,
    add_swing_bus_constraints,
    add_voltage_drop_constraints,
    add_regulator_constraints,
    add_generator_limits,
    add_voltage_limits,
)

print("=" * 80)
print("DER PORTFOLIO FORMULATION WITH BATTERIES")
print("(Using ieee123_30der_bat case with battery support)")
print("=" * 80)

# =============================================================================
# PHASE 1: BASELINE OPF (Known Working)
# =============================================================================

print("\nPHASE 1: Load case and build baseline OPF model")
print("-" * 80)

case = create_case(
    data_path=opf.CASES_DIR / "csv" / "ieee123_30der_bat",
    ignore_schedule=True,
    n_steps=1,
    start_step=0,
)
s_base = case.bus_data["s_base"].iloc[0]
bus_data_by_id = case.bus_data.set_index("id", drop=False)

print(f"Case: ieee123_30der")
print(f"  Buses: {case.bus_data.shape[0]}")
print(f"  Branches: {case.branch_data.shape[0]}")
print(f"  Generators: {case.gen_data.shape[0]}")
print(f"  System base: {s_base / 1e6:.2f} MVA")
print(f"  Time steps: {case.n_steps}")

# Create baseline model
model_baseline = create_lindist_model(case)
print(f"\nBaseline model created")
print(f"  Bus-phase set: {len(model_baseline.bus_phase_set)}")
print(f"  Branch-phase set: {len(model_baseline.branch_phase_set)}")
print(f"  Time set: {len(model_baseline.time_set)}")

# Add all standard constraints (from pyomo_example.py)
add_p_flow_constraints(model_baseline)
add_q_flow_constraints(model_baseline)
add_voltage_drop_constraints(model_baseline)
add_swing_bus_constraints(model_baseline)
add_cvr_load_constraints(model_baseline)
add_generator_constant_p_constraints_q_control(model_baseline)
add_generator_constant_q_constraints_p_control(model_baseline)
add_circular_generator_constraints_pq_control(model_baseline)
add_capacitor_constraints(model_baseline)
add_regulator_constraints(model_baseline)
add_voltage_limits(model_baseline)
add_generator_limits(model_baseline)

# Add battery constraints
from distopf.pyomo_models.constraints import (
    add_battery_power_limits,
    add_battery_soc_limits,
    add_battery_net_p_bat_equal_phase_constraints,
    add_battery_constant_q_constraints_p_control,
    add_battery_energy_constraints,
)

add_battery_constant_q_constraints_p_control(model_baseline)
add_battery_energy_constraints(model_baseline)
add_battery_net_p_bat_equal_phase_constraints(model_baseline)
add_battery_power_limits(model_baseline)
add_battery_soc_limits(model_baseline)


# Set loss minimization objective (simple, known to work)
def loss_objective_rule(model):
    """Loss minimization from pyomo_example.py - includes time dimension"""
    total_loss = 0
    for _id, ph in model.branch_phase_set:
        for t in model.time_set:
            total_loss += (model.p_flow[_id, ph, t] ** 2) * model.r[_id, ph + ph]
            total_loss += (model.q_flow[_id, ph, t] ** 2) * model.r[_id, ph + ph]
    return total_loss


model_baseline.objective = pyo.Objective(rule=loss_objective_rule, sense=pyo.minimize)

# Solve baseline
print(f"\nSolving baseline (loss minimization)...")
opt = pyo.SolverFactory("ipopt")
if not opt.available(exception_flag=False):
    raise RuntimeError("IPOPT not available")

results_baseline = opt.solve(model_baseline, tee=False)

if results_baseline.solver.status == pyo.SolverStatus.ok:
    obj_baseline = pyo.value(model_baseline.objective)
    print(f"✓ Baseline OPF successful")
    print(f"  Loss (p.u.): {obj_baseline:.6f}")
    print(f"  Loss (kW): {obj_baseline * s_base / 1e3 / 1e6:.2f}")
else:
    raise ValueError(f"Baseline solve failed: {results_baseline.solver.status}")

# =============================================================================
# PHASE 2: ADD DER PORTFOLIO VARIABLES
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 2: Add DER Portfolio Variables and Constraints")
print("=" * 80)

# Compute capacity target from total load
total_load_p = case.bus_data[["pl_a", "pl_b", "pl_c"]].sum().sum()
T_DER = 0.10 * total_load_p  # 10% of load (PV capacity budget)
T_BATT = T_DER  # 10% of load (new battery energy budget, same scale)

print(f"\nCapacity Planning Parameters:")
print(
    f"  Total system load: {total_load_p:.6f} p.u. ({total_load_p * s_base / 1e6:.2f} MW)"
)
print(
    f"  Target new PV capacity (10%):   {T_DER:.6f} p.u. ({T_DER * s_base / 1e6:.4f} MW)"
)
print(
    f"  Target new BATT energy (10%):   {T_BATT:.6f} p.u. ({T_BATT * s_base / 1e6:.4f} MWh)"
)

# Define zones by partitioning the network across switch sw2.
swing_bus_id = int(
    case.bus_data.loc[case.bus_data["bus_type"] == opf.SWING_BUS, "id"].iloc[0]
)


def build_zones_from_switch_partition(case, switch_name="sw2"):
    """
    Split buses into two zones by removing a named switch edge.

    For ieee123 this uses sw2 to create the two planning areas on either side
    of that switch.
    """
    sw_rows = case.branch_data.loc[case.branch_data["name"] == switch_name]
    if sw_rows.empty:
        raise ValueError(f"Switch {switch_name} not found in branch_data")

    sw_fb = int(sw_rows.iloc[0]["fb"])
    sw_tb = int(sw_rows.iloc[0]["tb"])

    topo = nx.Graph()
    for _, row in case.branch_data.iterrows():
        topo.add_edge(int(row["fb"]), int(row["tb"]))

    if topo.has_edge(sw_fb, sw_tb):
        topo.remove_edge(sw_fb, sw_tb)

    components = [sorted(list(c)) for c in nx.connected_components(topo)]
    if len(components) != 2:
        raise ValueError(
            f"Removing {switch_name} produced {len(components)} components; expected 2"
        )

    zones_local = {"Z1": components[0], "Z2": components[1]}
    return zones_local, (sw_fb, sw_tb)


zones, sw2_edge = build_zones_from_switch_partition(case, switch_name="sw2")

# Exclude swing bus from candidate allocations while preserving sw2-based zoning.
for z_name in zones:
    zones[z_name] = [b for b in zones[z_name] if b != swing_bus_id]

print("\nZone Definition (sw2 partition):")
print(f"  Partition switch: sw2 edge {sw2_edge[0]}-{sw2_edge[1]}")
for z_name, z_buses in zones.items():
    print(f"  {z_name}: {len(z_buses)} buses, IDs {min(z_buses)}-{max(z_buses)}")

# Resource set: PV generation; existing batteries (not new capacity allocation)
resources = ["PV"]
c = {"PV": 1.0}  # 100% PV

# Get battery IDs from case (we'll optimize their dispatch but not add new capacity)
batt_ids = (
    case.bat_data["id"].tolist()
    if case.bat_data is not None and len(case.bat_data) > 0
    else []
)

print(f"\nResource Set:")
print(f"  Resources: {resources}")
print(f"  Resource mix: c[PV] = 1.0")
print(f"  Existing batteries (dispatch optimization): {len(batt_ids)} units")
if batt_ids:
    for bid in batt_ids:
        batt_row = case.bat_data[case.bat_data["id"] == bid].iloc[0]
        print(
            f"    - Bus {bid}: {batt_row['s_max']} MVA, {batt_row['energy_capacity']} MWh"
        )


# Allocation factors: load-proportional for PV and new BATT
def build_alpha(zones, resources, bus_data_by_id):
    """Build allocation factors load-proportionally for PV and BATT."""
    alpha = {}
    for z_name, z_buses in zones.items():
        z_load = bus_data_by_id.loc[z_buses, ["pl_a", "pl_b", "pl_c"]].sum().sum()
        for n in z_buses:
            n_load = bus_data_by_id.loc[n, ["pl_a", "pl_b", "pl_c"]].sum()
            frac = n_load / z_load if z_load > 0 else 1.0 / len(z_buses)
            alpha[(z_name, n, "PV")] = frac
            alpha[(z_name, n, "BATT")] = frac  # same load-proportional siting proxy
    return alpha


alpha = build_alpha(zones, resources, bus_data_by_id)

print(f"\nAllocation Factors Check:")
for z_name in zones.keys():
    for r in resources + ["BATT"]:
        total = sum(alpha.get((z_name, n, r), 0.0) for n in zones[z_name])
        print(f"  ∑_n α[{z_name},{r}] = {total:.10f}")


def compute_zone_weights(case, zones, alpha, resources, swing_bus_id):
    """
    Compute zone-level locational weights from electrical distance to swing bus.

    The current simplified portfolio formulation does not yet inject DER directly
    into the network power-balance equations, so without a locational signal the
    optimizer is indifferent across zones and tends to split capacity equally.
    """
    graph = nx.Graph()
    for _, row in case.branch_data.iterrows():
        # Use average self-phase resistance as a scalar proxy for electrical distance.
        # Fall back to a tiny positive value to keep the graph connected for path ops.
        r_vals = [row.get("raa", 0.0), row.get("rbb", 0.0), row.get("rcc", 0.0)]
        r_nonzero = [r for r in r_vals if r > 0]
        r_weight = float(np.mean(r_nonzero)) if r_nonzero else 1e-9
        graph.add_edge(int(row["fb"]), int(row["tb"]), weight=r_weight)

    distances = nx.single_source_dijkstra_path_length(
        graph, int(swing_bus_id), weight="weight"
    )

    zone_weights = {}
    for z_name, z_buses in zones.items():
        # Weighted by alpha so we preserve the same nodal allocation structure.
        weighted_distance = 0.0
        for n in z_buses:
            distance_n = distances.get(int(n), 0.0)
            alpha_n = alpha.get((z_name, n, resources[0]), 0.0)
            weighted_distance += alpha_n * distance_n
        zone_weights[z_name] = weighted_distance

    # Normalize to mean 1.0 for numerical stability and interpretability.
    mean_weight = np.mean(list(zone_weights.values()))
    if mean_weight > 0:
        zone_weights = {z: w / mean_weight for z, w in zone_weights.items()}
    else:
        zone_weights = {z: 1.0 for z in zones.keys()}

    return zone_weights


zone_weights = compute_zone_weights(case, zones, alpha, resources, swing_bus_id)

print(f"\nLocational Weights (electrical-distance proxy):")
for z_name in zones.keys():
    print(f"  w[{z_name}] = {zone_weights[z_name]:.6f}")

# Capacity factor
cf = 0.8  # Global CF for PV
c_max = 1.0  # Allow 20% curtailment

# Battery operating parameters
eta_c = 0.93  # Charge efficiency
eta_d = 0.93  # Discharge efficiency
delta_t = 1.0  # Time step (1 hour)
c_rate = 1.0  # C-rate for power limits (1C = full discharge/charge in 1 hour)

print(f"\nOperating Parameters:")
print(f"  Capacity factor cf = {cf:.2f}")
print(f"  Max curtailment c_max = {c_max:.2f}")
print(f"  Min generation = (1-c_max)*cf*p_max = {(1 - c_max) * cf:.2f} * p_max")

# =============================================================================
# Create augmented model with portfolio variables
# =============================================================================

model_portfolio = create_lindist_model(case)


# CUSTOM POWER BALANCE WITH DER INJECTION
# We need to replace add_p_flow_constraints with a version that includes p_der_inj
def add_p_flow_constraints_with_der(m, p_der_inj_var):
    """
    Add LinDistFlow power balance constraints with DER injection.
    Active power: P_ij = sum(P_jk) + p_L - p_D - p_der_inj
    """

    def p_balance_rule(m: LindistModelProtocol, _id, ph, t):
        load = m.p_load[_id, ph, t]
        generation = m.p_gen[_id, ph, t] if (_id, ph, t) in m.p_gen else 0
        p_bat = m.p_bat[_id, ph, t] if (_id, ph, t) in m.p_bat else 0
        der_inj = p_der_inj_var[_id, ph, t]  # DER injection (positive = generation)

        incoming_flow = m.p_flow[_id, ph, t]
        outgoing_flows = sum(
            m.p_flow[to_bus, ph, t]
            for to_bus in m.to_bus_map[_id]
            if (to_bus, ph) in m.branch_phase_set
        )
        # DER injection acts as negative load (reduces swing bus power needed)
        return incoming_flow == outgoing_flows + load - generation - p_bat - der_inj

    m.power_balance_p = pyo.Constraint(
        m.branch_phase_set, m.time_set, rule=p_balance_rule
    )


# Add portfolio index sets
model_portfolio.zone_set = pyo.Set(initialize=list(zones.keys()))
model_portfolio.resource_set = pyo.Set(initialize=resources)

# Portfolio variables
model_portfolio.p_max = pyo.Var(
    model_portfolio.zone_set,
    model_portfolio.resource_set,
    domain=pyo.NonNegativeReals,
    initialize=0,
    doc="Installed DER capacity by zone and resource",
)

model_portfolio.p_zr = pyo.Var(
    model_portfolio.zone_set,
    model_portfolio.resource_set,
    model_portfolio.time_set,
    domain=pyo.Reals,
    initialize=0,
    doc="Zonal DER dispatch by zone, resource, and time",
)

# Battery dispatch variables (optimizing existing batteries in the system)
model_portfolio.p_batt_ch = pyo.Var(
    model_portfolio.bat_set,
    model_portfolio.time_set,
    domain=pyo.NonNegativeReals,
    initialize=0,
    doc="Charging power for existing batteries",
)

model_portfolio.p_batt_disch = pyo.Var(
    model_portfolio.bat_set,
    model_portfolio.time_set,
    domain=pyo.NonNegativeReals,
    initialize=0,
    doc="Discharging power for existing batteries",
)

model_portfolio.soc_batt = pyo.Var(
    model_portfolio.bat_set,
    model_portfolio.time_set,
    domain=pyo.NonNegativeReals,
    initialize=0,
    doc="State of charge for existing batteries",
)

# New battery capacity expansion variables (zone-level investment decisions)
model_portfolio.e_max_new = pyo.Var(
    model_portfolio.zone_set,
    domain=pyo.NonNegativeReals,
    initialize=0,
    doc="New battery energy capacity to build by zone (p.u. MWh)",
)
model_portfolio.p_batt_new_ch = pyo.Var(
    model_portfolio.zone_set,
    model_portfolio.time_set,
    domain=pyo.NonNegativeReals,
    initialize=0,
    doc="Charging power of new battery capacity by zone and time",
)
model_portfolio.p_batt_new_disch = pyo.Var(
    model_portfolio.zone_set,
    model_portfolio.time_set,
    domain=pyo.NonNegativeReals,
    initialize=0,
    doc="Discharging power of new battery capacity by zone and time",
)
model_portfolio.soc_new = pyo.Var(
    model_portfolio.zone_set,
    model_portfolio.time_set,
    domain=pyo.NonNegativeReals,
    initialize=0,
    doc="State of charge of new battery capacity by zone and time",
)

# DER INJECTION VARIABLE (created before power flow constraints)
model_portfolio.p_der_inj = pyo.Var(
    model_portfolio.bus_phase_set,
    model_portfolio.time_set,
    domain=pyo.Reals,
    initialize=0,
    doc="DER injection at each bus-phase (positive = generation into grid)",
)

# NOW add power flow constraints with DER injection coupled in
add_p_flow_constraints_with_der(model_portfolio, model_portfolio.p_der_inj)

print(f"\nPortfolio variables added:")
print(f"  p_max:          {len(model_portfolio.p_max)} (zone × resource) [PV capacity]")
print(
    f"  p_zr:           {len(model_portfolio.p_zr)} (zone × resource × time) [PV dispatch]"
)
print(
    f"  e_max_new:      {len(model_portfolio.e_max_new)} (zone) [new BATT energy capacity]"
)
print(
    f"  p_batt_new_ch:  {len(model_portfolio.p_batt_new_ch)} (zone × time) [new BATT charge]"
)
print(
    f"  p_batt_new_disch:{len(model_portfolio.p_batt_new_disch)} (zone × time) [new BATT discharge]"
)
print(
    f"  p_der_inj:      {len(model_portfolio.p_der_inj)} (bus-phase × time) [COUPLED TO POWER FLOW]"
)

# Add remaining standard constraints
add_q_flow_constraints(model_portfolio)
add_voltage_drop_constraints(model_portfolio)
add_swing_bus_constraints(model_portfolio)
add_cvr_load_constraints(model_portfolio)
add_generator_constant_p_constraints_q_control(model_portfolio)
add_generator_constant_q_constraints_p_control(model_portfolio)
add_circular_generator_constraints_pq_control(model_portfolio)
add_capacitor_constraints(model_portfolio)
add_regulator_constraints(model_portfolio)
add_voltage_limits(model_portfolio)
add_generator_limits(model_portfolio)

# Add battery constraints
add_battery_constant_q_constraints_p_control(model_portfolio)
add_battery_energy_constraints(model_portfolio)
add_battery_net_p_bat_equal_phase_constraints(model_portfolio)
add_battery_power_limits(model_portfolio)
add_battery_soc_limits(model_portfolio)

# Add portfolio index sets

# =============================================================================
# Portfolio constraints
# =============================================================================


# Eq. 3: Exact capacity allocation for PV
def capacity_allocation_pv_rule(model):
    return sum(model.p_max[z, "PV"] for z in model.zone_set) == T_DER * c["PV"]


model_portfolio.capacity_allocation_pv = pyo.Constraint(
    rule=capacity_allocation_pv_rule
)


# Eq. 6: Generation dispatch envelope for PV
def gen_min_rule(model, z, r, t):
    return model.p_zr[z, r, t] >= (1 - c_max) * cf * model.p_max[z, r]


def gen_max_rule(model, z, r, t):
    return model.p_zr[z, r, t] <= cf * model.p_max[z, r]


model_portfolio.gen_min = pyo.Constraint(
    model_portfolio.zone_set,
    model_portfolio.resource_set,
    model_portfolio.time_set,
    rule=gen_min_rule,
)

model_portfolio.gen_max = pyo.Constraint(
    model_portfolio.zone_set,
    model_portfolio.resource_set,
    model_portfolio.time_set,
    rule=gen_max_rule,
)


# Battery energy balance for existing batteries (Eq. 7-8)
def battery_energy_balance_rule(model, bid, t):
    """Energy balance for existing batteries"""
    if t == model.start_step:
        soc_prev = pyo.value(model.start_soc[bid]) * pyo.value(
            model.energy_capacity[bid]
        )
    else:
        soc_prev = model.soc_batt[bid, t - 1]

    return (
        model.soc_batt[bid, t]
        == soc_prev
        + eta_c * delta_t * model.p_batt_ch[bid, t]
        - (1.0 / eta_d) * delta_t * model.p_batt_disch[bid, t]
    )


model_portfolio.battery_energy = pyo.Constraint(
    model_portfolio.bat_set,
    model_portfolio.time_set,
    rule=battery_energy_balance_rule,
)


# Battery charge/discharge limits use existing power limits
def battery_charge_limit_rule(model, bid, ph, t):
    return (0, model.p_charge[bid, t], model.s_bat_rated[bid, ph])


def battery_discharge_limit_rule(model, bid, ph, t):
    return (0, model.p_discharge[bid, t], model.s_bat_rated[bid, ph])


model_portfolio.battery_charge_limit = pyo.Constraint(
    model_portfolio.bat_phase_set,
    model_portfolio.time_set,
    rule=battery_charge_limit_rule,
)

model_portfolio.battery_discharge_limit = pyo.Constraint(
    model_portfolio.bat_phase_set,
    model_portfolio.time_set,
    rule=battery_discharge_limit_rule,
)

# Battery SOC limits handled by add_battery_soc_limits() above

# =============================================================================
# New battery capacity expansion constraints
# =============================================================================
soc_init_frac = 0.5  # New batteries start at 50% SOC
soc_min_frac = 0.2  # Lower SOC bound (20%)
soc_max_frac = 0.9  # Upper SOC bound (90%)


def batt_capacity_budget_rule(model):
    return sum(model.e_max_new[z] for z in model.zone_set) == T_BATT


model_portfolio.batt_capacity_budget = pyo.Constraint(rule=batt_capacity_budget_rule)


def new_batt_energy_rule(model, z, t):
    """SOC dynamics for new battery capacity (bilinear at t=start_step)."""
    if t == model.start_step:
        soc_prev = soc_init_frac * model.e_max_new[z]
    else:
        soc_prev = model.soc_new[z, t - 1]
    return (
        model.soc_new[z, t]
        == soc_prev
        + eta_c * delta_t * model.p_batt_new_ch[z, t]
        - (1.0 / eta_d) * delta_t * model.p_batt_new_disch[z, t]
    )


model_portfolio.new_batt_energy = pyo.Constraint(
    model_portfolio.zone_set, model_portfolio.time_set, rule=new_batt_energy_rule
)


def new_batt_ch_limit_rule(model, z, t):
    return model.p_batt_new_ch[z, t] <= c_rate * model.e_max_new[z]


def new_batt_disch_limit_rule(model, z, t):
    return model.p_batt_new_disch[z, t] <= c_rate * model.e_max_new[z]


model_portfolio.new_batt_ch_limit = pyo.Constraint(
    model_portfolio.zone_set, model_portfolio.time_set, rule=new_batt_ch_limit_rule
)
model_portfolio.new_batt_disch_limit = pyo.Constraint(
    model_portfolio.zone_set, model_portfolio.time_set, rule=new_batt_disch_limit_rule
)


def new_batt_soc_lower_rule(model, z, t):
    return model.soc_new[z, t] >= soc_min_frac * model.e_max_new[z]


def new_batt_soc_upper_rule(model, z, t):
    return model.soc_new[z, t] <= soc_max_frac * model.e_max_new[z]


model_portfolio.new_batt_soc_lower = pyo.Constraint(
    model_portfolio.zone_set, model_portfolio.time_set, rule=new_batt_soc_lower_rule
)
model_portfolio.new_batt_soc_upper = pyo.Constraint(
    model_portfolio.zone_set, model_portfolio.time_set, rule=new_batt_soc_upper_rule
)


# Eq. 4: Nodal injection mapping (PV + existing BATT + new BATT)
def der_injection_rule(model, n, ph, t):
    injection = 0

    # PV injection
    for z in model.zone_set:
        for r in model.resource_set:
            factor = alpha.get((z, n, r), 0.0)
            injection += factor * model.p_zr[z, r, t]

    # Existing battery injection (net discharge - charge, spread equally across phases)
    if n in batt_ids:
        bid = n
        n_phases = pyo.value(model.battery_n_phases[bid])
        injection += (model.p_discharge[bid, t] - model.p_charge[bid, t]) / n_phases

    # New battery capacity injection (load-proportional within zone)
    for z in model.zone_set:
        factor_batt = alpha.get((z, n, "BATT"), 0.0)
        injection += factor_batt * (
            model.p_batt_new_disch[z, t] - model.p_batt_new_ch[z, t]
        )

    return model.p_der_inj[n, ph, t] == injection


model_portfolio.der_injection_mapping = pyo.Constraint(
    model_portfolio.bus_phase_set, model_portfolio.time_set, rule=der_injection_rule
)

print(f"\nPortfolio constraints added:")
print(f"  PV capacity allocation (Eq. 3), T_DER={T_DER:.4f} p.u.")
print(f"  PV generation envelope (Eq. 6)")
print(f"  Existing battery energy balance + power limits")
print(f"  New BATT capacity budget, T_BATT={T_BATT:.4f} p.u.")
print(f"  New BATT energy balance + power/SOC limits")
print(f"  Nodal injection mapping (Eq. 4) - PV + existing BATT + new BATT")

# =============================================================================
# Objective: Minimize substation active power withdrawal
# =============================================================================


model_portfolio.objective = pyo.Objective(
    rule=substation_power_objective_rule, sense=pyo.minimize
)

print(f"\nObjective: Minimize substation active power withdrawal")
print(f"  Minimize sum of p_flow on branches leaving swing bus {swing_bus_id}")

# =============================================================================
# Solve portfolio model
# =============================================================================

print(f"\nSolving portfolio model...")
results_portfolio = opt.solve(model_portfolio, tee=False)

if results_portfolio.solver.status == pyo.SolverStatus.ok:
    obj_portfolio = pyo.value(model_portfolio.objective)
    print(f"✓ Portfolio OPF successful")
    print(f"  Substation power (p.u.): {obj_portfolio:.6f}")
else:
    raise ValueError(f"Portfolio solve failed: {results_portfolio.solver.status}")

# =============================================================================
# Results & Validation
# =============================================================================

print("\n" + "=" * 80)
print("RESULTS AND VALIDATION")
print("=" * 80)

# Compute actual substation power for baseline (for comparison)
obj_baseline_substation = sum(
    pyo.value(model_baseline.p_flow[to_bus, ph, 0])
    for to_bus, ph in model_baseline.branch_phase_set
    if model_baseline.from_bus_map[to_bus] == swing_bus_id
)

print(f"\nObjective Comparison:")
print(
    f"  Baseline substation power: {obj_baseline_substation:.6f} p.u. ({obj_baseline_substation * s_base / 1e6:.4f} MW)"
)
print(
    f"  Portfolio substation power: {obj_portfolio:.6f} p.u. ({obj_portfolio * s_base / 1e6:.4f} MW)"
)
print(
    f"  Reduction: {(obj_baseline_substation - obj_portfolio):.6f} p.u. ({100 * (obj_baseline_substation - obj_portfolio) / max(obj_baseline_substation, 1e-9):.1f}%)"
)

print(f"\nPV Capacity Allocation (p.u.):")
total_installed = 0
for r in resources:
    for z in zones.keys():
        p_max_val = pyo.value(model_portfolio.p_max[z, r])
        total_installed += p_max_val
        print(
            f"  p_max[{z},{r}] = {p_max_val:.6f} p.u. ({p_max_val * s_base / 1e6:.2f} MW)"
        )

print(f"\nCapacity Allocation Verification:")
print(f"  Total installed: {total_installed:.6f} p.u.")
print(f"  Target (T_DER): {T_DER:.6f} p.u.")
print(
    f"  Match: {abs(total_installed - T_DER) < 1e-5} (error: {abs(total_installed - T_DER):.2e})"
)

print(f"\nPV Dispatch (p.u.) - first time step:")
t = list(model_portfolio.time_set)[0]
for z in zones.keys():
    for r in resources:
        p_zr_val = pyo.value(model_portfolio.p_zr[z, r, t])
        p_max_val = pyo.value(model_portfolio.p_max[z, r])
        bounds = f"[{(1 - c_max) * cf * p_max_val:.6f}, {cf * p_max_val:.6f}]"
        print(f"  p_zr[{z},{r},{t}] = {p_zr_val:.6f} p.u. (bounds: {bounds})")

print(f"\nBattery Operation (existing systems) - first time step:")
t = list(model_portfolio.time_set)[0]
for bid in model_portfolio.bat_set:
    p_ch = pyo.value(model_portfolio.p_charge[bid, t])
    p_disch = pyo.value(model_portfolio.p_discharge[bid, t])
    soc = pyo.value(model_portfolio.soc[bid, t])
    energy_cap = pyo.value(model_portfolio.energy_capacity[bid])
    print(
        f"  Battery {bid}: charge={p_ch:.4f}, discharge={p_disch:.4f} p.u., SOC={soc:.4f}/{energy_cap:.4f} p.u."
    )

print(f"\nNew Battery Capacity Expansion (p.u.):")
total_e_max = 0
for z in zones.keys():
    e_max_val = pyo.value(model_portfolio.e_max_new[z])
    total_e_max += e_max_val
    disch_val = pyo.value(model_portfolio.p_batt_new_disch[z, t])
    ch_val = pyo.value(model_portfolio.p_batt_new_ch[z, t])
    soc_val = pyo.value(model_portfolio.soc_new[z, t])
    print(
        f"  e_max_new[{z}] = {e_max_val:.6f} p.u. | disch={disch_val:.4f}, ch={ch_val:.4f}, SOC={soc_val:.4f}"
    )
print(f"  Total new BATT energy: {total_e_max:.6f} p.u. (target: {T_BATT:.6f})")

# Check constraints
print(f"\nConstraint Verification (first time step):")
t = list(model_portfolio.time_set)[0]
checks_pass = True

# PV capacity allocation
for r in resources:
    total = sum(pyo.value(model_portfolio.p_max[z, r]) for z in zones.keys())
    expected = T_DER * c[r]
    match = abs(total - expected) < 1e-5
    print(
        f"  PV capacity allocation [{r}]: {match} (total={total:.6f}, expect={expected:.6f})"
    )
    checks_pass = checks_pass and match

# Battery capacity budget
total_e = sum(pyo.value(model_portfolio.e_max_new[z]) for z in zones.keys())
batt_match = abs(total_e - T_BATT) < 1e-5
print(
    f"  BATT capacity budget: {batt_match} (total={total_e:.6f}, expect={T_BATT:.6f})"
)
checks_pass = checks_pass and batt_match

# Generation bounds
for z in zones.keys():
    for r in resources:
        p_zr_val = pyo.value(model_portfolio.p_zr[z, r, t])
        p_max_val = pyo.value(model_portfolio.p_max[z, r])
        min_bound = (1 - c_max) * cf * p_max_val - 1e-6
        max_bound = cf * p_max_val + 1e-6
        in_bounds = min_bound <= p_zr_val <= max_bound
        print(f"  Dispatch bounds [{z},{r}]: {in_bounds}")
        checks_pass = checks_pass and in_bounds

# Allocation normalization
print(f"\nAllocation Factor Normalization:")
for z in zones.keys():
    for r in resources + ["BATT"]:
        total_alpha = sum(alpha.get((z, n, r), 0.0) for n in zones[z])
        print(
            f"  ∑_n α[{z},{r}] = {total_alpha:.10f} (normalized: {abs(total_alpha - 1.0) < 1e-9})"
        )
        checks_pass = checks_pass and abs(total_alpha - 1.0) < 1e-9

print("\n" + "=" * 80)
if checks_pass:
    print("✓ ALL VALIDATION CHECKS PASSED")
else:
    print("✗ SOME VALIDATION CHECKS FAILED")
print("=" * 80)
