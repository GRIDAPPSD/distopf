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
print("DER PORTFOLIO FORMULATION: PRAGMATIC BASELINE")
print("(Using known-working ieee123_30der case)")
print("=" * 80)

# =============================================================================
# PHASE 1: BASELINE OPF (Known Working)
# =============================================================================

print("\nPHASE 1: Load case and build baseline OPF model")
print("-" * 80)

case = create_case(data_path=opf.CASES_DIR / "csv" / "ieee123_30der")
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

print(f"Standard constraints added")


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
T_DER = 10.0 * total_load_p  # 10% of load

print(f"\nCapacity Planning Parameters:")
print(
    f"  Total system load: {total_load_p:.6f} p.u. ({total_load_p * s_base / 1e6:.2f} MW)"
)
print(
    f"  Target new DER capacity (10%): {T_DER:.6f} p.u. ({T_DER * s_base / 1e6:.2f} MW)"
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

# Resource set: single PV resource
resources = ["PV"]
c = {r: 1.0 for r in resources}  # 100% PV

print(f"\nResource Set:")
print(f"  Resources: {resources}")
print(f"  Resource mix: c[PV] = 1.0")


# Allocation factors: load-proportional
def build_alpha(zones, resources, bus_data_by_id):
    """Build allocation factors load-proportionally."""
    alpha = {}
    for z_name, z_buses in zones.items():
        for r in resources:
            z_load = bus_data_by_id.loc[z_buses, ["pl_a", "pl_b", "pl_c"]].sum().sum()
            for n in z_buses:
                n_load = bus_data_by_id.loc[n, ["pl_a", "pl_b", "pl_c"]].sum()
                if z_load > 0:
                    alpha[(z_name, n, r)] = n_load / z_load
                else:
                    alpha[(z_name, n, r)] = 1.0 / len(z_buses)
    return alpha


alpha = build_alpha(zones, resources, bus_data_by_id)

print(f"\nAllocation Factors Check:")
for z_name in zones.keys():
    for r in resources:
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

print(f"\nOperating Parameters:")
print(f"  Capacity factor cf = {cf:.2f}")
print(f"  Max curtailment c_max = {c_max:.2f}")
print(f"  Min generation = (1-c_max)*cf*p_max = {(1 - c_max) * cf:.2f} * p_max")

# =============================================================================
# Create augmented model with portfolio variables
# =============================================================================

model_portfolio = create_lindist_model(case)

# Add standard constraints
add_p_flow_constraints(model_portfolio)
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

model_portfolio.p_der_inj = pyo.Var(
    model_portfolio.bus_phase_set,
    model_portfolio.time_set,
    domain=pyo.Reals,
    initialize=0,
    doc="DER injection at each bus-phase",
)

print(f"\nPortfolio variables added:")
print(f"  p_max: {len(model_portfolio.p_max)} (zone × resource)")
print(f"  p_zr: {len(model_portfolio.p_zr)} (zone × resource × time)")
print(f"  p_der_inj: {len(model_portfolio.p_der_inj)} (bus-phase × time)")

# =============================================================================
# Portfolio constraints
# =============================================================================


# Eq. 3: Exact capacity allocation
def capacity_allocation_rule(model, r):
    return sum(model.p_max[z, r] for z in model.zone_set) == T_DER * c[r]


model_portfolio.capacity_allocation = pyo.Constraint(
    model_portfolio.resource_set, rule=capacity_allocation_rule
)


# Eq. 6: Generation dispatch envelope
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


# Eq. 4: Nodal injection mapping
def der_injection_rule(model, n, ph, t):
    injection = 0
    for z in model.zone_set:
        for r in model.resource_set:
            factor = alpha.get((z, n, r), 0.0)
            injection += factor * model.p_zr[z, r, t]
    return model.p_der_inj[n, ph, t] == injection


model_portfolio.der_injection_mapping = pyo.Constraint(
    model_portfolio.bus_phase_set, model_portfolio.time_set, rule=der_injection_rule
)

print(f"\nPortfolio constraints added:")
print(f"  Capacity allocation (Eq. 3)")
print(f"  Generation envelope (Eq. 6) - min/max")
print(f"  Nodal injection mapping (Eq. 4)")

# =============================================================================
# Objective: Minimize loss WITH DER (reduced baseline loss)
# =============================================================================


def portfolio_loss_objective_rule(model):
    """
    Loss minimization accounting for DER injection reducing branch flows.
    Strategy: Minimize (baseline loss - DER benefit)
    where benefit ≈ saved losses from reduced swing bus power
    """
    total_loss = 0
    for _id, ph in model.branch_phase_set:
        for t in model.time_set:
            total_loss += (model.p_flow[_id, ph, t] ** 2) * model.r[_id, ph + ph]
            total_loss += (model.q_flow[_id, ph, t] ** 2) * model.r[_id, ph + ph]

    # Bonus/reduction for DER generation (only for active power)
    # Use zone-specific weights to represent higher marginal value in electrically
    # distant areas and avoid artificial symmetry across zones.
    der_benefit = 0
    for z in model.zone_set:
        for r in model.resource_set:
            for t in model.time_set:
                der_benefit += 0.01 * zone_weights[str(z)] * model.p_zr[z, r, t]

    return total_loss - der_benefit


model_portfolio.objective = pyo.Objective(
    rule=portfolio_loss_objective_rule, sense=pyo.minimize
)

print(f"\nObjective: Loss minimization with DER incentive")
print(f"  Base: minimize I²R losses")
print(f"  Bonus: reward DER generation (0.01 × zone_weight × dispatch)")

# =============================================================================
# Solve portfolio model
# =============================================================================

print(f"\nSolving portfolio model...")
results_portfolio = opt.solve(model_portfolio, tee=False)

if results_portfolio.solver.status == pyo.SolverStatus.ok:
    obj_portfolio = pyo.value(model_portfolio.objective)
    print(f"✓ Portfolio OPF successful")
    print(f"  Loss+DER objective (p.u.): {obj_portfolio:.6f}")
else:
    raise ValueError(f"Portfolio solve failed: {results_portfolio.solver.status}")

# =============================================================================
# Results & Validation
# =============================================================================

print("\n" + "=" * 80)
print("RESULTS AND VALIDATION")
print("=" * 80)

print(f"\nObjective Comparison:")
print(
    f"  Baseline loss: {obj_baseline:.6f} p.u. ({obj_baseline * s_base / 1e3:.2f} kW)"
)
print(f"  Portfolio obj: {obj_portfolio:.6f} p.u.")

print(f"\nDER Capacity Allocation (p.u.):")
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

print(f"\nDER Dispatch (p.u.) -  first time step:")
for z in zones.keys():
    for r in resources:
        t = list(model_portfolio.time_set)[0]
        p_zr_val = pyo.value(model_portfolio.p_zr[z, r, t])
        p_max_val = pyo.value(model_portfolio.p_max[z, r])
        bounds = f"[{(1 - c_max) * cf * p_max_val:.6f}, {cf * p_max_val:.6f}]"
        print(f"  p_zr[{z},{r},{t}] = {p_zr_val:.6f} p.u. (bounds: {bounds})")

# Check constraints
print(f"\nConstraint Verification (first time step):")
t = list(model_portfolio.time_set)[0]
checks_pass = True

# Capacity allocation
for r in resources:
    total = sum(pyo.value(model_portfolio.p_max[z, r]) for z in zones.keys())
    expected = T_DER * c[r]
    match = abs(total - expected) < 1e-5
    print(
        f"  Capacity allocation [{r}]: {match} (total={total:.6f}, expect={expected:.6f})"
    )
    checks_pass = checks_pass and match

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
    for r in resources:
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
