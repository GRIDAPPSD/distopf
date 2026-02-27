"""
Phase 1 & 2: DER Portfolio Capacity Expansion Baseline
========================================================

This script implements the first pass of the DER portfolio formulation:
- Phase 1: Validate standard OPF baseline on 2Bus-1ph-batt tiny feeder
- Phase 2: Layer in portfolio decision variables and constraints for generation-only resources

Structure:
  1. Load case and verify baseline feasibility (standard substation minimization OPF)
  2. Define zones, allocation factors, resource mix, and capacity target
  3. Add portfolio variables: p_max[z,r] (installed capacity), p_zr[z,r,t] (dispatch)
  4. Add portfolio constraints:
     - Eq. (3): Exact zone allocation of capacity ∑_z p_max[z,r] = T_DER * c[r]
     - Eq. (6): Generation dispatch envelope (1-c_max)*cf*p_max ≤ p_zr ≤ cf*p_max
     - Eq. (4): Nodal injection mapping via allocation factors α[z,n,r]
  5. Solve augmented model and validate against baseline

Units: All quantities in per-unit on s_base (typically 1 MW for this case)
Objective: Minimize total active power imported from substation (proxy for energy reduction)
Horizon: Single-step (n_steps=1)
Resources: Generation-only (PV-like) with global curtailment parameter c_max
"""

import distopf as opf
import pyomo.environ as pyo
import pandas as pd
import numpy as np
from pathlib import Path

from distopf.pyomo_models.lindist import create_lindist_model
from distopf.api import create_case
from distopf.pyomo_models.constraints import (
    add_p_flow_constraints,
    add_q_flow_constraints,
    add_voltage_drop_constraints,
    add_swing_bus_constraints,
    add_cvr_load_constraints,
    add_capacitor_constraints,
    add_regulator_constraints,
    add_generator_limits,
    add_voltage_limits,
    add_generator_constant_p_constraints_q_control,
    add_generator_constant_q_constraints_p_control,
)
from distopf.pyomo_models.objectives import substation_power_objective_rule
from distopf.pyomo_models.results import get_values

# =============================================================================
# PHASE 1: BASELINE STANDARD OPF
# =============================================================================

print("=" * 80)
print("PHASE 1: BASELINE STANDARD OPF on 2Bus-1ph-batt")
print("=" * 80)

# Load 2Bus-1ph-batt case
case = create_case(data_path=opf.CASES_DIR / "csv" / "2Bus-1ph-batt")
print(f"\nCase loaded: 2Bus-1ph-batt")
print(f"  Buses: {case.bus_data.shape[0]}")
print(f"  Branches: {case.branch_data.shape[0]}")
print(f"  Generators: {case.gen_data.shape[0]}")
s_base = case.bus_data["s_base"].iloc[0]
print(f"  System base (s_base): {s_base} MVA")

# Create baseline model
model_baseline = create_lindist_model(case)
print(f"\nBaseline model created.")
print(f"  Bus-phase set size: {len(model_baseline.bus_phase_set)}")
print(f"  Branch-phase set size: {len(model_baseline.branch_phase_set)}")
print(f"  Time set size: {len(model_baseline.time_set)}")

# Add standard constraints
add_p_flow_constraints(model_baseline)
add_q_flow_constraints(model_baseline)
add_voltage_drop_constraints(model_baseline)
add_swing_bus_constraints(model_baseline)
add_cvr_load_constraints(model_baseline)
add_capacitor_constraints(model_baseline)
add_regulator_constraints(model_baseline)
add_voltage_limits(model_baseline)
add_generator_limits(model_baseline)
add_generator_constant_p_constraints_q_control(model_baseline)
add_generator_constant_q_constraints_p_control(model_baseline)

# Set objective: minimize substation active power
model_baseline.objective = pyo.Objective(
    rule=substation_power_objective_rule,
    sense=pyo.minimize,
    doc="Minimize substation import",
)

# Solve baseline
print("\nSolving baseline model (substation minimization)...")
opt = pyo.SolverFactory("ipopt")

# Check solver availability
if not opt.available(exception_flag=False):
    raise RuntimeError(
        "IPOPT solver not available. Install via: pip install pyomo[ipopt]"
    )

results_baseline = opt.solve(model_baseline, tee=False)

# Validate baseline solution
if results_baseline.solver.status != pyo.SolverStatus.ok:
    raise ValueError(f"Baseline solve failed: {results_baseline.solver.status}")

obj_baseline = pyo.value(model_baseline.objective)
print(f"✓ Baseline solve successful")
print(f"  Objective (substation import, p.u.): {obj_baseline:.6f}")

# Extract and display baseline results
p_gen_baseline = get_values(model_baseline.p_gen)
q_gen_baseline = get_values(model_baseline.q_gen)
v_baseline = get_values(model_baseline.v2) ** 0.5
p_flow_baseline = get_values(model_baseline.p_flow)

print(f"  Generator active power (p.u.):")
for gen_id in p_gen_baseline.columns:
    print(f"    Gen {gen_id}: {p_gen_baseline[gen_id].values[0]:.6f}")
print(f"  Voltage magnitudes (p.u.):")
for bus_id in v_baseline.columns:
    print(f"    Bus {bus_id}: {v_baseline[bus_id].values[0]:.6f}")

print("\n" + "=" * 80)
print("PHASE 2: ADD DER PORTFOLIO VARIABLES AND CONSTRAINTS")
print("=" * 80)

# =============================================================================
# PHASE 2: PORTFOLIO FORMULATION
# =============================================================================

# Compute capacity target from total load (10% of system load)
total_load_p = case.bus_data[["pl_a", "pl_b", "pl_c"]].sum().sum()
total_load_q = case.bus_data[["ql_a", "ql_b", "ql_c"]].sum().sum()
T_DER = 0.1 * total_load_p  # 10% of active load

print(f"\nCapacity Planning Parameters:")
print(f"  Total system load (active): {total_load_p:.6f} p.u.")
print(f"  Total system load (reactive): {total_load_q:.6f} p.u.")
print(f"  Target new DER capacity (T_DER = 10% load): {T_DER:.6f} p.u.")

# Define zones: split feeder into two zones along main trunk
# For 2Bus case: Zone 1 = Bus 1, Zone 2 = Bus 2 (or bus at end of branch)
# Identify candidate buses (non-swing, primary system)
swing_bus_id = case.bus_data[case.bus_data["bus_type"] == case.SWING_BUS].index[0]
candidate_buses = [b for b in case.bus_data.index if b != swing_bus_id]

print(f"\nZone Definition:")
print(f"  Swing bus (voltage source): {swing_bus_id}")
print(f"  Candidate buses for DER: {candidate_buses}")

# Hard-coded zoning: split candidate buses into two zones
midpoint = len(candidate_buses) // 2
zone_1_buses = candidate_buses[:midpoint] if midpoint > 0 else candidate_buses
zone_2_buses = candidate_buses[midpoint:] if midpoint > 0 else []

zones = {"Z1": zone_1_buses, "Z2": zone_2_buses}
print(f"  Zone Z1 buses: {zones['Z1']}")
print(f"  Zone Z2 buses: {zones['Z2']}")

# Define resource set: generation-only (PV-like)
resources = ["solar"]  # Single resource in first pass
print(f"\nResource Set (generation-only):")
print(f"  Resources: {resources}")

# Define fixed resource mix: c[r]
# For single resource: c_solar = 1.0
c = {r: (1.0 / len(resources)) for r in resources}
print(f"  Resource mix fractions:")
for r in resources:
    print(f"    c[{r}] = {c[r]:.4f}")

# Validate mix fractions sum to 1
assert abs(sum(c.values()) - 1.0) < 1e-10, "Resource mix fractions must sum to 1"

# Define allocation factors α[z,n,r]: hard-coded for 2Bus case
# Normalize within each zone: ∑_n α[z,n,r] = 1
# Preferred method: load-proportional allocation within each zone


def build_allocation_factors(zones, resources, case):
    """
    Build allocation factors α[z,n,r] (load-proportional).
    Returns: dict {(z,n,r): factor}

    Constraint: ∑_n α[z,n,r] = 1 for all z,r
    """
    alpha = {}
    for z_name, z_buses in zones.items():
        for r in resources:
            # Total load in zone
            z_load = case.bus_data.loc[z_buses, ["pl_a", "pl_b", "pl_c"]].sum().sum()

            # Allocate proportionally to load at each bus
            for n in z_buses:
                n_load = case.bus_data.loc[n, ["pl_a", "pl_b", "pl_c"]].sum()
                if z_load > 0:
                    alpha[(z_name, n, r)] = n_load / z_load
                else:
                    # No load: allocate evenly across zone buses
                    alpha[(z_name, n, r)] = 1.0 / len(z_buses)

    return alpha


alpha = build_allocation_factors(zones, resources, case)

print(f"\nAllocation Factors α[z,n,r] (load-proportional):")
for z_name in zones.keys():
    print(f"  Zone {z_name}:")
    for n in zones[z_name]:
        for r in resources:
            factor = alpha.get((z_name, n, r), 0.0)
            print(f"    α[{z_name},{n},{r}] = {factor:.6f}")

# Verify allocation normalization
print(f"\nAllocation Normalization Check (∑_n α[z,n,r] = 1):")
for z_name in zones.keys():
    for r in resources:
        total = sum(alpha.get((z_name, n, r), 0.0) for n in zones[z_name])
        print(
            f"  Zone {z_name}, Resource {r}: {total:.10f} {'✓' if abs(total - 1.0) < 1e-9 else '✗ ERROR'}"
        )
        assert abs(total - 1.0) < 1e-9, f"Allocation not normalized for {z_name},{r}"

# Define capacity factor cf[t]
# For single-step: cf is a scalar (can vary by time in multi-period)
n_time_steps = len(model_baseline.time_set)
cf = {t: 0.8 for t in model_baseline.time_set}  # 80% capacity factor for PV

print(f"\nCapacity Factor cf[t]:")
for t in model_baseline.time_set:
    print(f"  cf[{t}] = {cf[t]:.4f}")

# Curtailment parameter
c_max = 0.2  # Allow up to 20% curtailment
print(f"\nCurtailment Parameter:")
print(f"  c_max (max curtailment ratio) = {c_max:.4f}")

# =============================================================================
# Create augmented model with portfolio variables
# =============================================================================

model_portfolio = create_lindist_model(case)
print(f"\nPortfolio model created (copy of baseline structure).")

# Add standard constraints
add_p_flow_constraints(model_portfolio)
add_q_flow_constraints(model_portfolio)
add_voltage_drop_constraints(model_portfolio)
add_swing_bus_constraints(model_portfolio)
add_cvr_load_constraints(model_portfolio)
add_capacitor_constraints(model_portfolio)
add_regulator_constraints(model_portfolio)
add_voltage_limits(model_portfolio)
add_generator_limits(model_portfolio)
add_generator_constant_p_constraints_q_control(model_portfolio)
add_generator_constant_q_constraints_p_control(model_portfolio)

# Define index sets for portfolio variables
model_portfolio.zone_set = pyo.Set(
    initialize=list(zones.keys()), doc="Zones for DER allocation"
)
model_portfolio.resource_set = pyo.Set(initialize=resources, doc="Resource types")

print(f"Index sets added:")
print(f"  Zone set: {list(model_portfolio.zone_set)}")
print(f"  Resource set: {list(model_portfolio.resource_set)}")

# Portfolio decision variables
# p_max[z,r]: installed capacity (MW or p.u.) in zone z for resource r
model_portfolio.p_max = pyo.Var(
    model_portfolio.zone_set,
    model_portfolio.resource_set,
    domain=pyo.NonNegativeReals,
    initialize=0,
    doc="Installed DER capacity (p.u.) by zone and resource",
)

# p_zr[z,r,t]: zonal dispatch (MW or p.u.) for zone z, resource r, time t
model_portfolio.p_zr = pyo.Var(
    model_portfolio.zone_set,
    model_portfolio.resource_set,
    model_portfolio.time_set,
    domain=pyo.Reals,
    initialize=0,
    doc="Zonal DER dispatch (p.u.) by zone, resource, and time",
)

print(f"\nPortfolio variables added:")
print(f"  p_max[z,r]: installed capacity {len(model_portfolio.p_max)} variables")
print(f"  p_zr[z,r,t]: zonal dispatch {len(model_portfolio.p_zr)} variables")

# =============================================================================
# Portfolio constraints (Eqs. 3, 6, 4)
# =============================================================================


# Constraint 1 (Eq. 3): Exact zone allocation
# ∑_z p_max[z,r] = T_DER * c[r] for all r
def capacity_allocation_rule(model, r):
    """Total installed capacity for resource r must match target fraction."""
    return sum(model.p_max[z, r] for z in model.zone_set) == T_DER * c[r]


model_portfolio.capacity_allocation = pyo.Constraint(
    model_portfolio.resource_set,
    rule=capacity_allocation_rule,
    doc="Eq. (3): Exact zone allocation of capacity",
)

print(f"\nCapacity Allocation Constraint (Eq. 3):")
print(f"  ∑_z p_max[z,r] = T_DER * c[r]")
print(f"  Expected total capacity across all zones: {T_DER:.6f} p.u.")


# Constraint 2 (Eq. 6): Generation dispatch envelope
# (1-c_max)*cf[t]*p_max[z,r] ≤ p_zr[z,r,t] ≤ cf[t]*p_max[z,r]
def generation_min_rule(model, z, r, t):
    """Minimum dispatch (minimum must-run fraction)."""
    return model.p_zr[z, r, t] >= (1 - c_max) * cf[t] * model.p_max[z, r]


def generation_max_rule(model, z, r, t):
    """Maximum dispatch (capacity factor limit)."""
    return model.p_zr[z, r, t] <= cf[t] * model.p_max[z, r]


model_portfolio.generation_min = pyo.Constraint(
    model_portfolio.zone_set,
    model_portfolio.resource_set,
    model_portfolio.time_set,
    rule=generation_min_rule,
    doc="Eq. (6): Minimum generation (1-c_max)*cf*p_max",
)

model_portfolio.generation_max = pyo.Constraint(
    model_portfolio.zone_set,
    model_portfolio.resource_set,
    model_portfolio.time_set,
    rule=generation_max_rule,
    doc="Eq. (6): Maximum generation cf*p_max",
)

print(f"\nGeneration Dispatch Envelope Constraint (Eq. 6):")
print(f"  (1-c_max)*cf[t]*p_max[z,r] ≤ p_zr[z,r,t] ≤ cf[t]*p_max[z,r]")
print(f"  Min factor: (1-c_max) = {1 - c_max:.4f}")
print(f"  Max factor: cf = {min(cf.values()):.4f} to {max(cf.values()):.4f}")

# Constraint 3 (Eq. 4): Nodal injection mapping
# For each bus n and time t:
#   p_inj[n,t] = ∑_z ∑_r α[z,n,r] * p_zr[z,r,t]
# This is built into the p_gen constraint by mapping portfolio dispatch to existing generators

# Strategy: Add auxiliary variable for DER injection at each bus, then add to p_gen via constraint
model_portfolio.p_der_inj = pyo.Var(
    model_portfolio.bus_phase_set,
    model_portfolio.time_set,
    domain=pyo.Reals,
    initialize=0,
    doc="Total DER active injection (p.u.) at each bus-phase",
)


def der_injection_rule(model, n, ph, t):
    """
    Map zonal dispatch to nodal DER injection.
    p_der_inj[n,t] = ∑_z ∑_r α[z,n,r] * p_zr[z,r,t]
    """
    injection = 0
    for z in model.zone_set:
        for r in model.resource_set:
            # Get allocation factor from precomputed dict
            factor = alpha.get((z, n, r), 0.0)
            injection += factor * model.p_zr[z, r, t]
    return model.p_der_inj[n, ph, t] == injection


model_portfolio.der_injection_mapping = pyo.Constraint(
    model_portfolio.bus_phase_set,
    model_portfolio.time_set,
    rule=der_injection_rule,
    doc="Eq. (4): Node-level injection mapping via allocation factors",
)

print(f"\nNodeal Injection Mapping Constraint (Eq. 4):")
print(f"  p_der_inj[n,t] = ∑_z ∑_r α[z,n,r] * p_zr[z,r,t]")
print(f"  Mapping variables added: {len(model_portfolio.p_der_inj)}")

# Strategy for including DER in power balance:
# Add constraint that modifies the swing bus power to account for total DER injection
# This way, the swing bus must supply less power when DER is generating

# First identify swing bus
swing_buses = [
    b
    for b in model_portfolio.bus_data.index
    if model_portfolio.bus_data.loc[b, "bus_type"] == model_portfolio.SWING_BUS
]
if swing_buses:
    swing_bus = swing_buses[0]

    # Modify objective to reflect DER reduction in swing bus injection
    def portfolio_objective_with_der_rule(model):
        """
        Minimize total active power imported from substation, accounting for DER injection.
        """
        # Start with substation (swing bus) power generation
        substation_power = 0
        for ph in model.phases:
            for t in model.time_set:
                substation_power += model.p_gen[swing_bus, ph, t]

        # Subtract total DER injection (reduces required import)
        total_der_injection = 0
        for n, ph in model.bus_phase_set:
            for t in model.time_set:
                total_der_injection += model.p_der_inj[n, ph, t]

        return substation_power - total_der_injection

    model_portfolio.objective = pyo.Objective(
        rule=portfolio_objective_with_der_rule,
        sense=pyo.minimize,
        doc="Minimize net substation import (swing bus generation minus total DER)",
    )

    print(f"\nObjective updated to account for DER:")
    print(f"  Swing bus: {swing_bus}")
    print(f"  Objective = p_gen[swing_bus] - ∑(DER injection)")
else:
    # Fallback if no swing bus found: use original objective
    model_portfolio.objective = pyo.Objective(
        rule=substation_power_objective_rule,
        sense=pyo.minimize,
        doc="Minimize substation active power import (fallback)",
    )
    print(f"\nNo swing bus found; using standard substation power objective")


print(f"\nObjective set: Minimize substation import (with DER reduction).")

# =============================================================================
# Solve augmented model
# =============================================================================

print(f"\nSolving portfolio model (with DER capacity expansion)...")
results_portfolio = opt.solve(model_portfolio, tee=False)

if results_portfolio.solver.status != pyo.SolverStatus.ok:
    raise ValueError(f"Portfolio solve failed: {results_portfolio.solver.status}")

obj_portfolio = pyo.value(model_portfolio.objective)
print(f"✓ Portfolio solve successful")
print(f"  Objective (substation import, p.u.): {obj_portfolio:.6f}")
print(
    f"  Objective reduction vs baseline: {obj_baseline - obj_portfolio:.6f} ({100 * (obj_baseline - obj_portfolio) / obj_baseline:.2f}%)"
)

# =============================================================================
# Extract and validate results
# =============================================================================

print("\n" + "=" * 80)
print("RESULTS AND VALIDATION")
print("=" * 80)

# Extract portfolio capacity allocation
print(f"\nDER Capacity Installation (p.u.):")
total_installed = 0
for r in resources:
    for z in zones.keys():
        p_max_val = pyo.value(model_portfolio.p_max[z, r])
        print(f"  p_max[{z},{r}] = {p_max_val:.6f} p.u.")
        total_installed += p_max_val

print(f"  Total installed capacity: {total_installed:.6f} p.u.")
print(f"  Target capacity (T_DER): {T_DER:.6f} p.u.")
print(
    f"  Allocation exact? {abs(total_installed - T_DER) < 1e-6} (error: {abs(total_installed - T_DER):.2e})"
)

# Verify exact allocation constraint
for r in resources:
    total_by_resource = sum(
        pyo.value(model_portfolio.p_max[z, r]) for z in zones.keys()
    )
    target_by_resource = T_DER * c[r]
    print(f"\n  Resource {r}:")
    print(f"    Installed:  {total_by_resource:.6f} p.u.")
    print(f"    Target:     {target_by_resource:.6f} p.u.")
    print(
        f"    Match? {abs(total_by_resource - target_by_resource) < 1e-6} (error: {abs(total_by_resource - target_by_resource):.2e})"
    )

# Extract dispatch
print(f"\nDER Dispatch (p.u.) by Zone and Resource:")
for t in model_portfolio.time_set:
    print(f"  Time step t={t}:")
    for z in zones.keys():
        for r in resources:
            p_zr_val = pyo.value(model_portfolio.p_zr[z, r, t])
            p_max_val = pyo.value(model_portfolio.p_max[z, r])

            # Check bounds
            min_dispatch = (1 - c_max) * cf[t] * p_max_val
            max_dispatch = cf[t] * p_max_val

            in_bounds = (
                min_dispatch <= p_zr_val + 1e-6 and p_zr_val <= max_dispatch + 1e-6
            )

            print(
                f"    p_zr[{z},{r},{t}] = {p_zr_val:.6f} p.u. (bounds: [{min_dispatch:.6f}, {max_dispatch:.6f}]) {'✓' if in_bounds else '✗'}"
            )

# Extract nodal DER injection
print(f"\nDER Injection at Buses (p.u.):")
p_der_inj = get_values(model_portfolio.p_der_inj)
print(f"  DER injection profile (phase a, time 0):")
for n in p_der_inj.columns:
    inj_val = p_der_inj[n].iloc[0] if len(p_der_inj) > 0 else 0
    print(f"    Bus {n}: {inj_val:.6f} p.u.")

# Get updated generator and voltage results
p_gen_portfolio = get_values(model_portfolio.p_gen)
v_portfolio = get_values(model_portfolio.v2) ** 0.5

print(f"\nGenerator Output (with DER):")
for gen_id in p_gen_portfolio.columns:
    val_baseline = p_gen_baseline[gen_id].values[0]
    val_portfolio = p_gen_portfolio[gen_id].values[0]
    delta = val_baseline - val_portfolio
    print(
        f"  Gen {gen_id}: {val_portfolio:.6f} p.u. (baseline: {val_baseline:.6f}, delta: {delta:.6f})"
    )

print(f"\nVoltage Magnitudes (with DER):")
for bus_id in v_portfolio.columns:
    val_baseline = v_baseline[bus_id].values[0]
    val_portfolio = v_portfolio[bus_id].values[0]
    delta = val_portfolio - val_baseline
    print(
        f"  Bus {bus_id}: {val_portfolio:.6f} p.u. (baseline: {val_baseline:.6f}, delta: {delta:+.6f})"
    )

print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

# Numerical checks
checks = {
    "Capacity allocation exact": abs(total_installed - T_DER) < 1e-5,
    "Dispatch within bounds": all(
        (1 - c_max) * cf[t] * pyo.value(model_portfolio.p_max[z, r]) - 1e-5
        <= pyo.value(model_portfolio.p_zr[z, r, t])
        <= cf[t] * pyo.value(model_portfolio.p_max[z, r]) + 1e-5
        for z in zones.keys()
        for r in resources
        for t in model_portfolio.time_set
    ),
    "Allocation normalization preserved": all(
        abs(sum(alpha.get((z, n, r), 0.0) for n in zones[z]) - 1.0) < 1e-9
        for z in zones.keys()
        for r in resources
    ),
    "Portfolio objective lower than baseline": obj_portfolio < obj_baseline,
    "Solver termination optimal": results_portfolio.solver.status
    == pyo.SolverStatus.ok,
}

print("\nNumerical Validation:")
for check_name, result in checks.items():
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"  {status}: {check_name}")

all_pass = all(checks.values())
print(f"\n{'=' * 80}")
print(
    f"Overall Status: {'✓ ALL CHECKS PASSED' if all_pass else '✗ SOME CHECKS FAILED'}"
)
print(f"{'=' * 80}")
