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
    add_octagonal_inverter_constraints_pq_control,
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
from distopf.pyomo_models.capacity_expansion_constraints import (
    create_zones_from_edge_names,
    add_capacity_expansion_as_fraction_of_load,
    add_pv_parameters,
    add_bess_parameters,
    add_capacity_expansion_with_slack_constraints,
)
from distopf.pyomo_models.slack_constraints import (
    add_voltage_slack_constraints,
    add_thermal_slack_constraints,
    thermal_slack_penalty,
    voltage_slack_penalty,
)
from distopf.pyomo_models.results import PyoResult

# ---------------------------------------------------------------------------
# Case + system data
# ---------------------------------------------------------------------------
case = create_case(
    data_path=opf.CASES_DIR / "csv" / "ieee123_30der_bat",
    ignore_schedule=True,
    ignore_bat=True,
    ignore_gen=True,
    n_steps=1,
    start_step=0,
)


# ---------------------------------------------------------------------------
# Baseline OPF (for comparison)
# ---------------------------------------------------------------------------
def run_baseline_opf():
    def _add_standard_constraints(m):
        add_p_flow_constraints(m)
        add_q_flow_constraints(m)
        add_voltage_drop_constraints(m)
        add_swing_bus_constraints(m)
        add_cvr_load_constraints(m)
        add_generator_constant_p_constraints_q_control(m)
        add_generator_constant_q_constraints_p_control(m)
        add_octagonal_inverter_constraints_pq_control(m)
        add_capacitor_constraints(m)
        add_regulator_constraints(m)
        # add_voltage_limits(m)
        add_generator_limits(m)
        add_battery_constant_q_constraints_p_control(m)
        add_battery_energy_constraints(m)
        add_battery_net_p_bat_equal_phase_constraints(m)
        add_battery_power_limits(m)
        add_battery_soc_limits(m)
        # slack constraints
        add_voltage_slack_constraints(m)
        add_thermal_slack_constraints(m)

    opt = pyo.SolverFactory("highs")

    m_base = create_lindist_model(case)
    _add_standard_constraints(m_base)

    m_base.objective = pyo.Objective(
        rule=lambda _m: substation_power_objective_rule(_m)
        + thermal_slack_penalty(_m)
        + voltage_slack_penalty(_m),
        sense=pyo.minimize,
    )
    res = opt.solve(m_base, tee=False)
    if res.solver.status != pyo.SolverStatus.ok:
        raise RuntimeError("Baseline solve failed")
    return PyoResult(m_base, pyo.value(substation_power_objective_rule(m_base)))


# ---------------------------------------------------------------------------
# Portfolio model
# ---------------------------------------------------------------------------
def run_der_expansion():
    zones = create_zones_from_edge_names(case, ["sw2", "sw3", "sw4"])
    del zones[0]
    m = create_lindist_model(case)

    add_capacity_expansion_as_fraction_of_load(
        m, case, fraction=1.0, relative_capacity={"PV": 0.5, "BESS": 0.5}
    )
    add_pv_parameters(m, curtailment_max=0.0, capacity_factor=1.0)
    add_bess_parameters(m, e_max=10, soc=0.5, discharge_derate=1, charge_derate=1)
    add_capacity_expansion_with_slack_constraints(m, case, zones)

    # --- objective and solve --------------------------------------------------
    m.objective = pyo.Objective(
        rule=lambda _m: substation_power_objective_rule(_m)
        + thermal_slack_penalty(_m)
        + voltage_slack_penalty(_m),
        sense=pyo.minimize,
    )

    opt = pyo.SolverFactory("highs")
    res = opt.solve(m, tee=True)
    if res.solver.status != pyo.SolverStatus.ok:
        raise RuntimeError(f"Portfolio solve failed: {res.solver.status}")

    # Results
    # ---------------------------------------------------------------------------
    p_sub = pyo.value(substation_power_objective_rule(m))
    result = PyoResult(m, pyo.value(substation_power_objective_rule(m)))
    return m, zones, result


if __name__ == "__main__":
    t0 = case.start_step
    result_baseline = run_baseline_opf()
    m, zones, result_portfolio = run_der_expansion()
    p_sub_baseline = result_baseline.objective_value
    p_sub = result_portfolio.objective_value
    print(f"Substation power  baseline : {p_sub_baseline} p.u.")
    print(
        f"Substation power  portfolio: {p_sub:.4f} p.u.  ({100 * (p_sub_baseline - p_sub) / p_sub_baseline:.1f}% reduction)"
    )
    print()
    for z in zones:
        for r in m.resource_set:
            print(
                f"  {r}   Zone-{z}: p_max={pyo.value(m.p_max[z, r]):.4f}  dispatch={pyo.value(m.p_zr[z, r, t0]):.4f} p.u."
            )

    print("\nNode-level DER allocation:")
    print(
        "node  zone  pv_cap_alloc  bess_cap_alloc  pv_dispatch_alloc  bess_dispatch_alloc"
    )
    has_pv = "PV" in set(m.resource_set)
    has_bess = "BESS" in set(m.resource_set)
    # Iterate through zones and their buses directly
    for z in sorted(zones.keys()):
        for _id in sorted(zones[z]):
            pv_cap = pyo.value(m.alpha[_id, "PV"] * m.p_max[z, "PV"]) if has_pv else 0.0
            bess_cap = (
                pyo.value(m.alpha[_id, "BESS"] * m.p_max[z, "BESS"])
                if has_bess
                else 0.0
            )
            pv_dispatch = (
                pyo.value(m.alpha[_id, "PV"] * m.p_zr[z, "PV", t0]) if has_pv else 0.0
            )
            bess_dispatch = (
                pyo.value(m.alpha[_id, "BESS"] * m.p_zr[z, "BESS", t0])
                if has_bess
                else 0.0
            )
            print(
                f"{_id:4d}  {z:4d}  {pv_cap:12.6f}  {bess_cap:14.6f}  {pv_dispatch:17.6f}  {bess_dispatch:19.6f}"
            )

    # Extract and report voltage slack violations
    print("\nVoltage Slack Variables (Violations):")
    voltage_slack_violations = {}
    total_voltage_slack = 0
    if hasattr(m, "v2_slack"):
        for bus_id, ph in m.bus_phase_set:
            for t in m.time_set:
                slack_val = pyo.value(m.v2_slack[bus_id, ph, t])
                if slack_val > 1e-6:  # Only show non-zero slacks
                    voltage_slack_violations[(bus_id, ph, t)] = slack_val
                    total_voltage_slack += slack_val
    print(f"  Total slack (sum of violations): {total_voltage_slack:.6f}")
    print(f"  Number of violations: {len(voltage_slack_violations)}")
    if voltage_slack_violations:
        print("\n  Detailed Voltage Slack Values:")
        print("  Bus  Phase  Time  Slack Value")
        print("  " + "-" * 30)
        for (bus_id, ph, t), slack_val in sorted(voltage_slack_violations.items())[:20]:
            print(f"  {bus_id:3d}  {ph:5s}  {t:4d}  {slack_val:.8f}")
        if len(voltage_slack_violations) > 20:
            print(f"  ... and {len(voltage_slack_violations) - 20} more")

    # Extract and report thermal slack violations
    print("\nThermal Slack Variables (Violations):")
    thermal_slack_violations = {}
    total_thermal_slack = 0
    if hasattr(m, "thermal_slack"):
        for branch_id, ph in m.branch_phase_set:
            for t in m.time_set:
                slack_val = pyo.value(m.thermal_slack[branch_id, ph, t])
                if slack_val > 1e-6:  # Only show non-zero slacks
                    thermal_slack_violations[(branch_id, ph, t)] = slack_val
                    total_thermal_slack += slack_val
    print(f"  Total slack (sum of violations): {total_thermal_slack:.6f}")
    print(f"  Number of violations: {len(thermal_slack_violations)}")
    if thermal_slack_violations:
        print("\n  Detailed Thermal Slack Values:")
        print("  Branch  Phase  Time  Slack Value")
        print("  " + "-" * 32)
        for (branch_id, ph, t), slack_val in sorted(thermal_slack_violations.items())[
            :20
        ]:
            print(f"  {branch_id:6d}  {ph:5s}  {t:4d}  {slack_val:.8f}")
        if len(thermal_slack_violations) > 20:
            print(f"  ... and {len(thermal_slack_violations) - 20} more")

    dif_flow = (
        result_portfolio.p_flow.loc[:, ["a", "b", "c"]]
        - result_baseline.p_flow.loc[:, ["a", "b", "c"]]
    )
    print(dif_flow.max())
    vb = result_baseline.voltages
    vp = result_portfolio.voltages
    dif_v = vp.loc[:, ["a", "b", "c"]] - vb.loc[:, ["a", "b", "c"]]
    print(dif_v.max())
    print()
    case.plot_network().show(renderer="browser")
    # opf.compare_flows(result_baseline.p_flows, result_portfolio.p_flows).show(renderer="browser")
