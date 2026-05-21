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
import numpy as np
import pandas as pd
from pathlib import Path

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
)
from distopf.pyomo_models.capacity_expansion_constraints import (
    create_zones_from_edge_names,
    add_capacity_expansion_as_fraction_of_load,
    add_pv_parameters,
    add_bess_parameters,
    add_capacity_expansion_with_slack_constraints,
    add_capacity_expansion_from_absolute_capacity,
)
from distopf.pyomo_models.slack_constraints import (
    add_voltage_slack_constraints,
    add_thermal_slack_constraints,
    thermal_slack_penalty,
    voltage_slack_penalty,
)
from distopf.pyomo_models.results import PyoResult


def substation_power_at_step(model: LindistModelProtocol, t: int):
    total_power = 0
    for _id, ph in model.branch_phase_set:
        if model.from_bus_map[_id] in model.swing_bus_set:
            total_power += model.p_flow[_id, ph, t]
    return total_power


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
        + thermal_slack_penalty(_m, weight=1e6)
        + voltage_slack_penalty(_m, weight=1e0),
        sense=pyo.minimize,
    )
    res = opt.solve(m_base, tee=False)
    if res.solver.status != pyo.SolverStatus.ok:
        raise RuntimeError("Baseline solve failed")
    return m_base, PyoResult(m_base, pyo.value(substation_power_objective_rule(m_base)))


# ---------------------------------------------------------------------------
# Portfolio model
# ---------------------------------------------------------------------------
def run_der_expansion(case, absolute_capacity):
    zone_breaks = [
        "jn-a",  # orange
        "jn-c",  # blue
        "span_11126",  # pink
        "3003288581-a",  # green
    ]
    zones = create_zones_from_edge_names(
        case,
        zone_breaks,
        # [
        #     "jn-a",
        #     "jn-b",
        #     "jn-c",
        #     # "jn-d",  # very small area
        # ],
    )
    del zones[0]
    m = create_lindist_model(case)

    # add_capacity_expansion_as_fraction_of_load(
    #     m, case, fraction=1.0, relative_capacity={"PV": 0.5, "BESS": 0.5}
    # )
    add_capacity_expansion_from_absolute_capacity(
        m,
        # absolute_capacity={"PV": 0.5, "BESS": 0.5},
        absolute_capacity=absolute_capacity,
    )
    add_pv_parameters(m, curtailment_max=0.0, capacity_factor=1.0, case=case)
    add_bess_parameters(m, e_max=10, soc=0.5, discharge_derate=1, charge_derate=1)
    add_capacity_expansion_with_slack_constraints(m, case, zones)

    # --- objective and solve --------------------------------------------------
    m.objective = pyo.Objective(
        rule=lambda _m: substation_power_objective_rule(_m)
        + thermal_slack_penalty(_m, weight=1e6)
        + voltage_slack_penalty(_m, weight=1e2),
        sense=pyo.minimize,
    )

    opt = pyo.SolverFactory("highs")
    res = opt.solve(m, tee=False)
    if res.solver.status != pyo.SolverStatus.ok:
        raise RuntimeError(f"Portfolio solve failed: {res.solver.status}")

    # Results
    # ---------------------------------------------------------------------------
    p_sub = pyo.value(substation_power_objective_rule(m))
    result = PyoResult(m, pyo.value(substation_power_objective_rule(m)))
    return m, zones, result


# ---------------------------------------------------------------------------
# Results extraction and reporting functions
# ---------------------------------------------------------------------------
def extract_system_info(case):
    """Extract system description from case."""
    n_buses = len(case.bus_data)
    n_branches = len(case.branch_data)
    n_loads = (case.bus_data[["pl_a", "pl_b", "pl_c"]].sum(axis=1) > 0).sum()
    swing_bus = case.bus_data[case.bus_data.bus_type == "SWING"]["id"].values[0]
    n_steps = case.n_steps
    delta_t = case.delta_t

    return {
        "n_buses": n_buses,
        "n_branches": n_branches,
        "n_loads": n_loads,
        "swing_bus": swing_bus,
        "n_steps": n_steps,
        "delta_t": delta_t,
    }


def print_system_description(case):
    """Print system description."""
    info = extract_system_info(case)
    print("## System Description\n")
    print(f"- Number of buses: {info['n_buses']}")
    print(f"- Number of branches: {info['n_branches']}")
    print(f"- Number of loads: {info['n_loads']}")
    print(f"- Swing bus: Bus {info['swing_bus']}")
    print(f"- Time steps: {info['n_steps']}")
    print(f"- Time step duration: {info['delta_t']} hours\n")


def print_load_summary(case, s_base_mva):
    """Print load model summary."""
    print("## Load Model Summary\n")
    total_load_p = case.bus_data[["pl_a", "pl_b", "pl_c"]].sum().sum()
    total_load_q = case.bus_data[["ql_a", "ql_b", "ql_c"]].sum().sum()
    loads_by_phase = {
        "a": case.bus_data["pl_a"].sum(),
        "b": case.bus_data["pl_b"].sum(),
        "c": case.bus_data["pl_c"].sum(),
    }

    total_load_mw = total_load_p * s_base_mva
    total_load_q_mvar = total_load_q * s_base_mva

    print(f"- Total active load: {total_load_mw:.1f} MW ({total_load_p:.4f} p.u.)")
    print(
        f"- Total reactive load: {total_load_q_mvar:.1f} MVAR ({total_load_q:.4f} p.u.)"
    )
    print(
        f"- Active load by phase: a={loads_by_phase['a'] * s_base_mva:.1f} MW, b={loads_by_phase['b'] * s_base_mva:.1f} MW, c={loads_by_phase['c'] * s_base_mva:.1f} MW\n"
    )


def print_schedule_info(case):
    """Print schedule information."""
    print("## Schedule Information\n")
    if case.schedules is not None and not case.schedules.empty:
        print("| Time Step | PV Factor | Load Multiplier |")
        print("|---:|---:|---:|")
        for t in case.schedules.index:
            pv_fac = (
                case.schedules.loc[t, "PV"] if "PV" in case.schedules.columns else "N/A"
            )
            load_mult = (
                case.schedules.loc[t, "default"]
                if "default" in case.schedules.columns
                else "N/A"
            )
            print(f"| {t} | {pv_fac} | {load_mult} |")
        print()
    else:
        print("No schedule data available.\n")


def extract_case_results(m, result, zones, case, case_name="Case"):
    """Extract results from a portfolio model."""
    objective_value = result.objective_value
    p_sub0 = pyo.value(substation_power_at_step(m, 0))
    p_sub1 = pyo.value(substation_power_at_step(m, 1))
    allocation_data = []
    for z in sorted(zones.keys()):
        for r in m.resource_set:
            p_max = pyo.value(m.p_max[z, r])
            dispatch_by_t = [pyo.value(m.p_zr[z, r, t]) for t in sorted(m.time_set)]
            allocation_data.append(
                {
                    "zone": z,
                    "resource": r,
                    "p_max": p_max,
                    "dispatch_by_t": dispatch_by_t,
                }
            )

    voltage_violations = {}
    total_voltage_slack = 0.0
    if hasattr(m, "v2_slack"):
        for bus_id, ph in m.bus_phase_set:
            for t in m.time_set:
                slack_val = pyo.value(m.v2_slack[bus_id, ph, t])
                if slack_val > 1e-6:
                    voltage_violations[(bus_id, ph, t)] = slack_val
                    total_voltage_slack += slack_val

    thermal_violations = {}
    thermal_violations_pct = {}
    total_thermal_slack = 0.0
    if hasattr(m, "thermal_slack"):
        for branch_id, ph in m.branch_phase_set:
            for t in m.time_set:
                slack_val = pyo.value(m.thermal_slack[branch_id, ph, t])
                rating_val = pyo.value(m.s_branch_max[branch_id, ph])
                if slack_val > 1e-6:
                    # Slack value represents the violation magnitude
                    thermal_violations[(branch_id, ph, t)] = slack_val
                    thermal_violations_pct[(branch_id, ph, t)] = (
                        slack_val / rating_val * 100
                    )
                    total_thermal_slack += slack_val

    # Calculate quantiles of thermal violations
    thermal_quantiles = {}
    if thermal_violations_pct:
        violations_list = list(thermal_violations_pct.values())
        # Calculate quantiles at every 5%
        thermal_quantiles = {
            f"q{int(p * 100)}": float(np.quantile(violations_list, p))
            for p in np.arange(0.05, 1.0, 0.05)
        }
        thermal_quantiles["mean"] = float(np.mean(violations_list))
        thermal_quantiles["max"] = float(np.max(violations_list))

    return {
        "objective_value": objective_value,
        "p_sub0": p_sub0,
        "p_sub1": p_sub1,
        "allocation": allocation_data,
        "voltage_violations": voltage_violations,
        "total_voltage_slack": total_voltage_slack,
        "thermal_violations": thermal_violations,
        "thermal_violations_pct": thermal_violations_pct,
        "total_thermal_slack": total_thermal_slack,
        "thermal_quantiles": thermal_quantiles,
    }


def print_portfolio_results(
    case_name, abs_capacity, baseline_p_tuple, case_results, zones, s_base_mva
):
    """Print portfolio results for a case."""
    objective_value = case_results["objective_value"]
    p_sub0 = case_results["p_sub0"]
    p_sub1 = case_results["p_sub1"]
    allocation = case_results["allocation"]
    voltage_violations = case_results["voltage_violations"]
    total_voltage_slack = case_results["total_voltage_slack"]
    thermal_violations = case_results["thermal_violations"]
    thermal_violations_pct = case_results["thermal_violations_pct"]
    total_thermal_slack = case_results["total_thermal_slack"]
    thermal_quantiles = case_results["thermal_quantiles"]

    baseline_p0, baseline_p1 = baseline_p_tuple

    print(f"## {case_name}\n")
    print(f"**Capacity expansion inputs:**")
    pv_mw = abs_capacity.get("PV", 0) * s_base_mva
    bess_mw = abs_capacity.get("BESS", 0) * s_base_mva
    print(f"- PV capacity: {pv_mw:.2f} MW ({abs_capacity.get('PV', 0):.4f} p.u.)")
    print(
        f"- BESS capacity: {bess_mw:.2f} MW ({abs_capacity.get('BESS', 0):.4f} p.u.)\n"
    )

    print(f"**Substation power and objective:**")
    p_sub0_mw = p_sub0 * s_base_mva
    p_sub1_mw = p_sub1 * s_base_mva
    print(f"- Substation power at t=0: {p_sub0_mw:.2f} MW ({p_sub0:.4f} p.u.)")
    print(f"- Substation power at t=1: {p_sub1_mw:.2f} MW ({p_sub1:.4f} p.u.)")
    print(f"- Objective value (with penalties): {objective_value:.4f}")

    if baseline_p0 is not None and baseline_p1 is not None:
        baseline_p0_mw = baseline_p0 * s_base_mva
        baseline_p1_mw = baseline_p1 * s_base_mva
        reduction0 = (
            100 * (baseline_p0 - p_sub0) / baseline_p0 if baseline_p0 != 0 else 0
        )
        reduction1 = (
            100 * (baseline_p1 - p_sub1) / baseline_p1 if baseline_p1 != 0 else 0
        )
        print(
            f"- Reduction vs. baseline at t=0: {reduction0:.1f}% ({baseline_p0_mw:.2f} → {p_sub0_mw:.2f} MW)"
        )
        print(
            f"- Reduction vs. baseline at t=1: {reduction1:.1f}% ({baseline_p1_mw:.2f} → {p_sub1_mw:.2f} MW)\n"
        )
    else:
        print()

    print(f"**Allocation by zone and resource:**\n")
    print(
        "| Zone | Resource | Capacity (MW) | Dispatch at t=0 (MW) | Dispatch at t=1 (MW) |"
    )
    print("|---:|---|---:|---:|---:|")
    for entry in allocation:
        z = entry["zone"]
        r = entry["resource"]
        p_max = entry["p_max"]
        dispatch = entry["dispatch_by_t"]
        p_max_mw = p_max * s_base_mva
        dispatch_mw = [d * s_base_mva for d in dispatch]
        time_steps_str = " | ".join(f"{d:.2f}" for d in dispatch_mw)
        print(f"| {z} | {r} | {p_max_mw:.2f} | {time_steps_str} |")
    print()

    print(f"**Constraint violations:**\n")
    print(
        f"- Voltage slack total: {total_voltage_slack:.6f} ({len(voltage_violations)} violations)"
    )
    if voltage_violations and len(voltage_violations) > 0:
        print("  - Top violations (Bus, Phase, Time, Slack):")
        for (bus_id, ph, t), slack_val in sorted(
            voltage_violations.items(), key=lambda x: -x[1]
        )[:10]:
            print(f"    - Bus {bus_id}, Phase {ph}, Time {t}: {slack_val:.8f}")
    print()

    print(
        f"- Thermal slack total: {total_thermal_slack * s_base_mva:.6f} MW ({len(thermal_violations)} violations)"
    )
    if thermal_quantiles:
        print("  - Thermal violation quantiles (slack in p.u.):")
        print(f"    - Mean: {thermal_quantiles['mean']:.8f}")
        print(f"    - 25th percentile: {thermal_quantiles['q25']:.8f}")
        print(f"    - Median (50th): {thermal_quantiles['q50']:.8f}")
        print(f"    - 75th percentile: {thermal_quantiles['q75']:.8f}")
        print(f"    - 90th percentile: {thermal_quantiles['q90']:.8f}")
        print(f"    - 95th percentile: {thermal_quantiles['q95']:.8f}")
        print(f"    - Max: {thermal_quantiles['max']:.8f}")
    if thermal_violations and len(thermal_violations) > 0:
        print("  - Top violations (Branch, Phase, Time, Slack (% of rating), MW slack):")
        for (branch_id, ph, t), slack_val in sorted(
            thermal_violations_pct.items(), key=lambda x: -x[1]
        )[:10]:
            pu = thermal_violations.get((branch_id, ph, t), 0)
            print(
                f"    - Branch {branch_id}, Phase {ph}, Time {t}: {slack_val:.2f}% ({pu * s_base_mva:.8f} MW)"
            )
    print()


# =========================================================================
# Results saving functions
# =========================================================================
def create_results_directory():
    """Create the results directory if it doesn't exist."""
    results_dir = Path("scratch/foresight_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_schedule_table(case, results_dir):
    """Save schedule information to CSV."""
    if case.schedules is not None and not case.schedules.empty:
        df = case.schedules.reset_index().rename(columns={"index": "TimeStep"})
        df.to_csv(results_dir / "schedule.csv", index=False)


def save_allocation_table(case_results, case_name, s_base_mva, results_dir):
    """Save allocation table to CSV."""
    allocation_data = []
    for entry in case_results["allocation"]:
        z = entry["zone"]
        r = entry["resource"]
        p_max = entry["p_max"]
        dispatch = entry["dispatch_by_t"]
        p_max_mw = p_max * s_base_mva
        dispatch_mw = [d * s_base_mva for d in dispatch]
        for t_idx, d_mw in enumerate(dispatch_mw):
            allocation_data.append(
                {
                    "Zone": z,
                    "Resource": r,
                    "Capacity_MW": p_max_mw,
                    "TimeStep": t_idx,
                    "Dispatch_MW": d_mw,
                }
            )
    if allocation_data:
        df = pd.DataFrame(allocation_data)
        df.to_csv(
            results_dir / f"allocation_{case_name.lower().replace(' ', '_')}.csv",
            index=False,
        )


def save_thermal_violations_table(case_results, case_name, s_base_mva, results_dir):
    """Save thermal violations to CSV."""
    violations_data = []
    for (branch_id, ph, t), pct in case_results["thermal_violations_pct"].items():
        slack_val = case_results["thermal_violations"][(branch_id, ph, t)]
        violations_data.append(
            {
                "Branch": branch_id,
                "Phase": ph,
                "TimeStep": t,
                "Slack_PU": slack_val,
                "Violation_Percent": pct,
                "Violation_MW": slack_val * s_base_mva,
            }
        )
    if violations_data:
        df = pd.DataFrame(violations_data).sort_values(
            "Violation_Percent", ascending=False
        )
        df.to_csv(
            results_dir
            / f"thermal_violations_{case_name.lower().replace(' ', '_')}.csv",
            index=False,
        )


def save_thermal_quantiles_table(case_results, case_name, results_dir):
    """Save thermal quantiles to CSV."""
    if case_results["thermal_quantiles"]:
        quantiles = case_results["thermal_quantiles"]
        # Build list of metrics and values
        metrics = []
        values = []

        # Add percentiles in order (5%, 10%, ..., 95%)
        for p in np.arange(0.05, 1.0, 0.05):
            key = f"q{int(p * 100)}"
            metrics.append(f"{int(p * 100)}th Percentile")
            values.append(quantiles[key])

        # Add mean and max
        metrics.extend(["Mean", "Max"])
        values.extend([quantiles["mean"], quantiles["max"]])

        df = pd.DataFrame(
            {
                "Metric": metrics,
                "Violation_Percent": values,
            }
        )
        df.to_csv(
            results_dir
            / f"thermal_quantiles_{case_name.lower().replace(' ', '_')}.csv",
            index=False,
        )


def save_summary_comparison_tables(
    baseline_p0,
    baseline_p1,
    objective_value_baseline,
    case1_results,
    case2_results,
    s_base_mva,
    results_dir,
):
    """Save summary comparison tables to CSV."""
    # Substation power comparison
    substation_data = [
        {
            "Scenario": "Baseline",
            "Substation_P_t0_MW": baseline_p0 * s_base_mva,
            "Substation_P_t1_MW": baseline_p1 * s_base_mva,
            "Reduction_t0_Percent": None,
            "Reduction_t1_Percent": None,
        },
        {
            "Scenario": "Case 1",
            "Substation_P_t0_MW": case1_results["p_sub0"] * s_base_mva,
            "Substation_P_t1_MW": case1_results["p_sub1"] * s_base_mva,
            "Reduction_t0_Percent": 100
            * (baseline_p0 - case1_results["p_sub0"])
            / baseline_p0
            if baseline_p0 != 0
            else 0,
            "Reduction_t1_Percent": 100
            * (baseline_p1 - case1_results["p_sub1"])
            / baseline_p1
            if baseline_p1 != 0
            else 0,
        },
        {
            "Scenario": "Case 2",
            "Substation_P_t0_MW": case2_results["p_sub0"] * s_base_mva,
            "Substation_P_t1_MW": case2_results["p_sub1"] * s_base_mva,
            "Reduction_t0_Percent": 100
            * (baseline_p0 - case2_results["p_sub0"])
            / baseline_p0
            if baseline_p0 != 0
            else 0,
            "Reduction_t1_Percent": 100
            * (baseline_p1 - case2_results["p_sub1"])
            / baseline_p1
            if baseline_p1 != 0
            else 0,
        },
    ]
    df_substation = pd.DataFrame(substation_data)
    df_substation.to_csv(results_dir / "summary_substation_power.csv", index=False)

    # Objective value comparison
    objective_data = [
        {
            "Scenario": "Baseline",
            "Objective_Value_MW": objective_value_baseline * s_base_mva,
            "Reduction_Percent": None,
        },
        {
            "Scenario": "Case 1",
            "Objective_Value_MW": case1_results["objective_value"] * s_base_mva,
            "Reduction_Percent": 100
            * (objective_value_baseline - case1_results["objective_value"])
            / objective_value_baseline
            if objective_value_baseline != 0
            else 0,
        },
        {
            "Scenario": "Case 2",
            "Objective_Value_MW": case2_results["objective_value"] * s_base_mva,
            "Reduction_Percent": 100
            * (objective_value_baseline - case2_results["objective_value"])
            / objective_value_baseline
            if objective_value_baseline != 0
            else 0,
        },
    ]
    df_objective = pd.DataFrame(objective_data)
    df_objective.to_csv(results_dir / "summary_objective_value.csv", index=False)


def save_markdown_report(
    case,
    baseline_results,
    p_sub_baseline0,
    p_sub_baseline1,
    objective_value_baseline,
    case1_capacity,
    case1_results,
    case2_capacity,
    case2_results,
    s_base_mva,
    results_dir,
):
    """Generate and save a comprehensive markdown report."""
    report = []

    # Header
    report.append("# DER Portfolio Capacity Expansion Analysis\n")
    report.append("Johnson Creek Feeder - Multi-Period OPF Results\n")

    # System Description
    info = extract_system_info(case)
    report.append("\n## System Description\n")
    report.append(f"- Number of buses: {info['n_buses']}\n")
    report.append(f"- Number of branches: {info['n_branches']}\n")
    report.append(f"- Number of loads: {info['n_loads']}\n")
    report.append(f"- Swing bus: Bus {info['swing_bus']}\n")
    report.append(f"- Time steps: {info['n_steps']}\n")
    report.append(f"- Time step duration: {info['delta_t']} hours\n")

    # Load Summary
    total_load_p = case.bus_data[["pl_a", "pl_b", "pl_c"]].sum().sum()
    total_load_q = case.bus_data[["ql_a", "ql_b", "ql_c"]].sum().sum()
    loads_by_phase = {
        "a": case.bus_data["pl_a"].sum(),
        "b": case.bus_data["pl_b"].sum(),
        "c": case.bus_data["pl_c"].sum(),
    }
    total_load_mw = total_load_p * s_base_mva
    total_load_q_mvar = total_load_q * s_base_mva

    report.append("\n## Load Model Summary\n")
    report.append(
        f"- Total active load: {total_load_mw:.1f} MW ({total_load_p:.4f} p.u.)\n"
    )
    report.append(
        f"- Total reactive load: {total_load_q_mvar:.1f} MVAR ({total_load_q:.4f} p.u.)\n"
    )
    report.append(
        f"- Active load by phase: a={loads_by_phase['a'] * s_base_mva:.1f} MW, b={loads_by_phase['b'] * s_base_mva:.1f} MW, c={loads_by_phase['c'] * s_base_mva:.1f} MW\n"
    )

    # Schedule
    report.append("\n## Schedule Information\n")
    if case.schedules is not None and not case.schedules.empty:
        report.append("| Time Step | PV Factor | Load Multiplier |\n")
        report.append("|---:|---:|---:|\n")
        for t in case.schedules.index:
            pv_fac = (
                case.schedules.loc[t, "PV"] if "PV" in case.schedules.columns else "N/A"
            )
            load_mult = (
                case.schedules.loc[t, "default"]
                if "default" in case.schedules.columns
                else "N/A"
            )
            report.append(f"| {t} | {pv_fac} | {load_mult} |\n")

    # Baseline Results
    report.append("\n## Baseline OPF Results\n")
    p_sub_baseline0_mw = p_sub_baseline0 * s_base_mva
    p_sub_baseline1_mw = p_sub_baseline1 * s_base_mva
    objective_value_baseline_mw = objective_value_baseline * s_base_mva
    report.append(
        f"- Substation power at t=0: {p_sub_baseline0_mw:.2f} MW ({p_sub_baseline0:.4f} p.u.)\n"
    )
    report.append(
        f"- Substation power at t=1: {p_sub_baseline1_mw:.2f} MW ({p_sub_baseline1:.4f} p.u.)\n"
    )
    report.append(
        f"- Objective value (with penalties): {objective_value_baseline_mw:.2f} MW ({objective_value_baseline:.4f} p.u.)\n"
    )

    # Case 1 Results
    pv_mw_c1 = case1_capacity.get("PV", 0) * s_base_mva
    bess_mw_c1 = case1_capacity.get("BESS", 0) * s_base_mva
    p_sub0_mw_c1 = case1_results["p_sub0"] * s_base_mva
    p_sub1_mw_c1 = case1_results["p_sub1"] * s_base_mva

    report.append("\n## Case 1: PV + BESS Expansion\n")
    report.append(f"**Capacity expansion:**\n")
    report.append(
        f"- PV capacity: {pv_mw_c1:.2f} MW ({case1_capacity.get('PV', 0):.4f} p.u.)\n"
    )
    report.append(
        f"- BESS capacity: {bess_mw_c1:.2f} MW ({case1_capacity.get('BESS', 0):.4f} p.u.)\n"
    )
    report.append(
        f"\n**Results:**\n- Substation power at t=0: {p_sub0_mw_c1:.2f} MW ({case1_results['p_sub0']:.4f} p.u.)\n"
    )
    report.append(
        f"- Substation power at t=1: {p_sub1_mw_c1:.2f} MW ({case1_results['p_sub1']:.4f} p.u.)\n"
    )
    report.append(
        f"- Objective value (with penalties): {case1_results['objective_value']:.4f}\n"
    )
    reduction0_c1 = (
        100 * (p_sub_baseline0 - case1_results["p_sub0"]) / p_sub_baseline0
        if p_sub_baseline0 != 0
        else 0
    )
    reduction1_c1 = (
        100 * (p_sub_baseline1 - case1_results["p_sub1"]) / p_sub_baseline1
        if p_sub_baseline1 != 0
        else 0
    )
    report.append(f"- Reduction vs. baseline at t=0: {reduction0_c1:.1f}%\n")
    report.append(f"- Reduction vs. baseline at t=1: {reduction1_c1:.1f}%\n")

    # Case 2 Results
    pv_mw_c2 = case2_capacity.get("PV", 0) * s_base_mva
    bess_mw_c2 = case2_capacity.get("BESS", 0) * s_base_mva
    p_sub0_mw_c2 = case2_results["p_sub0"] * s_base_mva
    p_sub1_mw_c2 = case2_results["p_sub1"] * s_base_mva
    obj_c1 = case1_results["objective_value"]
    obj_c2 = case2_results["objective_value"]

    report.append("\n## Case 2: Increased PV + BESS Expansion\n")
    report.append(f"**Capacity expansion:**\n")
    report.append(
        f"- PV capacity: {pv_mw_c2:.2f} MW ({case2_capacity.get('PV', 0):.4f} p.u.)\n"
    )
    report.append(
        f"- BESS capacity: {bess_mw_c2:.2f} MW ({case2_capacity.get('BESS', 0):.4f} p.u.)\n"
    )
    report.append(
        f"\n**Results:**\n- Substation power at t=0: {p_sub0_mw_c2:.2f} MW ({case2_results['p_sub0']:.4f} p.u.)\n"
    )
    report.append(
        f"- Substation power at t=1: {p_sub1_mw_c2:.2f} MW ({case2_results['p_sub1']:.4f} p.u.)\n"
    )
    report.append(
        f"- Objective value (with penalties): {case2_results['objective_value']:.4f}\n"
    )
    reduction0_c2 = (
        100 * (p_sub_baseline0 - case2_results["p_sub0"]) / p_sub_baseline0
        if p_sub_baseline0 != 0
        else 0
    )
    reduction1_c2 = (
        100 * (p_sub_baseline1 - case2_results["p_sub1"]) / p_sub_baseline1
        if p_sub_baseline1 != 0
        else 0
    )
    report.append(f"- Reduction vs. baseline at t=0: {reduction0_c2:.1f}%\n")
    report.append(f"- Reduction vs. baseline at t=1: {reduction1_c2:.1f}%\n")

    # Summary Comparison
    report.append("\n## Summary Comparison\n")
    report.append("\n### Substation Power Comparison\n")
    report.append(
        "| Scenario | Substation P at t=0 (MW) | Substation P at t=1 (MW) | Reduction @ t=0 | Reduction @ t=1 |\n"
    )
    report.append("|---|---:|---:|---:|---:|\n")
    report.append(
        f"| Baseline | {p_sub_baseline0_mw:.2f} | {p_sub_baseline1_mw:.2f} | - | - |\n"
    )
    report.append(
        f"| Case 1 | {p_sub0_mw_c1:.2f} | {p_sub1_mw_c1:.2f} | {reduction0_c1:.1f}% | {reduction1_c1:.1f}% |\n"
    )
    report.append(
        f"| Case 2 | {p_sub0_mw_c2:.2f} | {p_sub1_mw_c2:.2f} | {reduction0_c2:.1f}% | {reduction1_c2:.1f}% |\n"
    )

    report.append("\n### Objective Value Comparison\n")
    report.append("| Scenario | Objective Value (MW) | Reduction |\n")
    report.append("|---|---:|---:|\n")
    report.append(f"| Baseline | {objective_value_baseline_mw:.2f} | - |\n")
    obj_red_c1 = (
        100
        * (objective_value_baseline - case1_results["objective_value"])
        / objective_value_baseline
        if objective_value_baseline != 0
        else 0
    )
    report.append(f"| Case 1 | {obj_c1:.2f} | {obj_red_c1:.1f}% |\n")
    obj_red_c2 = (
        100
        * (objective_value_baseline - case2_results["objective_value"])
        / objective_value_baseline
        if objective_value_baseline != 0
        else 0
    )
    report.append(f"| Case 2 | {obj_c2:.2f} | {obj_red_c2:.1f}% |\n")

    report.append(
        "\n\n## Output Files\n\nAll results have been saved as CSV files:\n\n"
    )
    report.append(
        "- `schedule.csv` - Schedule information (PV factor, load multiplier)\n"
    )
    report.append("- `allocation_case_1.csv` - Case 1 resource allocation by zone\n")
    report.append("- `allocation_case_2.csv` - Case 2 resource allocation by zone\n")
    report.append(
        "- `thermal_violations_case_1.csv` - Case 1 thermal constraint violations\n"
    )
    report.append(
        "- `thermal_violations_case_2.csv` - Case 2 thermal constraint violations\n"
    )
    report.append(
        "- `thermal_quantiles_case_1.csv` - Case 1 thermal violation statistical summary\n"
    )
    report.append(
        "- `thermal_quantiles_case_2.csv` - Case 2 thermal violation statistical summary\n"
    )
    report.append(
        "- `summary_substation_power.csv` - Substation power comparison across all scenarios\n"
    )
    report.append(
        "- `summary_objective_value.csv` - Objective value comparison across all scenarios\n"
    )

    # Write markdown file
    report_path = results_dir / "report.md"
    with open(report_path, "w") as f:
        f.writelines(report)

    print(f"\nReport saved to: {report_path}")
    print(f"Results directory: {results_dir.resolve()}")


if __name__ == "__main__":
    case = create_case(
        data_path="scratch/JN_creek",
        ignore_schedule=False,
        ignore_bat=True,
        ignore_gen=True,
        n_steps=2,
        start_step=0,
    )
    case.modify(v_min=0.6, v_max=1.12)
    case.branch_data.loc[(case.branch_data.sa_max <= 0.0005) & (case.branch_data.sa_max >= 0.00049), "sa_max"] = 0.015
    case.branch_data.loc[(case.branch_data.sb_max <= 0.0005) & (case.branch_data.sb_max >= 0.00049), "sb_max"] = 0.015
    case.branch_data.loc[(case.branch_data.sc_max <= 0.0005) & (case.branch_data.sc_max >= 0.00049), "sc_max"] = 0.015
    s_base = case.branch_data.s_base.iloc[0]
    # Print system description
    s_base_mva = s_base / 1e6  # Convert kVA to MVA (72,000 MVA)
    print_system_description(case)
    print_load_summary(case, s_base_mva)
    print_schedule_info(case)

    # Run baseline OPF
    print("## Base")
    m_base, result_baseline = run_baseline_opf()
    objective_value_baseline = result_baseline.objective_value
    p_sub_baseline0 = pyo.value(substation_power_at_step(result_baseline._model, 0))
    p_sub_baseline1 = pyo.value(substation_power_at_step(result_baseline._model, 1))
    p_sub_baseline0_mw = p_sub_baseline0 * s_base_mva
    p_sub_baseline1_mw = p_sub_baseline1 * s_base_mva
    print(
        f"Baseline substation power at t=0: {p_sub_baseline0_mw:.2f} MW ({p_sub_baseline0:.4f} p.u.)"
    )
    print(
        f"Baseline substation power at t=1: {p_sub_baseline1_mw:.2f} MW ({p_sub_baseline1:.4f} p.u.)"
    )
    print(f"Baseline objective value: {objective_value_baseline:.2f}\n")
    base_results = extract_case_results(m_base, result_baseline, {}, case)
    print_portfolio_results(
        "Base",
        {},
        (p_sub_baseline0, p_sub_baseline1),
        base_results,
        None,
        s_base_mva,
    )
    # Run Case 1
    case1_capacity = {"PV": 0.020275, "BESS": 0.020275}
    print(
        f"## Running Case 1 (PV={case1_capacity['PV'] * s_base_mva:.2f} MW, BESS={case1_capacity['BESS'] * s_base_mva:.2f} MW)"
    )
    m, zones, result_portfolio = run_der_expansion(
        case, absolute_capacity=case1_capacity
    )
    case1_results = extract_case_results(m, result_portfolio, zones, case)
    print_portfolio_results(
        "Case 1",
        case1_capacity,
        (p_sub_baseline0, p_sub_baseline1),
        case1_results,
        zones,
        s_base_mva,
    )

    case2_capacity = {"PV": 0.04055, "BESS": 0.04055}
    # Run Case 2
    print(
        f"## Running Case 2 (PV={case2_capacity['PV'] * s_base_mva:.2f} MW, BESS={case2_capacity['BESS'] * s_base_mva:.2f} MW)"
    )
    m3, zones3, result_portfolio3 = run_der_expansion(
        case, absolute_capacity=case2_capacity
    )
    case2_results = extract_case_results(m3, result_portfolio3, zones3, case)
    print_portfolio_results(
        "Case 2",
        case2_capacity,
        (p_sub_baseline0, p_sub_baseline1),
        case2_results,
        zones3,
        s_base_mva,
    )

    # Summary comparison
    print("## Summary Comparison\n")
    print(
        f"| Scenario | Substation P at t=0 (MW) | Substation P at t=1 (MW) | Reduction @ t=0 | Reduction @ t=1 |"
    )
    print(f"|---|---:|---:|---:|---:|")
    print(f"| Baseline | {p_sub_baseline0_mw:.2f} | {p_sub_baseline1_mw:.2f} | - | - |")

    case1_p_mw0 = case1_results["p_sub0"] * s_base_mva
    case1_p_mw1 = case1_results["p_sub1"] * s_base_mva
    case1_red0 = (
        100 * (p_sub_baseline0 - case1_results["p_sub0"]) / p_sub_baseline0
        if p_sub_baseline0 != 0
        else 0
    )
    case1_red1 = (
        100 * (p_sub_baseline1 - case1_results["p_sub1"]) / p_sub_baseline1
        if p_sub_baseline1 != 0
        else 0
    )
    print(
        f"| Case 1 | {case1_p_mw0:.2f} | {case1_p_mw1:.2f} | {case1_red0:.1f}% | {case1_red1:.1f}% |"
    )

    case2_p_mw0 = case2_results["p_sub0"] * s_base_mva
    case2_p_mw1 = case2_results["p_sub1"] * s_base_mva
    case2_red0 = (
        100 * (p_sub_baseline0 - case2_results["p_sub0"]) / p_sub_baseline0
        if p_sub_baseline0 != 0
        else 0
    )
    case2_red1 = (
        100 * (p_sub_baseline1 - case2_results["p_sub1"]) / p_sub_baseline1
        if p_sub_baseline1 != 0
        else 0
    )
    print(
        f"| Case 2 | {case2_p_mw0:.2f} | {case2_p_mw1:.2f} | {case2_red0:.1f}% | {case2_red1:.1f}% |"
    )

    print()
    print(f"| Scenario | Objective Value (MW) | Reduction |")
    print(f"|---|---:|---:|")
    print(f"| Baseline | {objective_value_baseline * s_base_mva:.2f} | - |")
    case1_obj_mw = case1_results["objective_value"] * s_base_mva
    case1_obj_red = (
        100
        * (objective_value_baseline - case1_results["objective_value"])
        / objective_value_baseline
        if objective_value_baseline != 0
        else 0
    )
    print(f"| Case 1 | {case1_obj_mw:.2f} | {case1_obj_red:.1f}% |")

    case2_obj_mw = case2_results["objective_value"] * s_base_mva
    case2_obj_red = (
        100
        * (objective_value_baseline - case2_results["objective_value"])
        / objective_value_baseline
        if objective_value_baseline != 0
        else 0
    )
    print(f"| Case 2 | {case2_obj_mw:.2f} | {case2_obj_red:.1f}% |")

    # ========================================================================
    # Save all results to CSV and markdown
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80 + "\n")

    results_dir = create_results_directory()

    save_schedule_table(case, results_dir)
    save_allocation_table(base_results, "Baseline", s_base_mva, results_dir)
    save_allocation_table(case1_results, "Case 1", s_base_mva, results_dir)
    save_allocation_table(case2_results, "Case 2", s_base_mva, results_dir)
    save_thermal_violations_table(base_results, "Baseline", s_base_mva, results_dir)
    save_thermal_violations_table(case1_results, "Case 1", s_base_mva, results_dir)
    save_thermal_violations_table(case2_results, "Case 2", s_base_mva, results_dir)
    save_thermal_quantiles_table(base_results, "Baseline", results_dir)
    save_thermal_quantiles_table(case1_results, "Case 1", results_dir)
    save_thermal_quantiles_table(case2_results, "Case 2", results_dir)
    print("  results saved")

    # Save summary comparison tables
    save_summary_comparison_tables(
        p_sub_baseline0,
        p_sub_baseline1,
        objective_value_baseline,
        case1_results,
        case2_results,
        s_base_mva,
        results_dir,
    )
