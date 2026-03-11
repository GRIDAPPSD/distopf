"""
Example script to test penalized objective functions.

This script creates a simple test case and compares solutions with:
1. Hard constraints only
2. Soft constraints (penalties) only
3. Both hard and soft constraints

Requires: pyomo, numpy, pandas, and a solver (ipopt for nonlinear, glpk/cbc for linear)
"""

import distopf as opf
import pyomo.environ as pyo
import pandas as pd
import numpy as np

# Import from distopf (adjust path as needed)
from distopf.api import Case
from distopf.pyomo_models.lindist import create_lindist_model
from distopf.pyomo_models import constraints
from distopf.pyomo_models import objectives


def create_test_case() -> Case:
    """
    Create a simple 4-bus test case.

    Topology:
        Bus 1 (Swing) --> Bus 2 --> Bus 3 --> Bus 4

    Bus 1: Swing bus (voltage source)
    Bus 2: Load bus
    Bus 3: Load bus with generator
    Bus 4: Load bus with capacitor
    """
    bus_data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["swing", "load1", "gen_bus", "cap_bus"],
            "phases": ["abc", "abc", "abc", "abc"],
            "bus_type": ["SWING", "PQ", "PQ", "PQ"],
            "v_a": [1.0, 1.0, 1.0, 1.0],
            "v_b": [1.0, 1.0, 1.0, 1.0],
            "v_c": [1.0, 1.0, 1.0, 1.0],
            "v_min": [0.95, 0.95, 0.95, 0.95],
            "v_max": [1.05, 1.05, 1.05, 1.05],
            "pl_a": [0.0, 0.5, 0.3, 0.2],  # MW
            "pl_b": [0.0, 0.5, 0.3, 0.2],
            "pl_c": [0.0, 0.5, 0.3, 0.2],
            "ql_a": [0.0, 0.2, 0.1, 0.1],  # MVAr
            "ql_b": [0.0, 0.2, 0.1, 0.1],
            "ql_c": [0.0, 0.2, 0.1, 0.1],
            "cvr_p": [0.0, 0.0, 0.0, 0.0],
            "cvr_q": [0.0, 0.0, 0.0, 0.0],
            "load_shape": ["default", "default", "default", "default"],
        }
    )

    branch_data = pd.DataFrame(
        {
            "fb": [1, 2, 3],
            "tb": [2, 3, 4],
            "phases": ["abc", "abc", "abc"],
            "status": [None, None, None],
            "raa": [0.01, 0.01, 0.01],
            "xaa": [0.02, 0.02, 0.02],
            "rab": [0.001, 0.001, 0.001],
            "xab": [0.002, 0.002, 0.002],
            "rac": [0.001, 0.001, 0.001],
            "xac": [0.002, 0.002, 0.002],
            "rbb": [0.01, 0.01, 0.01],
            "xbb": [0.02, 0.02, 0.02],
            "rbc": [0.001, 0.001, 0.001],
            "xbc": [0.002, 0.002, 0.002],
            "rcc": [0.01, 0.01, 0.01],
            "xcc": [0.02, 0.02, 0.02],
            "sa_max": [1.0, 1.0, 1.0],  # MVA thermal limits
            "sb_max": [1.0, 1.0, 1.0],
            "sc_max": [1.0, 1.0, 1.0],
        }
    )

    gen_data = pd.DataFrame(
        {
            "id": [3],
            "name": ["gen1"],
            "phases": ["abc"],
            "pa": [0.4],  # MW available
            "pb": [0.4],
            "pc": [0.4],
            "qa": [0.0],
            "qb": [0.0],
            "qc": [0.0],
            "sa_max": [0.5],  # MVA rating
            "sb_max": [0.5],
            "sc_max": [0.5],
            "qa_max": [0.3],
            "qb_max": [0.3],
            "qc_max": [0.3],
            "qa_min": [-0.3],
            "qb_min": [-0.3],
            "qc_min": [-0.3],
            "control_variable": ["PQ"],
        }
    )

    cap_data = pd.DataFrame(
        {
            "id": [4],
            "name": ["cap1"],
            "phases": ["abc"],
            "qa": [0.1],  # MVAr at 1.0 p.u.
            "qb": [0.1],
            "qc": [0.1],
        }
    )

    reg_data = pd.DataFrame(
        {
            "fb": pd.Series([], dtype=int),
            "tb": pd.Series([], dtype=int),
            "phases": pd.Series([], dtype=str),
        }
    )

    bat_data = pd.DataFrame(
        {
            "id": pd.Series([], dtype=int),
            "phases": pd.Series([], dtype=str),
        }
    )

    schedules = pd.DataFrame(
        {
            "time": [1],
            "default": [1.0],
        },
        index=[0],
    )

    case = Case(
        bus_data=bus_data,
        branch_data=branch_data,
        gen_data=gen_data,
        cap_data=cap_data,
        reg_data=reg_data,
        bat_data=bat_data,
        schedules=schedules,
        start_step=0,
        n_steps=1,
        delta_t=1.0,
    )

    return opf.create_case(data_path=opf.CASES_DIR / "csv" / "ieee13", n_steps=1)


def build_model_with_hard_constraints(case: Case) -> pyo.ConcreteModel:
    """Build model with standard hard constraints."""
    model = create_lindist_model(
        case, control_capacitors=False, control_regulators=False
    )

    # Power flow
    constraints.add_p_flow_constraints(model)
    constraints.add_q_flow_constraints(model)

    # Voltage
    constraints.add_voltage_limits(model)
    constraints.add_voltage_drop_constraints(model)
    constraints.add_swing_bus_constraints(model)

    # Loads and devices
    constraints.add_cvr_load_constraints(model)
    constraints.add_capacitor_constraints(model)

    # Generators
    constraints.add_generator_limits(model)
    constraints.add_generator_constant_p_constraints_q_control(model)
    constraints.add_generator_constant_q_constraints_p_control(model)
    constraints.add_octagonal_inverter_constraints_pq_control(model)

    # Thermal limits
    constraints.add_octagonal_thermal_constraints(model)

    return model


def build_model_without_hard_limits(case: Case) -> pyo.ConcreteModel:
    """Build model without voltage/thermal hard constraints (for penalty testing)."""
    model = create_lindist_model(
        case, control_capacitors=False, control_regulators=False
    )

    # Power flow
    constraints.add_p_flow_constraints(model)
    constraints.add_q_flow_constraints(model)

    # Voltage (no limits, just drop equations)
    constraints.add_voltage_drop_constraints(model)
    constraints.add_swing_bus_constraints(model)

    # Loads and devices
    constraints.add_cvr_load_constraints(model)
    constraints.add_capacitor_constraints(model)

    # Generators (no octagonal constraints - will use penalty)
    constraints.add_generator_constant_p_constraints_q_control(model)
    constraints.add_generator_constant_q_constraints_p_control(model)

    # No thermal constraints - will use penalty

    return model


def solve_model(model: pyo.ConcreteModel, solver_name: str = "ipopt") -> bool:
    """
    Solve the model and return success status.

    Parameters
    ----------
    model : pyo.ConcreteModel
        Pyomo model to solve
    solver_name : str
        Solver to use ('ipopt' for nonlinear, 'glpk' or 'cbc' for linear)

    Returns
    -------
    bool
        True if optimal solution found
    """
    solver = pyo.SolverFactory(solver_name)
    if solver is None:
        print(f"Solver '{solver_name}' not available")
        return False

    try:
        result = solver.solve(model, tee=False)
        return result.solver.termination_condition == pyo.TerminationCondition.optimal
    except Exception as e:
        print(f"Solver error: {e}")
        return False


def extract_results(model: pyo.ConcreteModel) -> dict:
    """Extract key results from solved model."""
    results = {
        "voltages": {},
        "power_flows": {},
        "generator_output": {},
        "objective_value": None,
    }

    # Objective value
    if hasattr(model, "objective"):
        results["objective_value"] = pyo.value(model.objective)

    # Voltages
    for _id, ph in model.bus_phase_set:
        for t in model.time_set:
            v2 = pyo.value(model.v2[_id, ph, t])
            v = v2**0.5 if v2 is not None else None
            results["voltages"][(_id, ph, t)] = v

    # Power flows
    for _id, ph in model.branch_phase_set:
        for t in model.time_set:
            p = pyo.value(model.p_flow[_id, ph, t])
            q = pyo.value(model.q_flow[_id, ph, t])
            results["power_flows"][(_id, ph, t)] = {"p": p, "q": q}

    # Generator output
    for _id, ph in model.gen_phase_set:
        for t in model.time_set:
            p = pyo.value(model.p_gen[_id, ph, t])
            q = pyo.value(model.q_gen[_id, ph, t])
            results["generator_output"][(_id, ph, t)] = {"p": p, "q": q}

    return results


def print_results(results: dict, label: str) -> None:
    """Print results in a formatted way."""
    print(f"\n{'=' * 60}")
    print(f"Results: {label}")
    print("=" * 60)

    print(f"\nObjective value: {results['objective_value']:.6f}")

    print("\nVoltages (p.u.):")
    print("-" * 40)
    for (_id, ph, t), v in sorted(results["voltages"].items()):
        if v is not None:
            status = ""
            if v < 0.95:
                status = " [UNDERVOLTAGE]"
            elif v > 1.05:
                status = " [OVERVOLTAGE]"
            print(f"  Bus {_id}, phase {ph}, t={t}: {v:.4f}{status}")

    print("\nGenerator Output:")
    print("-" * 40)
    for (_id, ph, t), pq in sorted(results["generator_output"].items()):
        p, q = pq["p"], pq["q"]
        if p is not None and q is not None:
            s = (p**2 + q**2) ** 0.5
            print(f"  Gen {_id}, phase {ph}, t={t}: P={p:.4f}, Q={q:.4f}, |S|={s:.4f}")

    # print("\nBranch Power Flows:")
    # print("-" * 40)
    # for (_id, ph, t), pq in sorted(results["power_flows"].items()):
    #     p, q = pq["p"], pq["q"]
    #     if p is not None and q is not None:
    #         s = (p**2 + q**2) ** 0.5
    #         print(
    #             f"  Branch to {_id}, phase {ph}, t={t}: P={p:.4f}, Q={q:.4f}, |S|={s:.4f}"
    #         )


def test_hard_constraints_only():
    """Test with hard constraints and simple loss objective."""
    print("\n" + "=" * 70)
    print("TEST 1: Hard Constraints Only (Loss Minimization)")
    print("=" * 70)

    # case = opf.create_case(data_path=opf.CASES_DIR / "csv" / "ieee123_30der", n_steps=24)
    case = create_test_case()
    model = build_model_with_hard_constraints(case)

    # Add simple loss objective
    objectives.add_loss_objective(model)

    success = solve_model(model, "ipopt")
    if success:
        results = extract_results(model)
        print_results(results, "Hard Constraints + Loss Objective")
    else:
        print("Failed to solve model with hard constraints")

    return success


def test_soft_constraints_only():
    """Test with soft constraints (penalties) instead of hard limits."""
    print("\n" + "=" * 70)
    print("TEST 2: Soft Constraints Only (Penalized Loss Minimization)")
    print("=" * 70)

    case = create_test_case()
    model = build_model_without_hard_limits(case)

    # Add penalized loss objective
    objectives.add_penalized_loss_objective(
        model,
        voltage_weight=1e4,
        thermal_weight=1e3,
        generator_weight=1e3,
        battery_weight=0.0,  # No batteries in test case
        soc_weight=0.0,
    )

    success = solve_model(model, "ipopt")
    if success:
        results = extract_results(model)
        print_results(results, "Soft Constraints (Penalties) + Loss Objective")
    else:
        print("Failed to solve model with soft constraints")

    return success


def test_combined_constraints():
    """Test with both hard and soft constraints."""
    print("\n" + "=" * 70)
    print("TEST 3: Combined Hard + Soft Constraints")
    print("=" * 70)

    case = create_test_case()
    model = build_model_with_hard_constraints(case)

    # Add penalized objective (penalties act as regularization)
    objectives.add_penalized_loss_objective(
        model,
        voltage_weight=1e2,  # Lower weight since hard constraints exist
        thermal_weight=1e2,
        generator_weight=1e2,
        battery_weight=0.0,
        soc_weight=0.0,
    )

    success = solve_model(model, "ipopt")
    if success:
        results = extract_results(model)
        print_results(results, "Hard + Soft Constraints + Loss Objective")
    else:
        print("Failed to solve model with combined constraints")

    return success


def test_custom_penalized_objective():
    """Test with custom primary objective and selective penalties."""
    print("\n" + "=" * 70)
    print("TEST 4: Custom Penalized Objective (Minimize Substation Power)")
    print("=" * 70)

    case = create_test_case()
    model = build_model_without_hard_limits(case)

    # Create custom objective: minimize substation power + voltage penalty
    obj = objectives.create_penalized_objective(
        objectives.substation_power_objective_rule,
        voltage_weight=1e4,
        generator_weight=1e3,
    )
    objectives.set_objective(model, obj)

    success = solve_model(model, "ipopt")
    if success:
        results = extract_results(model)
        print_results(results, "Custom: Substation Power + Voltage Penalty")
    else:
        print("Failed to solve model with custom objective")

    return success


def test_infeasible_case():
    """Test penalty behavior when hard constraints would be infeasible."""
    print("\n" + "=" * 70)
    print("TEST 5: Infeasible Case (High Load, Penalties Allow Violation)")
    print("=" * 70)

    # Create case with very high load (would be infeasible with hard constraints)
    case = create_test_case()
    case.bus_data.loc[:, ["pl_a", "pl_b", "pl_c"]] *= 1.5  # Increase load significantly

    model = build_model_without_hard_limits(case)

    # Add penalized objective - should find solution with violations
    objectives.add_penalized_loss_objective(
        model,
        voltage_weight=1e4,
        thermal_weight=1e3,
        generator_weight=1e3,
        battery_weight=1e3,
        soc_weight=1e3,
    )

    success = solve_model(model, "ipopt")
    if success:
        results = extract_results(model)
        print_results(results, "High Load Case (Violations Expected)")

        # Get the pure loss objective (without penalties)
        pure_loss = pyo.value(objectives.loss_objective_rule(model))
        total_obj = results["objective_value"]
        print("\nObjective Breakdown:")
        print("-" * 40)
        print(f"  Pure loss objective: {pure_loss:.6f}")
        print(f"  Total objective (with penalties): {total_obj:.6f}")
        print(f"  Penalty contribution: {total_obj - pure_loss:.6f}")

        # Check for violations
        print("\nViolation Summary:")
        print("-" * 40)
        undervoltage_count = sum(
            1 for v in results["voltages"].values() if v is not None and v < 0.95
        )
        overvoltage_count = sum(
            1 for v in results["voltages"].values() if v is not None and v > 1.05
        )
        print(f"  Undervoltage violations: {undervoltage_count}")
        print(f"  Overvoltage violations: {overvoltage_count}")
    else:
        print("Failed to solve high-load model")

    return success


def test_weight_sensitivity():
    """Test how different penalty weights affect the solution."""
    print("\n" + "=" * 70)
    print("TEST 6: Weight Sensitivity Analysis")
    print("=" * 70)

    case = create_test_case()
    # Increase load slightly to create some voltage stress

    weights_to_test = [1e1, 1e2, 1e3, 1e4, 1e5]
    case.bus_data.loc[:, ["pl_a", "pl_b", "pl_c"]] *= 1.2  # Increase load significantly

    print("\nVoltage penalty weight sensitivity:")
    print("-" * 60)
    print(f"{'Weight':<12} {'Obj Value':<15} {'Min Voltage':<15} {'Max Voltage':<15}")
    print("-" * 60)

    for weight in weights_to_test:
        model = build_model_without_hard_limits(case)
        objectives.add_penalized_loss_objective(
            model,
            voltage_weight=weight,
            thermal_weight=1e3,
            generator_weight=1e3,
        )

        success = solve_model(model, "ipopt")
        if success:
            results = extract_results(model)
            voltages = [v for v in results["voltages"].values() if v is not None]
            min_v = min(voltages) if voltages else None
            max_v = max(voltages) if voltages else None
            obj_val = results["objective_value"]
            print(f"{weight:<12.0e} {obj_val:<15.6f} {min_v:<15.4f} {max_v:<15.4f}")
        else:
            print(f"{weight:<12.0e} {'FAILED':<15}")

    return success


def main():
    """Run all tests."""
    print("\n" + "#" * 70)
    print("# PENALIZED OBJECTIVE FUNCTION TESTS")
    print("#" * 70)

    # Check for IPOPT solver
    solver = pyo.SolverFactory("ipopt")
    if solver is None or not solver.available():
        print("\nWARNING: IPOPT solver not available.")
        print("Please install IPOPT to run these tests.")
        print("Try: conda install -c conda-forge ipopt")
        return

    tests = [
        # ("Hard Constraints Only", test_hard_constraints_only),
        # ("Soft Constraints Only", test_soft_constraints_only),
        # ("Combined Constraints", test_combined_constraints),
        # ("Custom Penalized Objective", test_custom_penalized_objective),
        ("Infeasible Case", test_infeasible_case),
        # ("Weight Sensitivity", test_weight_sensitivity),
    ]

    results_summary = []
    for name, test_func in tests:
        # try:
        success = test_func()
        results_summary.append((name, "PASSED" if success else "FAILED"))
        # except Exception as e:
        #     print(f"\nError in {name}: {e}")
        #     results_summary.append((name, f"ERROR: {e}"))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, status in results_summary:
        print(f"  {name:<40} {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
