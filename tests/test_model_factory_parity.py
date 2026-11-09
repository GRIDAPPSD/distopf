"""Parity checks for the active legacy-compatible model factories."""

from itertools import combinations_with_replacement, product

import distopf as opf

from distopf.pyomo_models.common.factory import (
    create_lindist_model as create_refactored_lindist_model,
    create_nl_branchflow_model as create_refactored_nl_branchflow_model,
)
from distopf.pyomo_models.common.data import parse_phases
from distopf.wrappers.pyomo_wrapper import PyomoWrapper


def test_lindist_factory_has_required_network_and_device_components():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
    model = create_refactored_lindist_model(case)

    for name in (
        "time_set",
        "bus_set",
        "branch_set",
        "bus_phase_set",
        "branch_phase_set",
        "gen_device_set",
        "gen_device_phase_set",
        "cap_device_set",
        "cap_device_phase_set",
        "bat_phase_set",
        "v2",
        "p_flow",
        "q_flow",
        "p_load",
        "q_load",
        "p_gen",
        "q_gen",
        "v_min",
        "v_max",
        "v_swing",
    ):
        assert hasattr(model, name), name


def test_branchflow_factory_has_formulation_specific_components():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
    model = create_refactored_nl_branchflow_model(case)

    for name in (
        "time_set",
        "bus_set",
        "branch_phase_pair_set",
        "branch_angle_phase_pair_set",
        "l_flow",
        "d",
        "v2",
        "p_flow",
        "q_flow",
    ):
        assert hasattr(model, name), name


def test_refactored_linear_factory_omits_branchflow_components():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
    model = create_refactored_lindist_model(case)

    for name in (
        "l_flow",
        "d",
        "branch_phase_pair_set",
        "branch_angle_phase_pair_set",
    ):
        assert not hasattr(model, name), name


def test_branchflow_factory_builds_phase_pair_sets_from_case_data():
    for case_name in ("ieee13", "minimal_triplex", "triplex_pv"):
        case = opf.create_case(opf.CASES_DIR / "csv" / case_name)
        model = create_refactored_nl_branchflow_model(case)

        expected_pairs = set()
        expected_angle_pairs = set()
        for _, row in case.branch_data.iterrows():
            fb, tb = int(row.fb), int(row.tb)
            phases = parse_phases(str(row.phases))
            expected_pairs.update(
                (fb, tb, phase_1 + phase_2)
                for phase_1, phase_2 in combinations_with_replacement(sorted(phases), 2)
            )
            if tb not in model.swing_bus_set:
                expected_angle_pairs.update(
                    (fb, tb, phase_1 + phase_2)
                    for phase_1, phase_2 in product(phases, repeat=2)
                )

        assert set(model.branch_phase_pair_set) == expected_pairs
        assert set(model.branch_angle_phase_pair_set) == expected_angle_pairs
        assert hasattr(model, "current_constraint")
        assert hasattr(model, "current_sqr_constraint")
        assert hasattr(model, "voltage_drop")


def test_refactored_branchflow_equations_use_current_and_angle_state():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
    linear_model = create_refactored_lindist_model(case)
    branchflow_model = create_refactored_nl_branchflow_model(case)
    fb, tb, phase = next(
        (fb, tb, phase)
        for fb, tb, phase in branchflow_model.branch_phase_set
        if phase == "a" and tb not in branchflow_model.swing_bus_set
    )
    time = next(iter(branchflow_model.time_set))

    p_balance = str(branchflow_model.power_balance_p[fb, tb, phase, time].body)
    voltage_drop = str(branchflow_model.voltage_drop[fb, tb, phase, time].body)
    linear_voltage_drop = str(linear_model.voltage_drop[fb, tb, phase, time].body)

    assert "l_flow" in p_balance
    assert "l_flow" in voltage_drop
    assert "d[" in voltage_drop
    assert "l_flow" not in linear_voltage_drop

    if branchflow_model.reg_phase_set:
        reg_fb, reg_tb, reg_phase = next(iter(branchflow_model.reg_phase_set))
        reg_voltage_drop = str(
            branchflow_model.voltage_drop[reg_fb, reg_tb, reg_phase, time].body
        )
        assert "v2_reg" in reg_voltage_drop
        assert "l_flow" in reg_voltage_drop


def test_refactored_battery_circle_respects_control_mode():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
    assert set(case.bat_data.control_variable) == {"P"}
    p_control_model = create_refactored_nl_branchflow_model(case)
    assert len(p_control_model.bat_circle_constraint) == 0

    case.bat_data.loc[:, "control_variable"] = "PQ"
    pq_control_model = create_refactored_nl_branchflow_model(case)
    assert len(pq_control_model.bat_circle_constraint) == len(
        pq_control_model.bat_phase_set
    )


def test_pyomo_wrapper_initializes_branchflow_state_from_fbs():
    import math
    import pyomo.environ as pyo

    for case_name in ("ieee13", "minimal_triplex", "triplex_pv"):
        case = opf.create_case(opf.CASES_DIR / "csv" / case_name)
        wrapper = PyomoWrapper(case)
        wrapper.model = create_refactored_nl_branchflow_model(case)
        wrapper._initialize_from_fbs()

        assert all(pyo.value(var) is not None for var in wrapper.model.l_flow.values())
        assert all(pyo.value(param) is not None for param in wrapper.model.d.values())
        assert all(
            math.isfinite(pyo.value(var)) for var in wrapper.model.l_flow.values()
        )
        assert all(
            math.isfinite(pyo.value(param)) for param in wrapper.model.d.values()
        )

        if case_name != "ieee13":
            for fb, tb, pair in wrapper.model.branch_phase_pair_set:
                phases = parse_phases(pair)
                if len(phases) == 2 and phases[0] != phases[1]:
                    for time in wrapper.model.time_set:
                        expected = math.sqrt(
                            pyo.value(
                                wrapper.model.l_flow[
                                    fb, tb, phases[0] + phases[0], time
                                ]
                            )
                            * pyo.value(
                                wrapper.model.l_flow[
                                    fb, tb, phases[1] + phases[1], time
                                ]
                            )
                        )
                        assert math.isclose(
                            pyo.value(wrapper.model.l_flow[fb, tb, pair, time]),
                            expected,
                            rel_tol=1e-12,
                        )
