import numpy as np
import pandas as pd
import pytest

import distopf as opf
from distopf.fbs import (
    FBS,
    _apply_boundary_load_setpoints,
    _apply_gen_setpoints_to_case,
    _apply_schedule_to_case,
    _replay_periods,
)


class _Case:
    def __init__(self, bus_data, schedules, n_steps=1, start_step=0):
        self.bus_data = bus_data
        self.schedules = schedules
        self.n_steps = n_steps
        self.start_step = start_step


def test_native_battery_is_split_over_declared_phases_and_affects_current():
    fbs = FBS.__new__(FBS)
    fbs.bat_data = pd.DataFrame([{"id": 2, "phases": "b", "p": 0.3, "q": -0.1}])
    fbs.phase_connections = {2: [1]}
    batteries = fbs._build_node_batteries()
    np.testing.assert_allclose(batteries[2][1], 0.3 - 0.1j)
    np.testing.assert_allclose(batteries[2][[0, 2, 3, 4]], 0)

    fbs.node_batteries = batteries
    fbs.node_loads = {}
    fbs.node_generations = {}
    fbs.node_capacitors = {}
    fbs.bus_data = pd.DataFrame([{"id": 2, "cvr_p": 0.0, "cvr_q": 0.0}])
    current = fbs._calculate_node_injection_current(2, np.ones(5, dtype=complex))
    np.testing.assert_allclose(current[1], 0.3 + 0.1j)


def test_fbs_applies_cvr_to_regular_phase_loads():
    fbs = FBS.__new__(FBS)
    fbs.node_loads = {2: np.array([1.0 + 0.4j, 0, 0, 0, 0, 0], dtype=complex)}
    fbs.node_generations = {}
    fbs.node_capacitors = {}
    fbs.node_batteries = {}
    fbs.phase_connections = {2: [0]}
    fbs.bus_data = pd.DataFrame([{"id": 2, "cvr_p": 0.8, "cvr_q": 0.5}])
    voltage = np.array([0.9, 1, 1, 0, 0], dtype=complex)

    current = fbs._calculate_node_injection_current(2, voltage)
    expected_load = 0.924 + 0.381j
    np.testing.assert_allclose(current[0], -np.conj(expected_load / voltage[0]))


def test_fbs_applies_cvr_to_secondary_legs_and_pair_loads():
    fbs = FBS.__new__(FBS)
    fbs.node_loads = {2: np.array([0, 0, 0, 0.2 + 0.1j, 0.3 + 0.15j, 1.0 + 0.4j])}
    fbs.node_generations = {}
    fbs.bus_data = pd.DataFrame([{"id": 2, "cvr_p": 0.8, "cvr_q": 0.5}])
    voltage = np.array([0, 0, 0, 0.8, 1.0], dtype=complex)

    current = fbs._calculate_triplex_node_injection_current(2, voltage)[3:]
    leg_v2_mean = (abs(voltage[3]) ** 2 + abs(voltage[4]) ** 2) / 2
    load_power = np.array(
        [
            0.2 * (1 + 0.8 / 2 * (abs(voltage[3]) ** 2 - 1))
            + 1j * 0.1 * (1 + 0.5 / 2 * (abs(voltage[3]) ** 2 - 1)),
            0.3 * (1 + 0.8 / 2 * (abs(voltage[4]) ** 2 - 1))
            + 1j * 0.15 * (1 + 0.5 / 2 * (abs(voltage[4]) ** 2 - 1)),
            1.0 * (1 + 0.8 / 2 * (leg_v2_mean - 1))
            + 1j * 0.4 * (1 + 0.5 / 2 * (leg_v2_mean - 1)),
        ]
    )
    load_voltage = np.array([voltage[3], voltage[4], voltage[3] + voltage[4]])
    load_current = np.conj(load_power / load_voltage)
    expected = -np.array([[1, 0, 1], [0, -1, -1]]) @ load_current
    np.testing.assert_allclose(current, expected)


def test_schedule_replay_uses_phase_specific_columns():
    bus = pd.DataFrame([{"id": 1, "load_shape": "default", "pl_a": 2.0, "ql_a": 3.0}])
    schedules = pd.DataFrame(
        [{"time": 0, "default.a.p": 0.25, "default.a.q": 0.5}]
    ).set_index("time")
    case = _Case(bus, schedules)
    _apply_schedule_to_case(case, 0)
    assert case.bus_data.at[0, "pl_a"] == 0.5
    assert case.bus_data.at[0, "ql_a"] == 1.5


def test_replay_periods_uses_case_horizon_without_frames_or_schedules():
    case = _Case(pd.DataFrame(), pd.DataFrame(), n_steps=3, start_step=4)
    assert _replay_periods(case, [None, None]) == [4, 5, 6]


def test_gen_setpoints_are_noop_without_generators():
    case = _Case(pd.DataFrame(), pd.DataFrame())
    case.gen_data = None
    _apply_gen_setpoints_to_case(case, pd.DataFrame({"id": [1], "a": [0.1]}), None)

    case.gen_data = pd.DataFrame()
    _apply_gen_setpoints_to_case(case, None, pd.DataFrame({"id": [1], "a": [0.1]}))


def test_schedule_replay_scales_s1s2_loads_gen_p_and_swing_voltage():
    bus = pd.DataFrame(
        [
            {
                "id": 1,
                "bus_type": "SWING",
                "load_shape": "",
                "v_a": 1.0,
                "v_b": 1.0,
                "v_c": 1.0,
            },
            {
                "id": 2,
                "bus_type": "PQ",
                "load_shape": "M",
                "v_a": 1.0,
                "v_b": 1.0,
                "v_c": 1.0,
                "pl_s1": 1.0,
                "pl_s2": 2.0,
                "pl_s1s2": 4.0,
                "ql_s1s2": 2.0,
            },
        ]
    )
    schedules = pd.DataFrame(
        [{"time": 0, "M": 0.5, "PV": 0.25, "v_a": 1.02, "v_b": np.nan, "v_c": 0.99}]
    ).set_index("time", drop=False)
    case = _Case(bus, schedules)
    case.gen_data = pd.DataFrame([{"id": 2, "p_a": 8.0, "q_a": 3.0, "gen_shape": "PV"}])
    _apply_schedule_to_case(case, 0)
    assert case.bus_data.at[1, "pl_s1s2"] == 2.0
    assert case.bus_data.at[1, "ql_s1s2"] == 1.0
    assert case.bus_data.at[1, "pl_s1"] == 0.5
    assert case.bus_data[["v_a", "v_b", "v_c"]].iloc[0].tolist() == [1.02, 1.0, 0.99]
    assert case.bus_data.at[1, "v_a"] == 1.0
    assert case.gen_data.at[0, "p_a"] == 2.0
    assert case.gen_data.at[0, "q_a"] == 3.0


def test_gen_setpoints_must_cover_every_generator():
    case = _Case(pd.DataFrame(), pd.DataFrame())
    case.gen_data = pd.DataFrame({"id": [1, 2], "p_a": [0.0, 0.0], "q_a": [0.0, 0.0]})
    p_gens = pd.DataFrame({"id": [1], "a": [0.1]})
    with pytest.raises(ValueError, match=r"missing generator id\(s\): \[2\]"):
        _apply_gen_setpoints_to_case(case, p_gens, None)


def test_boundary_load_setpoints_only_touch_out_buses():
    bus = pd.DataFrame(
        [
            {
                "id": 1,
                "bus_type": "PQ",
                "pl_a": 1.0,
                "ql_a": 1.0,
                "cvr_p": 1.0,
                "cvr_q": 1.0,
            },
            {
                "id": 2,
                "bus_type": "OUT",
                "pl_a": 1.0,
                "ql_a": 1.0,
                "cvr_p": 1.0,
                "cvr_q": 1.0,
            },
        ]
    )
    case = _Case(bus, pd.DataFrame())
    p_loads = pd.DataFrame({"id": [1, 2], "a": [9.0, 7.0]})
    q_loads = pd.DataFrame({"id": [1, 2], "a": [9.0, 5.0]})
    _apply_boundary_load_setpoints(case, p_loads, q_loads)
    assert case.bus_data["pl_a"].tolist() == [1.0, 7.0]
    assert case.bus_data["ql_a"].tolist() == [1.0, 5.0]
    assert case.bus_data["cvr_p"].tolist() == [1.0, 0.0]


def test_replay_ignores_saved_loads_on_triplex_buses():
    case = opf.create_case(opf.CASES_DIR / "csv" / "triplex_pv")
    bus, gen = case.bus_data, case.gen_data
    folded = pd.DataFrame(
        {
            "id": bus["id"],
            "t": 0,
            "s1": bus["pl_s1"] + bus["pl_s1s2"] / 2,
            "s2": bus["pl_s2"] + bus["pl_s1s2"] / 2,
        }
    )
    gens = pd.DataFrame({"id": gen["id"], "t": 0, "s1": gen["p_s1"], "s2": gen["p_s2"]})
    result = opf.PowerFlowResult(
        active_power_loads=folded,
        reactive_power_loads=folded.assign(s1=0.0, s2=0.0),
        active_power_generation=gens,
        reactive_power_generation=gens.assign(s1=0.0, s2=0.0),
    )
    replay = opf.run_fbs_with_opf_setpoints(case, result)
    reference = case.run_fbs()
    cols = ["a", "b", "c", "s1", "s2"]
    np.testing.assert_allclose(
        replay.active_power_flows[cols].to_numpy(float),
        reference.active_power_flows[cols].to_numpy(float),
        equal_nan=True,
    )


def test_fbs_load_results_split_s1s2_between_reported_legs():
    case = opf.create_case(opf.CASES_DIR / "csv" / "triplex_pv")
    raw_loads = case.bus_data.set_index("id")

    result = case.run_fbs()

    for result_frame, prefix in (
        (result.active_power_loads, "pl"),
        (result.reactive_power_loads, "ql"),
    ):
        reported = result_frame.set_index("id")
        for phase in ("s1", "s2"):
            expected = raw_loads[f"{prefix}_{phase}"] + raw_loads[f"{prefix}_s1s2"] / 2
            np.testing.assert_allclose(
                reported[phase].reindex(expected.index), expected, equal_nan=True
            )
        assert "s1s2" not in result_frame.columns

    pd.testing.assert_frame_equal(case.bus_data.set_index("id"), raw_loads)
