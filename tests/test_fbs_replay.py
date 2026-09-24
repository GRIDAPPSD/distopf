import numpy as np
import pandas as pd

from distopf.fbs import (
    FBS,
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
    fbs.bat_data = pd.DataFrame(
        [{"id": 2, "phases": "b", "p": 0.3, "q": -0.1}]
    )
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


def test_schedule_replay_uses_phase_specific_columns():
    bus = pd.DataFrame(
        [{"id": 1, "load_shape": "default", "pl_a": 2.0, "ql_a": 3.0}]
    )
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
