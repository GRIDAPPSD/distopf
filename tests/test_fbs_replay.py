"""FBS device injections, CVR, schedules and replay of OPF setpoints."""

import re

import numpy as np
import pandas as pd
import pytest

import distopf as opf
from distopf.fbs import FBS, _replay_times

CASES = opf.CASES_DIR / "csv"


class _Case:
    def __init__(self, n_steps=1, start_step=0):
        self.n_steps = n_steps
        self.start_step = start_step


def _node(fbs: FBS, bus_id: int) -> int:
    """Row of ``bus_id`` in the solver's node-ordered arrays."""
    return int(np.flatnonzero(fbs.node_ids == bus_id)[0])


def _flat_voltages(fbs: FBS) -> np.ndarray:
    return np.ones((len(fbs.node_ids), 5), dtype=complex)


def _set_bus(case, bus_id: int, **values) -> None:
    bus = case.bus_data
    bus.loc[bus.id == bus_id, list(values)] = list(values.values())


def test_battery_is_split_over_declared_phases_and_injects_current():
    case = opf.create_case(CASES / "ieee13")
    case.bat_data = pd.DataFrame([{"id": 5, "phases": "b", "p": 0.3, "q": -0.1}])
    fbs = FBS(case)
    k = _node(fbs, 5)
    battery = int(np.flatnonzero(fbs._bat_nodes == k)[0])
    np.testing.assert_allclose(fbs._bat[battery], [0, 0.3 - 0.1j, 0, 0, 0])

    v = _flat_voltages(fbs)
    with_battery = fbs._injection(v, fbs._step(0))
    fbs._bat_t[:] = 0
    without_battery = fbs._injection(v, fbs._step(0))
    np.testing.assert_allclose(
        with_battery[k] - without_battery[k], [0, 0.3 + 0.1j, 0, 0, 0]
    )


def test_cvr_scales_phase_loads_with_voltage_squared():
    case = opf.create_case(CASES / "ieee13")
    _set_bus(case, 5, pl_a=1.0, ql_a=0.4, cvr_p=0.8, cvr_q=0.5)
    fbs = FBS(case)
    k = _node(fbs, 5)
    v = _flat_voltages(fbs)
    v[k, 0] = 0.9

    current = fbs._injection(v, fbs._step(0))[k, 0]

    # p = p0 * (1 + cvr_p / 2 * (|v|^2 - 1)), and the same for q with cvr_q
    expected_load = 0.924 + 0.381j
    np.testing.assert_allclose(current, -np.conj(expected_load / 0.9))


def test_cvr_scales_secondary_leg_and_pair_loads():
    case = opf.create_case(CASES / "minimal_triplex")
    _set_bus(
        case,
        4,
        pl_s1=0.2,
        ql_s1=0.1,
        pl_s2=0.3,
        ql_s2=0.15,
        pl_s1s2=1.0,
        ql_s1s2=0.4,
        cvr_p=0.8,
        cvr_q=0.5,
    )
    fbs = FBS(case)
    k = _node(fbs, 4)
    v1, v2 = 0.8, 1.0
    v = _flat_voltages(fbs)
    v[k, 3:5] = [v1, v2]

    current = fbs._injection(v, fbs._step(0))[k, 3:5]

    def scaled(p, q, v_squared):
        return p * (1 + 0.8 / 2 * (v_squared - 1)) + 1j * q * (
            1 + 0.5 / 2 * (v_squared - 1)
        )

    # The pair load sees the mean of the two legs' squared voltages.
    load_power = np.array(
        [
            scaled(0.2, 0.1, v1**2),
            scaled(0.3, 0.15, v2**2),
            scaled(1.0, 0.4, (v1**2 + v2**2) / 2),
        ]
    )
    load_current = np.conj(load_power / np.array([v1, v2, v1 + v2]))
    expected = -np.array([[1, 0, 1], [0, -1, -1]]) @ load_current
    np.testing.assert_allclose(current, expected)


def test_schedules_use_phase_specific_then_shape_level_columns():
    case = opf.create_case(CASES / "ieee13")
    _set_bus(case, 5, load_shape="res", pl_a=2.0, ql_a=3.0, pl_b=2.0, ql_b=3.0)
    case.schedules = pd.DataFrame(
        {
            "time": [0],
            "res.a.p": [0.25],
            "res.a.q": [0.5],
            "res.p": [0.1],
            "res.q": [0.2],
        }
    ).set_index("time", drop=False)
    fbs = FBS(case)
    k = _node(fbs, 5)

    step = fbs._step(0)

    assert step.pl[k, 0] == pytest.approx(2.0 * 0.25)  # res.a.p
    assert step.ql[k, 0] == pytest.approx(3.0 * 0.5)  # res.a.q
    assert step.pl[k, 1] == pytest.approx(2.0 * 0.1)  # no res.b.p: res.p
    assert step.ql[k, 1] == pytest.approx(3.0 * 0.2)  # no res.b.q: res.q


def test_replay_times_use_case_horizon_without_labelled_frames():
    assert _replay_times(_Case(n_steps=3, start_step=4), [None, None]) == [4, 5, 6]


@pytest.mark.parametrize("gen_data", [None, pd.DataFrame()])
def test_generator_setpoints_are_ignored_without_generators(gen_data):
    case = opf.create_case(CASES / "ieee13")
    case.gen_data = gen_data
    fbs = FBS(case)
    saved = pd.DataFrame({"id": [1], "a": [0.1]})

    fbs.apply_setpoints(p_gens=saved, q_gens=saved)

    assert fbs._gen_p_t.size == 0
    assert fbs._gen_q_t.size == 0


def test_schedules_scale_secondary_loads_generator_p_and_swing_voltage():
    case = opf.create_case(CASES / "triplex_pv")
    bus = case.bus_data
    bus["load_shape"] = "M"
    case.schedules = pd.DataFrame(
        {
            "time": [0],
            "M": [0.5],
            "PV": [0.25],
            "v_a": [1.02],
            "v_b": [np.nan],
            "v_c": [0.99],
        }
    ).set_index("time", drop=False)
    fbs = FBS(case)
    k = _node(fbs, 5)
    secondary = list(fbs._sec_idx).index(k)

    step = fbs._step(0)

    house = bus.set_index("id").loc[5]
    for part, prefix in ((step.sec_load.real, "pl"), (step.sec_load.imag, "ql")):
        nominal = house[[f"{prefix}_s1", f"{prefix}_s2", f"{prefix}_s1s2"]]
        np.testing.assert_allclose(part[secondary], 0.5 * nominal.to_numpy(float))
    np.testing.assert_allclose(fbs._gen_p_t[0], 0.25 * fbs._gen_p)
    np.testing.assert_allclose(fbs._gen_q_t[0], fbs._gen_q)  # q is not scheduled
    swing_v_b = float(bus.loc[bus.bus_type == "SWING", "v_b"].iloc[0])
    np.testing.assert_allclose(np.abs(fbs._v_swing[0, :3]), [1.02, swing_v_b, 0.99])


def test_generator_setpoints_must_cover_every_generator():
    case = opf.create_case(CASES / "triplex_pv")
    ids = case.gen_data["id"].tolist()
    fbs = FBS(case)
    missing = sorted(ids[1:])

    with pytest.raises(
        ValueError, match=re.escape(f"missing generator id(s): {missing}")
    ):
        fbs.apply_setpoints(p_gens=pd.DataFrame({"id": ids[:1], "s1": [0.001]}))


def test_load_setpoints_apply_only_at_out_buses_and_disable_their_cvr():
    case = opf.create_case(CASES / "ieee13")
    out_bus, pq_bus = 5, 6
    _set_bus(case, out_bus, bus_type="OUT")
    case.bus_data["cvr_p"] = 1.0
    case.bus_data["cvr_q"] = 1.0
    fbs = FBS(case)
    k_out, k_pq = _node(fbs, out_bus), _node(fbs, pq_bus)
    before = fbs._step(0)

    fbs.apply_setpoints(
        p_loads=pd.DataFrame({"id": [out_bus, pq_bus], "a": [7.0, 9.0]}),
        q_loads=pd.DataFrame({"id": [out_bus, pq_bus], "a": [5.0, 9.0]}),
    )
    after = fbs._step(0)

    assert (after.pl[k_out, 0], after.ql[k_out, 0]) == (7.0, 5.0)
    assert after.pl[k_pq, 0] == before.pl[k_pq, 0]
    assert after.ql[k_pq, 0] == before.ql[k_pq, 0]
    assert (fbs._cvr_p[k_out], fbs._cvr_q[k_out]) == (0.0, 0.0)
    assert (fbs._cvr_p[k_pq], fbs._cvr_q[k_pq]) == (1.0, 1.0)


def test_replay_ignores_saved_loads_on_triplex_buses():
    case = opf.create_case(CASES / "triplex_pv")
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


def test_multistep_replay_warm_starts_each_period_from_the_previous_one():
    case = opf.create_case(CASES / "ieee13", n_steps=2, ignore_schedule=True)

    replay = opf.run_fbs_with_opf_setpoints(case)

    steps = replay.metadata["steps"]
    assert replay.converged
    assert [step["t"] for step in steps] == [0, 1]
    # The periods are identical, so the second starts at the first's solution.
    assert steps[0]["iterations"] > 1
    assert steps[1]["iterations"] == 1


def test_fbs_load_results_split_s1s2_between_reported_legs():
    case = opf.create_case(CASES / "triplex_pv")
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
