# Temporary validation of distopf.fbs_new against distopf.fbs; delete once fbs_new replaces it.
import time

import numpy as np
import pandas as pd
import pytest

import distopf as opf
from distopf.fbs import FBS, run_fbs_with_opf_setpoints
from distopf.fbs_new import FBS, fbs_new_solve
from distopf.utils.input_handlers import handle_schedules_input

FRAMES = (
    "voltages",
    "voltage_angles",
    "active_power_flows",
    "reactive_power_flows",
    "currents",
    "current_angles",
    "active_power_generation",
    "reactive_power_generation",
    "active_power_loads",
    "reactive_power_loads",
)


_PHASE_COLUMNS = ("a", "b", "c", "s1", "s2")


def _case(name, **kwargs):
    return opf.create_case(opf.CASES_DIR / "csv" / name, **kwargs)


def _with_schedules(case, frame):
    case.schedules = handle_schedules_input(frame)
    return case


def assert_same(new, old, atol=1e-8):
    for name in FRAMES:
        a, b = getattr(new, name), getattr(old, name)
        if b is None:
            assert a is None, name
            continue
        assert list(a.columns) == list(b.columns), name
        if name == "current_angles":
            # The angle of an exactly zero current is a signed-zero artifact.
            zero = old.currents[list(_PHASE_COLUMNS)].to_numpy() < 1e-12
            a, b = a.copy(), b.copy()
            for k, column in enumerate(_PHASE_COLUMNS):
                a.loc[zero[:, k], column] = np.nan
                b.loc[zero[:, k], column] = np.nan
        pd.testing.assert_frame_equal(
            a.reset_index(drop=True),
            b.reset_index(drop=True),
            check_exact=False,
            check_dtype=False,
            atol=atol,
            rtol=1e-7,
            obj=name,
        )


@pytest.mark.parametrize(
    "name",
    [
        "ieee13",
        "ieee123",
        "ieee123_30der",
        "ieee34",
        "triplex_pv",
        "triplex_3ph",
        "minimal_triplex",
    ],
)
def test_single_step_matches_fbs(name):
    case = _case(name, ignore_schedule=True)
    old = FBS(case).solve(tolerance=1e-10)
    new = fbs_new_solve(case, tolerance=1e-10)
    assert new.converged == old.converged
    assert new.iterations == old.iterations
    assert_same(new, old)


def test_result_reports_iterations_and_per_step_metadata():
    result = fbs_new_solve(_case("ieee13", n_steps=3, ignore_schedule=True))
    steps = result.metadata["steps"]
    assert [s["t"] for s in steps] == [0, 1, 2]
    assert result.iterations == sum(s["iterations"] for s in steps)
    assert result.solve_time == pytest.approx(sum(s["solve_time"] for s in steps))
    assert set(result.voltages["t"]) == {0, 1, 2}


def test_multistep_with_schedules_matches_replay():
    case = _case("ieee123_bat", n_steps=24)
    old = run_fbs_with_opf_setpoints(case, tolerance=1e-10)
    new = fbs_new_solve(case, tolerance=1e-10)
    assert new.converged and old.converged
    assert_same(new, old, atol=1e-7)


def test_start_step_labels_follow_case_horizon():
    case = _case("ieee123_bat", n_steps=3, start_step=5)
    new = fbs_new_solve(case, tolerance=1e-10)
    old = run_fbs_with_opf_setpoints(case, tolerance=1e-10)
    assert sorted(new.voltages["t"].unique()) == [5, 6, 7]
    assert_same(new, old, atol=1e-7)


def test_warm_started_steps_match_independent_solves():
    case = _case("ieee123_bat", n_steps=6)
    chained = fbs_new_solve(case, tolerance=1e-10)
    for t in range(6):
        single = case.copy()
        single.start_step, single.n_steps = t, 1
        alone = fbs_new_solve(single, tolerance=1e-10)
        a = chained.voltages[chained.voltages["t"] == t].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            a, alone.voltages, check_exact=False, atol=1e-8, check_dtype=False
        )


def _triplex_schedule(times):
    return pd.DataFrame(
        {
            "time": times,
            "M.s1.p": [1.0, 0.7, 1.2][: len(times)],
            "M.s1.q": [1.0, 0.8, 1.1][: len(times)],
            "M.s2.p": [1.0, 1.3, 0.6][: len(times)],
            "M.s2.q": [1.0, 1.2, 0.9][: len(times)],
        }
    )


def test_triplex_leg_schedules_match_replay_and_scale_pair_load_by_mean():
    case = _case("triplex_pv", n_steps=3)
    case.bus_data["load_shape"] = "M"
    _with_schedules(case, _triplex_schedule([0, 1, 2]))
    case.gen_data["gen_shape"] = ""
    new = fbs_new_solve(case, tolerance=1e-10)
    old = run_fbs_with_opf_setpoints(case, tolerance=1e-10)
    assert_same(new, old, atol=1e-7)

    base = case.bus_data.set_index("id")
    loads = new.active_power_loads
    step = loads[loads["t"] == 1].set_index("id")
    mean = (0.7 + 1.3) / 2
    expected = base["pl_s1"] * 0.7 + base["pl_s1s2"] * mean / 2
    np.testing.assert_allclose(step["s1"].reindex(expected.index), expected)


def test_plain_shape_column_takes_precedence_over_phase_columns():
    case = _case("ieee13", n_steps=2)
    schedule = pd.DataFrame(
        {"time": [0, 1], "default": [0.5, 0.25], "default.a.p": [9.0, 9.0]}
    )
    _with_schedules(case, schedule)
    result = fbs_new_solve(case)
    loads = result.active_power_loads
    base = case.bus_data.set_index("id")["pl_a"]
    got = loads[loads["t"] == 1].set_index("id")["a"].reindex(base.index)
    np.testing.assert_allclose(got, base * 0.25)


def test_generator_p_is_scheduled_and_q_is_not():
    case = _case("ieee123_30der", n_steps=2)
    _with_schedules(
        case, pd.DataFrame({"time": [0, 1], "PV": [1.0, 0.5], "default": [1.0, 1.0]})
    )
    result = fbs_new_solve(case)
    gens = case.gen_data.set_index("id")
    p = result.active_power_generation
    q = result.reactive_power_generation
    p1 = p[p["t"] == 1].set_index("id")
    q1 = q[q["t"] == 1].set_index("id")
    np.testing.assert_allclose(p1["a"], gens["p_a"].reindex(p1.index) * 0.5)
    np.testing.assert_allclose(q1["a"], gens["q_a"].reindex(q1.index))


def test_unknown_gen_shape_raises_like_the_opf():
    case = _case("ieee123_30der")
    _with_schedules(case, pd.DataFrame({"time": [0], "default": [1.0]}))
    with pytest.raises(ValueError, match="gen_shape"):
        FBS(case)


def test_missing_schedule_row_raises_like_the_opf():
    case = _case("ieee13", n_steps=3)
    _with_schedules(case, pd.DataFrame({"time": [0, 1], "default": [1.0, 0.9]}))
    with pytest.raises(ValueError, match="no row"):
        FBS(case)


def test_swing_voltage_schedule_ignores_blank_cells():
    case = _case("ieee13", n_steps=2)
    schedule = pd.DataFrame(
        {"time": [0, 1], "default": [1.0, 1.0], "v_a": [1.02, np.nan]}
    )
    _with_schedules(case, schedule)
    result = fbs_new_solve(case)
    swing = case.bus_data.loc[case.bus_data["bus_type"] == "SWING", "id"].iloc[0]
    v = result.voltages[result.voltages["id"] == swing].set_index("t")["a"]
    assert v[0] == pytest.approx(1.02)
    assert v[1] == pytest.approx(float(case.bus_data["v_a"].iloc[swing - 1]))


def _timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


@pytest.mark.performance
def test_performance_ieee123_bat_24_steps():
    case = _case("ieee123_bat", n_steps=24)
    old, t_old = _timed(lambda: run_fbs_with_opf_setpoints(case))
    new, t_new = _timed(lambda: fbs_new_solve(case))
    print(
        f"\nieee123_bat x24  old={t_old:.2f}s new={t_new:.2f}s "
        f"speedup={t_old / t_new:.1f}x  iterations old={old.iterations} new={new.iterations}"
    )
    pd.testing.assert_frame_equal(
        new.voltages, old.voltages, check_exact=False, atol=1e-5, check_dtype=False
    )
    assert t_new < t_old


@pytest.mark.slow
@pytest.mark.performance
def test_performance_ieee9500_feasible_unchanging_schedule():
    case = _case("ieee9500_feasible", n_steps=24)
    new, t_new = _timed(lambda: fbs_new_solve(case))
    two = case.copy()
    two.n_steps = 2
    old, t_old = _timed(lambda: run_fbs_with_opf_setpoints(two))
    steps = new.metadata["steps"]
    print(
        f"\nieee9500_feasible  new x24={t_new:.2f}s "
        f"(iterations/step {[s['iterations'] for s in steps]})"
        f"\n                   old x2 ={t_old:.2f}s -> ~{t_old / 2 * 24:.0f}s for 24 steps"
        f"\n                   speedup ~{t_old / 2 * 24 / t_new:.0f}x"
    )
    first = new.voltages[new.voltages["t"] == 0].reset_index(drop=True)
    ref = old.voltages[old.voltages["t"] == 0].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        first, ref, check_exact=False, atol=1e-6, check_dtype=False
    )
    assert t_new < t_old
