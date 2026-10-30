"""FBS replay of regulator tap and capacitor switch decisions from an OPF result."""

import numpy as np
import pandas as pd
import pytest

import distopf as opf

PHASES = ["a", "b", "c"]
TAP_STEP = 0.00625


def _case(n_steps=1):
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
    case.n_steps = n_steps
    return case


def _taps_frame(case, taps_by_t):
    """Pyomo-style reg_taps: one row per regulator and period, a/b/c integer steps."""
    rows = []
    for t, taps in taps_by_t.items():
        for _, reg in case.reg_data.iterrows():
            row = {"fb": reg["fb"], "tb": reg["tb"], "t": t}
            for phase in PHASES:
                row[phase] = taps.get(phase, 0) if phase in reg["phases"] else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _with_taps(case, taps):
    out = case.copy()
    reg = out.reg_data.copy()
    for phase in PHASES:
        reg[f"tap_{phase}"] = taps.get(phase, 0)
        reg[f"ratio_{phase}"] = 1 + TAP_STEP * taps.get(phase, 0)
    out.reg_data = reg
    return out


def _caps_frame(case, states_by_t):
    """Pyomo-style u_caps: one row per capacitor and period, a/b/c in {0, 1}."""
    rows = []
    for t, state in states_by_t.items():
        for _, cap in case.cap_data.iterrows():
            row = {"id": cap["id"], "name": cap["name"], "t": t}
            for phase in PHASES:
                row[phase] = state[cap["id"]] if cap[f"q_{phase}"] != 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _with_caps(case, state):
    out = case.copy()
    cap = out.cap_data.copy()
    for index, row in cap.iterrows():
        for phase in PHASES:
            cap.at[index, f"q_{phase}"] = row[f"q_{phase}"] * state[row["id"]]
    out.cap_data = cap
    return out


def _voltages(result):
    return result.voltages[PHASES].to_numpy(float)


def _empty_result(**frames):
    return opf.PowerFlowResult(**frames)


def test_constant_taps_match_case_with_those_taps():
    case = _case()
    taps = {"a": 4, "b": -3, "c": 7}
    replay = opf.run_fbs_with_opf_setpoints(
        case, _empty_result(reg_taps=_taps_frame(case, {0: taps})), tolerance=1e-10
    )
    expected = _with_taps(case, taps).run_fbs(tolerance=1e-10)
    np.testing.assert_allclose(
        _voltages(replay), _voltages(expected), equal_nan=True, atol=1e-8
    )
    default = case.run_fbs(tolerance=1e-10)
    assert not np.allclose(_voltages(replay), _voltages(default), equal_nan=True)


def test_time_varying_taps_match_single_step_runs():
    case = _case(n_steps=3)
    taps_by_t = {
        0: {"a": 0, "b": 0, "c": 0},
        1: {"a": 5, "b": 2, "c": -4},
        2: {"a": -6},
    }
    replay = opf.run_fbs_with_opf_setpoints(
        case, _empty_result(reg_taps=_taps_frame(case, taps_by_t)), tolerance=1e-10
    )
    for t, taps in taps_by_t.items():
        single = _case()
        single.start_step = t
        single.schedules = case.schedules
        expected = _with_taps(single, taps).run_fbs(tolerance=1e-10)
        got = replay.voltages[replay.voltages["t"] == t]
        np.testing.assert_allclose(
            got[PHASES].to_numpy(float), _voltages(expected), equal_nan=True, atol=1e-8
        )


def test_matrix_style_taps_without_time_apply_to_every_period():
    case = _case(n_steps=2)
    taps = {"a": 3, "b": 3, "c": 3}
    matrix_taps = case.reg_data[["fb", "tb"]].copy()
    for phase in PHASES:
        matrix_taps[f"tap_{phase}"] = taps[phase]
    replay = opf.run_fbs_with_opf_setpoints(
        case, _empty_result(reg_taps=matrix_taps), tolerance=1e-10
    )
    expected = _with_taps(case, taps).run_fbs(tolerance=1e-10)
    np.testing.assert_allclose(
        _voltages(replay), _voltages(expected), equal_nan=True, atol=1e-8
    )


def test_flows_use_the_tap_of_each_period():
    case = _case(n_steps=2)
    taps_by_t = {0: {"a": 0, "b": 0, "c": 0}, 1: {"a": 8, "b": 8, "c": 8}}
    replay = opf.run_fbs_with_opf_setpoints(
        case, _empty_result(reg_taps=_taps_frame(case, taps_by_t)), tolerance=1e-10
    )
    single = _with_taps(_case(), taps_by_t[1]).run_fbs(tolerance=1e-10)
    flows = replay.active_power_flows
    got = flows[flows["t"] == 1][PHASES].to_numpy(float)
    np.testing.assert_allclose(
        got,
        single.active_power_flows[PHASES].to_numpy(float),
        equal_nan=True,
        atol=1e-8,
    )


def test_capacitor_switching_via_u_caps_matches_single_step_runs():
    case = _case(n_steps=2)
    states = {0: {10: 1, 11: 1}, 1: {10: 0, 11: 1}}
    replay = opf.run_fbs_with_opf_setpoints(
        case, _empty_result(u_caps=_caps_frame(case, states)), tolerance=1e-10
    )
    for t, state in states.items():
        single = _case()
        single.start_step = t
        single.schedules = case.schedules
        expected = _with_caps(single, state).run_fbs(tolerance=1e-10)
        got = replay.voltages[replay.voltages["t"] == t]
        np.testing.assert_allclose(
            got[PHASES].to_numpy(float), _voltages(expected), equal_nan=True, atol=1e-8
        )


def test_u_caps_tolerate_solver_noise():
    case = _case()
    noisy = _caps_frame(case, {0: {10: 0, 11: 1}})
    noisy[PHASES] = noisy[PHASES].replace({0.0: 1e-9, 1.0: 0.9999})
    replay = opf.run_fbs_with_opf_setpoints(
        case, _empty_result(u_caps=noisy), tolerance=1e-10
    )
    expected = _with_caps(case, {10: 0, 11: 1}).run_fbs(tolerance=1e-10)
    np.testing.assert_allclose(
        _voltages(replay), _voltages(expected), equal_nan=True, atol=1e-8
    )


def _q_caps_frame(case, states_by_t, scale=1.0):
    rows = []
    for t, state in states_by_t.items():
        for _, cap in case.cap_data.iterrows():
            row = {"id": cap["id"], "name": cap["name"], "t": t}
            for phase in PHASES:
                row[phase] = cap[f"q_{phase}"] * state[cap["id"]] * scale
            rows.append(row)
    return pd.DataFrame(rows)


def test_capacitor_state_is_inferred_from_reactive_power_without_u_caps():
    case = _case(n_steps=2)
    states = {0: {10: 1, 11: 0}, 1: {10: 0, 11: 1}}
    # On capacitors report nominal * v^2, so scale by a plausible v^2 below 1.
    q_caps = _q_caps_frame(case, states, scale=0.96)
    replay = opf.run_fbs_with_opf_setpoints(
        case, _empty_result(capacitor_reactive_power=q_caps), tolerance=1e-10
    )
    for t, state in states.items():
        single = _case()
        single.start_step = t
        single.schedules = case.schedules
        expected = _with_caps(single, state).run_fbs(tolerance=1e-10)
        got = replay.voltages[replay.voltages["t"] == t]
        np.testing.assert_allclose(
            got[PHASES].to_numpy(float), _voltages(expected), equal_nan=True, atol=1e-8
        )


def test_reactive_power_values_are_not_replayed_when_u_caps_given():
    case = _case()
    states = {0: {10: 1, 11: 0}}
    garbage = _q_caps_frame(case, {0: {10: 0, 11: 1}}, scale=5.0)
    with_garbage = opf.run_fbs_with_opf_setpoints(
        case,
        _empty_result(
            u_caps=_caps_frame(case, states), capacitor_reactive_power=garbage
        ),
        tolerance=1e-10,
    )
    clean = opf.run_fbs_with_opf_setpoints(
        case, _empty_result(u_caps=_caps_frame(case, states)), tolerance=1e-10
    )
    np.testing.assert_allclose(
        _voltages(with_garbage), _voltages(clean), equal_nan=True, atol=1e-12
    )


def test_unknown_regulator_or_capacitor_ids_raise():
    case = _case()
    bad_taps = _taps_frame(case, {0: {"a": 1}})
    bad_taps.loc[0, "tb"] = 999
    with pytest.raises(ValueError, match="unknown regulator"):
        opf.run_fbs_with_opf_setpoints(case, _empty_result(reg_taps=bad_taps))
    bad_caps = _caps_frame(case, {0: {10: 1, 11: 1}})
    bad_caps.loc[0, "id"] = 999
    with pytest.raises(ValueError, match="unknown capacitor"):
        opf.run_fbs_with_opf_setpoints(case, _empty_result(u_caps=bad_caps))


def test_missing_periods_keep_case_values_and_case_is_not_mutated():
    case = _case(n_steps=2)
    reg_before = case.reg_data.copy()
    cap_before = case.cap_data.copy()
    taps = _taps_frame(case, {0: {"a": 6, "b": 6, "c": 6}, 1: {"a": 6, "b": 6, "c": 6}})
    taps = taps[~((taps["t"] == 0) & (taps["tb"] != 2))]
    replay = opf.run_fbs_with_opf_setpoints(
        case, _empty_result(reg_taps=taps), tolerance=1e-10
    )
    partial = case.copy()
    reg = partial.reg_data.copy()
    for phase in PHASES:
        reg.loc[reg["tb"] == 2, f"tap_{phase}"] = 6
        reg.loc[reg["tb"] == 2, f"ratio_{phase}"] = 1 + TAP_STEP * 6
    partial.reg_data = reg
    expected = partial.run_fbs(tolerance=1e-10)
    first = replay.voltages[replay.voltages["t"] == 0][PHASES].to_numpy(float)
    reference = expected.voltages[expected.voltages["t"] == 0][PHASES].to_numpy(float)
    np.testing.assert_allclose(first, reference, equal_nan=True, atol=1e-8)
    pd.testing.assert_frame_equal(case.reg_data, reg_before)
    pd.testing.assert_frame_equal(case.cap_data, cap_before)


def test_result_without_device_states_is_unchanged():
    case = _case()
    replay = opf.run_fbs_with_opf_setpoints(case, tolerance=1e-10)
    default = case.run_fbs(tolerance=1e-10)
    np.testing.assert_allclose(
        _voltages(replay), _voltages(default), equal_nan=True, atol=1e-12
    )


@pytest.mark.parametrize("wrapper", ["pyomo"])
def test_opf_round_trip_applies_the_decisions_the_opf_made(wrapper):
    case = _case(n_steps=2)
    result = case.run_opf(
        objective=None,
        wrapper=wrapper,
        control_variable="",
        control_regulators=True,
        control_capacitors=True,
        solver="highs",
    )
    assert result.reg_taps is not None
    assert result.u_caps is not None
    assert result.tap_ratios is not None
    np.testing.assert_allclose(
        result.tap_ratios[PHASES].to_numpy(float),
        1 + TAP_STEP * result.reg_taps[PHASES].to_numpy(float),
        equal_nan=True,
    )
    replay = opf.run_fbs_with_opf_setpoints(case, result, tolerance=1e-10)
    assert replay.converged
    for t in (0, 1):
        taps = result.reg_taps[result.reg_taps["t"] == t]
        state = result.u_caps[result.u_caps["t"] == t].set_index("id")
        expected = _case()
        expected.start_step = t
        expected.schedules = case.schedules
        reg = expected.reg_data.copy()
        for index, row in reg.iterrows():
            chosen = taps[taps["tb"] == row["tb"]].iloc[0]
            for phase in PHASES:
                if phase in row["phases"]:
                    reg.at[index, f"tap_{phase}"] = chosen[phase]
                    reg.at[index, f"ratio_{phase}"] = 1 + TAP_STEP * chosen[phase]
        expected.reg_data = reg
        cap = expected.cap_data.copy()
        for index, row in cap.iterrows():
            for phase in PHASES:
                on = state.at[row["id"], phase] > 0.5
                cap.at[index, f"q_{phase}"] = row[f"q_{phase}"] * float(on)
        expected.cap_data = cap
        exact = expected.run_fbs(tolerance=1e-10)
        got = replay.voltages[replay.voltages["t"] == t]
        np.testing.assert_allclose(
            got[PHASES].to_numpy(float),
            exact.voltages[PHASES].to_numpy(float),
            equal_nan=True,
            atol=1e-6,
        )


def test_saved_results_round_trip_applies_taps_and_capacitors(tmp_path):
    case = _case(n_steps=2)
    taps_by_t = {0: {"a": 4, "b": 4, "c": 4}, 1: {"a": -2, "b": -2, "c": -2}}
    states = {0: {10: 0, 11: 1}, 1: {10: 1, 11: 1}}
    result = _empty_result(
        reg_taps=_taps_frame(case, taps_by_t),
        u_caps=_caps_frame(case, states),
        active_power_generation=pd.DataFrame(columns=["id", "t", "a", "b", "c"]),
        reactive_power_generation=pd.DataFrame(columns=["id", "t", "a", "b", "c"]),
    )
    result.reg_taps.to_csv(tmp_path / "reg_taps.csv", index=False)
    result.u_caps.to_csv(tmp_path / "u_caps.csv", index=False)
    result.active_power_generation.to_csv(
        tmp_path / "active_power_generation.csv", index=False
    )
    result.reactive_power_generation.to_csv(
        tmp_path / "reactive_power_generation.csv", index=False
    )
    saved = opf.fbs.run_fbs_from_saved_results(tmp_path, case=case, tolerance=1e-10)
    direct = opf.run_fbs_with_opf_setpoints(case, result, tolerance=1e-10)
    np.testing.assert_allclose(
        _voltages(saved), _voltages(direct), equal_nan=True, atol=1e-12
    )


def test_matrix_bess_cap_mi_result_reports_capacitor_states_per_period():
    case = _case(n_steps=1)
    case.modify(v_min=0.90, v_max=1.10)
    result = case.run_opf(
        objective=None,
        wrapper="matrix_bess",
        formulation="lindist_cap_mi_mp",
        control_variable="",
    )
    assert result.u_caps is not None and "t" in result.u_caps.columns
    assert result.z_caps is not None
    assert result.capacitor_reactive_power is not None
    replay = opf.run_fbs_with_opf_setpoints(case, result, tolerance=1e-10)
    assert replay.converged


def test_matrix_result_reports_capacitor_reactive_power():
    result = _case().run_opf(objective=None, wrapper="matrix", control_variable="")
    assert result.capacitor_reactive_power is not None
    assert len(result.capacitor_reactive_power) > 0


def test_cvxpy_result_decisions_replay_through_fbs():
    case = _case(n_steps=2)
    result = case.run_opf(
        objective=None,
        wrapper="cvxpy",
        solver="highs",
        control_variable="",
        control_regulators=True,
        control_capacitors=True,
    )
    assert result.reg_taps is not None and "t" in result.reg_taps.columns
    assert result.u_caps is not None and "t" in result.u_caps.columns
    assert result.tap_ratios is not None
    replay = opf.run_fbs_with_opf_setpoints(case, result, tolerance=1e-10)
    assert replay.converged
    assert set(replay.voltages["t"]) == {0, 1}
